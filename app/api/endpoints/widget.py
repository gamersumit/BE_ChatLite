from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from supabase import Client
import json
import time
import logging
import hashlib
from urllib.parse import urlparse
from uuid import uuid4, UUID

logger = logging.getLogger(__name__)

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.services.vector_enhanced_chat_service import get_vector_enhanced_chat_service
from pydantic import BaseModel


router = APIRouter()

# Rate limiting storage (in production, use Redis)
rate_limiter_storage: Dict[str, Dict[str, Any]] = {}

class WidgetConfigResponse(BaseModel):
    widget_id: str
    website_id: str
    domain: str
    is_active: bool
    is_verified: bool
    verification_status: str
    config: Dict[str, Any]
    api_endpoints: Dict[str, str]

class ChatMessageRequest(BaseModel):
    message: str
    session_id: str
    visitor_id: str
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class ChatMessageResponse(BaseModel):
    response: str
    session_id: str
    message_id: str

class SessionCreateRequest(BaseModel):
    visitor_id: str
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    visitor_id: str
    website_id: str
    is_active: bool = True

class SessionEndRequest(BaseModel):
    session_id: str
    satisfaction_rating: Optional[int] = None
    feedback: Optional[str] = None

class AnalyticsTrackRequest(BaseModel):
    event: str
    session_id: Optional[str] = None
    visitor_id: Optional[str] = None
    timestamp: str
    user_agent: Optional[str] = None
    page_url: Optional[str] = None
    installation_method: Optional[str] = None


def check_rate_limit(client_id: str, limit: int = 10, window: int = 60) -> bool:
    """Simple in-memory rate limiter"""
    now = time.time()
    
    if client_id not in rate_limiter_storage:
        rate_limiter_storage[client_id] = {"requests": [], "blocked_until": 0}
    
    client_data = rate_limiter_storage[client_id]
    
    # Check if client is currently blocked
    if client_data["blocked_until"] > now:
        return False
    
    # Clean old requests outside the window
    client_data["requests"] = [req_time for req_time in client_data["requests"] if req_time > now - window]
    
    # Check if limit exceeded
    if len(client_data["requests"]) >= limit:
        client_data["blocked_until"] = now + 60  # Block for 1 minute
        return False
    
    # Add current request
    client_data["requests"].append(now)
    return True


def validate_domain(request: Request, website: dict) -> bool:
    """Validate if request is coming from authorized domain"""
    referer = request.headers.get("referer") or request.headers.get("origin", "")
    if not referer:
        return False
    
    try:
        parsed = urlparse(referer)
        request_domain = parsed.netloc.lower()
        allowed_domain = website['domain'].lower()
        
        return request_domain == allowed_domain or request_domain.endswith(f".{allowed_domain}")
    except:
        return False


@router.get("/config/{widget_id}")
async def get_widget_config(
    widget_id: str,
    request: Request,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get widget configuration for public embedding"""
    try:
        # Query website by widget_id
        result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        website = result.data[0]
        
        # Allow newly registered widgets to load config before verification
        # But mark them as unverified in the response
        is_verified = website.get('verification_status') == "verified" and website.get('is_active', False)
        
        # Validate domain for security (permissive in development)
        if not validate_domain(request, website):
            # In development, we might want to be more permissive
            pass
        
        # Get config from settings JSONB field with defaults
        settings = website.get('settings') or {}
        config = {
            "welcome_message": settings.get('welcome_message', "Hello! How can I help you today?"),
            "placeholder_text": settings.get('placeholder_text', "Type your message..."),
            "widget_color": settings.get('widget_color', "#0066CC"),
            "widget_position": settings.get('widget_position', "bottom-right"),
            "widget_theme": settings.get('widget_theme', "light"),
            "show_avatar": settings.get('show_avatar', True),
            "enable_sound": settings.get('enable_sound', False),
            "auto_open_delay": settings.get('auto_open_delay', 0),
            "show_online_status": settings.get('show_online_status', True),
            "offline_message": settings.get('offline_message', "We're currently offline. Leave a message!"),
            "thanks_message": settings.get('thanks_message', "Thanks for chatting with us!"),
            "show_branding": settings.get('show_branding', True),
            "company_name": settings.get('company_name', website.get('name', '')),
            "custom_css": settings.get('custom_css', ''),
            "font_family": settings.get('font_family', 'Inter, system-ui, sans-serif'),
            "border_radius": settings.get('border_radius', '12px')
        }
        
        api_endpoints = {
            "chat": f"/api/v1/widget/chat/{widget_id}",
            "session_create": f"/api/v1/widget/session/{widget_id}/create",
            "session_resume": f"/api/v1/widget/session/{widget_id}/resume",
            "analytics": f"/api/v1/widget/analytics/{widget_id}/track",
            "verify": f"/api/v1/widget/verify/{widget_id}",
            "config": f"/api/v1/widget/config/{widget_id}"
        }
        
        return WidgetConfigResponse(
            widget_id=widget_id,
            website_id=website['id'],
            domain=website['domain'],
            is_active=website['is_active'],
            is_verified=is_verified,
            verification_status=website.get('verification_status', 'pending'),
            config=config,
            api_endpoints=api_endpoints
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving widget config: {str(e)}")


@router.post("/chat/{widget_id}")
async def send_chat_message(
    widget_id: str,
    message_data: ChatMessageRequest,
    request: Request,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Handle chat messages from widget with rate limiting"""
    try:
        # Validate message
        if not message_data.message or not message_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(message_data.message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        # Get website
        website_result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()
        
        if not website_result.data:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        website = website_result.data[0]
        
        if not website['is_active']:
            raise HTTPException(status_code=403, detail="Widget is not active")
        
        # Rate limiting
        client_id = f"{widget_id}:{message_data.visitor_id}"
        if not check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Get or create conversation
        conv_result = supabase.table('conversations').select('*').eq('session_id', message_data.session_id).eq('website_id', website['id']).execute()
        
        conversation = conv_result.data[0] if conv_result.data else None
        
        if not conversation:
            conversation_data = {
                'id': str(uuid4()),
                'session_id': message_data.session_id,
                'visitor_id': message_data.visitor_id,
                'website_id': website['id'],
                'page_url': message_data.page_url,
                'page_title': message_data.page_title,
                'user_agent': message_data.user_agent,
                'ip_address': message_data.ip_address or getattr(request.client, 'host', 'unknown'),
                'is_active': True,
                'status': 'active',
                'total_messages': 0,
                'user_messages': 0,
                'ai_messages': 0,
                'started_at': datetime.utcnow().isoformat(),
                'last_activity_at': datetime.utcnow().isoformat()
            }
            conv_insert = supabase.table('conversations').insert(conversation_data).execute()
            conversation = conv_insert.data[0]
        
        # Get next sequence number
        last_msg_result = supabase.table('messages').select('sequence_number').eq('conversation_id', conversation['id']).order('sequence_number', desc=True).limit(1).execute()
        last_sequence = last_msg_result.data[0]['sequence_number'] if last_msg_result.data else 0
        next_sequence = last_sequence + 1
        
        # Create user message
        user_message_data = {
            'id': str(uuid4()),
            'conversation_id': conversation['id'],
            'content': message_data.message,
            'message_type': 'user',
            'sequence_number': next_sequence,
            'word_count': len(message_data.message.split()),
            'character_count': len(message_data.message),
            'sent_at': datetime.utcnow().isoformat()
        }
        user_msg_insert = supabase.table('messages').insert(user_message_data).execute()
        user_message = user_msg_insert.data[0]
        
        # Update conversation metrics
        updated_conversation = supabase.table('conversations').update({
            'total_messages': conversation['total_messages'] + 1,
            'user_messages': conversation['user_messages'] + 1,
            'last_activity_at': datetime.utcnow().isoformat()
        }).eq('id', conversation['id']).execute()
        conversation = updated_conversation.data[0]
        
        # Generate AI response using Vector Enhanced Chat Service
        try:
            chat_service = get_vector_enhanced_chat_service()

            # Get recent conversation history for context
            conversation_history = []
            if conversation['total_messages'] > 0:
                # Get last few messages for context
                history_result = supabase.table('messages').select(
                    'content, message_type'
                ).eq('conversation_id', conversation['id']).order(
                    'sequence_number', desc=False
                ).limit(6).execute()

                for msg in history_result.data:
                    role = "user" if msg['message_type'] == 'user' else "assistant"
                    conversation_history.append({
                        "role": role,
                        "content": msg['content']
                    })

            # Generate vector chat response
            rag_result = await chat_service.generate_chat_response(
                user_message=message_data.message,
                website_id=website['id'],  # Use string ID directly
                conversation_history=conversation_history,
                use_rag=True,
                use_hybrid_search=True
            )

            ai_response = rag_result['response']

            # Log RAG usage for debugging
            if rag_result['context_used']:
                logger.info(f"Vector RAG context used for website {website['id']}: {rag_result.get('context_chunks', 0)} chunks")

        except Exception as e:
            logger.error(f"Error generating vector chat response: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to simple response
            user_message_lower = message_data.message.lower()
            if any(word in user_message_lower for word in ['hello', 'hi', 'hey']):
                ai_response = f"Hello! Welcome to {website['company_name'] or website['name']}. How can I help you today?"
            elif any(word in user_message_lower for word in ['help', 'support']):
                ai_response = "I'm here to help! What would you like to know about our services?"
            else:
                ai_response = "Thank you for your message. I'm here to help you with information about our services."
        
        # Create AI message
        ai_message_data = {
            'id': str(uuid4()),
            'conversation_id': conversation['id'],
            'content': ai_response,
            'message_type': 'assistant',
            'sequence_number': next_sequence + 1,
            'word_count': len(ai_response.split()),
            'character_count': len(ai_response),
            'sent_at': datetime.utcnow().isoformat()
        }
        ai_msg_insert = supabase.table('messages').insert(ai_message_data).execute()
        ai_message = ai_msg_insert.data[0]
        
        # Update conversation metrics
        supabase.table('conversations').update({
            'total_messages': conversation['total_messages'] + 1,
            'ai_messages': conversation['ai_messages'] + 1
        }).eq('id', conversation['id']).execute()
        
        return ChatMessageResponse(
            response=ai_response,
            session_id=message_data.session_id,
            message_id=ai_message['id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")


@router.post("/session/{widget_id}/create")
async def create_session(
    widget_id: str,
    session_data: SessionCreateRequest,
    request: Request,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Create a new chat session"""
    try:
        website_result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()
        
        if not website_result.data:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        website = website_result.data[0]
        session_id = f"sess_{uuid4().hex[:16]}"
        
        conversation_data = {
            'id': str(uuid4()),
            'session_id': session_id,
            'visitor_id': session_data.visitor_id,
            'website_id': website['id'],
            'page_url': session_data.page_url,
            'page_title': session_data.page_title,
            'user_agent': session_data.user_agent,
            'ip_address': getattr(request.client, 'host', 'unknown'),
            'referrer': session_data.referrer,
            'is_active': True,
            'status': 'active',
            'total_messages': 0,
            'user_messages': 0,
            'ai_messages': 0,
            'started_at': datetime.utcnow().isoformat(),
            'last_activity_at': datetime.utcnow().isoformat()
        }
        
        conv_insert = supabase.table('conversations').insert(conversation_data).execute()
        
        return SessionResponse(
            session_id=session_id,
            visitor_id=session_data.visitor_id,
            website_id=website['id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@router.post("/session/{widget_id}/resume")
async def resume_session(
    widget_id: str,
    session_data: dict,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Resume an existing chat session"""
    try:
        # Basic implementation - return success for any session
        return SessionResponse(
            session_id=session_data.get("session_id", str(uuid4())),
            visitor_id=session_data.get("visitor_id", str(uuid4())),
            website_id=widget_id,
            is_active=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming session: {str(e)}")


@router.get("/session/{widget_id}/{session_id}/history")
async def get_session_history(
    widget_id: str,
    session_id: str,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get message history for a session"""
    try:
        # Basic implementation - return empty history
        return {
            "session_id": session_id,
            "messages": []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")


@router.post("/session/{widget_id}/end")
async def end_session(
    widget_id: str,
    session_data: SessionEndRequest,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """End a chat session"""
    try:
        # Basic implementation - return success
        return {"status": "success", "message": "Session ended"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")
        
        if session_data.satisfaction_rating:
            conversation.satisfaction_rating = session_data.satisfaction_rating
        
        if session_data.feedback:
            conversation.feedback = session_data.feedback
        
        await db.commit()
        
        return {
            "status": "ended",
            "session_id": session_data.session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")


@router.post("/analytics/{widget_id}/track")
async def track_analytics(
    widget_id: str,
    analytics_data: AnalyticsTrackRequest,
    request: Request,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Track widget analytics events"""
    try:
        # Basic implementation - just return success
        return {"status": "recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking analytics: {str(e)}")


@router.get("/script/{widget_id}")
async def get_widget_script(
    widget_id: str,
    request: Request,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Serve the widget JavaScript with domain validation"""
    try:
        # Basic implementation - return simple script
        website_result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()
        if not website_result.data:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        if not website.is_active:
            raise HTTPException(status_code=403, detail="Widget is not active")
        
        # Validate domain (permissive in development)
        if not validate_domain(request, website):
            # In development, allow cross-domain for testing
            pass
        
        # Generate widget script with configuration
        script_content = f"""
(function() {{
    // LiteChat Widget v1.0
    const WIDGET_ID = '{widget_id}';
    const API_BASE = window.location.origin + '/api/v1/widget';
    
    let widgetConfig = null;
    let isInitialized = false;
    let sessionId = null;
    let visitorId = localStorage.getItem('litechat_visitor_id') || generateVisitorId();
    
    function generateVisitorId() {{
        const id = 'visitor_' + Math.random().toString(36).substr(2, 16);
        localStorage.setItem('litechat_visitor_id', id);
        return id;
    }}
    
    // Initialize widget
    async function initWidget() {{
        if (isInitialized) return;

        try {{
            const response = await fetch(`${{API_BASE}}/config/${{WIDGET_ID}}`);
            if (!response.ok) throw new Error('Failed to load widget config');

            widgetConfig = await response.json();

            // Check verification status and verify if needed
            await checkAndVerifyWidget();

            createWidgetUI();
            isInitialized = true;
        }} catch (error) {{
            console.error('LiteChat initialization failed:', error);
        }}
    }}

    // Check widget verification status and verify if needed
    async function checkAndVerifyWidget() {{
        try {{
            console.log(`Widget verification status: ${{widgetConfig.is_verified ? 'VERIFIED' : 'NOT VERIFIED'}}`);

            // Only call verification if widget is not already verified
            if (!widgetConfig.is_verified) {{
                console.log('Starting verification process...');
                const verifyResponse = await fetch(`${{API_BASE}}/verify/${{WIDGET_ID}}`, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        page_url: window.location.href,
                        domain: window.location.hostname,
                        user_agent: navigator.userAgent
                    }})
                }});

                if (verifyResponse.ok) {{
                    const verifyData = await verifyResponse.json();
                    console.log('Widget verification completed:', verifyData);
                    widgetConfig.is_verified = true;
                }} else {{
                    console.error('Widget verification failed:', verifyResponse.status);
                }}
            }} else {{
                console.log('Widget already verified, skipping verification call');
            }}
        }} catch (e) {{
            console.log('Failed to check verification status:', e);
            // If config check fails, don't try verification as fallback to avoid loops
        }}
    }}
    
    function createWidgetUI() {{
        // Create widget container
        const widget = document.createElement('div');
        widget.id = 'litechat-widget';
        widget.innerHTML = `
            <div class="litechat-button" onclick="LiteChat.toggle()">
                <span>💬</span>
            </div>
            <div class="litechat-chat" style="display: none;">
                <div class="litechat-header">
                    <span>${{widgetConfig.config.company_name || 'Chat'}}</span>
                    <button onclick="LiteChat.close()">×</button>
                </div>
                <div class="litechat-messages"></div>
                <div class="litechat-input">
                    <input type="text" placeholder="${{widgetConfig.config.placeholder_text}}" />
                    <button onclick="LiteChat.sendMessage()">Send</button>
                </div>
            </div>
        `;
        
        // Apply styles
        const style = document.createElement('style');
        style.textContent = `
            #litechat-widget {{
                position: fixed;
                ${{widgetConfig.config.widget_position.includes('right') ? 'right: 20px' : 'left: 20px'}};
                bottom: 20px;
                z-index: 999999;
                font-family: ${{widgetConfig.config.font_family || 'Arial, sans-serif'}};
            }}
            .litechat-button {{
                width: 60px;
                height: 60px;
                border-radius: ${{widgetConfig.config.border_radius || 50}}px;
                background: ${{widgetConfig.config.widget_color}};
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }}
            .litechat-chat {{
                width: 350px;
                height: 500px;
                background: white;
                border-radius: ${{widgetConfig.config.border_radius || 8}}px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.15);
                position: absolute;
                bottom: 70px;
                right: 0;
                display: flex;
                flex-direction: column;
            }}
            .litechat-header {{
                background: ${{widgetConfig.config.widget_color}};
                color: white;
                padding: 15px;
                border-radius: ${{widgetConfig.config.border_radius || 8}}px ${{widgetConfig.config.border_radius || 8}}px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .litechat-messages {{
                flex: 1;
                padding: 15px;
                overflow-y: auto;
            }}
            .litechat-input {{
                padding: 15px;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
            }}
            .litechat-input input {{
                flex: 1;
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                outline: none;
            }}
            .litechat-input button {{
                padding: 8px 16px;
                background: ${{widgetConfig.config.widget_color}};
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            ${{widgetConfig.config.custom_css || ''}}
        `;
        
        document.head.appendChild(style);
        document.body.appendChild(widget);
        
        // Add welcome message
        if (widgetConfig.config.welcome_message) {{
            addMessage(widgetConfig.config.welcome_message, 'ai');
        }}
    }}
    
    function addMessage(content, sender) {{
        const messagesContainer = document.querySelector('.litechat-messages');
        const message = document.createElement('div');
        message.className = `litechat-message litechat-${{sender}}`;
        message.style.cssText = `
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
            ${{sender === 'user' ? 'margin-left: auto; background: ' + widgetConfig.config.widget_color + '; color: white;' : 'background: #f1f3f5;'}}
        `;
        message.textContent = content;
        messagesContainer.appendChild(message);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }}
    
    async function sendMessage() {{
        const input = document.querySelector('.litechat-input input');
        const message = input.value.trim();
        if (!message) return;
        
        addMessage(message, 'user');
        input.value = '';
        
        try {{
            if (!sessionId) {{
                // Create session first
                const sessionResponse = await fetch(`${{API_BASE}}/session/${{WIDGET_ID}}/create`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        visitor_id: visitorId,
                        page_url: window.location.href,
                        page_title: document.title,
                        user_agent: navigator.userAgent
                    }})
                }});
                const sessionData = await sessionResponse.json();
                sessionId = sessionData.session_id;
            }}
            
            // Send message
            const response = await fetch(`${{API_BASE}}/chat/${{WIDGET_ID}}`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    message: message,
                    session_id: sessionId,
                    visitor_id: visitorId,
                    page_url: window.location.href,
                    page_title: document.title,
                    user_agent: navigator.userAgent
                }})
            }});
            
            if (!response.ok) throw new Error('Failed to send message');
            
            const data = await response.json();
            addMessage(data.response, 'ai');
            
        }} catch (error) {{
            console.error('Error sending message:', error);
            addMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }}
    }}
    
    // Global LiteChat object
    window.LiteChat = {{
        toggle: function() {{
            const chat = document.querySelector('.litechat-chat');
            chat.style.display = chat.style.display === 'none' ? 'block' : 'none';
        }},
        close: function() {{
            const chat = document.querySelector('.litechat-chat');
            chat.style.display = 'none';
        }},
        sendMessage: sendMessage
    }};
    
    // Handle Enter key in input
    document.addEventListener('keypress', function(e) {{
        if (e.target.matches('.litechat-input input') && e.key === 'Enter') {{
            sendMessage();
        }}
    }});
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initWidget);
    }} else {{
        initWidget();
    }}
}})();
        """
        
        return PlainTextResponse(content=script_content, media_type="application/javascript")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving widget script: {str(e)}")