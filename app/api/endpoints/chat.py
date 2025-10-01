from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from supabase import Client
from ...core.database import get_supabase_client
from ...core.config import settings
from ...services.vector_enhanced_chat_service import get_vector_enhanced_chat_service
from uuid import UUID, uuid4
import json
import asyncio


router = APIRouter()

# Simple chat response models (basic implementation)
from pydantic import BaseModel

class ChatMessage(BaseModel):
    session_id: str
    message: str
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    processing_time_ms: int = 0


@router.post("/test")
async def test_db(supabase: Client = Depends(get_supabase_client)):
    """
    Test Supabase database connection.
    """
    try:
        result = supabase.table('websites').select('id, widget_id').limit(1).execute()
        if result.data:
            website = result.data[0]
            return {"status": "success", "database": "supabase", "website_id": website['id'], "widget_id": website['widget_id']}
        else:
            return {"status": "no_websites", "database": "supabase"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase Error: {str(e)}")

@router.post("/message", response_model=ChatResponse)
async def send_message(
    chat_request: ChatMessage,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Send a message to the AI assistant and get a response using RAG (Retrieval Augmented Generation).
    """
    try:
        # Initialize vector enhanced chat service
        chat_service = get_vector_enhanced_chat_service()

        # Extract website_id from the request or use a default
        # For demo purposes, we'll use a default website_id
        try:
            # Try to extract UUID from session_id if it contains one
            parts = chat_request.session_id.split('_')
            website_id = parts[0] if len(parts) > 0 else "demo-website"
        except (ValueError, IndexError):
            # Default website_id for demo
            website_id = "demo-website"

        # Get conversation history from Supabase
        conversation_history = []
        try:
            # Get recent messages for context (last 6 messages)
            history_result = supabase.table('conversations')\
                .select('*')\
                .eq('session_token', chat_request.session_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()

            if history_result.data:
                # Get messages for the conversation
                conv_id = history_result.data[0]['id']
                messages_result = supabase.table('messages')\
                    .select('*')\
                    .eq('conversation_id', conv_id)\
                    .order('created_at', desc=False)\
                    .limit(10)\
                    .execute()

                if messages_result.data:
                    for msg in messages_result.data[-6:]:  # Last 6 messages
                        role = "user" if msg['message_type'] == 'user' else "assistant"
                        conversation_history.append({
                            "role": role,
                            "content": msg['content']
                        })
        except Exception as e:
            print(f"Warning: Could not fetch conversation history: {e}")

        # Generate AI response using vector enhanced chat
        rag_response = await chat_service.generate_chat_response(
            user_message=chat_request.message,
            website_id=website_id,
            conversation_history=conversation_history,
            use_rag=True,
            use_hybrid_search=True
        )

        ai_response = rag_response.get('response', 'I apologize, but I encountered an issue generating a response. Please try again.')
        
        # Store the conversation in Supabase
        try:
            # Create or get conversation
            conv_result = supabase.table('conversations').upsert({
                'id': str(uuid4()),
                'session_token': chat_request.session_id,
                'website_id': str(website_id),
                'visitor_id': 'demo_user',  # This would be dynamic in production
                'created_at': 'now()',
                'updated_at': 'now()'
            }).execute()
            
            if conv_result.data:
                conv_id = conv_result.data[0]['id']
                
                # Store user message
                supabase.table('messages').insert({
                    'id': str(uuid4()),
                    'conversation_id': conv_id,
                    'content': chat_request.message,
                    'message_type': 'user',
                    'created_at': 'now()'
                }).execute()
                
                # Store AI response
                supabase.table('messages').insert({
                    'id': str(uuid4()),
                    'conversation_id': conv_id,
                    'content': ai_response,
                    'message_type': 'assistant',
                    'created_at': 'now()'
                }).execute()
        except Exception as e:
            print(f"Warning: Could not store conversation: {e}")
        
        return ChatResponse(
            response=ai_response,
            session_id=chat_request.session_id,
            message_id=f"msg_{hash(chat_request.message)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Retrieve chat history for a session (basic implementation).
    """
    try:
        # Basic implementation - return empty history for now
        return {"session_id": session_id, "messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@router.websocket("/stream/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat streaming (basic implementation).
    """
    await websocket.accept()
    print(f"✅ WebSocket connected for session: {session_id}")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                message = message_data.get("message", "")
                
                # Send typing indicator
                await websocket.send_text(json.dumps({
                    "type": "typing",
                    "is_typing": True
                }))
                
                # Generate response using vector enhanced chat service
                try:
                    chat_service = get_vector_enhanced_chat_service()
                    website_id = session_id.split('_')[0] if '_' in session_id else "demo-website"

                    # Stream response using vector chat service
                    async for chunk in chat_service.stream_chat_response(
                        user_message=message,
                        website_id=website_id,
                        use_rag=True
                    ):
                        if chunk.get("type") == "content":
                            await websocket.send_text(json.dumps({
                                "type": "content",
                                "content": chunk.get("content", "")
                            }))
                        elif chunk.get("type") == "complete":
                            await websocket.send_text(json.dumps({
                                "type": "complete",
                                "content": chunk.get("full_response", "")
                            }))
                            break
                        elif chunk.get("type") == "error":
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "error": chunk.get("error", "Unknown error")
                            }))
                            break

                except Exception as e:
                    # Fallback to simple response
                    response_text = f"Thank you for your message: '{message}'. This is a response from ChatLite."
                    await websocket.send_text(json.dumps({
                        "type": "complete",
                        "content": response_text
                    }))
                
                # Stop typing
                await websocket.send_text(json.dumps({
                    "type": "typing",
                    "is_typing": False
                }))
                    
            except json.JSONDecodeError as e:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "error": f"Invalid JSON: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected for session: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error for session {session_id}: {str(e)}")
        try:
            await websocket.close(code=1000)
        except:
            pass


@router.get("/service-status")
async def get_service_status():
    """
    Get chat service status and configuration.
    """
    return {
        "status": "success",
        "database": "supabase",
        "endpoints_available": [
            "/chat/message (POST)",
            "/chat/history/{session_id} (GET)", 
            "/chat/service-status (GET)",
            "/chat/test (POST)",
            "/chat/stream/{session_id} (WebSocket)"
        ]
    }