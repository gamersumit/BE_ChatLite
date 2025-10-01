"""
Chat service using Supabase for database operations.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from ..services.supabase_service import SupabaseService
from ..schemas.chat import ChatMessage, ChatResponse, ChatHistory
from .enhanced_chat_service import EnhancedChatService


class SupabaseChatService:
    """Chat service using Supabase backend."""
    
    def __init__(self, enhanced_chat_service: EnhancedChatService):
        self.chat_service = enhanced_chat_service
        self.db_service = SupabaseService()
    
    async def process_chat_message(self, chat_request: ChatMessage) -> ChatResponse:
        """Process incoming chat message and generate AI response."""
        
        # Get or create conversation
        conversation = self._get_or_create_conversation(chat_request)
        
        # Save user message
        user_message = self._save_user_message(
            conversation["id"], 
            chat_request.message
        )
        
        # Get conversation history for context
        history = self._get_conversation_history(
            conversation["id"], 
            limit=5
        )
        
        # Get website context
        website_context = self._get_website_context(conversation["website_id"])
        
        # Generate AI response
        ai_response = await self.chat_service.generate_chat_response(
            user_message=chat_request.message,
            conversation_history=history,
            website_context=website_context
        )
        
        # Save AI response
        ai_message = self._save_ai_message(
            conversation["id"],
            ai_response
        )
        
        return ChatResponse(
            message_id=ai_message["id"],
            response=ai_response["response"],
            confidence_score=ai_response.get("confidence_score"),
            processing_time_ms=ai_response["processing_time_ms"],
            sources=ai_response.get("sources", [])
        )
    
    async def get_chat_history(self, session_id: str) -> Optional[ChatHistory]:
        """Retrieve chat history for a session."""
        
        # Get conversation
        conversation = self.db_service.get_conversation_by_session(session_id)
        if not conversation:
            return None
        
        # Get messages
        messages = self.db_service.get_messages_by_conversation(
            conversation["id"]
        )
        
        return ChatHistory(
            conversation_id=conversation["id"],
            session_id=conversation["session_id"],
            messages=[
                {
                    "id": msg["id"],
                    "role": msg.get("message_type", "user"),  # Use message_type from actual table
                    "content": msg["content"],
                    "created_at": msg["created_at"]
                }
                for msg in messages
            ],
            total_messages=len(messages),
            started_at=conversation["created_at"],
            last_activity_at=conversation["updated_at"]
        )
    
    async def stream_chat_response(
        self, 
        chat_request: ChatMessage
    ):
        """Stream chat response for WebSocket connections."""
        
        # Get or create conversation
        conversation = self._get_or_create_conversation(chat_request)
        
        # Save user message
        self._save_user_message(conversation["id"], chat_request.message)
        
        # Get conversation history for context
        history = self._get_conversation_history(conversation["id"], limit=5)
        
        # Get website context
        website_context = self._get_website_context(conversation["website_id"])
        
        # Stream AI response
        full_response = ""
        async for chunk in self.chat_service.stream_chat_response(
            user_message=chat_request.message,
            conversation_history=history,
            website_context=website_context
        ):
            if chunk.get("type") == "content":
                full_response += chunk.get("content", "")
            yield chunk
        
        # Save the complete AI response
        if full_response:
            self._save_ai_message(conversation["id"], {
                "response": full_response,
                "processing_time_ms": 0,  # Will be calculated in OpenAI service
                "model_used": "gpt-4o"
            })
    
    def _get_or_create_conversation(self, chat_request: ChatMessage) -> Dict[str, Any]:
        """Get existing conversation or create new one."""
        
        # Try to get existing conversation
        conversation = self.db_service.get_conversation_by_session(
            chat_request.session_id
        )
        
        if conversation:
            return conversation
        
        # Get website by domain or use default
        website = None
        if chat_request.page_url:
            from urllib.parse import urlparse
            domain = urlparse(chat_request.page_url).netloc
            website = self.db_service.get_website_by_domain(domain)
        
        if not website:
            # Use default website for localhost/development
            website = self.db_service.get_website_by_domain("localhost")
            if not website:
                # Create default website
                website = self.db_service.create_website({
                    "domain": "localhost",
                    "widget_id": "default-widget-id",
                    "name": "Local Development",
                    "description": "Default website for local development"
                })
        
        # Create new conversation
        conversation_data = {
            "session_id": chat_request.session_id,
            "website_id": website["id"],
            "user_id": chat_request.user_id,
            "page_url": chat_request.page_url,
            "page_title": chat_request.page_title
        }
        
        return self.db_service.create_conversation(conversation_data)
    
    def _save_user_message(self, conversation_id: str, content: str) -> Dict[str, Any]:
        """Save user message to database."""
        return self.db_service.create_message({
            "conversation_id": conversation_id,
            "role": "user",
            "content": content,
            "metadata": {}
        })
    
    def _save_ai_message(self, conversation_id: str, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Save AI response message to database."""
        return self.db_service.create_message({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": ai_response["response"],
            "metadata": {
                "processing_time_ms": ai_response.get("processing_time_ms"),
                "model_used": ai_response.get("model_used"),
                "tokens_used": ai_response.get("tokens_used"),
                "confidence_score": ai_response.get("confidence_score"),
                "cost_usd": ai_response.get("cost_usd")
            }
        })
    
    def _get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        
        messages = self.db_service.get_messages_by_conversation(
            conversation_id, 
            limit=limit
        )
        
        return [
            {
                "role": msg.get("message_type", "user"),  # Use message_type from actual table
                "content": msg["content"]
            }
            for msg in messages
        ]
    
    def _get_website_context(self, website_id: str) -> Optional[str]:
        """Get website context for RAG."""
        
        # For MVP, return basic context
        # In production, this would fetch scraped content
        return "This is a helpful AI assistant that can answer questions about the website and provide general assistance."