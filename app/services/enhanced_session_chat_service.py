"""
Enhanced Session-Aware Chat Service that provides continuous conversation experience
with persistent context and session management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime, timezone

from ..core.config import get_settings
from ..schemas.chat import ChatMessage, ChatResponse
from .session_service import SessionService
from .context_service import ContextService
from .rag_chat_service import RAGChatService
from .vector_search_service import VectorSearchService

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedSessionChatService:
    """
    Enhanced chat service with persistent session management and continuous context.
    """
    
    def __init__(self):
        self.session_service = SessionService()
        self.context_service = ContextService()
        self.rag_chat_service = RAGChatService()
        self.vector_service = VectorSearchService()
    
    async def process_chat_with_session(
        self,
        message: str,
        website_id: UUID,
        visitor_id: str,
        session_token: Optional[str] = None,
        page_url: Optional[str] = None,
        page_title: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        referrer: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process chat message with full session management and context continuity.
        
        Args:
            message: User's message
            website_id: UUID of the website
            visitor_id: Unique visitor identifier
            session_token: Optional existing session token
            page_url: URL where conversation is happening
            page_title: Title of the page
            user_agent: Browser user agent
            ip_address: Client IP address
            referrer: HTTP referrer
            user_id: Optional authenticated user ID
            
        Returns:
            Dictionary containing AI response and session information
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Get or create session
            session_info = await self.session_service.get_or_create_session(
                website_id=website_id,
                visitor_id=visitor_id,
                session_token=session_token,
                page_url=page_url,
                page_title=page_title,
                user_agent=user_agent,
                ip_address=ip_address,
                referrer=referrer,
                user_id=user_id
            )
            
            conversation_id = UUID(session_info['conversation_id'])
            session_token = session_info['session_token']
            
            logger.info(f"🔄 CHAT: Processing message for session {session_token}")
            
            # Step 2: Add user message to context
            await self.context_service.add_message_to_context(
                conversation_id=conversation_id,
                message_content=message,
                message_type="user"
            )
            
            # Step 3: Prepare conversation context
            context_data = await self.context_service.prepare_conversation_context(
                conversation_id=conversation_id,
                max_tokens=settings.context_window_size
            )
            
            # Step 4: Generate AI response using RAG with conversation context
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in context_data.get('messages', [])
            ]
            
            # Generate RAG response with website context and conversation history
            ai_response_data = await self.rag_chat_service.generate_rag_response(
                user_message=message,
                website_id=website_id,
                conversation_history=conversation_history
            )
            
            ai_response = ai_response_data.get('response', 'I apologize, but I am having trouble processing your request right now.')
            
            # Step 5: Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Step 6: Add AI response to context with metadata
            await self.context_service.add_message_to_context(
                conversation_id=conversation_id,
                message_content=ai_response,
                message_type="assistant",
                metadata={
                    'processing_time_ms': int(processing_time),
                    'model_used': 'gpt-3.5-turbo',  # This could be dynamic
                    'context_sources': ai_response_data.get('context_preview', ''),
                    'confidence_score': 0.85  # This could be calculated
                }
            )
            
            # Step 7: Extend session if it was reused
            if not session_info.get('is_new', False):
                await self.session_service.extend_session(session_token)
            
            logger.info(f"✅ CHAT: Generated response for session {session_token} ({processing_time:.0f}ms)")
            
            return {
                # Response data
                'response': ai_response,
                'response_type': 'ai',
                'processing_time_ms': int(processing_time),
                
                # Session data
                'session_token': session_token,
                'session_expires_at': session_info['expires_at'],
                'is_new_session': session_info.get('is_new', False),
                
                # Context data
                'context_used': ai_response_data.get('context_used', False),
                'context_preview': ai_response_data.get('context_preview', ''),
                'messages_in_context': context_data.get('messages_included', 0),
                'total_context_tokens': context_data.get('total_tokens', 0),
                'has_context_summary': context_data.get('has_summary', False),
                
                # Conversation metadata
                'conversation_id': str(conversation_id),
                'total_messages': context_data.get('conversation_metadata', {}).get('total_messages', 0),
                'visitor_id': visitor_id,
                'website_id': str(website_id),
                
                # Source information
                'source': ai_response_data.get('source', 'enhanced_session_chat')
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT: Error processing chat message: {e}")
            
            # Return error response while maintaining session
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again.",
                'response_type': 'error',
                'session_token': session_token or None,
                'error': str(e),
                'source': 'error'
            }
    
    async def get_session_history(
        self, 
        session_token: str, 
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Get conversation history for a session.
        
        Args:
            session_token: Session identifier
            limit: Maximum number of messages to return
            include_metadata: Whether to include message metadata
            
        Returns:
            Dictionary containing session info and message history
        """
        try:
            # Get session information
            session_info = await self.session_service.get_session(session_token)
            if not session_info:
                return {
                    'error': 'Session not found or expired',
                    'messages': [],
                    'session_info': None
                }
            
            # Get conversation history
            conversation_id = UUID(session_info['conversation_id'])
            messages = await self.context_service.get_conversation_history(
                conversation_id=conversation_id,
                limit=limit,
                include_metadata=include_metadata
            )
            
            return {
                'messages': messages,
                'session_info': {
                    'session_token': session_token,
                    'visitor_id': session_info['visitor_id'],
                    'website_id': session_info['website_id'],
                    'started_at': session_info['started_at'],
                    'last_activity_at': session_info['last_activity_at'],
                    'total_messages': session_info['total_messages'],
                    'expires_at': session_info['expires_at'],
                    'is_active': session_info['is_active']
                },
                'total_messages': len(messages)
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT: Error getting session history for {session_token}: {e}")
            return {
                'error': str(e),
                'messages': [],
                'session_info': None
            }
    
    async def extend_session_duration(
        self, 
        session_token: str, 
        extension_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extend session duration.
        
        Args:
            session_token: Session to extend
            extension_days: Number of days to extend
            
        Returns:
            Dictionary containing success status and new expiration
        """
        try:
            success = await self.session_service.extend_session(session_token, extension_days)
            
            if success:
                # Get updated session info
                session_info = await self.session_service.get_session(session_token)
                
                return {
                    'success': True,
                    'session_token': session_token,
                    'expires_at': session_info['expires_at'] if session_info else None,
                    'message': 'Session extended successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to extend session',
                    'session_token': session_token
                }
                
        except Exception as e:
            logger.error(f"❌ CHAT: Error extending session {session_token}: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_token': session_token
            }
    
    async def end_session(self, session_token: str, reason: str = "manual") -> Dict[str, Any]:
        """
        End a chat session.
        
        Args:
            session_token: Session to end
            reason: Reason for ending
            
        Returns:
            Dictionary containing success status
        """
        try:
            success = await self.session_service.end_session(session_token, reason)
            
            return {
                'success': success,
                'session_token': session_token,
                'message': 'Session ended successfully' if success else 'Failed to end session'
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT: Error ending session {session_token}: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_token': session_token
            }
    
    async def get_website_session_analytics(
        self, 
        website_id: UUID, 
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get session analytics for a website.
        
        Args:
            website_id: Website UUID
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        try:
            analytics = await self.session_service.get_session_analytics(website_id, days)
            
            return {
                'success': True,
                'website_id': str(website_id),
                'analytics': analytics
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT: Error getting analytics for website {website_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'website_id': str(website_id)
            }
    
    async def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """
        Clean up expired sessions.
        
        Returns:
            Cleanup results
        """
        try:
            cleaned_count = await self.session_service.cleanup_expired_sessions()
            
            return {
                'success': True,
                'cleaned_sessions': cleaned_count,
                'message': f'Cleaned up {cleaned_count} expired sessions'
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT: Error cleaning up expired sessions: {e}")
            return {
                'success': False,
                'error': str(e),
                'cleaned_sessions': 0
            }
    
    async def validate_session(self, session_token: str) -> Dict[str, Any]:
        """
        Validate if a session is active and not expired.
        
        Args:
            session_token: Session to validate
            
        Returns:
            Validation results
        """
        try:
            session_info = await self.session_service.get_session(session_token)
            
            if session_info:
                return {
                    'valid': True,
                    'session_token': session_token,
                    'expires_at': session_info['expires_at'],
                    'visitor_id': session_info['visitor_id'],
                    'website_id': session_info['website_id']
                }
            else:
                return {
                    'valid': False,
                    'session_token': session_token,
                    'error': 'Session not found or expired'
                }
                
        except Exception as e:
            logger.error(f"❌ CHAT: Error validating session {session_token}: {e}")
            return {
                'valid': False,
                'session_token': session_token,
                'error': str(e)
            }