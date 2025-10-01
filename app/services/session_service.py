"""
Session Service for managing chat sessions with persistent history and context.
Provides continuous conversation experience across browser sessions.
"""

import asyncio
import logging
import secrets
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
from dateutil import tz

from ..core.config import get_settings
from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionService:
    """Service for managing chat sessions with enhanced context and continuity."""
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        
    async def create_session(
        self,
        website_id: UUID,
        visitor_id: str,
        page_url: Optional[str] = None,
        page_title: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        referrer: Optional[str] = None,
        user_id: Optional[str] = None,
        session_duration_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat session with enhanced tracking capabilities.
        
        Args:
            website_id: UUID of the website
            visitor_id: Unique visitor identifier
            page_url: URL where conversation started
            page_title: Title of the page
            user_agent: Browser user agent
            ip_address: Client IP address
            referrer: HTTP referrer
            user_id: Optional authenticated user ID
            session_duration_days: Custom session duration (defaults to config)
            
        Returns:
            Dictionary containing session information including session_token
        """
        try:
            # Generate secure session token
            session_token = self._generate_session_token()
            
            # Calculate session expiration
            duration_days = session_duration_days or settings.session_duration_days
            session_expires_at = datetime.now(timezone.utc) + timedelta(days=duration_days)
            
            # Create conversation record with enhanced session fields
            conversation_data = {
                'session_id': session_token,  # Using session_token as session_id for compatibility
                'visitor_id': visitor_id,
                'website_id': str(website_id),
                'page_url': page_url,
                'page_title': page_title,
                'user_agent': user_agent,
                'ip_address': ip_address,
                'referrer': referrer,
                'is_active': True,
                'status': 'active',
                'session_token': session_token,
                'session_expires_at': session_expires_at.isoformat(),
                'context_tokens_used': 0,
                'context_summary': None,
                'parent_session_id': None,
                'last_context_update': datetime.now(timezone.utc).isoformat()
            }
            
            # Add user_id only if provided (not all tables may have this column)
            if user_id:
                # For now, we'll store user_id in session metadata or skip if column doesn't exist
                pass  # The column doesn't exist in Supabase yet
            
            result = self.supabase.table('conversations').insert(conversation_data).execute()
            
            if not result.data:
                raise Exception("Failed to create conversation record")
                
            conversation = result.data[0]
            
            logger.info(f"✅ SESSION: Created new session {session_token} for visitor {visitor_id}")
            
            return {
                'session_token': session_token,
                'conversation_id': conversation['id'],
                'expires_at': session_expires_at.isoformat(),
                'visitor_id': visitor_id,
                'website_id': str(website_id),
                'is_new': True
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION: Failed to create session: {e}")
            raise Exception(f"Failed to create session: {str(e)}")
    
    async def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information by session token.
        
        Args:
            session_token: Unique session identifier
            
        Returns:
            Session data if valid, None if expired or not found
        """
        try:
            # Query conversation by session_token (stored in session_id field)
            result = self.supabase.table('conversations').select('*').eq('session_id', session_token).execute()
            
            if not result.data:
                logger.warning(f"⚠️ SESSION: Session not found: {session_token}")
                return None
                
            conversation = result.data[0]
            
            # Check if session has expired
            if self._is_session_expired(conversation.get('session_expires_at')):
                logger.info(f"⏰ SESSION: Session expired: {session_token}")
                await self._mark_session_expired(session_token)
                return None
            
            # Update last activity
            await self._update_last_activity(session_token)
            
            return {
                'session_token': session_token,
                'conversation_id': conversation['id'],
                'visitor_id': conversation['visitor_id'],
                'website_id': conversation['website_id'],
                'user_id': None,  # user_id column doesn't exist in current schema
                'expires_at': conversation.get('session_expires_at'),
                'is_active': conversation.get('is_active', True),
                'context_tokens_used': conversation.get('context_tokens_used', 0),
                'context_summary': conversation.get('context_summary'),
                'parent_session_id': conversation.get('parent_session_id'),
                'total_messages': conversation.get('total_messages', 0),
                'started_at': conversation['started_at'],
                'last_activity_at': conversation['last_activity_at']
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION: Error retrieving session {session_token}: {e}")
            return None
    
    async def extend_session(
        self, 
        session_token: str, 
        extension_days: Optional[int] = None
    ) -> bool:
        """
        Extend session expiration time.
        
        Args:
            session_token: Session to extend
            extension_days: Days to extend (defaults to config default)
            
        Returns:
            True if successfully extended, False otherwise
        """
        try:
            # Calculate new expiration time
            extension = extension_days or settings.session_duration_days
            max_extension = settings.max_session_duration_days
            
            new_expires_at = datetime.now(timezone.utc) + timedelta(days=min(extension, max_extension))
            
            result = self.supabase.table('conversations').update({
                'session_expires_at': new_expires_at.isoformat(),
                'last_activity_at': datetime.now(timezone.utc).isoformat()
            }).eq('session_id', session_token).execute()
            
            if result.data:
                logger.info(f"⏰ SESSION: Extended session {session_token} until {new_expires_at}")
                return True
            else:
                logger.warning(f"⚠️ SESSION: Failed to extend session {session_token}")
                return False
                
        except Exception as e:
            logger.error(f"❌ SESSION: Error extending session {session_token}: {e}")
            return False
    
    async def get_or_create_session(
        self,
        website_id: UUID,
        visitor_id: str,
        session_token: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get existing session or create new one if needed.
        
        Args:
            website_id: UUID of the website
            visitor_id: Unique visitor identifier  
            session_token: Optional existing session token
            **kwargs: Additional parameters for session creation
            
        Returns:
            Session information
        """
        # Try to retrieve existing session
        if session_token:
            existing_session = await self.get_session(session_token)
            if existing_session:
                logger.info(f"♻️ SESSION: Reusing existing session {session_token}")
                return existing_session
        
        # Create new session
        logger.info(f"🆕 SESSION: Creating new session for visitor {visitor_id}")
        return await self.create_session(website_id, visitor_id, **kwargs)
    
    async def end_session(self, session_token: str, reason: str = "manual") -> bool:
        """
        End a chat session and mark it as inactive.
        
        Args:
            session_token: Session to end
            reason: Reason for ending (manual, timeout, error, etc.)
            
        Returns:
            True if successfully ended, False otherwise
        """
        try:
            result = self.supabase.table('conversations').update({
                'is_active': False,
                'status': 'completed',
                'ended_at': datetime.now(timezone.utc).isoformat(),
                'last_activity_at': datetime.now(timezone.utc).isoformat()
            }).eq('session_id', session_token).execute()
            
            if result.data:
                logger.info(f"🏁 SESSION: Ended session {session_token} (reason: {reason})")
                return True
            else:
                logger.warning(f"⚠️ SESSION: Failed to end session {session_token}")
                return False
                
        except Exception as e:
            logger.error(f"❌ SESSION: Error ending session {session_token}: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from the database.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Find expired sessions
            expired_result = self.supabase.table('conversations').select('session_id').lt('session_expires_at', current_time).eq('is_active', True).execute()
            
            if not expired_result.data:
                return 0
            
            expired_sessions = [session['session_id'] for session in expired_result.data]
            
            # Mark as expired
            cleanup_result = self.supabase.table('conversations').update({
                'is_active': False,
                'status': 'expired',
                'ended_at': current_time
            }).lt('session_expires_at', current_time).eq('is_active', True).execute()
            
            count = len(expired_sessions) if cleanup_result.data else 0
            
            if count > 0:
                logger.info(f"🧹 SESSION: Cleaned up {count} expired sessions")
                
            return count
            
        except Exception as e:
            logger.error(f"❌ SESSION: Error cleaning up expired sessions: {e}")
            return 0
    
    async def get_session_analytics(
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
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get session counts and metrics
            sessions_result = self.supabase.table('conversations').select(
                'id, visitor_id, total_messages, started_at, ended_at, status'
            ).eq('website_id', str(website_id)).gte('started_at', cutoff_date).execute()
            
            if not sessions_result.data:
                return self._empty_analytics()
            
            sessions = sessions_result.data
            
            # Calculate analytics
            total_sessions = len(sessions)
            active_sessions = len([s for s in sessions if s.get('status') == 'active'])
            unique_visitors = len(set(s['visitor_id'] for s in sessions))
            total_messages = sum(s.get('total_messages', 0) for s in sessions)
            
            # Calculate average session duration
            completed_sessions = [s for s in sessions if s.get('ended_at')]
            avg_duration_minutes = 0
            if completed_sessions:
                durations = []
                for session in completed_sessions:
                    start = datetime.fromisoformat(session['started_at'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(session['ended_at'].replace('Z', '+00:00'))
                    durations.append((end - start).total_seconds() / 60)
                avg_duration_minutes = sum(durations) / len(durations)
            
            return {
                'period_days': days,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'unique_visitors': unique_visitors,
                'total_messages': total_messages,
                'avg_messages_per_session': total_messages / total_sessions if total_sessions > 0 else 0,
                'avg_session_duration_minutes': round(avg_duration_minutes, 2),
                'return_visitor_rate': ((total_sessions - unique_visitors) / total_sessions * 100) if total_sessions > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION: Error getting analytics for website {website_id}: {e}")
            return self._empty_analytics()
    
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return f"session_{secrets.token_urlsafe(32)}"
    
    def _is_session_expired(self, expires_at_str: Optional[str]) -> bool:
        """Check if session has expired."""
        if not expires_at_str:
            return False
            
        try:
            expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
            return datetime.now(timezone.utc) > expires_at
        except (ValueError, TypeError):
            return True
    
    async def _mark_session_expired(self, session_token: str):
        """Mark session as expired."""
        try:
            self.supabase.table('conversations').update({
                'is_active': False,
                'status': 'expired',
                'ended_at': datetime.now(timezone.utc).isoformat()
            }).eq('session_id', session_token).execute()
        except Exception as e:
            logger.error(f"Error marking session expired: {e}")
    
    async def _update_last_activity(self, session_token: str):
        """Update last activity timestamp."""
        try:
            self.supabase.table('conversations').update({
                'last_activity_at': datetime.now(timezone.utc).isoformat()
            }).eq('session_id', session_token).execute()
        except Exception as e:
            logger.error(f"Error updating last activity: {e}")
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            'period_days': 0,
            'total_sessions': 0,
            'active_sessions': 0,
            'unique_visitors': 0,
            'total_messages': 0,
            'avg_messages_per_session': 0,
            'avg_session_duration_minutes': 0,
            'return_visitor_rate': 0
        }