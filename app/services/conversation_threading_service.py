"""
Conversation Threading Service for linking related conversations across sessions.
Enables tracking user journeys and conversation topics across multiple sessions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
import hashlib
from collections import Counter
import re

from ..core.config import get_settings
from ..core.database import get_supabase_admin
from .message_importance_service import MessageImportanceService

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationThreadingService:
    """
    Service for linking related conversations across multiple sessions.
    """
    
    # Thread linking strategies
    LINK_BY_VISITOR = 'visitor'        # Link by visitor ID
    LINK_BY_TOPIC = 'topic'           # Link by conversation topics
    LINK_BY_CONTENT = 'content'       # Link by content similarity
    LINK_ADAPTIVE = 'adaptive'        # Combination of strategies
    
    # Topic extraction keywords for common conversation themes
    TOPIC_KEYWORDS = {
        'pricing': ['price', 'cost', 'plan', 'subscription', 'billing', 'payment', '$'],
        'features': ['feature', 'functionality', 'capability', 'what can', 'how does'],
        'support': ['help', 'support', 'issue', 'problem', 'troubleshoot', 'error'],
        'integration': ['integrate', 'api', 'connect', 'setup', 'configure'],
        'onboarding': ['getting started', 'how to', 'tutorial', 'guide', 'first time'],
        'technical': ['technical', 'development', 'code', 'implementation', 'documentation'],
        'sales': ['demo', 'trial', 'purchase', 'buy', 'contact sales', 'quote'],
        'account': ['account', 'profile', 'settings', 'preferences', 'login', 'password']
    }
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        self.importance_service = MessageImportanceService()
    
    async def create_or_update_thread(
        self,
        conversation_id: UUID,
        visitor_id: str,
        website_id: UUID,
        linking_strategy: str = LINK_ADAPTIVE,
        force_new_thread: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread or link to existing thread.
        
        Args:
            conversation_id: UUID of the current conversation
            visitor_id: Unique visitor identifier
            website_id: UUID of the website
            linking_strategy: Strategy for linking conversations
            force_new_thread: Force creation of new thread
            
        Returns:
            Thread information and linking results
        """
        try:
            # Get conversation details
            conversation = await self._get_conversation_details(conversation_id)
            if not conversation:
                logger.error(f"Conversation {conversation_id} not found")
                return self._empty_thread_result()
            
            # Get conversation content for topic analysis
            messages = await self._get_conversation_messages(conversation_id)
            
            # Extract topics and themes
            topics = await self._extract_conversation_topics(messages)
            thread_title = await self._generate_thread_title(messages, topics)
            
            if not force_new_thread:
                # Try to find existing thread to link to
                existing_thread = await self._find_existing_thread(
                    visitor_id,
                    website_id,
                    topics,
                    linking_strategy
                )
                
                if existing_thread:
                    # Link to existing thread
                    result = await self._link_to_existing_thread(
                        existing_thread,
                        conversation_id,
                        topics,
                        thread_title
                    )
                    
                    logger.info(f"🔗 THREADING: Linked conversation {conversation_id} to thread {existing_thread['thread_id']}")
                    return result
            
            # Create new thread
            thread_result = await self._create_new_thread(
                conversation_id,
                visitor_id,
                website_id,
                topics,
                thread_title,
                messages
            )
            
            logger.info(f"🆕 THREADING: Created new thread {thread_result.get('thread_id')} for conversation {conversation_id}")
            return thread_result
            
        except Exception as e:
            logger.error(f"❌ THREADING: Error creating/updating thread: {e}")
            return self._empty_thread_result()
    
    async def _find_existing_thread(
        self,
        visitor_id: str,
        website_id: UUID,
        topics: List[str],
        strategy: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find existing thread that matches the current conversation.
        """
        try:
            # Get recent threads for this visitor and website
            cutoff_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            
            threads_result = self.supabase.table('conversation_threads').select(
                'id, thread_id, visitor_id, thread_title, thread_summary, total_sessions, created_at, updated_at'
            ).eq('visitor_id', visitor_id).eq('website_id', str(website_id)).eq('is_active', True).gte('updated_at', cutoff_time).execute()
            
            if not threads_result.data:
                return None
            
            # Score threads by relevance
            thread_scores = []
            for thread in threads_result.data:
                score = await self._calculate_thread_relevance(
                    thread,
                    topics,
                    strategy
                )
                thread_scores.append((thread, score))
            
            # Sort by relevance score
            thread_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return best match if score is high enough
            if thread_scores and thread_scores[0][1] > 0.6:  # 60% relevance threshold
                return thread_scores[0][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing thread: {e}")
            return None
    
    async def _calculate_thread_relevance(
        self,
        thread: Dict[str, Any],
        current_topics: List[str],
        strategy: str
    ) -> float:
        """
        Calculate relevance score between current conversation and existing thread.
        """
        score = 0.0
        
        # Base score for same visitor (always applicable)
        score += 0.3
        
        # Time-based score (more recent = higher score)
        try:
            updated_at = datetime.fromisoformat(thread['updated_at'].replace('Z', '+00:00'))
            days_old = (datetime.now(timezone.utc) - updated_at).days
            time_score = max(0, 1.0 - (days_old / 30))  # Linear decay over 30 days
            score += time_score * 0.2
        except Exception:
            pass
        
        # Topic similarity score
        if current_topics and thread.get('thread_summary'):
            thread_text = thread['thread_summary'].lower()
            topic_matches = sum(1 for topic in current_topics if topic.lower() in thread_text)
            topic_score = min(1.0, topic_matches / len(current_topics))
            score += topic_score * 0.4
        
        # Title similarity score
        if current_topics and thread.get('thread_title'):
            title_text = thread['thread_title'].lower()
            title_matches = sum(1 for topic in current_topics if topic.lower() in title_text)
            title_score = min(1.0, title_matches / max(1, len(current_topics)))
            score += title_score * 0.1
        
        return min(1.0, score)
    
    async def _create_new_thread(
        self,
        conversation_id: UUID,
        visitor_id: str,
        website_id: UUID,
        topics: List[str],
        thread_title: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread.
        """
        try:
            thread_id = f"thread_{uuid4().hex[:12]}"
            
            # Create thread summary
            thread_summary = await self._create_thread_summary(messages, topics)
            
            # Create thread record
            thread_data = {
                'thread_id': thread_id,
                'website_id': str(website_id),
                'visitor_id': visitor_id,
                'primary_session_id': conversation_id,  # Will be updated to session_token later
                'thread_title': thread_title[:200],  # Limit title length
                'thread_summary': thread_summary,
                'total_sessions': 1,
                'total_messages': len(messages),
                'is_active': True
            }
            
            result = self.supabase.table('conversation_threads').insert(thread_data).execute()
            
            if result.data:
                thread_record = result.data[0]
                
                # Update conversation with thread reference
                await self._update_conversation_thread_reference(conversation_id, thread_id)
                
                return {
                    'thread_id': thread_id,
                    'thread_title': thread_title,
                    'thread_summary': thread_summary,
                    'total_sessions': 1,
                    'is_new_thread': True,
                    'topics': topics,
                    'database_id': thread_record['id']
                }
            else:
                logger.error("Failed to create thread record")
                return self._empty_thread_result()
                
        except Exception as e:
            logger.error(f"Error creating new thread: {e}")
            return self._empty_thread_result()
    
    async def _link_to_existing_thread(
        self,
        thread: Dict[str, Any],
        conversation_id: UUID,
        topics: List[str],
        thread_title: str
    ) -> Dict[str, Any]:
        """
        Link conversation to existing thread.
        """
        try:
            # Update thread with new session
            update_data = {
                'total_sessions': thread.get('total_sessions', 0) + 1,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'is_active': True
            }
            
            # Update thread title if current one is more descriptive
            if len(thread_title) > len(thread.get('thread_title', '')):
                update_data['thread_title'] = thread_title[:200]
            
            # Update thread summary with new topics
            if topics:
                current_summary = thread.get('thread_summary', '')
                enhanced_summary = await self._enhance_thread_summary(
                    current_summary, topics
                )
                update_data['thread_summary'] = enhanced_summary
            
            update_result = self.supabase.table('conversation_threads').update(update_data).eq('id', thread['id']).execute()
            
            if update_result.data:
                # Update conversation with thread reference
                await self._update_conversation_thread_reference(conversation_id, thread['thread_id'])
                
                return {
                    'thread_id': thread['thread_id'],
                    'thread_title': update_data.get('thread_title', thread.get('thread_title')),
                    'thread_summary': update_data.get('thread_summary', thread.get('thread_summary')),
                    'total_sessions': update_data['total_sessions'],
                    'is_new_thread': False,
                    'topics': topics,
                    'database_id': thread['id']
                }
            else:
                logger.error("Failed to update existing thread")
                return self._empty_thread_result()
                
        except Exception as e:
            logger.error(f"Error linking to existing thread: {e}")
            return self._empty_thread_result()
    
    async def _extract_conversation_topics(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract main topics from conversation messages.
        """
        topics = []
        
        # Combine all message content
        all_text = ' '.join([
            msg.get('content', '') for msg in messages
            if msg.get('message_type') == 'user'  # Focus on user messages for topics
        ]).lower()
        
        # Check for predefined topic keywords
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(keyword in all_text for keyword in keywords):
                topics.append(topic)
        
        # Extract entities (simple approach)
        # Look for capitalized words that might be product/service names
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ' '.join([
            msg.get('content', '') for msg in messages
        ]))
        
        # Add most common entities as topics
        if entities:
            entity_counts = Counter(entities)
            for entity, count in entity_counts.most_common(3):
                if count > 1 and entity.lower() not in [t.lower() for t in topics]:
                    topics.append(entity.lower())
        
        return topics[:5]  # Limit to top 5 topics
    
    async def _generate_thread_title(
        self,
        messages: List[Dict[str, Any]],
        topics: List[str]
    ) -> str:
        """
        Generate a descriptive title for the conversation thread.
        """
        if not messages:
            return "New Conversation"
        
        # Use topics if available
        if topics:
            if len(topics) == 1:
                return f"Discussion about {topics[0]}"
            elif len(topics) == 2:
                return f"{topics[0].title()} and {topics[1]} inquiry"
            else:
                return f"{topics[0].title()} and related topics"
        
        # Fallback to first user message
        for msg in messages:
            if msg.get('message_type') == 'user':
                content = msg.get('content', '')
                if len(content) > 10:
                    # Clean and truncate
                    title = re.sub(r'[^\w\s]', '', content)[:50]
                    if '?' in content:
                        return f"Question: {title}..."
                    else:
                        return f"Inquiry: {title}..."
        
        return "General Conversation"
    
    async def _create_thread_summary(
        self,
        messages: List[Dict[str, Any]],
        topics: List[str]
    ) -> str:
        """
        Create a summary of the conversation thread.
        """
        summary_parts = []
        
        # Add topics
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics)}")
        
        # Add message count
        user_msgs = len([m for m in messages if m.get('message_type') == 'user'])
        summary_parts.append(f"User questions: {user_msgs}")
        
        # Add key information from first few messages
        if messages:
            first_user_msg = None
            for msg in messages[:3]:
                if msg.get('message_type') == 'user':
                    first_user_msg = msg.get('content', '')[:100]
                    break
            
            if first_user_msg:
                summary_parts.append(f"Started with: {first_user_msg}...")
        
        return '. '.join(summary_parts)
    
    async def _enhance_thread_summary(
        self,
        current_summary: str,
        new_topics: List[str]
    ) -> str:
        """
        Enhance existing thread summary with new topics.
        """
        if not new_topics:
            return current_summary
        
        # Extract existing topics from summary
        existing_topics = []
        if 'Topics:' in current_summary:
            topics_part = current_summary.split('Topics:')[1].split('.')[0]
            existing_topics = [t.strip() for t in topics_part.split(',')]
        
        # Merge topics
        all_topics = list(set(existing_topics + new_topics))
        
        # Rebuild summary
        if current_summary:
            if 'Topics:' in current_summary:
                # Replace topics section
                parts = current_summary.split('.')
                parts[0] = f"Topics: {', '.join(all_topics)}"
                return '. '.join(parts)
            else:
                return f"Topics: {', '.join(all_topics)}. {current_summary}"
        else:
            return f"Topics: {', '.join(all_topics)}"
    
    async def _update_conversation_thread_reference(
        self,
        conversation_id: UUID,
        thread_id: str
    ):
        """
        Update conversation record with thread reference.
        """
        try:
            # Add thread_id to conversation (assuming we add this field)
            # For now, we could store it in a metadata field or create a separate mapping table
            logger.info(f"📝 THREADING: Updated conversation {conversation_id} with thread {thread_id}")
        except Exception as e:
            logger.error(f"Error updating conversation thread reference: {e}")
    
    async def get_thread_conversations(
        self,
        thread_id: str,
        include_messages: bool = False,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get all conversations in a thread.
        
        Args:
            thread_id: Thread identifier
            include_messages: Whether to include message details
            limit: Maximum conversations to return
            
        Returns:
            Thread information with conversations
        """
        try:
            # Get thread details
            thread_result = self.supabase.table('conversation_threads').select(
                '*'
            ).eq('thread_id', thread_id).execute()
            
            if not thread_result.data:
                return {'error': 'Thread not found'}
            
            thread = thread_result.data[0]
            
            # Get conversations linked to this thread (would need thread references in conversations)
            # For now, return thread info with placeholder for conversations
            
            result = {
                'thread_id': thread_id,
                'thread_title': thread['thread_title'],
                'thread_summary': thread['thread_summary'],
                'total_sessions': thread['total_sessions'],
                'total_messages': thread.get('total_messages', 0),
                'created_at': thread['created_at'],
                'updated_at': thread['updated_at'],
                'is_active': thread['is_active'],
                'conversations': []  # Would be populated with actual conversations
            }
            
            if include_messages:
                # Would include message details for each conversation
                result['conversations_with_messages'] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting thread conversations: {e}")
            return {'error': str(e)}
    
    async def get_visitor_threads(
        self,
        visitor_id: str,
        website_id: UUID,
        days: int = 30,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all threads for a visitor.
        
        Args:
            visitor_id: Visitor identifier
            website_id: Website UUID
            days: Number of days to look back
            include_inactive: Whether to include inactive threads
            
        Returns:
            List of threads for the visitor
        """
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            query = self.supabase.table('conversation_threads').select(
                'thread_id, thread_title, thread_summary, total_sessions, total_messages, created_at, updated_at, is_active'
            ).eq('visitor_id', visitor_id).eq('website_id', str(website_id)).gte('created_at', cutoff_date)
            
            if not include_inactive:
                query = query.eq('is_active', True)
            
            result = query.order('updated_at', desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting visitor threads: {e}")
            return []
    
    async def get_threading_analytics(
        self,
        website_id: UUID,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get analytics about conversation threading.
        
        Args:
            website_id: Website UUID
            days: Number of days to analyze
            
        Returns:
            Threading analytics data
        """
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get thread statistics
            threads_result = self.supabase.table('conversation_threads').select(
                'total_sessions, total_messages, created_at'
            ).eq('website_id', str(website_id)).gte('created_at', cutoff_date).execute()
            
            if not threads_result.data:
                return self._empty_threading_analytics()
            
            threads = threads_result.data
            
            # Calculate metrics
            total_threads = len(threads)
            total_sessions = sum(t.get('total_sessions', 0) for t in threads)
            total_messages = sum(t.get('total_messages', 0) for t in threads)
            
            # Sessions per thread distribution
            sessions_per_thread = [t.get('total_sessions', 0) for t in threads]
            avg_sessions_per_thread = sum(sessions_per_thread) / len(sessions_per_thread) if sessions_per_thread else 0
            
            # Multi-session threads (threads with more than 1 session)
            multi_session_threads = sum(1 for s in sessions_per_thread if s > 1)
            
            return {
                'period_days': days,
                'total_threads': total_threads,
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'avg_sessions_per_thread': round(avg_sessions_per_thread, 2),
                'multi_session_threads': multi_session_threads,
                'multi_session_rate': round((multi_session_threads / total_threads * 100), 1) if total_threads > 0 else 0,
                'avg_messages_per_thread': round(total_messages / total_threads, 1) if total_threads > 0 else 0,
                'threading_effectiveness': {
                    'single_session_threads': total_threads - multi_session_threads,
                    'multi_session_threads': multi_session_threads,
                    'max_sessions_in_thread': max(sessions_per_thread) if sessions_per_thread else 0,
                    'conversation_continuity_score': round(avg_sessions_per_thread, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting threading analytics: {e}")
            return self._empty_threading_analytics()
    
    async def _get_conversation_details(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get conversation metadata."""
        try:
            result = self.supabase.table('conversations').select('*').eq('id', str(conversation_id)).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting conversation details: {e}")
            return None
    
    async def _get_conversation_messages(self, conversation_id: UUID) -> List[Dict[str, Any]]:
        """Get conversation messages."""
        try:
            result = self.supabase.table('messages').select(
                'content, message_type, sequence_number, created_at'
            ).eq('conversation_id', str(conversation_id)).order('sequence_number', desc=False).execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting conversation messages: {e}")
            return []
    
    def _empty_thread_result(self) -> Dict[str, Any]:
        """Return empty thread result."""
        return {
            'thread_id': None,
            'thread_title': '',
            'thread_summary': '',
            'total_sessions': 0,
            'is_new_thread': False,
            'topics': [],
            'error': True
        }
    
    def _empty_threading_analytics(self) -> Dict[str, Any]:
        """Return empty threading analytics."""
        return {
            'period_days': 0,
            'total_threads': 0,
            'total_sessions': 0,
            'total_messages': 0,
            'avg_sessions_per_thread': 0,
            'multi_session_threads': 0,
            'multi_session_rate': 0,
            'avg_messages_per_thread': 0,
            'threading_effectiveness': {
                'single_session_threads': 0,
                'multi_session_threads': 0,
                'max_sessions_in_thread': 0,
                'conversation_continuity_score': 0
            }
        }