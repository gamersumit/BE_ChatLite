"""
Context Service for managing conversation context, history, and token optimization.
Provides intelligent context window management for AI chat responses.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from datetime import datetime, timezone
import tiktoken

from ..core.config import get_settings
from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)
settings = get_settings()


class ContextService:
    """Service for managing conversation context and history optimization."""
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def prepare_conversation_context(
        self,
        conversation_id: UUID,
        max_tokens: Optional[int] = None,
        include_system_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare optimized conversation context for AI response generation.
        
        Args:
            conversation_id: UUID of the conversation
            max_tokens: Maximum tokens to include (defaults to config)
            include_system_prompt: Whether to include system prompt in token count
            
        Returns:
            Dictionary containing optimized context and metadata
        """
        try:
            max_tokens = max_tokens or settings.context_window_size
            
            # Get conversation metadata
            conversation = await self._get_conversation_details(conversation_id)
            if not conversation:
                return self._empty_context()
            
            # Get conversation messages
            messages = await self._get_conversation_messages(conversation_id)
            if not messages:
                return self._empty_context()
            
            # Get existing context summary if available
            context_summary = conversation.get('context_summary')
            
            # Decide if we need to summarize older context
            should_summarize = len(messages) >= settings.context_summary_trigger
            
            if should_summarize and not context_summary:
                context_summary = await self._create_context_summary(messages[:-10])  # Summarize older messages
                await self._save_context_summary(conversation_id, context_summary)
            
            # Build optimized message history
            optimized_messages = self._optimize_message_history(
                messages, 
                max_tokens, 
                context_summary,
                include_system_prompt
            )
            
            # Calculate token usage
            total_tokens = self._count_tokens_in_messages(optimized_messages)
            if context_summary:
                total_tokens += self._count_tokens(context_summary)
            
            # Update conversation context tracking
            await self._update_context_tracking(conversation_id, total_tokens)
            
            logger.info(f"📋 CONTEXT: Prepared context for conversation {conversation_id} ({total_tokens} tokens)")
            
            return {
                'messages': optimized_messages,
                'context_summary': context_summary,
                'total_tokens': total_tokens,
                'messages_included': len(optimized_messages),
                'has_summary': bool(context_summary),
                'conversation_metadata': {
                    'total_messages': conversation.get('total_messages', 0),
                    'started_at': conversation.get('started_at'),
                    'last_activity_at': conversation.get('last_activity_at'),
                    'website_id': conversation.get('website_id')
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CONTEXT: Error preparing context for conversation {conversation_id}: {e}")
            return self._empty_context()
    
    async def add_message_to_context(
        self,
        conversation_id: UUID,
        message_content: str,
        message_type: str,  # "user" or "assistant"
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new message to the conversation context.
        
        Args:
            conversation_id: UUID of the conversation
            message_content: Content of the message
            message_type: Type of message ("user" or "assistant")
            metadata: Additional metadata (tokens_used, model_used, etc.)
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Get conversation to update sequence number
            conversation_result = self.supabase.table('conversations').select('total_messages, user_messages, ai_messages').eq('id', str(conversation_id)).execute()
            
            if not conversation_result.data:
                logger.error(f"❌ CONTEXT: Conversation not found: {conversation_id}")
                return False
            
            conversation = conversation_result.data[0]
            current_total = conversation.get('total_messages', 0)
            current_user = conversation.get('user_messages', 0)
            current_ai = conversation.get('ai_messages', 0)
            
            # Calculate message metrics
            word_count = len(message_content.split())
            character_count = len(message_content)
            token_count = self._count_tokens(message_content) if self.tokenizer else None
            
            # Create message record
            message_data = {
                'conversation_id': str(conversation_id),
                'content': message_content,
                'message_type': message_type,
                'sequence_number': current_total + 1,
                'word_count': word_count,
                'character_count': character_count
            }
            
            # Add metadata fields if provided (only add columns that exist in the table)
            if metadata:
                if 'tokens_used' in metadata:
                    message_data['tokens_used'] = metadata['tokens_used']
                # Skip columns that don't exist in current schema:
                # - model_used
                # - processing_time_ms  
                # - cost_usd
                # - context_sources
                # - confidence_score
            
            # Insert message
            message_result = self.supabase.table('messages').insert(message_data).execute()
            
            if not message_result.data:
                logger.error(f"❌ CONTEXT: Failed to insert message")
                return False
            
            # Update conversation counters
            new_totals = {
                'total_messages': current_total + 1,
                'last_activity_at': datetime.now(timezone.utc).isoformat()
            }
            
            if message_type == 'user':
                new_totals['user_messages'] = current_user + 1
            elif message_type == 'assistant':
                new_totals['ai_messages'] = current_ai + 1
            
            # Update conversation
            update_result = self.supabase.table('conversations').update(new_totals).eq('id', str(conversation_id)).execute()
            
            if update_result.data:
                logger.info(f"📝 CONTEXT: Added {message_type} message to conversation {conversation_id}")
                return True
            else:
                logger.warning(f"⚠️ CONTEXT: Message added but failed to update conversation counters")
                return True  # Message was still added
                
        except Exception as e:
            logger.error(f"❌ CONTEXT: Error adding message to conversation {conversation_id}: {e}")
            return False
    
    async def get_conversation_history(
        self, 
        conversation_id: UUID, 
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation message history.
        
        Args:
            conversation_id: UUID of the conversation
            limit: Maximum number of messages to return
            include_metadata: Whether to include message metadata
            
        Returns:
            List of messages with content and metadata
        """
        try:
            limit = limit or settings.max_context_messages
            
            # Build select query
            select_fields = 'id, content, message_type, sequence_number, created_at'
            if include_metadata:
                select_fields += ', word_count, character_count, tokens_used'
                # Note: model_used, processing_time_ms, confidence_score, context_sources don't exist in current schema
            
            # Query messages
            result = self.supabase.table('messages').select(select_fields).eq('conversation_id', str(conversation_id)).order('sequence_number', desc=False).limit(limit).execute()
            
            if not result.data:
                return []
            
            messages = []
            for msg in result.data:
                message_dict = {
                    'id': msg['id'],
                    'content': msg['content'],
                    'type': msg['message_type'],
                    'sequence': msg['sequence_number'],
                    'created_at': msg['created_at']
                }
                
                if include_metadata:
                    message_dict.update({
                        'word_count': msg.get('word_count'),
                        'character_count': msg.get('character_count'),
                        'tokens_used': msg.get('tokens_used')
                        # Note: model_used, processing_time_ms, confidence_score, context_sources not in current schema
                    })
                
                messages.append(message_dict)
            
            logger.info(f"📚 CONTEXT: Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages
            
        except Exception as e:
            logger.error(f"❌ CONTEXT: Error getting conversation history for {conversation_id}: {e}")
            return []
    
    async def _get_conversation_details(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get conversation metadata."""
        try:
            result = self.supabase.table('conversations').select('*').eq('id', str(conversation_id)).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting conversation details: {e}")
            return None
    
    async def _get_conversation_messages(self, conversation_id: UUID) -> List[Dict[str, Any]]:
        """Get all conversation messages ordered by sequence."""
        try:
            result = self.supabase.table('messages').select(
                'id, content, message_type, sequence_number, created_at, tokens_used'
            ).eq('conversation_id', str(conversation_id)).order('sequence_number', desc=False).execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting conversation messages: {e}")
            return []
    
    async def _create_context_summary(self, older_messages: List[Dict[str, Any]]) -> str:
        """
        Create a summary of older messages to save context space.
        For now, this is a simple implementation. Can be enhanced with AI summarization.
        """
        if not older_messages:
            return ""
        
        # Simple summarization: extract key topics and user intents
        user_messages = [msg for msg in older_messages if msg['message_type'] == 'user']
        ai_messages = [msg for msg in older_messages if msg['message_type'] == 'assistant']
        
        if len(user_messages) == 0:
            return ""
        
        # Extract key themes (simple keyword analysis)
        all_user_text = " ".join([msg['content'] for msg in user_messages])
        all_ai_text = " ".join([msg['content'] for msg in ai_messages])
        
        summary = f"Previous conversation covered: {len(user_messages)} user questions, {len(ai_messages)} AI responses. "
        
        # Add first and last user message as context
        if len(user_messages) > 0:
            summary += f"Initial topic: {user_messages[0]['content'][:100]}... "
        if len(user_messages) > 1:
            summary += f"Recent topic: {user_messages[-1]['content'][:100]}..."
        
        return summary
    
    async def _save_context_summary(self, conversation_id: UUID, summary: str):
        """Save context summary to conversation record."""
        try:
            self.supabase.table('conversations').update({
                'context_summary': summary,
                'last_context_update': datetime.now(timezone.utc).isoformat()
            }).eq('id', str(conversation_id)).execute()
        except Exception as e:
            logger.error(f"Error saving context summary: {e}")
    
    async def _update_context_tracking(self, conversation_id: UUID, tokens_used: int):
        """Update context token usage tracking."""
        try:
            self.supabase.table('conversations').update({
                'context_tokens_used': tokens_used,
                'last_context_update': datetime.now(timezone.utc).isoformat()
            }).eq('id', str(conversation_id)).execute()
        except Exception as e:
            logger.error(f"Error updating context tracking: {e}")
    
    def _optimize_message_history(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int,
        context_summary: Optional[str] = None,
        include_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """
        Optimize message history to fit within token limits.
        
        Returns:
            List of messages in OpenAI format [{"role": "user|assistant", "content": "..."}]
        """
        if not messages:
            return []
        
        # Reserve tokens for context summary and system prompt
        reserved_tokens = 0
        if context_summary:
            reserved_tokens += self._count_tokens(context_summary)
        if include_system_prompt:
            reserved_tokens += 200  # Estimate for system prompt
        
        available_tokens = max_tokens - reserved_tokens
        
        # Convert to OpenAI format and count tokens
        formatted_messages = []
        current_tokens = 0
        
        # Start from the most recent messages and work backwards
        for message in reversed(messages):
            role = "user" if message['message_type'] == 'user' else "assistant"
            content = message['content']
            
            message_tokens = self._count_tokens(content)
            
            # Check if adding this message would exceed the limit
            if current_tokens + message_tokens > available_tokens and len(formatted_messages) > 0:
                break
            
            formatted_messages.insert(0, {"role": role, "content": content})
            current_tokens += message_tokens
        
        logger.info(f"📊 CONTEXT: Optimized to {len(formatted_messages)} messages ({current_tokens} tokens)")
        return formatted_messages
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer or not text:
            # Fallback: rough estimate (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.33)
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return int(len(text.split()) * 1.33)
    
    def _count_tokens_in_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in message list."""
        total_tokens = 0
        for message in messages:
            total_tokens += self._count_tokens(message.get('content', ''))
            total_tokens += 3  # Add overhead for role and formatting
        return total_tokens
    
    def _empty_context(self) -> Dict[str, Any]:
        """Return empty context structure."""
        return {
            'messages': [],
            'context_summary': None,
            'total_tokens': 0,
            'messages_included': 0,
            'has_summary': False,
            'conversation_metadata': {}
        }