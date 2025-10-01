"""
Conversation Summarization Service for intelligent context optimization.
Uses LLM to create concise summaries of long conversations to save tokens.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
import tiktoken
from openai import AsyncOpenAI

from ..core.config import get_settings
from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationSummarizationService:
    """
    Service for creating intelligent conversation summaries to optimize context windows.
    """
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if hasattr(settings, 'openai_api_key') else None
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ SUMMARY: Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int = 500,
        preserve_key_points: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Create an intelligent summary of conversation messages.
        
        Args:
            messages: List of conversation messages to summarize
            target_tokens: Target token count for the summary
            preserve_key_points: Whether to preserve key decision points
            include_metadata: Whether to include conversation metadata in summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            if not messages or len(messages) < 3:
                return self._empty_summary()
            
            # Separate user and assistant messages
            user_messages = [msg for msg in messages if msg.get('type') == 'user' or msg.get('message_type') == 'user']
            assistant_messages = [msg for msg in messages if msg.get('type') == 'assistant' or msg.get('message_type') == 'assistant']
            
            # Use LLM for intelligent summarization if available
            if self.openai_client:
                summary = await self._llm_summarization(
                    user_messages, 
                    assistant_messages, 
                    target_tokens,
                    preserve_key_points
                )
            else:
                # Fallback to rule-based summarization
                summary = self._rule_based_summarization(
                    user_messages,
                    assistant_messages,
                    target_tokens
                )
            
            # Calculate summary metrics
            summary_tokens = self._count_tokens(summary)
            compression_ratio = self._calculate_compression_ratio(messages, summary)
            
            result = {
                'summary': summary,
                'summary_tokens': summary_tokens,
                'original_message_count': len(messages),
                'compression_ratio': compression_ratio,
                'summarization_method': 'llm' if self.openai_client else 'rule_based',
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            if include_metadata:
                result['metadata'] = self._extract_conversation_metadata(messages)
            
            logger.info(f"📝 SUMMARY: Created summary ({summary_tokens} tokens, {compression_ratio:.1f}x compression)")
            return result
            
        except Exception as e:
            logger.error(f"❌ SUMMARY: Error creating conversation summary: {e}")
            return self._empty_summary()
    
    async def _llm_summarization(
        self,
        user_messages: List[Dict[str, Any]],
        assistant_messages: List[Dict[str, Any]],
        target_tokens: int,
        preserve_key_points: bool
    ) -> str:
        """
        Use LLM to create an intelligent conversation summary.
        """
        try:
            # Prepare conversation for summarization
            conversation_text = self._format_conversation_for_llm(user_messages, assistant_messages)
            
            # Create summarization prompt
            prompt = self._create_summarization_prompt(target_tokens, preserve_key_points)
            
            # Call OpenAI for summarization
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=target_tokens,
                temperature=0.3  # Lower temperature for consistent summaries
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ SUMMARY: LLM summarization failed: {e}")
            # Fallback to rule-based
            return self._rule_based_summarization(user_messages, assistant_messages, target_tokens)
    
    def _rule_based_summarization(
        self,
        user_messages: List[Dict[str, Any]],
        assistant_messages: List[Dict[str, Any]],
        target_tokens: int
    ) -> str:
        """
        Create a rule-based summary when LLM is not available.
        """
        summary_parts = []
        
        # Extract key topics from user messages
        if user_messages:
            topics = self._extract_topics(user_messages)
            summary_parts.append(f"User discussed: {', '.join(topics[:5])}")
        
        # Extract key information from assistant responses
        if assistant_messages:
            key_points = self._extract_key_points(assistant_messages)
            if key_points:
                summary_parts.append(f"Key information provided: {', '.join(key_points[:3])}")
        
        # Add conversation flow summary
        if user_messages:
            summary_parts.append(f"Conversation included {len(user_messages)} questions")
        
        # Add initial and recent context
        if user_messages:
            first_topic = self._extract_main_topic(user_messages[0]['content'])
            summary_parts.append(f"Started with: {first_topic}")
            
            if len(user_messages) > 1:
                last_topic = self._extract_main_topic(user_messages[-1]['content'])
                summary_parts.append(f"Most recent: {last_topic}")
        
        summary = ". ".join(summary_parts)
        
        # Truncate to target tokens
        while self._count_tokens(summary) > target_tokens and len(summary_parts) > 1:
            summary_parts.pop()
            summary = ". ".join(summary_parts)
        
        return summary
    
    def _format_conversation_for_llm(
        self,
        user_messages: List[Dict[str, Any]],
        assistant_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Format conversation messages for LLM processing.
        """
        all_messages = []
        
        # Combine and sort messages by sequence
        for msg in user_messages:
            all_messages.append({
                'sequence': msg.get('sequence', msg.get('sequence_number', 0)),
                'role': 'User',
                'content': msg['content'][:200]  # Limit each message
            })
        
        for msg in assistant_messages:
            all_messages.append({
                'sequence': msg.get('sequence', msg.get('sequence_number', 0)),
                'role': 'Assistant',
                'content': msg['content'][:200]
            })
        
        # Sort by sequence
        all_messages.sort(key=lambda x: x['sequence'])
        
        # Format as conversation
        formatted = []
        for msg in all_messages:
            formatted.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def _create_summarization_prompt(self, target_tokens: int, preserve_key_points: bool) -> str:
        """
        Create the prompt for LLM summarization.
        """
        base_prompt = f"""You are a conversation summarization expert. Create a concise summary of the following conversation in approximately {target_tokens // 4} words.

Focus on:
1. Main topics discussed
2. Key questions asked by the user
3. Important information provided
4. Any decisions or conclusions reached"""

        if preserve_key_points:
            base_prompt += """
5. Preserve ALL critical information like:
   - Specific product/feature names mentioned
   - Price points or numbers discussed
   - Action items or next steps agreed upon
   - Contact information or references shared"""

        base_prompt += """

Provide the summary in a clear, narrative format that captures the essence of the conversation while maintaining context for future reference."""

        return base_prompt
    
    def _extract_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract main topics from user messages.
        """
        topics = []
        
        for msg in messages:
            content = msg['content'].lower()
            
            # Extract question words as topics
            for question_word in ['what', 'how', 'why', 'when', 'where', 'which']:
                if question_word in content:
                    # Extract the phrase after the question word
                    parts = content.split(question_word)
                    if len(parts) > 1:
                        topic = parts[1].split('?')[0].strip()[:50]
                        if topic and topic not in topics:
                            topics.append(topic)
                            break
        
        return topics
    
    def _extract_key_points(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key points from assistant messages.
        """
        key_points = []
        
        for msg in messages:
            content = msg['content']
            
            # Look for structured information
            if any(marker in content for marker in [':', '•', '-', '1.', '*']):
                # Extract first structured point
                for marker in [':', '•', '-', '1.']:
                    if marker in content:
                        point = content.split(marker)[1].split('\n')[0].strip()[:100]
                        if point and len(point) > 20:
                            key_points.append(point)
                            break
        
        return key_points
    
    def _extract_main_topic(self, content: str) -> str:
        """
        Extract the main topic from a message.
        """
        # Remove common phrases
        content = content.lower()
        for phrase in ['i need', 'i want', 'can you', 'could you', 'please', 'help me']:
            content = content.replace(phrase, '')
        
        # Take first meaningful part
        sentences = content.split('.')
        if sentences:
            return sentences[0].strip()[:100]
        
        return content.strip()[:100]
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        """
        if not self.tokenizer or not text:
            return int(len(text.split()) * 1.33)
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return int(len(text.split()) * 1.33)
    
    def _calculate_compression_ratio(self, messages: List[Dict[str, Any]], summary: str) -> float:
        """
        Calculate compression ratio.
        """
        original_tokens = sum(self._count_tokens(msg.get('content', '')) for msg in messages)
        summary_tokens = self._count_tokens(summary)
        
        if summary_tokens == 0:
            return 0.0
        
        return original_tokens / summary_tokens
    
    def _extract_conversation_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata about the conversation.
        """
        return {
            'total_messages': len(messages),
            'user_messages': len([m for m in messages if m.get('type') == 'user' or m.get('message_type') == 'user']),
            'assistant_messages': len([m for m in messages if m.get('type') == 'assistant' or m.get('message_type') == 'assistant']),
            'avg_message_length': sum(len(m.get('content', '')) for m in messages) // len(messages) if messages else 0,
            'total_tokens': sum(self._count_tokens(m.get('content', '')) for m in messages)
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """
        Return empty summary structure.
        """
        return {
            'summary': '',
            'summary_tokens': 0,
            'original_message_count': 0,
            'compression_ratio': 0.0,
            'summarization_method': 'none',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    async def should_summarize_conversation(
        self,
        conversation_id: UUID,
        messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if a conversation should be summarized.
        
        Args:
            conversation_id: UUID of the conversation
            messages: Current conversation messages
            
        Returns:
            True if summarization should occur
        """
        # Check message count threshold
        if len(messages) < settings.context_summary_trigger:
            return False
        
        # Check token count
        total_tokens = sum(self._count_tokens(msg.get('content', '')) for msg in messages)
        if total_tokens < settings.context_window_size * 0.7:  # 70% of window
            return False
        
        # Check if recently summarized
        try:
            result = self.supabase.table('conversations').select('last_context_update').eq('id', str(conversation_id)).execute()
            if result.data:
                last_update = result.data[0].get('last_context_update')
                if last_update:
                    last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    time_since_update = datetime.now(timezone.utc) - last_update_time
                    
                    # Don't summarize if updated in last hour
                    if time_since_update.total_seconds() < 3600:
                        return False
        except Exception as e:
            logger.error(f"Error checking last summarization time: {e}")
        
        return True
    
    async def store_conversation_summary(
        self,
        conversation_id: UUID,
        summary: str,
        summary_metadata: Dict[str, Any]
    ) -> bool:
        """
        Store conversation summary in database.
        
        Args:
            conversation_id: UUID of the conversation
            summary: The summary text
            summary_metadata: Metadata about the summary
            
        Returns:
            True if successfully stored
        """
        try:
            # Update conversation with summary
            update_data = {
                'context_summary': summary,
                'last_context_update': datetime.now(timezone.utc).isoformat(),
                'context_tokens_used': summary_metadata.get('summary_tokens', 0)
            }
            
            result = self.supabase.table('conversations').update(update_data).eq('id', str(conversation_id)).execute()
            
            if result.data:
                logger.info(f"✅ SUMMARY: Stored summary for conversation {conversation_id}")
                return True
            else:
                logger.error(f"❌ SUMMARY: Failed to store summary for conversation {conversation_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ SUMMARY: Error storing summary: {e}")
            return False