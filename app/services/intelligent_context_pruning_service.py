"""
Intelligent Context Pruning Service for optimal context window management.
Combines summarization and importance scoring to maintain conversation coherence.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
import tiktoken

from .conversation_summarization_service import ConversationSummarizationService
from .message_importance_service import MessageImportanceService
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IntelligentContextPruningService:
    """
    Service for intelligent context pruning that maintains conversation coherence
    while optimizing token usage.
    """
    
    # Pruning strategies
    STRATEGY_AGGRESSIVE = 'aggressive'  # Maximum compression, may lose some context
    STRATEGY_BALANCED = 'balanced'      # Balance between compression and context preservation
    STRATEGY_CONSERVATIVE = 'conservative'  # Minimal compression, preserve most context
    
    def __init__(self):
        self.summarization_service = ConversationSummarizationService()
        self.importance_service = MessageImportanceService()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ PRUNING: Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def prune_context(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        strategy: str = STRATEGY_BALANCED,
        preserve_recent: int = 5,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligently prune conversation context to fit within token limits.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum token budget
            strategy: Pruning strategy to use
            preserve_recent: Number of recent messages to always preserve
            include_summary: Whether to include summary of pruned content
            
        Returns:
            Dictionary containing pruned context and metadata
        """
        try:
            if not messages:
                return self._empty_pruned_context()
            
            # Calculate current token usage
            current_tokens = self._calculate_total_tokens(messages)
            
            # If already within limits, return as-is
            if current_tokens <= max_tokens:
                logger.info(f"✅ PRUNING: Context within limits ({current_tokens}/{max_tokens} tokens)")
                return {
                    'messages': messages,
                    'summary': None,
                    'total_tokens': current_tokens,
                    'pruned_messages': 0,
                    'strategy_used': 'none',
                    'compression_achieved': 1.0
                }
            
            logger.info(f"🔄 PRUNING: Optimizing context from {current_tokens} to {max_tokens} tokens")
            
            # Apply pruning based on strategy
            if strategy == self.STRATEGY_AGGRESSIVE:
                result = await self._aggressive_pruning(
                    messages, max_tokens, preserve_recent, include_summary
                )
            elif strategy == self.STRATEGY_CONSERVATIVE:
                result = await self._conservative_pruning(
                    messages, max_tokens, preserve_recent, include_summary
                )
            else:  # STRATEGY_BALANCED
                result = await self._balanced_pruning(
                    messages, max_tokens, preserve_recent, include_summary
                )
            
            # Calculate compression metrics
            final_tokens = self._calculate_total_tokens(result['messages'])
            if result.get('summary'):
                final_tokens += self._count_tokens(result['summary'])
            
            result['total_tokens'] = final_tokens
            result['compression_achieved'] = current_tokens / final_tokens if final_tokens > 0 else 1.0
            result['strategy_used'] = strategy
            
            logger.info(f"📊 PRUNING: Reduced from {current_tokens} to {final_tokens} tokens "
                       f"({result['compression_achieved']:.1f}x compression)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PRUNING: Error pruning context: {e}")
            return self._empty_pruned_context()
    
    async def _aggressive_pruning(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_recent: int,
        include_summary: bool
    ) -> Dict[str, Any]:
        """
        Aggressive pruning: Maximum compression with summarization.
        """
        # Separate recent messages to preserve
        recent_messages = messages[-preserve_recent:] if len(messages) > preserve_recent else messages
        older_messages = messages[:-preserve_recent] if len(messages) > preserve_recent else []
        
        # Calculate token budget
        recent_tokens = self._calculate_total_tokens(recent_messages)
        remaining_budget = max_tokens - recent_tokens
        
        # Summarize older messages aggressively
        summary = None
        summary_tokens = 0
        if older_messages and include_summary and remaining_budget > 100:
            summary_result = await self.summarization_service.summarize_conversation(
                older_messages,
                target_tokens=min(remaining_budget // 2, 300),
                preserve_key_points=False  # Aggressive: don't preserve all details
            )
            summary = summary_result['summary']
            summary_tokens = summary_result['summary_tokens']
            remaining_budget -= summary_tokens
        
        # Select only most important older messages
        selected_older = []
        if remaining_budget > 50 and older_messages:
            # Score and select messages
            importance_threshold = 1.5  # High threshold for aggressive pruning
            selected_older = self.importance_service.select_important_messages(
                older_messages,
                max_messages=3,  # Very few messages
                min_importance_score=importance_threshold
            )
            
            # Further filter by tokens
            filtered_older = []
            tokens_used = 0
            for msg in selected_older:
                msg_tokens = self._count_tokens(msg.get('content', ''))
                if tokens_used + msg_tokens <= remaining_budget:
                    filtered_older.append(msg)
                    tokens_used += msg_tokens
            selected_older = filtered_older
        
        # Combine results
        final_messages = selected_older + recent_messages
        
        return {
            'messages': final_messages,
            'summary': summary,
            'pruned_messages': len(messages) - len(final_messages),
            'summary_tokens': summary_tokens
        }
    
    async def _balanced_pruning(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_recent: int,
        include_summary: bool
    ) -> Dict[str, Any]:
        """
        Balanced pruning: Moderate compression with importance-based selection.
        """
        # Separate recent messages to preserve
        recent_messages = messages[-preserve_recent:] if len(messages) > preserve_recent else messages
        older_messages = messages[:-preserve_recent] if len(messages) > preserve_recent else []
        
        # Calculate token budget
        recent_tokens = self._calculate_total_tokens(recent_messages)
        remaining_budget = max_tokens - recent_tokens
        
        # Create moderate summary if needed
        summary = None
        summary_tokens = 0
        if len(older_messages) > 10 and include_summary and remaining_budget > 200:
            # Summarize very old messages (keep more recent ones)
            very_old = older_messages[:-5] if len(older_messages) > 5 else []
            if very_old:
                summary_result = await self.summarization_service.summarize_conversation(
                    very_old,
                    target_tokens=min(remaining_budget // 3, 400),
                    preserve_key_points=True  # Balanced: preserve key points
                )
                summary = summary_result['summary']
                summary_tokens = summary_result['summary_tokens']
                remaining_budget -= summary_tokens
                
                # Update older messages to exclude summarized ones
                older_messages = older_messages[-5:] if len(older_messages) > 5 else older_messages
        
        # Score and select important messages
        selected_older = []
        if remaining_budget > 100 and older_messages:
            # Use moderate importance threshold
            importance_threshold = 1.0  # Average importance
            scored_messages = self.importance_service.score_conversation(older_messages)
            
            # Select messages based on importance and token budget
            tokens_used = 0
            for msg, score in scored_messages:
                if score >= importance_threshold:
                    msg_tokens = self._count_tokens(msg.get('content', ''))
                    if tokens_used + msg_tokens <= remaining_budget:
                        selected_older.append(msg)
                        tokens_used += msg_tokens
                    elif tokens_used < remaining_budget // 2:
                        # Include at least some important messages
                        truncated_msg = self._truncate_message(msg, remaining_budget - tokens_used)
                        if truncated_msg:
                            selected_older.append(truncated_msg)
                            break
            
            # Sort back to conversation order
            selected_older.sort(key=lambda m: m.get('sequence_number', m.get('sequence', 0)))
        
        # Combine results
        final_messages = selected_older + recent_messages
        
        return {
            'messages': final_messages,
            'summary': summary,
            'pruned_messages': len(messages) - len(final_messages),
            'summary_tokens': summary_tokens
        }
    
    async def _conservative_pruning(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_recent: int,
        include_summary: bool
    ) -> Dict[str, Any]:
        """
        Conservative pruning: Minimal compression, preserve most context.
        """
        # Calculate how many messages we can keep
        tokens_per_message = self._calculate_total_tokens(messages) // len(messages)
        max_messages = max_tokens // tokens_per_message if tokens_per_message > 0 else len(messages)
        
        # Keep as many recent messages as possible
        if len(messages) <= max_messages:
            return {
                'messages': messages,
                'summary': None,
                'pruned_messages': 0,
                'summary_tokens': 0
            }
        
        # Minimal pruning: remove only oldest messages
        messages_to_keep = max(preserve_recent, min(max_messages, len(messages) - 2))
        kept_messages = messages[-messages_to_keep:]
        pruned_messages = messages[:-messages_to_keep]
        
        # Create brief summary of pruned content if requested
        summary = None
        summary_tokens = 0
        if pruned_messages and include_summary:
            # Very brief summary
            summary_result = await self.summarization_service.summarize_conversation(
                pruned_messages,
                target_tokens=100,
                preserve_key_points=True
            )
            summary = summary_result['summary']
            summary_tokens = summary_result['summary_tokens']
        
        return {
            'messages': kept_messages,
            'summary': summary,
            'pruned_messages': len(messages) - len(kept_messages),
            'summary_tokens': summary_tokens
        }
    
    def _truncate_message(self, message: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Truncate a message to fit within token limit.
        """
        content = message.get('content', '')
        tokens = self._count_tokens(content)
        
        if tokens <= max_tokens:
            return message
        
        # Estimate characters per token
        chars_per_token = len(content) / tokens if tokens > 0 else 4
        target_length = int(max_tokens * chars_per_token * 0.9)  # 90% to be safe
        
        if target_length < 50:  # Too short to be useful
            return None
        
        truncated_message = message.copy()
        truncated_message['content'] = content[:target_length] + "..."
        truncated_message['truncated'] = True
        
        return truncated_message
    
    def _calculate_total_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Calculate total tokens in messages.
        """
        total = 0
        for msg in messages:
            total += self._count_tokens(msg.get('content', ''))
            total += 3  # Account for message formatting overhead
        return total
    
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
    
    def _empty_pruned_context(self) -> Dict[str, Any]:
        """
        Return empty pruned context structure.
        """
        return {
            'messages': [],
            'summary': None,
            'total_tokens': 0,
            'pruned_messages': 0,
            'strategy_used': 'none',
            'compression_achieved': 1.0
        }
    
    async def optimize_context_for_response(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        max_total_tokens: int = 4000,
        response_reserve_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Optimize context specifically for generating an AI response.
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt to include
            max_total_tokens: Total token budget (including response)
            response_reserve_tokens: Tokens to reserve for AI response
            
        Returns:
            Optimized context ready for AI processing
        """
        # Calculate available tokens for context
        system_tokens = self._count_tokens(system_prompt)
        available_for_context = max_total_tokens - system_tokens - response_reserve_tokens
        
        # Determine pruning strategy based on how much we need to compress
        current_tokens = self._calculate_total_tokens(messages)
        compression_needed = current_tokens / available_for_context if available_for_context > 0 else float('inf')
        
        if compression_needed > 3:
            strategy = self.STRATEGY_AGGRESSIVE
        elif compression_needed > 1.5:
            strategy = self.STRATEGY_BALANCED
        else:
            strategy = self.STRATEGY_CONSERVATIVE
        
        # Prune context
        pruned_result = await self.prune_context(
            messages=messages,
            max_tokens=available_for_context,
            strategy=strategy,
            preserve_recent=min(5, len(messages) // 2),
            include_summary=True
        )
        
        # Format for AI processing
        formatted_messages = []
        
        # Add summary as system context if available
        if pruned_result.get('summary'):
            formatted_messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {pruned_result['summary']}"
            })
        
        # Add conversation messages
        for msg in pruned_result['messages']:
            role = "user" if msg.get('message_type', msg.get('type')) == 'user' else "assistant"
            formatted_messages.append({
                "role": role,
                "content": msg.get('content', '')
            })
        
        return {
            'messages': formatted_messages,
            'system_prompt': system_prompt,
            'total_context_tokens': pruned_result['total_tokens'] + system_tokens,
            'available_for_response': response_reserve_tokens,
            'pruning_strategy': pruned_result['strategy_used'],
            'messages_pruned': pruned_result['pruned_messages'],
            'compression_achieved': pruned_result['compression_achieved']
        }