"""
Optimized Context Service that integrates all context optimization features.
Combines summarization, importance scoring, and intelligent pruning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timezone

from .context_service import ContextService
from .conversation_summarization_service import ConversationSummarizationService
from .message_importance_service import MessageImportanceService
from .intelligent_context_pruning_service import IntelligentContextPruningService
from ..core.config import get_settings
from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)
settings = get_settings()


class OptimizedContextService(ContextService):
    """
    Enhanced context service with advanced optimization features.
    """
    
    def __init__(self):
        super().__init__()
        self.summarization_service = ConversationSummarizationService()
        self.importance_service = MessageImportanceService()
        self.pruning_service = IntelligentContextPruningService()
        self.supabase = get_supabase_admin()
    
    async def prepare_optimized_context(
        self,
        conversation_id: UUID,
        max_tokens: Optional[int] = None,
        optimization_level: str = 'balanced',
        include_importance_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare optimized conversation context with advanced features.
        
        Args:
            conversation_id: UUID of the conversation
            max_tokens: Maximum tokens to include (defaults to config)
            optimization_level: 'aggressive', 'balanced', or 'conservative'
            include_importance_scores: Whether to include message importance scores
            
        Returns:
            Dictionary containing optimized context and rich metadata
        """
        try:
            max_tokens = max_tokens or settings.context_window_size
            
            # Get conversation details
            conversation = await self._get_conversation_details(conversation_id)
            if not conversation:
                return self._empty_optimized_context()
            
            # Get all messages
            messages = await self._get_conversation_messages(conversation_id)
            if not messages:
                return self._empty_optimized_context()
            
            logger.info(f"🔄 CONTEXT: Optimizing context for conversation {conversation_id} "
                       f"({len(messages)} messages)")
            
            # Check if summarization is needed
            should_summarize = await self.summarization_service.should_summarize_conversation(
                conversation_id, messages
            )
            
            # Perform summarization if needed
            summary = None
            if should_summarize:
                summary_result = await self.summarization_service.summarize_conversation(
                    messages[:-10],  # Summarize older messages
                    target_tokens=300,
                    preserve_key_points=True,
                    include_metadata=True
                )
                summary = summary_result['summary']
                
                # Store summary
                await self.summarization_service.store_conversation_summary(
                    conversation_id, summary, summary_result
                )
            else:
                # Use existing summary if available
                summary = conversation.get('context_summary')
            
            # Score message importance
            importance_stats = None
            if include_importance_scores:
                importance_stats = self.importance_service.get_importance_statistics(messages)
            
            # Apply intelligent pruning
            pruning_strategy = {
                'aggressive': IntelligentContextPruningService.STRATEGY_AGGRESSIVE,
                'balanced': IntelligentContextPruningService.STRATEGY_BALANCED,
                'conservative': IntelligentContextPruningService.STRATEGY_CONSERVATIVE
            }.get(optimization_level, IntelligentContextPruningService.STRATEGY_BALANCED)
            
            pruned_result = await self.pruning_service.prune_context(
                messages=messages,
                max_tokens=max_tokens,
                strategy=pruning_strategy,
                preserve_recent=min(7, len(messages) // 3),
                include_summary=bool(summary)
            )
            
            # Format messages for AI processing
            formatted_messages = self._format_messages_for_ai(pruned_result['messages'])
            
            # Calculate final metrics
            total_tokens = pruned_result['total_tokens']
            
            # Update conversation tracking
            await self._update_optimized_context_tracking(
                conversation_id,
                total_tokens,
                len(pruned_result['messages']),
                pruned_result.get('pruned_messages', 0)
            )
            
            result = {
                'messages': formatted_messages,
                'raw_messages': pruned_result['messages'],
                'context_summary': summary or pruned_result.get('summary'),
                'total_tokens': total_tokens,
                'messages_included': len(pruned_result['messages']),
                'messages_pruned': pruned_result.get('pruned_messages', 0),
                'has_summary': bool(summary or pruned_result.get('summary')),
                'optimization_level': optimization_level,
                'compression_achieved': pruned_result.get('compression_achieved', 1.0),
                'conversation_metadata': {
                    'conversation_id': str(conversation_id),
                    'total_messages': conversation.get('total_messages', 0),
                    'started_at': conversation.get('started_at'),
                    'last_activity_at': conversation.get('last_activity_at'),
                    'website_id': conversation.get('website_id')
                }
            }
            
            # Add importance statistics if requested
            if importance_stats:
                result['importance_statistics'] = importance_stats
            
            logger.info(f"✅ CONTEXT: Optimized context ready - {total_tokens} tokens, "
                       f"{len(formatted_messages)} messages, "
                       f"{pruned_result.get('compression_achieved', 1.0):.1f}x compression")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ CONTEXT: Error preparing optimized context: {e}")
            return self._empty_optimized_context()
    
    async def add_message_with_optimization(
        self,
        conversation_id: UUID,
        message_content: str,
        message_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Add a message and automatically optimize context if needed.
        
        Args:
            conversation_id: UUID of the conversation
            message_content: Content of the message
            message_type: Type of message ("user" or "assistant")
            metadata: Additional metadata
            auto_optimize: Whether to automatically optimize context
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Calculate importance score for the new message
            importance_score = self.importance_service.score_message(
                {'content': message_content, 'message_type': message_type}
            )
            
            # Enhanced metadata with importance
            enhanced_metadata = metadata or {}
            enhanced_metadata['context_importance'] = importance_score
            
            # Add message using parent method
            success = await super().add_message_to_context(
                conversation_id,
                message_content,
                message_type,
                enhanced_metadata
            )
            
            if not success:
                return {
                    'success': False,
                    'error': 'Failed to add message'
                }
            
            result = {
                'success': True,
                'importance_score': importance_score,
                'message_type': message_type
            }
            
            # Check if optimization is needed
            if auto_optimize:
                optimization_result = await self._check_and_optimize(conversation_id)
                if optimization_result:
                    result['optimization_triggered'] = True
                    result['optimization_result'] = optimization_result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ CONTEXT: Error adding message with optimization: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _check_and_optimize(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Check if context optimization is needed and perform it.
        """
        try:
            # Get current conversation stats
            conversation = await self._get_conversation_details(conversation_id)
            if not conversation:
                return None
            
            total_messages = conversation.get('total_messages', 0)
            
            # Check optimization triggers
            needs_optimization = False
            
            # Trigger 1: Message count threshold
            if total_messages >= settings.context_summary_trigger:
                needs_optimization = True
            
            # Trigger 2: Token usage threshold
            if conversation.get('context_tokens_used', 0) > settings.context_window_size * 0.8:
                needs_optimization = True
            
            # Trigger 3: Time since last optimization
            last_update = conversation.get('last_context_update')
            if last_update:
                last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                time_since = (datetime.now(timezone.utc) - last_update_time).total_seconds()
                if time_since > 3600:  # More than 1 hour
                    needs_optimization = True
            
            if not needs_optimization:
                return None
            
            # Perform optimization
            logger.info(f"🔧 CONTEXT: Triggering automatic optimization for conversation {conversation_id}")
            
            messages = await self._get_conversation_messages(conversation_id)
            
            # Create summary of older messages
            if len(messages) > 15:
                summary_result = await self.summarization_service.summarize_conversation(
                    messages[:-10],
                    target_tokens=400,
                    preserve_key_points=True
                )
                
                # Store summary
                await self.summarization_service.store_conversation_summary(
                    conversation_id,
                    summary_result['summary'],
                    summary_result
                )
                
                return {
                    'optimization_performed': True,
                    'summary_created': True,
                    'compression_ratio': summary_result['compression_ratio']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking and optimizing context: {e}")
            return None
    
    def _format_messages_for_ai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format messages for AI processing with role mapping.
        """
        formatted = []
        for msg in messages:
            role = "user" if msg.get('message_type', msg.get('type')) == 'user' else "assistant"
            formatted.append({
                "role": role,
                "content": msg.get('content', '')
            })
        return formatted
    
    async def _update_optimized_context_tracking(
        self,
        conversation_id: UUID,
        tokens_used: int,
        messages_included: int,
        messages_pruned: int
    ):
        """
        Update conversation with optimization metrics.
        """
        try:
            update_data = {
                'context_tokens_used': tokens_used,
                'last_context_update': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase.table('conversations').update(update_data).eq('id', str(conversation_id)).execute()
            
            # Also log optimization metrics for analytics
            optimization_log = {
                'conversation_id': str(conversation_id),
                'tokens_used': tokens_used,
                'messages_included': messages_included,
                'messages_pruned': messages_pruned,
                'optimized_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Could store in a separate optimization_logs table if needed
            logger.info(f"📊 CONTEXT: Optimization metrics - {messages_included} messages, "
                       f"{messages_pruned} pruned, {tokens_used} tokens")
            
        except Exception as e:
            logger.error(f"Error updating optimization tracking: {e}")
    
    def _empty_optimized_context(self) -> Dict[str, Any]:
        """
        Return empty optimized context structure.
        """
        return {
            'messages': [],
            'raw_messages': [],
            'context_summary': None,
            'total_tokens': 0,
            'messages_included': 0,
            'messages_pruned': 0,
            'has_summary': False,
            'optimization_level': 'none',
            'compression_achieved': 1.0,
            'conversation_metadata': {}
        }
    
    async def get_optimization_statistics(
        self,
        conversation_id: UUID
    ) -> Dict[str, Any]:
        """
        Get detailed optimization statistics for a conversation.
        """
        try:
            conversation = await self._get_conversation_details(conversation_id)
            messages = await self._get_conversation_messages(conversation_id)
            
            if not conversation or not messages:
                return {
                    'error': 'Conversation not found',
                    'statistics': {}
                }
            
            # Get importance statistics
            importance_stats = self.importance_service.get_importance_statistics(messages)
            
            # Calculate token statistics
            total_tokens = sum(self._count_tokens(msg.get('content', '')) for msg in messages)
            avg_tokens_per_message = total_tokens // len(messages) if messages else 0
            
            # Check if summarization would help
            needs_summarization = await self.summarization_service.should_summarize_conversation(
                conversation_id, messages
            )
            
            return {
                'conversation_id': str(conversation_id),
                'total_messages': len(messages),
                'total_tokens': total_tokens,
                'avg_tokens_per_message': avg_tokens_per_message,
                'current_context_tokens': conversation.get('context_tokens_used', 0),
                'has_summary': bool(conversation.get('context_summary')),
                'needs_summarization': needs_summarization,
                'importance_statistics': importance_stats,
                'optimization_potential': {
                    'aggressive': f"{(total_tokens * 0.2 / total_tokens * 100):.0f}% size",
                    'balanced': f"{(total_tokens * 0.5 / total_tokens * 100):.0f}% size",
                    'conservative': f"{(total_tokens * 0.8 / total_tokens * 100):.0f}% size"
                },
                'last_optimization': conversation.get('last_context_update')
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization statistics: {e}")
            return {
                'error': str(e),
                'statistics': {}
            }