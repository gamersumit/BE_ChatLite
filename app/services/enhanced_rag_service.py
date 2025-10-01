"""
Enhanced RAG Service that integrates conversation history with website content.
Provides balanced context combining conversation history and website content.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
from openai import AsyncOpenAI

from .rag_chat_service import RAGChatService
from .optimized_context_service import OptimizedContextService
from .vector_search_service import VectorSearchService
from .redis_session_cache import RedisSessionCache
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedRAGService(RAGChatService):
    """
    Enhanced RAG service that balances conversation history and website content.
    """
    
    # Context allocation strategies
    CONTEXT_SPLIT_BALANCED = 'balanced'      # 50% conversation, 50% website
    CONTEXT_SPLIT_CONVERSATION = 'conversation'  # 70% conversation, 30% website
    CONTEXT_SPLIT_WEBSITE = 'website'        # 30% conversation, 70% website
    CONTEXT_SPLIT_ADAPTIVE = 'adaptive'      # Dynamic based on relevance
    
    def __init__(self):
        super().__init__()
        self.optimized_context = OptimizedContextService()
        self.vector_service = VectorSearchService()
        self.redis_cache = RedisSessionCache()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if hasattr(settings, 'openai_api_key') else None
    
    async def generate_enhanced_rag_response(
        self,
        user_message: str,
        conversation_id: UUID,
        website_id: UUID,
        context_strategy: str = CONTEXT_SPLIT_ADAPTIVE,
        max_total_tokens: int = 4000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate AI response using enhanced RAG with conversation history.
        
        Args:
            user_message: Current user message
            conversation_id: UUID of the conversation
            website_id: UUID of the website
            context_strategy: How to split context between conversation and website
            max_total_tokens: Maximum total token budget
            include_sources: Whether to include source information
            
        Returns:
            Enhanced response with conversation and website context
        """
        try:
            start_time = datetime.now()
            
            # Check cache first
            cache_key = f"rag_context_{conversation_id}_{hash(user_message)}"
            cached_context = await self.redis_cache.get_cached_context(conversation_id)
            
            # Prepare conversation context
            conversation_context = await self.optimized_context.prepare_optimized_context(
                conversation_id=conversation_id,
                max_tokens=max_total_tokens // 2,  # Initial allocation
                optimization_level='balanced',
                include_importance_scores=True
            )
            
            # Get website content context
            website_context = await self.vector_service.get_context_for_query_improved(
                user_message, website_id
            )
            
            # Determine context allocation
            context_allocation = await self._determine_context_allocation(
                user_message,
                conversation_context,
                website_context,
                strategy=context_strategy,
                max_tokens=max_total_tokens
            )
            
            # Build optimized system prompt
            system_prompt = await self._build_enhanced_system_prompt(
                conversation_context=conversation_context,
                website_context=website_context,
                allocation=context_allocation,
                user_message=user_message
            )
            
            # Prepare final message context
            final_context = await self._prepare_final_context(
                conversation_context,
                website_context,
                context_allocation,
                max_total_tokens
            )
            
            # Generate AI response
            if self.openai_client:
                ai_response = await self._generate_enhanced_openai_response(
                    system_prompt,
                    final_context['messages'],
                    user_message
                )
            else:
                ai_response = await self._generate_enhanced_fallback_response(
                    user_message,
                    conversation_context,
                    website_context
                )
            
            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build response
            response_data = {
                'response': ai_response,
                'processing_time_ms': int(processing_time),
                'context_allocation': context_allocation,
                'conversation_tokens': conversation_context['total_tokens'],
                'website_tokens': len(website_context.split()) * 1.33 if website_context else 0,
                'total_context_tokens': final_context['total_tokens'],
                'optimization_level': conversation_context['optimization_level'],
                'messages_included': conversation_context['messages_included'],
                'messages_pruned': conversation_context['messages_pruned'],
                'has_conversation_summary': conversation_context['has_summary'],
                'website_context_used': bool(website_context),
                'source': 'enhanced_rag'
            }
            
            # Add source information if requested
            if include_sources:
                response_data['sources'] = {
                    'conversation_sources': {
                        'messages_count': conversation_context['messages_included'],
                        'summary_included': conversation_context['has_summary'],
                        'importance_stats': conversation_context.get('importance_statistics'),
                        'compression_ratio': conversation_context['compression_achieved']
                    },
                    'website_sources': {
                        'context_preview': website_context[:200] + '...' if website_context and len(website_context) > 200 else website_context,
                        'context_length': len(website_context) if website_context else 0
                    }
                }
            
            # Cache the result for future use
            await self.redis_cache.cache_context(
                conversation_id,
                {
                    'last_response': ai_response[:500],  # Preview only
                    'context_allocation': context_allocation,
                    'processing_time': processing_time
                }
            )
            
            logger.info(f"✅ ENHANCED_RAG: Generated response ({processing_time:.0f}ms, "
                       f"{final_context['total_tokens']} tokens, "
                       f"{context_allocation['conversation_percentage']}% conv / "
                       f"{context_allocation['website_percentage']}% web)")
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ ENHANCED_RAG: Error generating response: {e}")
            return await self._generate_error_response(user_message, conversation_id, website_id)
    
    async def _determine_context_allocation(
        self,
        user_message: str,
        conversation_context: Dict[str, Any],
        website_context: str,
        strategy: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Determine how to allocate tokens between conversation and website context.
        """
        try:
            # Calculate available context tokens (reserve some for system prompt and response)
            available_tokens = max_tokens - 200  # System prompt - 500  # Response reserve
            
            if strategy == self.CONTEXT_SPLIT_BALANCED:
                conv_allocation = 0.5
                web_allocation = 0.5
            elif strategy == self.CONTEXT_SPLIT_CONVERSATION:
                conv_allocation = 0.7
                web_allocation = 0.3
            elif strategy == self.CONTEXT_SPLIT_WEBSITE:
                conv_allocation = 0.3
                web_allocation = 0.7
            else:  # ADAPTIVE
                conv_allocation, web_allocation = await self._calculate_adaptive_allocation(
                    user_message, conversation_context, website_context
                )
            
            # Calculate actual token allocations
            conv_tokens = int(available_tokens * conv_allocation)
            web_tokens = int(available_tokens * web_allocation)
            
            return {
                'strategy': strategy,
                'conversation_tokens': conv_tokens,
                'website_tokens': web_tokens,
                'conversation_percentage': int(conv_allocation * 100),
                'website_percentage': int(web_allocation * 100),
                'total_available': available_tokens
            }
            
        except Exception as e:
            logger.error(f"Error determining context allocation: {e}")
            # Default balanced allocation
            return {
                'strategy': 'balanced_fallback',
                'conversation_tokens': available_tokens // 2,
                'website_tokens': available_tokens // 2,
                'conversation_percentage': 50,
                'website_percentage': 50,
                'total_available': available_tokens
            }
    
    async def _calculate_adaptive_allocation(
        self,
        user_message: str,
        conversation_context: Dict[str, Any],
        website_context: str
    ) -> Tuple[float, float]:
        """
        Calculate adaptive allocation based on message content and context relevance.
        """
        user_message_lower = user_message.lower()
        
        # Factors that increase conversation context weight
        conv_weight = 0.5
        
        # If user refers to previous conversation
        if any(phrase in user_message_lower for phrase in [
            'you said', 'earlier', 'before', 'previously', 'we discussed',
            'as mentioned', 'continuing', 'follow up', 'more about'
        ]):
            conv_weight += 0.2
        
        # If user asks clarifying questions
        if any(phrase in user_message_lower for phrase in [
            'what did you mean', 'clarify', 'explain more', 'can you elaborate'
        ]):
            conv_weight += 0.15
        
        # If conversation has high importance messages recently
        importance_stats = conversation_context.get('importance_statistics')
        if importance_stats and importance_stats.get('avg_importance', 1.0) > 1.2:
            conv_weight += 0.1
        
        # Factors that increase website context weight
        web_weight = 1.0 - conv_weight
        
        # If user asks about new topics
        if any(phrase in user_message_lower for phrase in [
            'what is', 'tell me about', 'how does', 'what are',
            'information about', 'details on', 'features', 'pricing'
        ]):
            web_weight += 0.1
            conv_weight -= 0.1
        
        # If no website context available, favor conversation
        if not website_context or len(website_context) < 100:
            conv_weight += 0.2
            web_weight -= 0.2
        
        # Clamp values
        conv_weight = max(0.2, min(0.8, conv_weight))
        web_weight = max(0.2, min(0.8, 1.0 - conv_weight))
        
        logger.debug(f"📊 ADAPTIVE: Allocation {conv_weight:.1f} conv / {web_weight:.1f} web")
        
        return conv_weight, web_weight
    
    async def _build_enhanced_system_prompt(
        self,
        conversation_context: Dict[str, Any],
        website_context: str,
        allocation: Dict[str, Any],
        user_message: str
    ) -> str:
        """
        Build enhanced system prompt that incorporates both contexts.
        """
        base_prompt = """You are an intelligent AI assistant that helps users with questions about a website. You have access to both the website's content and the ongoing conversation history.

Guidelines for responses:
1. Use BOTH conversation history and website content to provide comprehensive answers
2. Reference previous conversation points when relevant
3. Provide specific information from the website when available
4. Maintain conversation continuity and context
5. If information conflicts between sources, prioritize the most recent and relevant
6. Be conversational and acknowledge the ongoing dialogue"""

        # Add context information
        if conversation_context.get('has_summary'):
            base_prompt += "\n\nPrevious conversation summary is provided to maintain context continuity."
        
        if website_context:
            base_prompt += "\n\nWebsite content is provided to answer specific questions about products, services, and information."
        
        # Add context allocation information
        base_prompt += f"\n\nContext allocation: {allocation['conversation_percentage']}% conversation history, {allocation['website_percentage']}% website content."
        
        return base_prompt
    
    async def _prepare_final_context(
        self,
        conversation_context: Dict[str, Any],
        website_context: str,
        allocation: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Prepare the final context for AI processing.
        """
        final_messages = []
        total_tokens = 0
        
        # Add conversation summary if available
        if conversation_context.get('context_summary'):
            summary_msg = {
                "role": "system",
                "content": f"Previous conversation summary: {conversation_context['context_summary']}"
            }
            final_messages.append(summary_msg)
            total_tokens += len(conversation_context['context_summary'].split()) * 1.33
        
        # Add website context if available
        if website_context:
            # Truncate website context to fit allocation
            web_token_limit = allocation['website_tokens']
            if len(website_context.split()) * 1.33 > web_token_limit:
                # Truncate website context
                words = website_context.split()
                target_words = int(web_token_limit / 1.33)
                website_context = ' '.join(words[:target_words]) + '...'
            
            web_msg = {
                "role": "system", 
                "content": f"Website information: {website_context}"
            }
            final_messages.append(web_msg)
            total_tokens += len(website_context.split()) * 1.33
        
        # Add conversation messages (already optimized)
        conversation_messages = conversation_context.get('messages', [])
        if conversation_messages:
            # Further limit if needed to fit allocation
            conv_token_limit = allocation['conversation_tokens']
            included_messages = []
            conv_tokens = 0
            
            for msg in reversed(conversation_messages):
                msg_tokens = len(msg.get('content', '').split()) * 1.33
                if conv_tokens + msg_tokens <= conv_token_limit:
                    included_messages.insert(0, msg)
                    conv_tokens += msg_tokens
                else:
                    break
            
            final_messages.extend(included_messages)
            total_tokens += conv_tokens
        
        return {
            'messages': final_messages,
            'total_tokens': int(total_tokens)
        }
    
    async def _generate_enhanced_openai_response(
        self,
        system_prompt: str,
        context_messages: List[Dict[str, str]],
        user_message: str
    ) -> str:
        """
        Generate response using OpenAI with enhanced context.
        """
        try:
            # Build messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(context_messages)
            messages.append({"role": "user", "content": user_message})
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating enhanced OpenAI response: {e}")
            return "I apologize, but I'm experiencing technical difficulties with my response generation. Please try again."
    
    async def _generate_enhanced_fallback_response(
        self,
        user_message: str,
        conversation_context: Dict[str, Any],
        website_context: str
    ) -> str:
        """
        Generate enhanced fallback response when OpenAI is not available.
        """
        user_lower = user_message.lower()
        
        response_parts = []
        
        # Check if we can use conversation context
        if conversation_context.get('messages'):
            recent_messages = conversation_context['messages'][-2:]
            if recent_messages:
                response_parts.append("Based on our conversation")
        
        # Check if we can use website context
        if website_context:
            if len(website_context) > 100:
                response_parts.append("according to the website information")
                
                # Try to extract relevant part
                context_preview = website_context[:300]
                response_parts.append(f": {context_preview}...")
        
        if not response_parts:
            return "Thank you for your question. I'd be happy to help, but I need a bit more context to provide you with the most accurate information."
        
        return " ".join(response_parts)
    
    async def _generate_error_response(
        self,
        user_message: str,
        conversation_id: UUID,
        website_id: UUID
    ) -> Dict[str, Any]:
        """
        Generate error response when processing fails.
        """
        return {
            'response': "I apologize, but I'm experiencing technical difficulties processing your request. Please try again in a moment.",
            'processing_time_ms': 0,
            'context_allocation': {'conversation_percentage': 0, 'website_percentage': 0},
            'error': True,
            'source': 'error_fallback'
        }
    
    async def get_context_statistics(
        self,
        conversation_id: UUID,
        website_id: UUID
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about context usage and optimization.
        """
        try:
            # Get conversation context stats
            conv_stats = await self.optimized_context.get_optimization_statistics(conversation_id)
            
            # Get cache stats
            cache_stats = await self.redis_cache.get_cache_stats()
            
            # Get website context availability
            website_test_result = await self.vector_service.get_context_for_query_improved(
                "test query", website_id
            )
            
            return {
                'conversation_statistics': conv_stats,
                'cache_statistics': cache_stats,
                'website_context_available': bool(website_test_result),
                'website_context_length': len(website_test_result) if website_test_result else 0,
                'enhanced_rag_features': {
                    'adaptive_allocation': True,
                    'conversation_optimization': True,
                    'redis_caching': cache_stats.get('available', False),
                    'importance_scoring': True,
                    'context_pruning': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting context statistics: {e}")
            return {
                'error': str(e),
                'statistics_available': False
            }