"""
Vector Enhanced Chat Service - Cloud-ready chat with vector RAG
Replaces local storage with vector database for context-aware responses
"""

import time
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio

from ..core.config import settings
from .vector_rag_service import get_vector_rag_service
from .openai_service import OpenAIService
from .simple_chat_service import SimpleChatService

logger = logging.getLogger(__name__)


class VectorEnhancedChatService:
    """Enhanced chat service with vector-based RAG and OpenAI integration"""

    def __init__(self):
        """Initialize vector enhanced chat service"""
        self.vector_rag_service = get_vector_rag_service()
        self.openai_service = OpenAIService()
        self.simple_service = SimpleChatService()
        self.use_openai = bool(settings.openai_api_key)

    async def generate_chat_response(
        self,
        user_message: str,
        website_id: str,
        conversation_history: List[Dict[str, str]] = None,
        max_tokens: int = 500,
        use_rag: bool = True,
        use_hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Generate AI response using vector RAG with fallback options

        Args:
            user_message: User's message
            website_id: Website identifier for context retrieval
            conversation_history: Previous conversation messages
            max_tokens: Maximum tokens for response
            use_rag: Whether to use RAG for context
            use_hybrid_search: Whether to use hybrid search (vector + keyword)

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        try:
            if use_rag and website_id:
                # Use vector RAG for context-aware response
                logger.info(f"Generating vector RAG response for website {website_id}")

                response = await self.vector_rag_service.generate_rag_response(
                    user_message=user_message,
                    website_id=website_id,
                    conversation_history=conversation_history,
                    use_hybrid_search=use_hybrid_search
                )

                # Add timing information
                response["response_time"] = round(time.time() - start_time, 2)
                response["service"] = "vector_rag"

                return response

            elif self.use_openai:
                # Fallback to OpenAI without RAG
                logger.info("Generating OpenAI response without RAG")

                try:
                    response = await self.openai_service.generate_chat_response(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        website_context=None,
                        max_tokens=max_tokens
                    )

                    # If OpenAI fails, fallback to simple service
                    if response.get("error") or "technical difficulties" in response.get("response", ""):
                        logger.warning("OpenAI failed, falling back to simple service")
                        return await self._fallback_to_simple(
                            user_message, conversation_history, max_tokens, start_time
                        )

                    response["response_time"] = round(time.time() - start_time, 2)
                    response["service"] = "openai_fallback"
                    return response

                except Exception as e:
                    logger.error(f"OpenAI service error: {e}")
                    return await self._fallback_to_simple(
                        user_message, conversation_history, max_tokens, start_time
                    )

            else:
                # Fallback to simple service
                return await self._fallback_to_simple(
                    user_message, conversation_history, max_tokens, start_time
                )

        except Exception as e:
            logger.error(f"Error in chat response generation: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "error": str(e),
                "service": "error",
                "response_time": round(time.time() - start_time, 2)
            }

    async def _fallback_to_simple(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        max_tokens: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Fallback to simple chat service"""
        response = await self.simple_service.generate_chat_response(
            user_message=user_message,
            conversation_history=conversation_history,
            website_context=None,
            max_tokens=max_tokens
        )
        response["response_time"] = round(time.time() - start_time, 2)
        response["service"] = "simple_fallback"
        return response

    async def stream_chat_response(
        self,
        user_message: str,
        website_id: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True,
        use_hybrid_search: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response using vector RAG

        Args:
            user_message: User's message
            website_id: Website identifier
            conversation_history: Previous conversation messages
            use_rag: Whether to use RAG for context
            use_hybrid_search: Whether to use hybrid search

        Yields:
            Streaming response chunks
        """
        try:
            if use_rag and website_id:
                # Get context using vector RAG
                logger.info(f"Streaming vector RAG response for website {website_id}")

                # First, retrieve context
                yield {
                    "type": "status",
                    "message": "Retrieving relevant context...",
                    "timestamp": time.time()
                }

                context_chunks = await self.vector_rag_service._retrieve_context(
                    query=user_message,
                    website_id=website_id,
                    use_hybrid=use_hybrid_search
                )

                yield {
                    "type": "context",
                    "chunks_found": len(context_chunks),
                    "sources": [
                        {
                            "url": chunk.get("url", ""),
                            "title": chunk.get("title", ""),
                            "similarity": round(chunk.get("similarity", 0), 3)
                        }
                        for chunk in context_chunks[:3]
                    ],
                    "timestamp": time.time()
                }

                # Build context and system prompt
                context_text = self.vector_rag_service._build_context_text(context_chunks)
                system_prompt = self.vector_rag_service._build_system_prompt(context_text, context_chunks)

                # Prepare messages
                messages = await self.vector_rag_service._prepare_messages(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    conversation_history=conversation_history
                )

                # Stream OpenAI response
                if self.use_openai and self.vector_rag_service.openai_client:
                    yield {
                        "type": "status",
                        "message": "Generating response...",
                        "timestamp": time.time()
                    }

                    async for chunk in self._stream_openai_response(messages):
                        yield chunk

                else:
                    # Non-streaming fallback
                    response_text = self.vector_rag_service._generate_fallback_response(
                        user_message, context_chunks
                    )
                    yield {
                        "type": "content",
                        "content": response_text,
                        "timestamp": time.time()
                    }

                # Final metadata
                yield {
                    "type": "complete",
                    "context_used": len(context_chunks) > 0,
                    "service": "vector_rag_stream",
                    "timestamp": time.time()
                }

            else:
                # Stream without RAG
                yield {
                    "type": "status",
                    "message": "Generating response...",
                    "timestamp": time.time()
                }

                if self.use_openai:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Provide concise and helpful responses."
                        },
                        {"role": "user", "content": user_message}
                    ]

                    async for chunk in self._stream_openai_response(messages):
                        yield chunk
                else:
                    # Simple service response
                    response = await self.simple_service.generate_chat_response(
                        user_message=user_message,
                        conversation_history=conversation_history
                    )
                    yield {
                        "type": "content",
                        "content": response.get("response", ""),
                        "timestamp": time.time()
                    }

                yield {
                    "type": "complete",
                    "context_used": False,
                    "service": "openai_stream" if self.use_openai else "simple_stream",
                    "timestamp": time.time()
                }

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    async def _stream_openai_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from OpenAI"""
        try:
            response = await self.vector_rag_service.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "content",
                        "content": chunk.choices[0].delta.content,
                        "timestamp": time.time()
                    }

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield {
                "type": "error",
                "error": f"Streaming failed: {str(e)}",
                "timestamp": time.time()
            }

    async def get_chat_suggestions(self, website_id: str) -> List[str]:
        """
        Get suggested questions based on website content

        Args:
            website_id: Website identifier

        Returns:
            List of suggested questions
        """
        try:
            # Get content stats to see if we have data
            stats = await self.vector_rag_service.get_content_summary(website_id)

            if not stats.get("rag_ready", False):
                return [
                    "What services do you offer?",
                    "How can I contact you?",
                    "Tell me about your company",
                    "What are your hours?"
                ]

            # Generate context-aware suggestions by searching for common topics
            common_queries = [
                "services", "pricing", "contact", "about",
                "features", "support", "location", "hours"
            ]

            suggestions = []
            for query in common_queries:
                try:
                    context_chunks = await self.vector_rag_service._retrieve_context(
                        query=query,
                        website_id=website_id,
                        use_hybrid=True
                    )

                    if context_chunks:
                        # Create suggestion based on found content
                        top_chunk = context_chunks[0]
                        title = top_chunk.get("title", "")

                        if "pricing" in query.lower() or "price" in title.lower():
                            suggestions.append("What are your pricing plans?")
                        elif "contact" in query.lower() or "contact" in title.lower():
                            suggestions.append("How can I contact you?")
                        elif "service" in query.lower() or "service" in title.lower():
                            suggestions.append("What services do you offer?")
                        elif "about" in query.lower() or "about" in title.lower():
                            suggestions.append("Tell me about your company")

                except Exception:
                    continue

            # Remove duplicates and limit to 4 suggestions
            unique_suggestions = list(dict.fromkeys(suggestions))[:4]

            # Fill with default suggestions if needed
            default_suggestions = [
                "What can you help me with?",
                "What makes you different?",
                "Do you have any special offers?",
                "How do I get started?"
            ]

            while len(unique_suggestions) < 4:
                for default in default_suggestions:
                    if default not in unique_suggestions:
                        unique_suggestions.append(default)
                        break
                if len(unique_suggestions) >= 4:
                    break

            return unique_suggestions[:4]

        except Exception as e:
            logger.error(f"Error generating chat suggestions: {e}")
            return [
                "What services do you offer?",
                "How can I contact you?",
                "Tell me about your company",
                "What are your hours?"
            ]

    async def test_chat_functionality(self, website_id: str) -> Dict[str, Any]:
        """
        Test chat functionality for a website

        Args:
            website_id: Website identifier

        Returns:
            Test results
        """
        try:
            # Test RAG functionality
            rag_test = await self.vector_rag_service.test_rag_functionality(website_id)

            # Test basic chat response
            test_response = await self.generate_chat_response(
                user_message="Hello, can you help me?",
                website_id=website_id,
                use_rag=True
            )

            # Get content summary
            content_summary = await self.vector_rag_service.get_content_summary(website_id)

            return {
                "website_id": website_id,
                "chat_test": {
                    "response_generated": bool(test_response.get("response")),
                    "context_used": test_response.get("context_used", False),
                    "service_used": test_response.get("service", "unknown"),
                    "response_time": test_response.get("response_time", 0)
                },
                "rag_test": rag_test,
                "content_summary": content_summary,
                "overall_status": "healthy" if content_summary.get("rag_ready") else "no_content"
            }

        except Exception as e:
            logger.error(f"Error testing chat functionality: {e}")
            return {
                "website_id": website_id,
                "error": str(e),
                "overall_status": "error"
            }


# Singleton instance
_vector_enhanced_chat_service = None

def get_vector_enhanced_chat_service() -> VectorEnhancedChatService:
    """Get or create vector enhanced chat service instance"""
    global _vector_enhanced_chat_service
    if _vector_enhanced_chat_service is None:
        _vector_enhanced_chat_service = VectorEnhancedChatService()
    return _vector_enhanced_chat_service