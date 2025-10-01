"""
Vector RAG Service - Cloud-ready RAG using Supabase vector storage
Replaces local storage with vector database for context retrieval
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from openai import AsyncOpenAI

from ..core.config import get_settings
from .vector_search_service import VectorSearchService

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorRAGService:
    """Chat service with Vector-based Retrieval Augmented Generation (RAG)"""

    def __init__(self):
        """Initialize vector RAG service"""
        self.vector_service = VectorSearchService()
        self.openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key
        ) if hasattr(settings, 'openai_api_key') else None

        # RAG configuration
        self.max_context_length = 4000  # Maximum context length in tokens
        self.context_chunk_limit = 5    # Maximum chunks to use for context
        self.similarity_threshold = 0.5  # Minimum similarity score for relevance (0.5 for hybrid, 0.7 for pure vector)

    async def generate_rag_response(
        self,
        user_message: str,
        website_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Generate AI response using vector-based context retrieval

        Args:
            user_message: User's question or message
            website_id: Website identifier for context retrieval
            conversation_history: Previous conversation messages
            use_hybrid_search: Whether to use hybrid (vector + keyword) search

        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Generating RAG response for website {website_id}")

            # Retrieve relevant context using vector search
            context_chunks = await self._retrieve_context(
                query=user_message,
                website_id=website_id,
                use_hybrid=use_hybrid_search
            )

            # Build context string from chunks
            context_text = self._build_context_text(context_chunks)

            # Build system prompt with context
            system_prompt = self._build_system_prompt(context_text, context_chunks)

            # Prepare conversation messages
            messages = await self._prepare_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation_history
            )

            # Generate AI response
            if self.openai_client:
                ai_response = await self._generate_openai_response(messages)
                response_source = "openai"
            else:
                ai_response = self._generate_fallback_response(user_message, context_chunks)
                response_source = "fallback"

            # Prepare response with metadata
            return {
                "response": ai_response,
                "context_used": len(context_chunks) > 0,
                "context_chunks": len(context_chunks),
                "context_sources": [
                    {
                        "url": chunk.get("url", ""),
                        "title": chunk.get("title", "Untitled"),
                        "similarity": round(chunk.get("similarity", 0), 3)
                    }
                    for chunk in context_chunks[:3]  # Show top 3 sources
                ],
                "source": response_source,
                "search_type": "hybrid" if use_hybrid_search else "vector"
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "context_used": False,
                "context_chunks": 0,
                "context_sources": [],
                "source": "error",
                "error": str(e)
            }

    async def _retrieve_context(
        self,
        query: str,
        website_id: str,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks using vector search

        Args:
            query: Search query
            website_id: Website identifier
            use_hybrid: Use hybrid search (vector + keyword)

        Returns:
            List of relevant context chunks
        """
        try:
            # Use vector similarity search (VectorSearchService doesn't have hybrid yet)
            results = await self.vector_service.similarity_search(
                query=query,
                website_id=UUID(website_id),
                limit=self.context_chunk_limit
            )
            similarity_key = "similarity"

            # Filter by similarity threshold and format results
            relevant_chunks = []
            for result in results:
                similarity = result.get(similarity_key, 0)
                if similarity >= self.similarity_threshold:
                    chunk = {
                        "content": result.get("chunk_text", ""),
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "similarity": similarity,
                        "metadata": result.get("metadata", {})
                    }
                    relevant_chunks.append(chunk)

            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks (threshold: {self.similarity_threshold})")
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def _build_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build context text from chunks

        Args:
            context_chunks: List of context chunks

        Returns:
            Formatted context text
        """
        if not context_chunks:
            return ""

        context_parts = []
        current_length = 0

        for chunk in context_chunks:
            content = chunk.get("content", "").strip()
            url = chunk.get("url", "")
            title = chunk.get("title", "")

            if not content:
                continue

            # Format chunk with source information
            chunk_text = f"From '{title}' ({url}):\n{content}\n"

            # Check if adding this chunk would exceed context limit
            if current_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n---\n".join(context_parts)

    def _build_system_prompt(
        self,
        context_text: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Build system prompt with website context

        Args:
            context_text: Formatted context text
            context_chunks: Original context chunks for metadata

        Returns:
            System prompt string
        """
        base_prompt = """You are a helpful website assistant. Give SHORT, SIMPLE answers.

CRITICAL RULES:
1. Maximum 2-3 sentences unless user asks for details
2. Answer ONLY what's asked - no extra information
3. Use simple everyday language
4. For contact info: give 1-2 main options, not everything
5. For lists: give 2-3 key items unless user asks for more

Be friendly but brief."""

        if context_text:
            sources_info = ""
            if context_chunks:
                unique_urls = list(set(chunk.get("url", "") for chunk in context_chunks if chunk.get("url")))
                if unique_urls:
                    sources_info = f"\n\nYou have access to information from {len(unique_urls)} page(s) on this website."

            context_prompt = f"""

WEBSITE CONTENT FOR REFERENCE:
{context_text}
{sources_info}

Use this content to answer the user's questions. If the answer requires information not provided above, let the user know you don't have that specific information available."""

            return base_prompt + context_prompt
        else:
            return base_prompt + "\n\nI don't have access to specific website content at the moment, but I'll do my best to help with general questions."

    async def _prepare_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API

        Args:
            system_prompt: System prompt with context
            user_message: Current user message
            conversation_history: Previous conversation messages

        Returns:
            Formatted messages list
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limited to recent messages)
        if conversation_history:
            # Keep last 8 messages for context while staying within token limits
            recent_history = conversation_history[-8:]
            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _generate_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using OpenAI API

        Args:
            messages: Conversation messages

        Returns:
            AI response text
        """
        try:
            # Log the system prompt to debug context inclusion
            if messages and messages[0].get("role") == "system":
                system_content = messages[0]["content"]
                logger.info(f"System prompt length: {len(system_content)} chars")
                if "WEBSITE CONTENT FOR REFERENCE:" in system_content:
                    logger.info("✅ Context is included in system prompt")
                else:
                    logger.warning("❌ Context NOT found in system prompt")

            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=100,
                temperature=0.5,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _generate_fallback_response(
        self,
        user_message: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate fallback response when OpenAI is unavailable

        Args:
            user_message: User's message
            context_chunks: Available context chunks

        Returns:
            Fallback response
        """
        if context_chunks:
            # Extract key information from context
            topics = []
            for chunk in context_chunks[:2]:  # Use top 2 chunks
                content = chunk.get("content", "")
                if content:
                    # Extract first sentence or first 100 characters
                    first_sentence = content.split('.')[0]
                    if len(first_sentence) < 100:
                        topics.append(first_sentence)
                    else:
                        topics.append(content[:100] + "...")

            if topics:
                return f"Based on the website content, here's what I found: {' '.join(topics)}. For more detailed information, please refer to the website pages directly."

        return "I apologize, but I'm currently unable to process your request. Please try again later or contact the website directly for assistance."

    async def get_content_summary(self, website_id: str) -> Dict[str, Any]:
        """
        Get a summary of available content for a website

        Args:
            website_id: Website identifier

        Returns:
            Content summary statistics
        """
        try:
            stats = await self.vector_service.get_content_stats(website_id)

            return {
                "status": "success",
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "chunks_with_embeddings": stats.get("chunks_with_embeddings", 0),
                "latest_crawl": stats.get("latest_crawl"),
                "rag_ready": stats.get("chunks_with_embeddings", 0) > 0
            }

        except Exception as e:
            logger.error(f"Error getting content summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "rag_ready": False
            }

    async def test_rag_functionality(self, website_id: str) -> Dict[str, Any]:
        """
        Test RAG functionality for a website

        Args:
            website_id: Website identifier

        Returns:
            Test results
        """
        test_queries = [
            "What services do you offer?",
            "How can I contact you?",
            "What are your prices?",
            "Tell me about your company"
        ]

        results = []
        for query in test_queries:
            try:
                context_chunks = await self._retrieve_context(
                    query=query,
                    website_id=website_id,
                    use_hybrid=True
                )

                results.append({
                    "query": query,
                    "chunks_found": len(context_chunks),
                    "avg_similarity": sum(c.get("similarity", 0) for c in context_chunks) / len(context_chunks) if context_chunks else 0,
                    "top_source": context_chunks[0].get("url", "") if context_chunks else None
                })

            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "chunks_found": 0
                })

        return {
            "website_id": website_id,
            "test_results": results,
            "overall_performance": {
                "queries_tested": len(test_queries),
                "successful_queries": len([r for r in results if r.get("chunks_found", 0) > 0]),
                "avg_chunks_per_query": sum(r.get("chunks_found", 0) for r in results) / len(results)
            }
        }


# Singleton instance
_vector_rag_service = None

def get_vector_rag_service() -> VectorRAGService:
    """Get or create vector RAG service instance"""
    global _vector_rag_service
    if _vector_rag_service is None:
        _vector_rag_service = VectorRAGService()
    return _vector_rag_service