"""
Embedding Service
Handles text embedding generation and caching
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional
import asyncio
import json

import openai
import tiktoken
import redis

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and caching text embeddings"""

    def __init__(self):
        """Initialize OpenAI client and Redis cache"""
        settings = get_settings()
        self.openai_client = openai.OpenAI(
            api_key=settings.openai_api_key
        )
        self.model = "text-embedding-ada-002"
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        # Initialize Redis cache (optional)
        self.redis_client = None
        try:
            redis_url = settings.redis_url
            self.redis_client = redis.from_url(redis_url)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected for embeddings")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")

        # Cache settings
        self.cache_ttl = 86400 * 7  # 7 days

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()

    async def _get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        if not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(f"embedding:{text_hash}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    async def _cache_embedding(self, text_hash: str, embedding: List[float]):
        """Cache embedding for future use"""
        if not self.redis_client:
            return

        try:
            self.redis_client.setex(
                f"embedding:{text_hash}",
                self.cache_ttl,
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with caching

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        text_hash = self._get_text_hash(text)
        cached_embedding = await self._get_cached_embedding(text_hash)
        if cached_embedding:
            logger.debug(f"Using cached embedding for text hash: {text_hash}")
            return cached_embedding

        try:
            # Generate new embedding
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model=self.model
            )

            embedding = response.data[0].embedding

            # Cache the result
            await self._cache_embedding(text_hash, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def batch_generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        texts_to_generate = []
        cache_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append([0.0] * 1536)  # Zero vector for empty text
                continue

            cached_embedding = None
            if use_cache:
                text_hash = self._get_text_hash(text)
                cached_embedding = await self._get_cached_embedding(text_hash)

            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                texts_to_generate.append(text)
                cache_indices.append(i)

        # Generate embeddings for uncached texts
        if texts_to_generate:
            try:
                response = await asyncio.to_thread(
                    self.openai_client.embeddings.create,
                    input=texts_to_generate,
                    model=self.model
                )

                new_embeddings = [item.embedding for item in response.data]

                # Fill in the results and cache them
                for j, (text, embedding) in enumerate(zip(texts_to_generate, new_embeddings)):
                    index = cache_indices[j]
                    embeddings[index] = embedding

                    # Cache the result
                    if use_cache:
                        text_hash = self._get_text_hash(text)
                        await self._cache_embedding(text_hash, embedding)

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Fill remaining None values with zero vectors
                for i, emb in enumerate(embeddings):
                    if emb is None:
                        embeddings[i] = [0.0] * 1536
                raise

        return embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    async def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0 to 1)
        """
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

            if norm_product == 0:
                return 0.0

            return dot_product / norm_product

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Find most similar embeddings to query

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embeddings to compare
            top_k: Number of top results

        Returns:
            List of similarity results with indices and scores
        """
        try:
            import numpy as np

            query_vec = np.array(query_embedding)
            similarities = []

            for i, embedding in enumerate(embeddings):
                similarity = await self.similarity(query_embedding, embedding)
                similarities.append({
                    'index': i,
                    'similarity': similarity
                })

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []

    async def get_cache_stats(self) -> Dict[str, any]:
        """Get embedding cache statistics"""
        if not self.redis_client:
            return {'cache_enabled': False}

        try:
            info = self.redis_client.info()
            embedding_keys = len(self.redis_client.keys("embedding:*"))

            return {
                'cache_enabled': True,
                'embedding_keys': embedding_keys,
                'memory_used': info.get('used_memory_human', 'Unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'cache_enabled': False, 'error': str(e)}

    async def clear_cache(self, pattern: str = "embedding:*") -> int:
        """Clear embedding cache"""
        if not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service