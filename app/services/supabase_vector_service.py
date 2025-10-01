"""
Supabase Vector Storage Service
Handles all vector database operations for crawled content
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import asyncio
import numpy as np

from supabase import create_client, Client
import tiktoken

from .embedding_service import get_embedding_service
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class SupabaseVectorService:
    """Service for managing vector storage in Supabase"""

    def __init__(self):
        """Initialize Supabase client and OpenAI"""
        settings = get_settings()
        self.supabase_url = settings.supabase_url
        self.supabase_key = settings.supabase_service_role_key

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

        self.client: Client = create_client(self.supabase_url, self.supabase_key)

        # Initialize embedding service
        self.embedding_service = get_embedding_service()
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        # Configuration
        self.max_chunk_tokens = 2000
        self.chunk_overlap_tokens = 200
        self.batch_size = 100

    def chunk_text(self, text: str, max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks for embedding

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of chunks with metadata
        """
        if max_tokens is None:
            max_tokens = self.max_chunk_tokens

        tokens = self.encoding.encode(text)
        chunks = []

        # If text is short enough, return as single chunk
        if len(tokens) <= max_tokens:
            return [{
                'text': text,
                'tokens': len(tokens),
                'start_index': 0,
                'end_index': len(text)
            }]

        # Split into overlapping chunks
        start = 0
        while start < len(tokens):
            end = start + max_tokens

            # Find a good breaking point (sentence boundary)
            if end < len(tokens):
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)

                # Try to break at sentence boundary
                last_period = chunk_text.rfind('. ')
                if last_period > len(chunk_text) * 0.5:  # Only if not too far back
                    chunk_text = chunk_text[:last_period + 1]
                    chunk_tokens = self.encoding.encode(chunk_text)
                    end = start + len(chunk_tokens)
            else:
                chunk_tokens = tokens[start:]
                chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                'text': chunk_text,
                'tokens': len(chunk_tokens),
                'start_index': start,
                'end_index': end
            })

            # Move start with overlap
            start = end - self.chunk_overlap_tokens if end < len(tokens) else end

        return chunks

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using embedding service

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self.embedding_service.generate_embedding(text)

    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self.embedding_service.batch_generate_embeddings(texts)

    async def store_content(
        self,
        website_id: str,
        url: str,
        title: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Store content with embeddings in Supabase

        Args:
            website_id: Website identifier
            url: Page URL
            title: Page title
            content: Full page content
            metadata: Additional metadata

        Returns:
            Storage result with document IDs
        """
        try:
            # Chunk the content
            chunks = self.chunk_text(content)
            total_chunks = len(chunks)

            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = await self.batch_generate_embeddings(chunk_texts)

            # Prepare documents for insertion
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    'website_id': website_id,
                    'url': url,
                    'title': title,
                    'content': content if i == 0 else '',  # Store full content only in first chunk
                    'content_chunk': chunk['text'],
                    'chunk_index': i,
                    'total_chunks': total_chunks,
                    'embedding': embedding,
                    'metadata': metadata or {},
                    'word_count': len(chunk['text'].split())
                }
                documents.append(doc)

            # Insert into Supabase
            result = self.client.table('crawled_content').insert(documents).execute()

            return {
                'success': True,
                'chunks_created': len(documents),
                'document_ids': [doc['id'] for doc in result.data]
            }

        except Exception as e:
            logger.error(f"Error storing content: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_created': 0
            }

    async def search_similar(
        self,
        query: str,
        website_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity

        Args:
            query: Search query
            website_id: Website to search in
            limit: Maximum results

        Returns:
            List of similar content chunks
        """
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)

            # Call Supabase search function
            result = self.client.rpc('search_similar_content', {
                'query_embedding': query_embedding,
                'match_website_id': website_id,
                'match_limit': limit
            }).execute()

            return result.data

        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        website_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword)

        Args:
            query: Search query
            website_id: Website to search in
            limit: Maximum results

        Returns:
            List of matching content chunks
        """
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)

            # Call Supabase hybrid search function
            result = self.client.rpc('hybrid_search_content', {
                'query_embedding': query_embedding,
                'keyword_query': query,
                'match_website_id': website_id,
                'match_limit': limit
            }).execute()

            return result.data

        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []

    async def update_content(
        self,
        website_id: str,
        url: str,
        new_content: str,
        new_title: Optional[str] = None
    ) -> bool:
        """
        Update existing content with new embeddings

        Args:
            website_id: Website identifier
            url: Page URL
            new_content: Updated content
            new_title: Updated title (optional)

        Returns:
            Success status
        """
        try:
            # Mark existing content as inactive
            self.client.table('crawled_content').update({
                'is_active': False
            }).eq('website_id', website_id).eq('url', url).execute()

            # Get current version
            current = self.client.table('crawled_content').select('version').eq(
                'website_id', website_id
            ).eq('url', url).order('version', desc=True).limit(1).execute()

            new_version = 1
            if current.data:
                new_version = current.data[0]['version'] + 1

            # Store new version
            result = await self.store_content(
                website_id=website_id,
                url=url,
                title=new_title or '',
                content=new_content,
                metadata={'version': new_version}
            )

            return result['success']

        except Exception as e:
            logger.error(f"Error updating content: {e}")
            return False

    async def delete_content(self, website_id: str, url: Optional[str] = None) -> bool:
        """
        Delete content from vector database

        Args:
            website_id: Website identifier
            url: Specific URL to delete (optional, deletes all if not provided)

        Returns:
            Success status
        """
        try:
            query = self.client.table('crawled_content').delete().eq(
                'website_id', website_id
            )

            if url:
                query = query.eq('url', url)

            query.execute()
            return True

        except Exception as e:
            logger.error(f"Error deleting content: {e}")
            return False

    async def get_content_stats(self, website_id: str) -> Dict[str, Any]:
        """
        Get statistics for stored content

        Args:
            website_id: Website identifier

        Returns:
            Content statistics
        """
        try:
            result = self.client.rpc('get_content_stats', {
                'match_website_id': website_id
            }).execute()

            if result.data:
                return result.data[0]

            return {
                'total_documents': 0,
                'total_chunks': 0,
                'chunks_with_embeddings': 0,
                'latest_crawl': None
            }

        except Exception as e:
            logger.error(f"Error getting content stats: {e}")
            return {}

    async def create_crawl_job(
        self,
        website_id: str,
        config: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Create a new crawl job

        Args:
            website_id: Website identifier
            config: Crawl configuration

        Returns:
            Job ID if successful
        """
        try:
            result = self.client.table('crawl_jobs').insert({
                'website_id': website_id,
                'status': 'pending',
                'config': config or {}
            }).execute()

            if result.data:
                return result.data[0]['id']
            return None

        except Exception as e:
            logger.error(f"Error creating crawl job: {e}")
            return None

    async def update_crawl_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        pages_found: Optional[int] = None,
        pages_processed: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update crawl job status

        Args:
            job_id: Job identifier
            status: New status
            pages_found: Number of pages found
            pages_processed: Number of pages processed
            error_message: Error message if failed

        Returns:
            Success status
        """
        try:
            update_data = {}

            if status:
                update_data['status'] = status
                if status == 'processing':
                    update_data['started_at'] = datetime.utcnow().isoformat()
                elif status in ['completed', 'failed']:
                    update_data['completed_at'] = datetime.utcnow().isoformat()

            if pages_found is not None:
                update_data['pages_found'] = pages_found

            if pages_processed is not None:
                update_data['pages_processed'] = pages_processed

            if error_message:
                update_data['error_message'] = error_message

            self.client.table('crawl_jobs').update(update_data).eq('id', job_id).execute()
            return True

        except Exception as e:
            logger.error(f"Error updating crawl job: {e}")
            return False


# Singleton instance
_vector_service = None

def get_vector_service() -> SupabaseVectorService:
    """Get or create vector service instance"""
    global _vector_service
    if _vector_service is None:
        _vector_service = SupabaseVectorService()
    return _vector_service