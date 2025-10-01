"""
Vector search service for RAG (Retrieval Augmented Generation) functionality.
Handles embeddings, similarity search, and context retrieval for AI chat responses.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import openai
from openai import AsyncOpenAI

from ..core.config import get_settings
from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorSearchService:
    """Service for vector embeddings and similarity search using Supabase pgvector."""
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if hasattr(settings, 'openai_api_key') else None
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI's embedding model."""
        try:
            if not self.openai_client:
                logger.warning("OpenAI API key not configured, returning dummy embedding")
                return [0.0] * 1536  # Dummy embedding for development
                
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text.strip()
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 1536  # Fallback to dummy embedding
    
    async def chunk_and_embed_content(self, page_id: UUID, content: str, website_id: str = None, chunk_size: int = 1000) -> List[str]:
        """Split content into chunks and generate embeddings for each chunk."""
        try:
            # If website_id not provided, fetch it from the page
            if not website_id:
                page_result = self.supabase.table('scraped_pages').select('scraped_website_id').eq('id', str(page_id)).single().execute()
                if page_result.data:
                    scraped_website_id = page_result.data['scraped_website_id']
                    # Get website_id from scraped_websites
                    sw_result = self.supabase.table('scraped_websites').select('website_id').eq('id', scraped_website_id).single().execute()
                    if sw_result.data:
                        website_id = sw_result.data['website_id']

            if not website_id:
                logger.error(f"Could not determine website_id for page {page_id}")
                return []

            # Simple chunking by characters (you can improve this with proper sentence splitting)
            chunks = []
            words = content.split()
            current_chunk = []
            current_length = 0

            for word in words:
                if current_length + len(word) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(word)
                current_length += len(word) + 1

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Generate embeddings and store chunks
            chunk_ids = []
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                embedding = await self.generate_embedding(chunk_text)

                chunk_data = {
                    'scraped_page_id': str(page_id),
                    'website_id': website_id,
                    'chunk_text': chunk_text,
                    'chunk_index': i,
                    'token_count': len(chunk_text.split()),
                    'embedding_vector': embedding,
                    'chunk_type': 'text'
                }

                result = self.supabase.table('scraped_content_chunks').insert(chunk_data).execute()
                if result.data:
                    chunk_ids.append(result.data[0]['id'])

            logger.info(f"Created {len(chunk_ids)} chunks for page {page_id}")
            return chunk_ids

        except Exception as e:
            logger.error(f"Error chunking and embedding content: {e}")
            return []
    
    async def similarity_search(
        self,
        query: str,
        website_id: UUID,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform similarity search to find relevant content chunks."""
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)

            # Perform vector similarity search using Supabase RPC
            result = self.supabase.rpc(
                'match_content_chunks',
                {
                    'query_embedding': query_embedding,
                    'website_id': str(website_id),
                    'match_threshold': similarity_threshold,
                    'match_count': limit
                }
            ).execute()

            if not result.data:
                return []

            # Format results by fetching page info for each chunk
            formatted_results = []
            for item in result.data:
                page_id = item.get('scraped_page_id')
                page_info = {'url': '', 'title': 'Untitled'}

                if page_id:
                    try:
                        page_result = self.supabase.table('scraped_pages').select('url,title').eq('id', page_id).execute()
                        if page_result.data:
                            page_info = page_result.data[0]
                    except Exception as page_err:
                        logger.warning(f"Could not fetch page info for {page_id}: {page_err}")

                formatted_results.append({
                    'chunk_text': item.get('chunk_text', ''),
                    'url': page_info.get('url', ''),
                    'title': page_info.get('title', 'Untitled'),
                    'similarity': item.get('similarity', 0),
                    'chunk_index': item.get('chunk_index', 0)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            # Fallback to basic text search
            return await self.fallback_text_search(query, website_id, limit)
    
    async def fallback_text_search(self, query: str, website_id: UUID, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback to basic text search when vector search is not available."""
        try:
            # Get all chunks for website (without ilike to avoid timeout)
            # Then filter in Python
            result = self.supabase.table('scraped_content_chunks').select(
                'id,chunk_text,chunk_index,scraped_page_id'
            ).eq('website_id', str(website_id)).limit(50).execute()  # Get more, filter in Python

            # Filter by query in Python (case-insensitive)
            query_lower = query.lower()
            matched_chunks = []
            for item in result.data:
                chunk_text = item.get('chunk_text', '')
                if query_lower in chunk_text.lower():
                    matched_chunks.append(item)
                    if len(matched_chunks) >= limit:
                        break

            # Get page info and format
            formatted_results = []
            for item in matched_chunks:
                page_id = item.get('scraped_page_id')
                page_info = {'url': '', 'title': 'Untitled'}
                if page_id:
                    try:
                        page_result = self.supabase.table('scraped_pages').select('url,title').eq('id', page_id).execute()
                        if page_result.data:
                            page_info = page_result.data[0]
                    except:
                        pass

                formatted_results.append({
                    'chunk_text': item.get('chunk_text', ''),
                    'url': page_info.get('url', ''),
                    'title': page_info.get('title', 'Untitled'),
                    'similarity': 0.5,  # Default similarity for text search
                    'chunk_index': item.get('chunk_index', 0)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def get_context_for_query(self, query: str, website_id: UUID) -> str:
        """Get relevant context from scraped content for a user query."""
        try:
            # Search for relevant chunks
            relevant_chunks = await self.similarity_search(query, website_id, limit=3)
            
            if not relevant_chunks:
                return ""
            
            # Build context string
            context_parts = []
            for chunk in relevant_chunks:
                # Include URL and title for context
                url = chunk.get('url', 'Unknown page')
                title = chunk.get('title', 'Untitled')
                content = chunk.get('chunk_text', '')
                
                context_parts.append(f"From {title} ({url}):\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Truncate if too long (keep under ~3000 chars for reasonable context)
            if len(context) > 3000:
                context = context[:3000] + "..."
                
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for query: {e}")
            return ""
    
    async def process_scraped_page(self, page_data: Dict[str, Any]) -> bool:
        """Process a newly scraped page - extract content and create embeddings."""
        try:
            page_id = UUID(page_data['id'])
            content = page_data.get('content_text', '')
            
            if not content or len(content.strip()) < 100:
                logger.warning(f"Page {page_id} has insufficient content for processing")
                return False
            
            # Create chunks and embeddings
            chunk_ids = await self.chunk_and_embed_content(page_id, content)
            
            logger.info(f"Successfully processed page {page_id} into {len(chunk_ids)} chunks")
            return len(chunk_ids) > 0
            
        except Exception as e:
            logger.error(f"Error processing scraped page: {e}")
            return False
    
    async def chunk_and_embed_content_improved(self, global_page_id: str, content: str) -> List[str]:
        """
        Create content chunks and embeddings for the improved schema (global pages).
        Similar to chunk_and_embed_content but works with global_scraped_pages.
        """
        try:
            if not content or len(content.strip()) < 100:
                logger.warning(f"Content too short for embedding: {len(content)} chars")
                return []
            
            # Create text chunks
            chunks = self._create_text_chunks(content)
            chunk_ids = []
            
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Generate embedding
                embedding = await self.generate_embedding(chunk_text)
                
                # Store chunk with embedding (using global_page_id instead of scraped_page_id)
                chunk_data = {
                    'global_page_id': global_page_id,
                    'chunk_text': chunk_text,
                    'chunk_index': i,
                    'token_count': len(chunk_text.split()),
                    'embedding_vector': embedding,
                    'chunk_type': 'content'
                }
                
                try:
                    result = self.supabase.table('scraped_content_chunks').insert(chunk_data).execute()
                except Exception as e:
                    error_msg = str(e).lower()
                    if ('global_page_id' in error_msg and 'does not exist' in error_msg):
                        # Column doesn't exist yet, skip this chunk
                        logger.warning(f"global_page_id column not available, skipping chunk {i}")
                        continue
                    elif ('scraped_page_id' in error_msg and ('null' in error_msg or 'not-null' in error_msg)):
                        # scraped_page_id is still required, fall back to legacy approach
                        logger.warning(f"Schema not fully migrated, cannot use global pages yet")
                        return []  # Return empty list to trigger fallback
                    else:
                        logger.error(f"Unexpected error creating chunk: {e}")
                        continue
                
                if result.data:
                    chunk_ids.append(result.data[0]['id'])
            
            logger.info(f"Created {len(chunk_ids)} chunks for global page {global_page_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error creating chunks for global page {global_page_id}: {e}")
            return []
    
    async def get_context_for_query_improved(self, query: str, website_id: UUID, limit: int = 3) -> str:
        """
        Get context using improved schema (tries global pages first, falls back to legacy).
        """
        try:
            # First, try to get context from global pages (improved schema)
            context = await self._get_global_context(query, website_id, limit)
            
            if context:
                print(f"🌍 VECTOR: Using global context ({len(context)} chars)")
                return context
            
            # Fallback to legacy schema
            print(f"📄 VECTOR: Falling back to legacy context")
            return await self.get_context_for_query(query, website_id)
            
        except Exception as e:
            logger.error(f"Error getting improved context: {e}")
            # Ultimate fallback
            return await self.get_context_for_query(query, website_id)
    
    async def _get_global_context(self, query: str, website_id: UUID, limit: int = 3) -> str:
        """Get context from global pages schema."""
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            
            # Search for similar content chunks linked to global pages for this website
            # Use a simpler approach: get chunks for global pages, then filter by website associations
            chunks_result = self.supabase.table('scraped_content_chunks').select(
                '''
                id,
                chunk_text,
                global_page_id,
                global_scraped_pages!inner(id, url, title)
                '''
            ).not_.is_('global_page_id', 'null').limit(limit * 3).execute()  # Get more to filter
            
            if not chunks_result.data:
                return ""
            
            # Now get website associations to filter chunks for this specific website
            associations_result = self.supabase.table('website_page_associations').select(
                '''
                global_page_id,
                scraped_websites!inner(website_id)
                '''
            ).eq('scraped_websites.website_id', str(website_id)).execute()
            
            if not associations_result.data:
                return ""
            
            # Get the global page IDs for this website
            website_page_ids = {assoc['global_page_id'] for assoc in associations_result.data}
            
            # Filter chunks to only those for pages associated with this website
            filtered_chunks = [
                chunk for chunk in chunks_result.data 
                if chunk.get('global_page_id') in website_page_ids
            ][:limit]
            
            if filtered_chunks:
                context_pieces = []
                for chunk in filtered_chunks:
                    page_title = chunk.get('global_scraped_pages', {}).get('title', 'Untitled')
                    page_url = chunk.get('global_scraped_pages', {}).get('url', '')
                    chunk_text = chunk.get('chunk_text', '')
                    
                    context_pieces.append(f"From: {page_title} ({page_url})\n{chunk_text}")
                
                return '\n\n'.join(context_pieces)
            
        except Exception as e:
            logger.error(f"Error getting global context: {e}")
            
        return ""
    
    def _create_text_chunks(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split content into overlapping chunks for embedding.
        
        Args:
            content: Text content to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not content or len(content.strip()) < 50:
            return []
        
        content = content.strip()
        chunks = []
        
        # Simple sentence-aware chunking
        sentences = content.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, truncate it
                    chunks.append(sentence[:chunk_size])
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]
        
        return chunks