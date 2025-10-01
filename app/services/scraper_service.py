"""
Service classes for website scraper using Supabase client.
Handles all database operations for scraped content.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from supabase import Client
from app.core.supabase_client import get_supabase, get_supabase_admin
from app.models.scraper_schemas import (
    ScrapedWebsite, ScrapedWebsiteCreate, ScrapedWebsiteUpdate,
    ScrapedPage, ScrapedPageCreate, ScrapedPageUpdate,
    ScrapedContentChunk, ScrapedContentChunkCreate, ScrapedContentChunkUpdate,
    ScrapedEntity, ScrapedEntityCreate, ScrapedEntityUpdate,
    ContentSearchQuery, SemanticSearchQuery, EntitySearchQuery,
    CrawlStatusUpdate
)


class ScrapedWebsiteService:
    """Service for managing scraped websites."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.client = supabase_client or get_supabase()
        self.admin_client = get_supabase_admin()
    
    async def create(self, data: ScrapedWebsiteCreate) -> ScrapedWebsite:
        """Create a new scraped website record."""
        try:
            result = self.client.table('scraped_websites').insert(data.model_dump()).execute()
            if result.data:
                return ScrapedWebsite(**result.data[0])
            raise Exception("Failed to create scraped website")
        except Exception as e:
            raise Exception(f"Error creating scraped website: {str(e)}")
    
    async def get_by_id(self, website_id: UUID) -> Optional[ScrapedWebsite]:
        """Get scraped website by ID."""
        try:
            result = self.client.table('scraped_websites').select("*").eq('id', str(website_id)).execute()
            if result.data:
                return ScrapedWebsite(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error fetching scraped website: {str(e)}")
    
    async def get_by_website_id(self, website_id: UUID) -> Optional[ScrapedWebsite]:
        """Get scraped website by main website ID."""
        try:
            result = self.client.table('scraped_websites').select("*").eq('website_id', str(website_id)).execute()
            if result.data:
                return ScrapedWebsite(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error fetching scraped website by website_id: {str(e)}")
    
    async def update(self, website_id: UUID, data: ScrapedWebsiteUpdate) -> Optional[ScrapedWebsite]:
        """Update scraped website."""
        try:
            update_data = {k: v for k, v in data.model_dump().items() if v is not None}
            result = self.client.table('scraped_websites').update(update_data).eq('id', str(website_id)).execute()
            if result.data:
                return ScrapedWebsite(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error updating scraped website: {str(e)}")
    
    async def update_crawl_status(self, website_id: UUID, status_update: CrawlStatusUpdate) -> Optional[ScrapedWebsite]:
        """Update crawl status and metrics."""
        try:
            update_data = status_update.model_dump(exclude_unset=True)
            result = self.client.table('scraped_websites').update(update_data).eq('id', str(website_id)).execute()
            if result.data:
                return ScrapedWebsite(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error updating crawl status: {str(e)}")
    
    async def delete(self, website_id: UUID) -> bool:
        """Delete scraped website and all related data."""
        try:
            result = self.client.table('scraped_websites').delete().eq('id', str(website_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            raise Exception(f"Error deleting scraped website: {str(e)}")
    
    async def list_by_status(self, status: str, limit: int = 100) -> List[ScrapedWebsite]:
        """List scraped websites by crawl status."""
        try:
            result = self.client.table('scraped_websites').select("*").eq('crawl_status', status).limit(limit).execute()
            return [ScrapedWebsite(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error listing scraped websites by status: {str(e)}")


class ScrapedPageService:
    """Service for managing scraped pages."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.client = supabase_client or get_supabase()
        self.admin_client = get_supabase_admin()
    
    async def create(self, data: ScrapedPageCreate) -> ScrapedPage:
        """Create a new scraped page record."""
        try:
            page_data = data.model_dump()
            # Convert Decimal to float for Supabase
            if 'relevance_score' in page_data and page_data['relevance_score'] is not None:
                page_data['relevance_score'] = float(page_data['relevance_score'])
            
            result = self.client.table('scraped_pages').insert(page_data).execute()
            if result.data:
                return ScrapedPage(**result.data[0])
            raise Exception("Failed to create scraped page")
        except Exception as e:
            raise Exception(f"Error creating scraped page: {str(e)}")
    
    async def get_by_id(self, page_id: UUID) -> Optional[ScrapedPage]:
        """Get scraped page by ID."""
        try:
            result = self.client.table('scraped_pages').select("*").eq('id', str(page_id)).execute()
            if result.data:
                return ScrapedPage(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error fetching scraped page: {str(e)}")
    
    async def get_by_website_and_url(self, scraped_website_id: UUID, url: str) -> Optional[ScrapedPage]:
        """Get scraped page by website ID and URL."""
        try:
            result = (self.client.table('scraped_pages').select("*")
                     .eq('scraped_website_id', str(scraped_website_id))
                     .eq('url', url).execute())
            if result.data:
                return ScrapedPage(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error fetching scraped page by URL: {str(e)}")
    
    async def list_by_website(self, scraped_website_id: UUID, limit: int = 100, offset: int = 0) -> List[ScrapedPage]:
        """List scraped pages by website."""
        try:
            result = (self.client.table('scraped_pages').select("*")
                     .eq('scraped_website_id', str(scraped_website_id))
                     .range(offset, offset + limit - 1).execute())
            return [ScrapedPage(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error listing scraped pages: {str(e)}")
    
    async def update(self, page_id: UUID, data: ScrapedPageUpdate) -> Optional[ScrapedPage]:
        """Update scraped page."""
        try:
            update_data = {k: v for k, v in data.model_dump().items() if v is not None}
            # Convert Decimal to float for Supabase
            if 'relevance_score' in update_data and update_data['relevance_score'] is not None:
                update_data['relevance_score'] = float(update_data['relevance_score'])
            
            result = self.client.table('scraped_pages').update(update_data).eq('id', str(page_id)).execute()
            if result.data:
                return ScrapedPage(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error updating scraped page: {str(e)}")
    
    async def search_content(self, query: ContentSearchQuery) -> List[ScrapedPage]:
        """Search pages using full-text search."""
        try:
            supabase_query = self.client.table('scraped_pages').select("*")
            
            # Apply filters
            if query.website_id:
                supabase_query = supabase_query.eq('scraped_website_id', str(query.website_id))
            if query.page_type:
                supabase_query = supabase_query.eq('page_type', query.page_type)
            
            # Full-text search using PostgreSQL
            supabase_query = supabase_query.text_search('title,meta_description,content_text', query.query)
            
            result = supabase_query.range(query.offset, query.offset + query.limit - 1).execute()
            return [ScrapedPage(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error searching content: {str(e)}")
    
    async def delete(self, page_id: UUID) -> bool:
        """Delete scraped page and all related data."""
        try:
            result = self.client.table('scraped_pages').delete().eq('id', str(page_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            raise Exception(f"Error deleting scraped page: {str(e)}")


class ScrapedContentChunkService:
    """Service for managing scraped content chunks."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.client = supabase_client or get_supabase()
        self.admin_client = get_supabase_admin()
    
    async def create(self, data: ScrapedContentChunkCreate) -> ScrapedContentChunk:
        """Create a new content chunk record."""
        try:
            result = self.client.table('scraped_content_chunks').insert(data.model_dump()).execute()
            if result.data:
                return ScrapedContentChunk(**result.data[0])
            raise Exception("Failed to create content chunk")
        except Exception as e:
            raise Exception(f"Error creating content chunk: {str(e)}")
    
    async def create_batch(self, chunks: List[ScrapedContentChunkCreate]) -> List[ScrapedContentChunk]:
        """Create multiple content chunks in batch."""
        try:
            chunk_data = [chunk.model_dump() for chunk in chunks]
            result = self.client.table('scraped_content_chunks').insert(chunk_data).execute()
            return [ScrapedContentChunk(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error creating content chunks batch: {str(e)}")
    
    async def get_by_page(self, page_id: UUID, limit: int = 100, offset: int = 0) -> List[ScrapedContentChunk]:
        """Get content chunks by page ID."""
        try:
            result = (self.client.table('scraped_content_chunks').select("*")
                     .eq('scraped_page_id', str(page_id))
                     .order('chunk_index')
                     .range(offset, offset + limit - 1).execute())
            return [ScrapedContentChunk(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error fetching content chunks: {str(e)}")
    
    async def semantic_search(self, query: SemanticSearchQuery) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings."""
        try:
            # Use RPC function for vector similarity search
            result = self.admin_client.rpc('search_content_chunks', {
                'query_embedding': query.embedding,
                'similarity_threshold': query.similarity_threshold,
                'website_filter': str(query.website_id) if query.website_id else None,
                'match_count': query.limit
            }).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error performing semantic search: {str(e)}")
    
    async def update_embedding(self, chunk_id: UUID, embedding: List[float]) -> Optional[ScrapedContentChunk]:
        """Update chunk embedding vector."""
        try:
            result = self.client.table('scraped_content_chunks').update({
                'embedding_vector': embedding
            }).eq('id', str(chunk_id)).execute()
            if result.data:
                return ScrapedContentChunk(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error updating chunk embedding: {str(e)}")
    
    async def delete_by_page(self, page_id: UUID) -> bool:
        """Delete all chunks for a page."""
        try:
            result = self.client.table('scraped_content_chunks').delete().eq('scraped_page_id', str(page_id)).execute()
            return True
        except Exception as e:
            raise Exception(f"Error deleting content chunks: {str(e)}")


class ScrapedEntityService:
    """Service for managing scraped entities."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.client = supabase_client or get_supabase()
        self.admin_client = get_supabase_admin()
    
    async def create(self, data: ScrapedEntityCreate) -> ScrapedEntity:
        """Create a new scraped entity record."""
        try:
            entity_data = data.model_dump()
            # Convert Decimal to float for Supabase
            if 'confidence_score' in entity_data and entity_data['confidence_score'] is not None:
                entity_data['confidence_score'] = float(entity_data['confidence_score'])
            
            result = self.client.table('scraped_entities').insert(entity_data).execute()
            if result.data:
                return ScrapedEntity(**result.data[0])
            raise Exception("Failed to create scraped entity")
        except Exception as e:
            raise Exception(f"Error creating scraped entity: {str(e)}")
    
    async def get_or_create(self, data: ScrapedEntityCreate) -> ScrapedEntity:
        """Get existing entity or create new one."""
        try:
            # Try to find existing entity
            existing = await self.get_by_name_and_type(
                data.scraped_website_id, 
                data.entity_name, 
                data.entity_type
            )
            
            if existing:
                # Update mention count
                await self.update(existing.id, ScrapedEntityUpdate(
                    mention_count=existing.mention_count + 1
                ))
                return existing
            else:
                # Create new entity
                return await self.create(data)
        except Exception as e:
            raise Exception(f"Error in get_or_create entity: {str(e)}")
    
    async def get_by_name_and_type(self, website_id: UUID, name: str, entity_type: Optional[str] = None) -> Optional[ScrapedEntity]:
        """Get entity by name and type within a website."""
        try:
            query = (self.client.table('scraped_entities').select("*")
                    .eq('scraped_website_id', str(website_id))
                    .eq('entity_name', name))
            
            if entity_type:
                query = query.eq('entity_type', entity_type)
            
            result = query.execute()
            if result.data:
                return ScrapedEntity(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error fetching entity: {str(e)}")
    
    async def list_by_website(self, website_id: UUID, query: EntitySearchQuery) -> List[ScrapedEntity]:
        """List entities by website with optional filters."""
        try:
            supabase_query = (self.client.table('scraped_entities').select("*")
                            .eq('scraped_website_id', str(website_id)))
            
            if query.entity_type:
                supabase_query = supabase_query.eq('entity_type', query.entity_type)
            
            if query.min_confidence:
                supabase_query = supabase_query.gte('confidence_score', float(query.min_confidence))
            
            result = (supabase_query.order('mention_count', desc=True)
                     .range(query.offset, query.offset + query.limit - 1).execute())
            return [ScrapedEntity(**item) for item in result.data]
        except Exception as e:
            raise Exception(f"Error listing entities: {str(e)}")
    
    async def update(self, entity_id: UUID, data: ScrapedEntityUpdate) -> Optional[ScrapedEntity]:
        """Update scraped entity."""
        try:
            update_data = {k: v for k, v in data.model_dump().items() if v is not None}
            # Convert Decimal to float for Supabase
            if 'confidence_score' in update_data and update_data['confidence_score'] is not None:
                update_data['confidence_score'] = float(update_data['confidence_score'])
            
            result = self.client.table('scraped_entities').update(update_data).eq('id', str(entity_id)).execute()
            if result.data:
                return ScrapedEntity(**result.data[0])
            return None
        except Exception as e:
            raise Exception(f"Error updating scraped entity: {str(e)}")
    
    async def delete_by_website(self, website_id: UUID) -> bool:
        """Delete all entities for a website."""
        try:
            result = self.client.table('scraped_entities').delete().eq('scraped_website_id', str(website_id)).execute()
            return True
        except Exception as e:
            raise Exception(f"Error deleting entities: {str(e)}")


# Service factory functions
def get_scraped_website_service() -> ScrapedWebsiteService:
    """Get scraped website service instance."""
    return ScrapedWebsiteService()


def get_scraped_page_service() -> ScrapedPageService:
    """Get scraped page service instance."""
    return ScrapedPageService()


def get_scraped_content_chunk_service() -> ScrapedContentChunkService:
    """Get scraped content chunk service instance."""
    return ScrapedContentChunkService()


def get_scraped_entity_service() -> ScrapedEntityService:
    """Get scraped entity service instance."""
    return ScrapedEntityService()