"""
Pydantic schemas for website scraper data models.
Used for validation and serialization with Supabase.
"""
from datetime import datetime
from typing import Optional, List, Any, Dict
from uuid import UUID
from pydantic import BaseModel, Field
from decimal import Decimal


class SupabaseCompatibleModel(BaseModel):
    """Base model with UUID serialization for Supabase compatibility."""
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to convert UUIDs and datetimes to strings for Supabase."""
        data = super().model_dump(**kwargs)
        return self._convert_objects_to_strings(data)
    
    def _convert_objects_to_strings(self, data: Any) -> Any:
        """Recursively convert UUID objects and datetime objects to strings."""
        if isinstance(data, UUID):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {key: self._convert_objects_to_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_objects_to_strings(item) for item in data]
        elif isinstance(data, Decimal):
            return float(data)
        return data


class ScrapedWebsiteBase(SupabaseCompatibleModel):
    """Base schema for scraped website."""
    website_id: UUID
    domain: str = Field(..., max_length=255)
    base_url: str = Field(..., max_length=500)
    crawl_status: str = Field(default="pending", max_length=50)
    last_crawled_at: Optional[datetime] = None
    total_pages_found: int = Field(default=0)
    pages_processed: int = Field(default=0)
    content_hash: Optional[str] = Field(None, max_length=64)
    robots_txt_content: Optional[str] = None
    sitemap_urls: Optional[List[str]] = None
    crawl_depth: int = Field(default=3)
    max_pages: int = Field(default=100)


class ScrapedWebsiteCreate(ScrapedWebsiteBase):
    """Schema for creating a scraped website."""
    pass


class ScrapedWebsiteUpdate(SupabaseCompatibleModel):
    """Schema for updating a scraped website."""
    crawl_status: Optional[str] = None
    last_crawled_at: Optional[datetime] = None
    total_pages_found: Optional[int] = None
    pages_processed: Optional[int] = None
    content_hash: Optional[str] = None
    robots_txt_content: Optional[str] = None
    sitemap_urls: Optional[List[str]] = None


class ScrapedWebsite(ScrapedWebsiteBase):
    """Complete scraped website schema."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScrapedPageBase(SupabaseCompatibleModel):
    """Base schema for scraped page."""
    scraped_website_id: UUID
    url: str = Field(..., max_length=1000)
    title: Optional[str] = Field(None, max_length=500)
    meta_description: Optional[str] = None
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    content_hash: Optional[str] = Field(None, max_length=64)
    page_type: Optional[str] = Field(None, max_length=50)
    depth_level: int = Field(default=0)
    word_count: int = Field(default=0)
    relevance_score: Decimal = Field(default=Decimal('0.0'))
    last_modified: Optional[datetime] = None
    status_code: Optional[int] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class ScrapedPageCreate(ScrapedPageBase):
    """Schema for creating a scraped page."""
    pass


class ScrapedPageUpdate(SupabaseCompatibleModel):
    """Schema for updating a scraped page."""
    title: Optional[str] = None
    meta_description: Optional[str] = None
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    content_hash: Optional[str] = None
    page_type: Optional[str] = None
    word_count: Optional[int] = None
    relevance_score: Optional[Decimal] = None
    last_modified: Optional[datetime] = None
    status_code: Optional[int] = None


class ScrapedPage(ScrapedPageBase):
    """Complete scraped page schema."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScrapedContentChunkBase(SupabaseCompatibleModel):
    """Base schema for scraped content chunk."""
    scraped_page_id: UUID
    chunk_text: str
    chunk_index: int
    token_count: Optional[int] = None
    embedding_vector: Optional[List[float]] = None
    chunk_type: Optional[str] = Field(None, max_length=50)


class ScrapedContentChunkCreate(ScrapedContentChunkBase):
    """Schema for creating a scraped content chunk."""
    pass


class ScrapedContentChunkUpdate(SupabaseCompatibleModel):
    """Schema for updating a scraped content chunk."""
    chunk_text: Optional[str] = None
    token_count: Optional[int] = None
    embedding_vector: Optional[List[float]] = None
    chunk_type: Optional[str] = None


class ScrapedContentChunk(ScrapedContentChunkBase):
    """Complete scraped content chunk schema."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScrapedEntityBase(SupabaseCompatibleModel):
    """Base schema for scraped entity."""
    scraped_website_id: UUID
    entity_type: Optional[str] = Field(None, max_length=100)
    entity_name: str = Field(..., max_length=300)
    entity_description: Optional[str] = None
    confidence_score: Optional[Decimal] = None
    mention_count: int = Field(default=1)
    first_seen_page_id: Optional[UUID] = None


class ScrapedEntityCreate(ScrapedEntityBase):
    """Schema for creating a scraped entity."""
    pass


class ScrapedEntityUpdate(SupabaseCompatibleModel):
    """Schema for updating a scraped entity."""
    entity_description: Optional[str] = None
    confidence_score: Optional[Decimal] = None
    mention_count: Optional[int] = None
    first_seen_page_id: Optional[UUID] = None


class ScrapedEntity(ScrapedEntityBase):
    """Complete scraped entity schema."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Search and query schemas
class ContentSearchQuery(SupabaseCompatibleModel):
    """Schema for content search queries."""
    query: str
    website_id: Optional[UUID] = None
    page_type: Optional[str] = None
    limit: int = Field(default=10, le=100)
    offset: int = Field(default=0, ge=0)


class SemanticSearchQuery(SupabaseCompatibleModel):
    """Schema for semantic search queries."""
    query: str
    embedding: List[float]
    website_id: Optional[UUID] = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=10, le=100)


class EntitySearchQuery(SupabaseCompatibleModel):
    """Schema for entity search queries."""
    entity_type: Optional[str] = None
    website_id: Optional[UUID] = None
    min_confidence: Optional[Decimal] = None
    limit: int = Field(default=10, le=100)
    offset: int = Field(default=0, ge=0)


class CrawlStatusUpdate(SupabaseCompatibleModel):
    """Schema for updating crawl status."""
    crawl_status: str = Field(..., max_length=50)
    pages_found: Optional[int] = None
    pages_processed: Optional[int] = None
    last_crawled_at: Optional[datetime] = None