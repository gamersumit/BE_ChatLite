"""
Database models for website content scraping functionality.
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from sqlalchemy import String, Text, Boolean, Integer, JSON, ForeignKey, DECIMAL, Index, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
from .base import Base


class ScrapedWebsite(Base):
    """Model for tracking scraped websites and their crawl status."""
    __tablename__ = "scraped_websites"
    
    # Foreign key to main website
    website_id: Mapped[UUID] = mapped_column(ForeignKey("websites.id", ondelete="CASCADE"), nullable=False)
    
    # Website information
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Crawl status and configuration
    crawl_status: Mapped[str] = mapped_column(String(50), default="pending", index=True)  # pending, crawling, completed, failed
    last_crawled_at: Mapped[Optional[datetime]] = mapped_column()
    
    # Crawl metrics
    total_pages_found: Mapped[int] = mapped_column(Integer, default=0)
    pages_processed: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))  # SHA-256 hash of overall site content
    
    # Crawl configuration
    robots_txt_content: Mapped[Optional[str]] = mapped_column(Text)
    sitemap_urls: Mapped[Optional[List[str]]] = mapped_column(JSON, default=lambda: [])
    crawl_depth: Mapped[int] = mapped_column(Integer, default=3)
    max_pages: Mapped[int] = mapped_column(Integer, default=100)
    
    # Relationships
    website = relationship("Website", back_populates="scraped_websites")
    scraped_pages = relationship("ScrapedPage", back_populates="scraped_website", cascade="all, delete-orphan")
    scraped_entities = relationship("ScrapedEntity", back_populates="scraped_website", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ScrapedWebsite {self.domain} ({self.crawl_status})>"


class ScrapedPage(Base):
    """Model for individual scraped pages and their content."""
    __tablename__ = "scraped_pages"
    
    # Foreign key to scraped website
    scraped_website_id: Mapped[UUID] = mapped_column(ForeignKey("scraped_websites.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Page information
    url: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    meta_description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Content
    content_text: Mapped[Optional[str]] = mapped_column(Text)  # Main extracted text content
    content_html: Mapped[Optional[str]] = mapped_column(Text)  # Cleaned HTML content
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))  # SHA-256 hash for change detection
    
    # Page metadata
    page_type: Mapped[Optional[str]] = mapped_column(String(50))  # home, product, service, contact, about, etc.
    depth_level: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    relevance_score: Mapped[float] = mapped_column(DECIMAL(5,3), default=0.0, index=True)  # 0.0 to 1.0
    
    # Crawl metadata
    last_modified: Mapped[Optional[datetime]] = mapped_column()
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    scraped_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # Relationships
    scraped_website = relationship("ScrapedWebsite", back_populates="scraped_pages")
    content_chunks = relationship("ScrapedContentChunk", back_populates="scraped_page", cascade="all, delete-orphan")
    
    # Constraints  
    __table_args__ = (
        UniqueConstraint('scraped_website_id', 'url', name='unique_url_per_website'),
        # Full-text search index for PostgreSQL/Supabase
        Index('idx_scraped_pages_content_search', text('to_tsvector(\'english\', title || \' \' || COALESCE(meta_description, \'\') || \' \' || COALESCE(content_text, \'\'))')),
    )
    
    def __repr__(self) -> str:
        return f"<ScrapedPage {self.url} ({self.page_type})>"


class ScrapedContentChunk(Base):
    """Model for chunked content from scraped pages for vector search."""
    __tablename__ = "scraped_content_chunks"
    
    # Foreign key to scraped page
    scraped_page_id: Mapped[UUID] = mapped_column(ForeignKey("scraped_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Content chunk data
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Order within the page
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Vector embedding for semantic search (pgvector extension)
    embedding_vector: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536) if PGVECTOR_AVAILABLE else JSON, 
        nullable=True,
        comment="OpenAI embedding vector for semantic search"
    )
    
    # Chunk metadata
    chunk_type: Mapped[Optional[str]] = mapped_column(String(50))  # paragraph, heading, list, table, etc.
    
    # Relationships
    scraped_page = relationship("ScrapedPage", back_populates="content_chunks")
    
    def __repr__(self) -> str:
        return f"<ScrapedContentChunk {self.scraped_page_id}:{self.chunk_index}>"


class ScrapedEntity(Base):
    """Model for extracted entities from scraped content."""
    __tablename__ = "scraped_entities"
    
    # Foreign key to scraped website
    scraped_website_id: Mapped[UUID] = mapped_column(ForeignKey("scraped_websites.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Entity information
    entity_type: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # PERSON, ORG, PRODUCT, SERVICE, LOCATION, etc.
    entity_name: Mapped[str] = mapped_column(String(300), nullable=False)
    entity_description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Extraction metadata
    confidence_score: Mapped[Optional[float]] = mapped_column(DECIMAL(5,3))  # NLP extraction confidence
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_page_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("scraped_pages.id"))
    
    # Relationships
    scraped_website = relationship("ScrapedWebsite", back_populates="scraped_entities")
    first_seen_page = relationship("ScrapedPage", foreign_keys=[first_seen_page_id])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('scraped_website_id', 'entity_type', 'entity_name', name='unique_entity_per_website'),
    )
    
    def __repr__(self) -> str:
        return f"<ScrapedEntity {self.entity_type}:{self.entity_name}>"