from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, JSON, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class Website(Base):
    __tablename__ = "websites"

    # User isolation - every website belongs to a user
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Core website information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False, unique=True, index=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Widget configuration
    widget_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Configuration and metadata (all custom config goes here as JSONB)
    settings: Mapped[Optional[dict]] = mapped_column(JSON)
    custom_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    # Website scraping configuration
    scraping_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Status tracking
    verification_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    last_crawled: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Usage tracking
    total_conversations: Mapped[int] = mapped_column(Integer, default=0)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    monthly_message_limit: Mapped[int] = mapped_column(Integer, default=100)
    
    # Relationships
    user = relationship("User", back_populates="websites")
    conversations = relationship("Conversation", back_populates="website", cascade="all, delete-orphan")
    scraped_websites = relationship("ScrapedWebsite", back_populates="website", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Website {self.name} ({self.domain})>"