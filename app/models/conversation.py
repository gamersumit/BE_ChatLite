from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, DateTime, ForeignKey, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
import json
from .base import Base


class Conversation(Base):
    __tablename__ = "conversations"
    
    # Core conversation information
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    visitor_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # For visitor tracking across sessions
    website_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("websites.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Visitor information (optional)
    visitor_name: Mapped[Optional[str]] = mapped_column(String(255))
    visitor_email: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Conversation state
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, completed, abandoned
    
    # Context and metadata
    page_url: Mapped[Optional[str]] = mapped_column(String(1000))  # Page where conversation started
    page_title: Mapped[Optional[str]] = mapped_column(String(500))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv4 or IPv6
    referrer: Mapped[Optional[str]] = mapped_column(String(1000))
    
    # Conversation metrics
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    user_messages: Mapped[int] = mapped_column(Integer, default=0)
    ai_messages: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timing information
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Lead generation
    lead_captured: Mapped[bool] = mapped_column(Boolean, default=False)
    lead_data: Mapped[Optional[str]] = mapped_column(String(1000))  # JSON string for contact info
    
    # Quality and feedback
    satisfaction_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 scale
    feedback: Mapped[Optional[str]] = mapped_column(Text)
    
    # Website context integration
    website_context_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    context_pages_referenced: Mapped[Optional[list[str]]] = mapped_column(JSON, default=lambda: [])
    
    # Relationships
    website = relationship("Website", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    media_files = relationship("RichMediaFile", back_populates="conversation", cascade="all, delete-orphan")
    context = relationship("ConversationContext", back_populates="conversation", uselist=False, cascade="all, delete-orphan")
    ux_analytics = relationship("UXAnalytics", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Conversation {self.session_id} for {self.website.domain if self.website else 'Unknown'}>"