from typing import Optional
from datetime import datetime, date
from sqlalchemy import String, Integer, DateTime, Date, ForeignKey, func, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSON
from uuid import UUID
from .base import Base


class WidgetAnalytics(Base):
    __tablename__ = "widget_analytics"
    
    # Core analytics information
    website_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), 
        ForeignKey("websites.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Time period
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    hour: Mapped[Optional[int]] = mapped_column(Integer)  # 0-23 for hourly analytics
    
    # Usage metrics
    page_views: Mapped[int] = mapped_column(Integer, default=0)  # Widget loaded
    widget_opens: Mapped[int] = mapped_column(Integer, default=0)  # Chat opened
    conversations_started: Mapped[int] = mapped_column(Integer, default=0)
    conversations_completed: Mapped[int] = mapped_column(Integer, default=0)
    
    # Message metrics
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    user_messages: Mapped[int] = mapped_column(Integer, default=0)
    ai_messages: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    average_response_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    total_processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    timeout_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # User engagement
    unique_users: Mapped[int] = mapped_column(Integer, default=0)
    returning_users: Mapped[int] = mapped_column(Integer, default=0)
    average_session_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    bounce_rate: Mapped[Optional[float]] = mapped_column(Float)  # % who close without messaging
    
    # Lead generation
    leads_captured: Mapped[int] = mapped_column(Integer, default=0)
    email_addresses_collected: Mapped[int] = mapped_column(Integer, default=0)
    conversion_rate: Mapped[Optional[float]] = mapped_column(Float)  # % of conversations to leads
    
    # Satisfaction metrics
    positive_feedback: Mapped[int] = mapped_column(Integer, default=0)
    negative_feedback: Mapped[int] = mapped_column(Integer, default=0)
    average_rating: Mapped[Optional[float]] = mapped_column(Float)  # 1-5 scale
    
    # Cost tracking
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_per_conversation: Mapped[Optional[float]] = mapped_column(Float)
    
    # Top pages and sources
    top_pages: Mapped[Optional[dict]] = mapped_column(JSON)  # Page URLs with counts
    top_referrers: Mapped[Optional[dict]] = mapped_column(JSON)  # Referrer URLs with counts
    user_agents: Mapped[Optional[dict]] = mapped_column(JSON)  # Browser/device stats
    
    # Geographic data (if available)
    countries: Mapped[Optional[dict]] = mapped_column(JSON)  # Country codes with counts
    cities: Mapped[Optional[dict]] = mapped_column(JSON)  # City names with counts
    
    # Widget specific
    load_time_ms: Mapped[Optional[float]] = mapped_column(Float)  # Average widget load time
    widget_failures: Mapped[int] = mapped_column(Integer, default=0)
    javascript_errors: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    website = relationship("Website", back_populates="analytics")
    
    def __repr__(self) -> str:
        return f"<WidgetAnalytics {self.date} for {self.website.domain if self.website else 'Unknown'}>"