from typing import Optional
from datetime import date
from sqlalchemy import String, DECIMAL, ForeignKey, Date, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSON
from uuid import UUID
from .base import Base


class UXAnalytics(Base):
    """
    Stores UX metrics and analytics data for performance monitoring.
    Part of Enhanced User Experience specification.
    Designed for time-series data with monthly partitioning.
    """
    __tablename__ = "ux_analytics"
    
    # Widget identification
    widget_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Optional conversation link
    conversation_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    
    # Metric information
    metric_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(DECIMAL(10, 4), nullable=False)
    
    # Additional metric metadata
    metric_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    # Timing (using custom recorded_at instead of base created_at for analytics)
    recorded_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
        index=True
    )
    
    # Date partition for partitioning (computed in PostgreSQL/Supabase)
    date_partition: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True
    )
    
    # Relationships
    conversation = relationship("Conversation", back_populates="ux_analytics")
    
    def __repr__(self) -> str:
        return f"<UXAnalytics {self.metric_type}={self.metric_value} widget={self.widget_id}>"