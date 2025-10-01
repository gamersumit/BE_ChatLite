from typing import Optional
from sqlalchemy import String, Text, ForeignKey, DECIMAL, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, JSON
from uuid import UUID
from .base import Base


class ConversationContext(Base):
    """
    Stores conversation context and AI-generated insights for personalization.
    Part of Enhanced User Experience specification.
    """
    __tablename__ = "conversation_context"
    
    # Foreign key relationship (unique per conversation)
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # AI-generated insights
    user_intent: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    conversation_summary: Mapped[Optional[str]] = mapped_column(Text)
    user_sentiment: Mapped[Optional[str]] = mapped_column(String(20))
    
    # AI suggestions
    suggested_responses: Mapped[Optional[list]] = mapped_column(JSON, default=[])
    
    # Context metadata
    context_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    # Confidence scoring
    confidence_score: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 2), default=0.0)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="context")
    
    def __repr__(self) -> str:
        return f"<ConversationContext intent={self.user_intent} confidence={self.confidence_score}>"