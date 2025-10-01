from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, DateTime, ForeignKey, func, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class Message(Base):
    __tablename__ = "messages"

    # User isolation - every message belongs to a user (through conversation)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Core message information
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Message content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "user" or "assistant"
    
    # Message metadata
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)  # Order in conversation
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    character_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Processing information (for AI messages)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer)  # Time to generate response
    model_used: Mapped[Optional[str]] = mapped_column(String(50))  # e.g., "gpt-4o"
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    
    # Context and sources (for RAG responses)
    context_sources: Mapped[Optional[dict]] = mapped_column(JSON)  # References to website content used
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)  # AI confidence in response
    
    # User interaction
    was_helpful: Mapped[Optional[bool]] = mapped_column(Boolean)  # User feedback
    user_reaction: Mapped[Optional[str]] = mapped_column(String(20))  # thumbs_up, thumbs_down, etc.
    
    # Status and delivery
    status: Mapped[str] = mapped_column(String(20), default="sent")  # sent, delivered, read, failed
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timing
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")
    media_files = relationship("RichMediaFile", back_populates="message", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message {self.message_type} #{self.sequence_number}: {content_preview}>"