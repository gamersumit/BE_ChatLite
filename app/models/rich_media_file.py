from typing import Optional
from sqlalchemy import String, Integer, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSON
from uuid import UUID
from .base import Base


class RichMediaFile(Base):
    """
    Stores metadata for uploaded files and media content.
    Part of Enhanced User Experience specification.
    """
    __tablename__ = "rich_media_files"
    
    # Foreign key relationships
    conversation_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    message_id: Mapped[Optional[UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    
    # File information
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Storage URLs
    file_url: Mapped[str] = mapped_column(Text, nullable=False)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text)
    
    # Accessibility
    alt_text: Mapped[Optional[str]] = mapped_column(Text)
    
    # Upload status
    upload_status: Mapped[str] = mapped_column(
        String(20), 
        default='processing',
        nullable=False
    )
    
    # Additional metadata
    file_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    # Relationships
    conversation = relationship("Conversation", back_populates="media_files")
    message = relationship("Message", back_populates="media_files")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("upload_status IN ('processing', 'completed', 'failed')", 
                       name='valid_upload_status'),
    )
    
    def __repr__(self) -> str:
        return f"<RichMediaFile {self.file_name} ({self.file_type})>"