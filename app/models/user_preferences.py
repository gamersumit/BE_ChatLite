from typing import Optional
from sqlalchemy import String, Boolean, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSON
from .base import Base


class UserPreferences(Base):
    """
    Stores user preferences for personalization and accessibility.
    Part of Enhanced User Experience specification.
    """
    __tablename__ = "user_preferences"
    
    # User identification (unique constraint)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    
    # Theme preferences
    theme: Mapped[str] = mapped_column(
        String(20), 
        default='auto',
        nullable=False
    )
    
    # Language preferences
    language: Mapped[str] = mapped_column(String(10), default='en', nullable=False)
    
    # Notification preferences
    notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Accessibility preferences
    accessibility_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    font_size: Mapped[str] = mapped_column(
        String(20), 
        default='medium',
        nullable=False
    )
    high_contrast: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reduced_motion: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Conversation style preferences
    conversation_style: Mapped[str] = mapped_column(
        String(20), 
        default='auto',
        nullable=False
    )
    
    # Custom settings storage
    custom_settings: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    # Table constraints for valid enum values
    __table_args__ = (
        CheckConstraint("theme IN ('light', 'dark', 'auto')", name='valid_theme'),
        CheckConstraint("conversation_style IN ('formal', 'casual', 'auto')", name='valid_conversation_style'),
        CheckConstraint("font_size IN ('small', 'medium', 'large', 'xlarge')", name='valid_font_size'),
    )
    
    def __repr__(self) -> str:
        return f"<UserPreferences {self.user_id} theme={self.theme}>"