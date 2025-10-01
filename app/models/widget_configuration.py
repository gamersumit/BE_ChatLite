"""
Widget Configuration Version model.
Tracks configuration changes and allows rollback functionality.
"""

from typing import Optional, TYPE_CHECKING
from datetime import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .website import Website


class WidgetConfigurationVersion(Base):
    """Model for storing widget configuration versions."""
    __tablename__ = "widget_configuration_versions"
    
    # Relationships
    website_id: Mapped[str] = mapped_column(ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    website: Mapped["Website"] = relationship("Website", back_populates="configuration_versions")
    
    # Version information
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration data (JSON snapshot)
    configuration_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # User who created this version
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)  # Current active version
    
    def __repr__(self) -> str:
        return f"<WidgetConfigurationVersion {self.website_id} v{self.version_number}>"
    
    def to_dict(self) -> dict:
        """Convert configuration version to dictionary."""
        return {
            "version_id": str(self.id),
            "website_id": str(self.website_id),
            "version_number": self.version_number,
            "version_name": self.version_name,
            "description": self.description,
            "configuration": self.configuration_data,
            "created_by": self.created_by,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }