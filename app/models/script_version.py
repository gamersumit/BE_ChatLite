"""
Script Version model for tracking installation script versions.
Enables versioning, rollback, and change tracking for widget scripts.
"""

from typing import Optional, TYPE_CHECKING
from datetime import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey, JSON, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .website import Website


class ScriptVersion(Base):
    """Model for storing script versions and enabling rollbacks."""
    __tablename__ = "script_versions"
    
    # Relationships
    website_id: Mapped[str] = mapped_column(ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    website: Mapped["Website"] = relationship("Website", back_populates="script_versions")
    
    # Version information
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Script content and metadata
    script_content: Mapped[str] = mapped_column(Text, nullable=False)
    script_url: Mapped[str] = mapped_column(String(500), nullable=False)
    script_size: Mapped[int] = mapped_column(Integer, nullable=False)  # Size in bytes
    
    # Configuration snapshot
    widget_config: Mapped[dict] = mapped_column(JSON, nullable=False)  # Configuration used to generate script
    
    # Version status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Generation metadata
    generated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # User who generated this version
    generation_time: Mapped[float] = mapped_column(nullable=True)  # Time taken to generate (seconds)
    
    # CDN and caching
    cdn_uploaded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cache_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<ScriptVersion {self.website_id} v{self.version_number}>"
    
    def to_dict(self) -> dict:
        """Convert script version to dictionary."""
        return {
            "version_id": str(self.id),
            "website_id": str(self.website_id),
            "version_number": self.version_number,
            "version_hash": self.version_hash,
            "script_url": self.script_url,
            "script_size": self.script_size,
            "widget_config": self.widget_config,
            "is_active": self.is_active,
            "is_published": self.is_published,
            "generated_by": self.generated_by,
            "generation_time": self.generation_time,
            "cdn_uploaded": self.cdn_uploaded,
            "cache_key": self.cache_key,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def get_installation_code(self) -> str:
        """Generate HTML installation code for this script version."""
        return f'''<script 
    src="{self.script_url}" 
    async 
    data-widget-id="{self.website.widget_id if self.website else ''}"
    data-version="{self.version_number}">
</script>'''