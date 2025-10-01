"""
User-Website relationship model.
Defines the many-to-many relationship between users and websites.
"""

from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .user import User
    from .website import Website


class UserWebsite(Base):
    """Association table for User-Website many-to-many relationship."""
    __tablename__ = "user_websites"
    
    # Foreign key relationships
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    website_id: Mapped[str] = mapped_column(ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Role and permissions
    role: Mapped[str] = mapped_column(String(50), default="owner", nullable=False)
    
    # Relationships
    user: Mapped["User"] = relationship("User")
    website: Mapped["Website"] = relationship("Website")
    
    def __repr__(self) -> str:
        return f"<UserWebsite user_id={self.user_id} website_id={self.website_id} role={self.role}>"
    
    def to_dict(self) -> dict:
        """Convert relationship to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "website_id": str(self.website_id),
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }