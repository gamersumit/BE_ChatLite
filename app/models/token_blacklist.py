"""
Token Blacklist model for JWT token revocation.
Stores revoked refresh tokens to prevent reuse.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .user import User


class TokenBlacklist(Base):
    """Model for tracking blacklisted (revoked) JWT tokens."""
    __tablename__ = "token_blacklist"

    # Token identification
    token_jti: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)

    # User relationship
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Timing information
    blacklisted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="blacklisted_tokens")

    def __repr__(self) -> str:
        return f"<TokenBlacklist {self.token_jti}>"

    @property
    def is_expired(self) -> bool:
        """Check if the blacklisted token has expired."""
        return datetime.utcnow() > self.expires_at