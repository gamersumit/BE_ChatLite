"""
Authentication utilities for JWT token management and password hashing.
Provides functions for creating, validating, and managing JWT tokens.
"""

import secrets
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import jwt
from app.models.token_blacklist import TokenBlacklist


# JWT Configuration
JWT_SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12
REFRESH_TOKEN_EXPIRE_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with 12-hour expiry."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    payload = {
        "user_id": user_id,
        "type": "access",
        "exp": expire.timestamp(),
        "iat": datetime.now(timezone.utc).timestamp(),
        "jti": secrets.token_urlsafe(32)
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT refresh token with 7-day expiry."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "user_id": user_id,
        "type": "refresh",
        "exp": expire.timestamp(),
        "iat": datetime.now(timezone.utc).timestamp(),
        "jti": secrets.token_urlsafe(32)
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    Raises jwt.InvalidTokenError if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.InvalidTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")


async def blacklist_token(session: AsyncSession, token_jti: str, user_id: str, expires_at_timestamp: float) -> None:
    """Add a token to the blacklist."""
    expires_at = datetime.fromtimestamp(expires_at_timestamp, tz=timezone.utc)

    blacklist_entry = TokenBlacklist(
        token_jti=token_jti,
        user_id=user_id,
        expires_at=expires_at
    )

    session.add(blacklist_entry)
    await session.commit()


async def is_token_blacklisted(session: AsyncSession, token_jti: str) -> bool:
    """Check if a token JTI is blacklisted."""
    result = await session.execute(
        select(TokenBlacklist).where(TokenBlacklist.token_jti == token_jti)
    )
    blacklist_entry = result.scalar_one_or_none()

    if blacklist_entry is None:
        return False

    # If token is expired, it's effectively not blacklisted anymore
    if blacklist_entry.is_expired:
        return False

    return True


async def cleanup_expired_blacklist_tokens(session: AsyncSession) -> int:
    """
    Remove expired tokens from blacklist.
    Returns number of deleted tokens.
    """
    now = datetime.now(timezone.utc)

    # Delete expired blacklist entries
    result = await session.execute(
        delete(TokenBlacklist).where(TokenBlacklist.expires_at < now)
    )

    deleted_count = result.rowcount
    await session.commit()

    return deleted_count


def extract_user_id_from_token(token: str) -> str:
    """
    Extract user ID from a valid JWT token.
    Returns None if token is invalid.
    """
    try:
        payload = decode_token(token)
        return payload.get("user_id")
    except jwt.InvalidTokenError:
        return None


async def refresh_access_token(session: AsyncSession, refresh_token: str) -> Optional[tuple[str, str]]:
    """
    Generate a new access token using a valid refresh token.
    Returns tuple of (access_token, new_refresh_token) or None if invalid.
    """
    try:
        payload = decode_token(refresh_token)

        # Verify this is a refresh token
        if payload.get("type") != "refresh":
            return None

        # Check if token is blacklisted
        if await is_token_blacklisted(session, payload["jti"]):
            return None

        user_id = payload["user_id"]

        # Blacklist the old refresh token
        await blacklist_token(session, payload["jti"], user_id, payload["exp"])

        # Create new tokens
        new_access_token = create_access_token(user_id)
        new_refresh_token = create_refresh_token(user_id)

        return new_access_token, new_refresh_token

    except jwt.InvalidTokenError:
        return None


def generate_verification_token() -> str:
    """Generate a secure verification token for email verification."""
    return secrets.token_urlsafe(32)


def generate_reset_token() -> str:
    """Generate a secure reset token for password resets."""
    return secrets.token_urlsafe(32)