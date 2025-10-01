from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from .config import settings
from .supabase_client import get_supabase, get_supabase_admin


async def get_supabase_client():
    """
    Dependency to get Supabase client.
    Use this in FastAPI route dependencies.
    """
    return get_supabase()


async def get_supabase_admin_client():
    """
    Dependency to get Supabase admin client.
    Use this in FastAPI route dependencies for admin operations.
    """
    return get_supabase_admin()


def set_user_context(session: AsyncSession, user_id: Optional[str]) -> None:
    """
    Set the current user context for database row-level security.
    This should be called at the beginning of each request with the authenticated user ID.
    """
    if user_id:
        # Store the user ID in the session info for RLS policies
        session.info["current_user_id"] = user_id
    else:
        # Clear the user context
        session.info.pop("current_user_id", None)


def get_user_context(session: AsyncSession) -> Optional[str]:
    """
    Get the current user context from the database session.
    Returns the UUID of the currently authenticated user or None.
    """
    return session.info.get("current_user_id")


def ensure_user_context(session: AsyncSession) -> str:
    """
    Ensure that user context is set and return the user ID.
    Raises ValueError if no user context is set.
    """
    user_id = get_user_context(session)
    if user_id is None:
        raise ValueError("User context not set. Authentication required.")
    return user_id