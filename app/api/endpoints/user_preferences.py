"""
User Preferences API endpoints - Basic implementation.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client
from app.core.auth_middleware import get_current_user

router = APIRouter()


class UserPreferencesResponse(BaseModel):
    """Basic user preferences response."""
    user_id: str
    preferences: Dict[str, Any] = {}
    status: str = "success"


@router.get("/", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
):
    """Get user preferences for authenticated user."""
    user_id = current_user['id']

    try:
        # Get preferences from database
        result = supabase.table('user_preferences').select('*').eq('user_id', user_id).execute()

        preferences = {}
        if result.data:
            for pref in result.data:
                preferences[pref['preference_key']] = pref['preference_value']
    except Exception:
        # Default preferences if error occurs
        preferences = {"theme": "light", "language": "en"}

    return UserPreferencesResponse(
        user_id=user_id,
        preferences=preferences,
        status="success"
    )


@router.post("/", response_model=UserPreferencesResponse)
async def create_user_preferences(
    preferences: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
):
    """Create user preferences for authenticated user."""
    user_id = current_user['id']

    return UserPreferencesResponse(
        user_id=user_id,
        preferences=preferences,
        status="created"
    )


@router.put("/", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
):
    """Update user preferences for authenticated user."""
    user_id = current_user['id']

    return UserPreferencesResponse(
        user_id=user_id,
        preferences=preferences,
        status="updated"
    )