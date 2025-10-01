"""
UX Analytics API endpoints - Basic implementation.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client

router = APIRouter()


class AnalyticsResponse(BaseModel):
    """Basic analytics response."""
    status: str = "success"
    data: Dict[str, Any] = {}


@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_analytics_dashboard(
    supabase: Client = Depends(get_supabase_client)
):
    """Get analytics dashboard data."""
    return AnalyticsResponse(
        status="success",
        data={"total_users": 0, "total_conversations": 0}
    )


@router.post("/track", response_model=AnalyticsResponse)
async def track_event(
    event_data: Dict[str, Any],
    supabase: Client = Depends(get_supabase_client)
):
    """Track analytics event."""
    return AnalyticsResponse(
        status="success",
        data={"event_tracked": True}
    )