"""
Rich Media API endpoints - Basic implementation.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client

router = APIRouter()


class MediaResponse(BaseModel):
    """Basic media response."""
    status: str = "success"
    message: str = "Media endpoint available"


@router.get("/status", response_model=MediaResponse)
async def get_media_status(
    supabase: Client = Depends(get_supabase_client)
):
    """Get media service status."""
    return MediaResponse(
        status="success",
        message="Rich media service is running"
    )


@router.post("/upload", response_model=MediaResponse) 
async def upload_media(
    supabase: Client = Depends(get_supabase_client)
):
    """Upload media file."""
    return MediaResponse(
        status="success", 
        message="Media upload endpoint available"
    )