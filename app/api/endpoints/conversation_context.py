"""
Conversation Context API endpoints - Basic implementation.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client

router = APIRouter()


class ContextResponse(BaseModel):
    """Basic context response."""
    status: str = "success"
    context: Dict[str, Any] = {}


@router.get("/{conversation_id}", response_model=ContextResponse)
async def get_conversation_context(
    conversation_id: str,
    supabase: Client = Depends(get_supabase_client)
):
    """Get conversation context."""
    return ContextResponse(
        status="success",
        context={"conversation_id": conversation_id, "messages": []}
    )