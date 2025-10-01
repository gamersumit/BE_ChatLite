"""
Website Management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_admin_client
from app.core.auth_middleware import get_current_user

router = APIRouter()


class WebsiteResponse(BaseModel):
    """Website operation response."""
    status: str
    message: str


@router.delete("/{website_id}", response_model=WebsiteResponse)
async def delete_website(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Delete a website owned by the authenticated user."""
    try:
        user_id = current_user['id']

        # First check if website exists and belongs to user
        website_result = supabase.table('websites').select('*').eq('id', website_id).eq('user_id', user_id).execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found or you don't have permission to delete it"
            )

        website = website_result.data[0]

        # Delete the website
        delete_result = supabase.table('websites').delete().eq('id', website_id).execute()

        if delete_result.data:
            return WebsiteResponse(
                status="success",
                message=f"Website '{website['name']}' deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete website"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting website: {str(e)}"
        )


@router.post("/{website_id}/activate", response_model=WebsiteResponse)
async def activate_website(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Activate a website after script integration is verified."""
    try:
        user_id = current_user['id']

        # Check if website exists and belongs to user
        website_result = supabase.table('websites').select('*').eq('id', website_id).eq('user_id', user_id).execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found or you don't have permission to activate it"
            )

        website = website_result.data[0]

        # Activate the website
        update_result = supabase.table('websites').update(
            {"is_active": True}
        ).eq('id', website_id).execute()

        if update_result.data:
            return WebsiteResponse(
                status="success",
                message=f"Website '{website['name']}' activated successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to activate website"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating website: {str(e)}"
        )