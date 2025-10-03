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


class ScreenshotUploadRequest(BaseModel):
    """Request model for screenshot upload."""
    screenshot_base64: str


class ScreenshotUploadResponse(BaseModel):
    """Response model for screenshot upload."""
    success: bool
    screenshot_url: str = None
    message: str = None


@router.post("/{website_id}/screenshot", response_model=ScreenshotUploadResponse)
async def upload_website_screenshot(
    website_id: str,
    request: ScreenshotUploadRequest,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """
    Upload website screenshot to Cloudinary and update database.
    This endpoint is called by the Celery worker after capturing a screenshot.
    """
    try:
        from app.services.cloudinary_service import CloudinaryService

        # Get website from database
        website_result = supabase.table('websites').select('*').eq('id', website_id).execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found"
            )

        website = website_result.data[0]
        domain = website['domain']

        # Upload to Cloudinary
        cloudinary_service = CloudinaryService()
        screenshot_url = cloudinary_service.upload_screenshot(
            screenshot_base64=request.screenshot_base64,
            website_domain=domain,
            website_id=website_id
        )

        if not screenshot_url:
            return ScreenshotUploadResponse(
                success=False,
                message="Failed to upload screenshot to Cloudinary"
            )

        # Update website with screenshot URL
        update_result = supabase.table('websites').update({
            'screenshot_url': screenshot_url
        }).eq('id', website_id).execute()

        if update_result.data:
            return ScreenshotUploadResponse(
                success=True,
                screenshot_url=screenshot_url,
                message="Screenshot uploaded successfully"
            )
        else:
            return ScreenshotUploadResponse(
                success=False,
                message="Failed to update website with screenshot URL"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading screenshot: {str(e)}"
        )