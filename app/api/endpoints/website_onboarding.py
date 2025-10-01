"""
Website Onboarding API endpoints - Enhanced for test site support.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel, Field, validator
import re

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.core.auth_middleware import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


class WebsiteRegistrationRequest(BaseModel):
    """Website registration request model."""
    name: str = Field(..., min_length=1, max_length=255, description="Website name")
    url: str = Field(..., description="Website URL")
    domain: Optional[str] = Field(None, description="Website domain")
    description: Optional[str] = Field(None, max_length=1000, description="Website description")
    admin_email: Optional[str] = Field(None, description="Admin email address")
    is_test_site: bool = Field(default=False, description="Flag for test sites")
    
    @validator('url')
    def validate_url(cls, v):
        # Allow localhost URLs for test sites
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('admin_email')
    def validate_email(cls, v):
        if v and not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError('Invalid email format')
        return v


class OnboardingResponse(BaseModel):
    """Enhanced onboarding response."""
    id: str  # For compatibility with frontend expecting 'id'
    message: str  # Success message
    website: Dict[str, Any]  # Full website object
    # Legacy fields kept for backwards compatibility
    status: Optional[str] = "registered"
    website_id: Optional[str] = None
    widget_id: Optional[str] = None
    next_steps: Optional[Dict[str, str]] = None
    script_url: Optional[str] = None
    verification_url: Optional[str] = None


class WebsiteStatusResponse(BaseModel):
    """Website status response."""
    status: str
    website_id: str
    widget_id: str
    registered_at: Optional[str] = None
    script_installed: bool = False
    last_verified: Optional[str] = None




@router.post("/register", response_model=OnboardingResponse)
async def register_website(
    website_data: WebsiteRegistrationRequest,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Register a new website for ChatLite integration with user authentication."""

    try:
        user_id = current_user['id']

        # Generate unique IDs
        website_id = str(uuid.uuid4())
        widget_id = f"widget_{uuid.uuid4().hex[:8]}"

        # Extract domain from URL
        domain = website_data.domain or website_data.url.replace('http://', '').replace('https://', '').split('/')[0]

        # Store in Supabase database
        site_data = {
            "id": website_id,
            "name": website_data.name,
            "domain": domain,
            "url": website_data.url,
            "widget_id": widget_id,
            "user_id": user_id,  # Proper foreign key to users table
            "is_active": False,  # Start inactive - user activates after setup
            "scraping_enabled": True,  # Enabled by default for automatic crawling
            "verification_status": "pending",  # Will be verified when widget loads
            "settings": {
                "description": website_data.description or "",
                "category": getattr(website_data, 'category', 'Other'),
                "maxPages": getattr(website_data, 'maxPages', 100),
                "scrapingFrequency": getattr(website_data, 'scrapingFrequency', 'daily')
            },
            "custom_metadata": {
                "admin_email": website_data.admin_email,
                "is_test_site": website_data.is_test_site
            }
        }

        # Insert into database
        result = supabase.table('websites').insert(site_data).execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register website"
            )

    except Exception as e:
        error_message = str(e)
        # Check for duplicate key violation
        if 'duplicate key' in error_message.lower() or '23505' in error_message:
            # Extract the field that caused the duplicate
            if 'websites_url_key' in error_message:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A website with this URL already exists. Please use a different URL."
                )
            elif 'websites_domain_key' in error_message:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A website with this domain already exists. Please use a different domain."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A website with these details already exists."
                )
        else:
            # For other errors, return 500
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error registering website: {error_message}"
            )
    
    next_steps = {
        "1": "Copy the generated widget script",
        "2": "Add the script before the closing </body> tag on your website",
        "3": "Verify the installation",
        "4": "Start using the chatbot"
    }
    
    # Construct the full website object as it would appear in the frontend
    website_object = {
        "id": website_id,
        "name": website_data.name,
        "domain": domain,
        "url": website_data.url,
        "description": website_data.description if hasattr(website_data, 'description') else '',
        "category": website_data.category if hasattr(website_data, 'category') else 'general',
        "status": "inactive",  # New websites start in inactive status until script is verified
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "widget_id": widget_id,
        "is_active": False,
        # Additional fields expected by frontend
        "crawl_status": None,
        "last_crawled": None,
        "total_pages": 0,
        "total_threads": 0,
        "active_sessions": 0,
        "response_rate": 0
    }

    return OnboardingResponse(
        id=website_id,
        message=f"Successfully registered website: {website_data.name}",
        website=website_object,
        # Legacy fields for backwards compatibility
        status="registered",
        website_id=website_id,
        widget_id=widget_id,
        next_steps=next_steps,
        script_url=f"/api/v1/generate/{widget_id}",
        verification_url=f"/api/v1/widget/verify/{widget_id}"
    )


@router.get("/status/{website_id}", response_model=WebsiteStatusResponse)
async def get_website_status(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
):
    """Get website onboarding status."""

    # Check database
    try:
        result = supabase.table('websites').select('*').eq('id', website_id).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found"
            )

        site_data = result.data[0]

        return WebsiteStatusResponse(
            status="active" if site_data.get("is_active") else "inactive",
            website_id=website_id,
            widget_id=site_data["widget_id"],
            registered_at=site_data.get("created_at"),
            script_installed=site_data.get("verification_status") == "verified",
            last_verified=site_data.get("updated_at")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching website status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching website status"
        )

