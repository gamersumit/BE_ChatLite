"""
Script Generation API endpoints - Enhanced for test site integration.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse
from supabase import Client
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.core.auth_middleware import get_current_user
from app.core.config import settings
from app.schemas.script_generation import (
    WidgetConfiguration,
    ScriptGenerationPostRequest,
    ScriptGenerationPostResponse
)

logger = logging.getLogger(__name__)


def trigger_website_crawl(
    website_id: str,
    website_url: str,
    user_id: Optional[str],
    job_type: str = "verification_crawl",
    max_pages: int = 25,
    max_depth: int = 3
) -> Optional[str]:
    """
    Shared function to trigger a website crawl.

    Args:
        website_id: ID of the website to crawl
        website_url: URL to crawl
        user_id: User ID (optional)
        job_type: Type of crawl job (verification_crawl, manual_crawl, etc.)
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth

    Returns:
        Job ID if successful, None if failed
    """
    try:
        from ...tasks.crawler_tasks import crawl_url
        from ...core.supabase_client import get_supabase_admin

        logger.info(f"🚀 Triggering {job_type} for website {website_id}, URL: {website_url}")

        # Get fresh supabase client in background task
        supabase = get_supabase_admin()

        # Create crawl job in crawling_jobs table
        crawl_job_id = str(uuid.uuid4())
        crawl_job = {
            'id': crawl_job_id,
            'website_id': website_id,
            'user_id': user_id,
            'job_type': job_type,
            'status': 'pending',
            'config': {
                'max_pages': max_pages,
                'max_depth': max_depth,
                'url': website_url
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"📝 Creating crawl job {crawl_job_id} in database")
        # Insert crawl job
        job_result = supabase.table('crawling_jobs').insert(crawl_job).execute()
        if not job_result.data:
            raise Exception("Failed to create crawl job in database")

        logger.info(f"📤 Dispatching Celery task for job {crawl_job_id}")
        # Start background crawl using Celery
        task = crawl_url.delay(
            job_id=crawl_job_id,
            website_id=website_id,
            url=website_url,
            max_pages=max_pages,
            max_depth=max_depth
        )
        logger.info(f"✅ Successfully dispatched {job_type} for website {website_id}, job_id: {crawl_job_id}, celery_task_id: {task.id}")
        return crawl_job_id

    except Exception as e:
        logger.error(f"❌ Failed to trigger {job_type} for website {website_id}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None

def escape_js_string(text: str) -> str:
    """Escape a string for safe embedding in JavaScript."""
    if not text:
        return ''
    # Replace problematic characters with their escaped equivalents
    return (text
            .replace('\\', '\\\\')  # Escape backslashes first
            .replace("'", "\\'")    # Escape single quotes
            .replace('"', '\\"')    # Escape double quotes
            .replace('\n', '\\n')   # Escape newlines
            .replace('\r', '\\r')   # Escape carriage returns
            .replace('\t', '\\t')   # Escape tabs
            )

router = APIRouter()


class ScriptResponse(BaseModel):
    """Enhanced script response."""
    script: str
    installation_instructions: Dict[str, str]
    widget_config: Optional[Dict[str, Any]] = None


class WidgetVerificationResponse(BaseModel):
    """Widget verification response."""
    verified: bool
    installation_detected: Optional[str] = None
    widget_status: str
    message: str



class WidgetVerifyRequest(BaseModel):
    """Widget verification request."""
    domain: str
    mode: Optional[str] = "embedded"
    page_url: Optional[str] = None
    user_agent: Optional[str] = None


@router.post("/verify/{widget_id}", response_model=WidgetVerificationResponse)
async def verify_widget_installation(
    widget_id: str,
    verification_data: WidgetVerifyRequest,
    background_tasks: BackgroundTasks,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Verify widget installation on website."""

    try:
        # Find the website in database by widget_id
        result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Widget not found"
            )

        website = result.data[0]
        website_db_id = website['id']

        # Check if widget is already verified to prevent redundant operations
        is_already_verified = website.get('verification_status') == "verified" and website.get('is_active', False)

        if is_already_verified:
            logger.info(f"Widget {widget_id} is already verified, skipping verification process")
            return WidgetVerificationResponse(
                verified=True,
                installation_detected=website.get('updated_at') or datetime.now(timezone.utc).isoformat(),
                widget_status="active",
                message="Already verified"
            )

        # Verify domain matches
        from urllib.parse import urlparse
        website_domain = website.get('domain', '').lower()
        request_domain = verification_data.domain.lower()

        # Parse URL if it's a full URL
        if website.get('url'):
            parsed = urlparse(website['url'])
            website_domain = parsed.netloc.lower() or website_domain

        # Check domain match (exact match or subdomain)
        domain_matches = (
            request_domain == website_domain or
            request_domain.endswith(f".{website_domain}") or
            website_domain.endswith(f".{request_domain}") or
            request_domain == "localhost"  # Allow localhost for development
        )

        if not domain_matches:
            logger.warning(f"Domain mismatch for widget {widget_id}: request={request_domain}, website={website_domain}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Domain mismatch: widget registered for {website_domain}, request from {request_domain}"
            )

        # Update installation and verification status in database
        from datetime import datetime, timezone

        # Only update fields that exist in the database schema
        updates = {
            'is_active': True,
            'verification_status': 'verified',
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        # Update in database
        update_result = supabase.table('websites').update(updates).eq('id', website_db_id).execute()

        if not update_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update verification status"
            )

        # Trigger initial crawl after verification in background (non-blocking)
        background_tasks.add_task(
            trigger_website_crawl,
            website_id=website_db_id,
            website_url=website.get('url'),
            user_id=website.get('user_id'),
            job_type='verification_crawl',
            max_pages=100,
            max_depth=3
        )

        return WidgetVerificationResponse(
            verified=True,
            installation_detected=datetime.now(timezone.utc).isoformat(),
            widget_status="active",
            message="Widget installation verified successfully and initial crawl started"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@router.get("/{widget_id}/status")
async def get_widget_status(
    widget_id: str,
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get widget status and configuration."""

    # Check database
    try:
        result = supabase.table('websites').select('*').eq('widget_id', widget_id).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Widget not found"
            )

        website = result.data[0]

        return {
            "widget_id": widget_id,
            "website_name": website.get("name", "Unknown"),
            "website_url": website.get("url", ""),
            "status": "active" if website.get("is_active") else "inactive",
            "verification_status": website.get("verification_status")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching widget status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching widget status"
        )