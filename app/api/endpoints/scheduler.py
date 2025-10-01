"""
Scheduler API endpoints for managing automated crawling schedules.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.core.auth_middleware import get_current_user
from app.services.automated_scheduler import get_automated_scheduler
from app.services.registration_scheduler import get_registration_scheduler

router = APIRouter()


class SchedulerStatusResponse(BaseModel):
    """Scheduler status response model."""
    status: str = "success"
    data: Dict[str, Any] = {}


class UpdateScheduleRequest(BaseModel):
    """Update schedule request model."""
    website_id: str
    frequency: str  # daily, weekly, monthly, manual
    enabled: bool = True


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get current status of the automated scheduler."""
    try:
        scheduler = get_automated_scheduler()
        status_info = scheduler.get_scheduler_status()

        return SchedulerStatusResponse(
            status="success",
            data=status_info
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting scheduler status: {str(e)}"
        )


@router.post("/start", response_model=SchedulerStatusResponse)
async def start_scheduler(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Start the automated scheduler system."""
    try:
        scheduler = get_automated_scheduler()
        result = scheduler.start_scheduler()

        return SchedulerStatusResponse(
            status="success" if result.get('success') else "error",
            data=result
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting scheduler: {str(e)}"
        )


@router.post("/check", response_model=SchedulerStatusResponse)
async def trigger_schedule_check(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Manually trigger a schedule check and update."""
    try:
        scheduler = get_automated_scheduler()
        result = scheduler.trigger_manual_schedule_check()

        return SchedulerStatusResponse(
            status="success" if result.get('success') else "error",
            data=result
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering schedule check: {str(e)}"
        )


@router.get("/upcoming", response_model=SchedulerStatusResponse)
async def get_upcoming_tasks(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get upcoming scheduled crawl tasks."""
    try:
        scheduler = get_automated_scheduler()
        upcoming_tasks = scheduler.get_upcoming_tasks(limit=limit)

        # Filter tasks to only show user's websites
        user_id = current_user['id']

        # Get user's websites
        websites_result = supabase.table('websites').select(
            'id'
        ).eq('user_id', user_id).execute()

        user_website_ids = {w['id'] for w in (websites_result.data or [])}

        # Filter upcoming tasks
        filtered_tasks = [
            task for task in upcoming_tasks
            if task.get('website_id') in user_website_ids
        ]

        return SchedulerStatusResponse(
            status="success",
            data={
                "upcoming_tasks": filtered_tasks,
                "total_tasks": len(filtered_tasks)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting upcoming tasks: {str(e)}"
        )


@router.put("/update", response_model=SchedulerStatusResponse)
async def update_website_schedule(
    request: UpdateScheduleRequest,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Update a website's crawl schedule."""
    try:
        user_id = current_user['id']

        # Verify user owns the website
        website_result = supabase.table('websites').select(
            'id, user_id, domain'
        ).eq('id', request.website_id).eq('user_id', user_id).single().execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found or access denied"
            )

        # Update schedule using registration scheduler
        reg_scheduler = get_registration_scheduler()
        update_result = reg_scheduler.update_website_crawl_schedule(
            website_id=request.website_id,
            new_frequency=request.frequency,
            enabled=request.enabled
        )

        if not update_result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=update_result.get('error', 'Schedule update failed')
            )

        # Update automated scheduler
        auto_scheduler = get_automated_scheduler()
        scheduler_update = auto_scheduler.update_website_schedule(
            website_id=request.website_id,
            new_frequency=request.frequency
        )

        return SchedulerStatusResponse(
            status="success",
            data={
                "website_id": request.website_id,
                "frequency": request.frequency,
                "enabled": request.enabled,
                "database_update": update_result,
                "scheduler_update": scheduler_update,
                "message": "Schedule updated successfully"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating schedule: {str(e)}"
        )


@router.delete("/remove/{website_id}", response_model=SchedulerStatusResponse)
async def remove_website_schedule(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Remove a website's schedule from the automated system."""
    try:
        user_id = current_user['id']

        # Verify user owns the website
        website_result = supabase.table('websites').select(
            'id, user_id, domain'
        ).eq('id', website_id).eq('user_id', user_id).single().execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found or access denied"
            )

        # Remove from automated scheduler
        scheduler = get_automated_scheduler()
        result = scheduler.remove_website_schedule(website_id)

        # Update database to disable scheduling
        supabase.table('websites').update({
            'scraping_enabled': False,
            'next_scheduled_crawl': None
        }).eq('id', website_id).execute()

        return SchedulerStatusResponse(
            status="success",
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing schedule: {str(e)}"
        )


@router.get("/website/{website_id}", response_model=SchedulerStatusResponse)
async def get_website_schedule(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get schedule information for a specific website."""
    try:
        user_id = current_user['id']

        # Get website with schedule info
        website_result = supabase.table('websites').select(
            'id, domain, name, scraping_frequency, scraping_enabled, '
            'next_scheduled_crawl, last_scheduled_crawl, max_pages, max_depth'
        ).eq('id', website_id).eq('user_id', user_id).single().execute()

        if not website_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Website not found or access denied"
            )

        website = website_result.data

        # Check if schedule is active in automated scheduler
        scheduler = get_automated_scheduler()
        task_name = f"crawl_website_{website_id}"
        is_active_in_scheduler = task_name in scheduler.active_schedules

        return SchedulerStatusResponse(
            status="success",
            data={
                "website": website,
                "is_active_in_scheduler": is_active_in_scheduler,
                "schedule_info": scheduler.active_schedules.get(task_name, {})
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting website schedule: {str(e)}"
        )