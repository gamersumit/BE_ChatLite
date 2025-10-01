"""
API endpoints for schedule status tracking and display.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.auth_middleware import get_current_user
from ...services.schedule_status import (
    get_schedule_status_service,
    get_website_status,
    get_all_website_statuses,
    get_system_health,
    ScheduleStatus,
    CrawlStatus
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ScheduleStatusResponse(BaseModel):
    """Response model for schedule status."""
    website_id: str = Field(..., description="Website identifier")
    website_domain: str = Field(..., description="Website domain")
    frequency: str = Field(..., description="Crawling frequency")
    schedule_status: str = Field(..., description="Current schedule status")
    last_crawl_time: Optional[str] = Field(None, description="Last crawl timestamp")
    last_crawl_status: Optional[str] = Field(None, description="Last crawl status")
    next_scheduled_time: Optional[str] = Field(None, description="Next scheduled crawl time")
    user_timezone: str = Field(..., description="User's timezone")
    pages_crawled: int = Field(default=0, description="Pages crawled in last run")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")
    created_at: str = Field(..., description="Website creation timestamp")


class ScheduleHealthResponse(BaseModel):
    """Response model for schedule health metrics."""
    total_schedules: int = Field(..., description="Total number of schedules")
    active_schedules: int = Field(..., description="Number of active schedules")
    failed_schedules: int = Field(..., description="Number of failed schedules")
    paused_schedules: int = Field(..., description="Number of paused schedules")
    average_success_rate: float = Field(..., description="Average success rate")
    total_executions_24h: int = Field(..., description="Total executions in last 24h")
    failed_executions_24h: int = Field(..., description="Failed executions in last 24h")
    last_updated: str = Field(..., description="Last update timestamp")


class ScheduleStatusUpdateRequest(BaseModel):
    """Request model for updating schedule status."""
    status: str = Field(..., description="New schedule status", pattern="^(active|paused|failed|idle|pending)$")
    reason: Optional[str] = Field(None, description="Reason for status change")


@router.get("/website/{website_id}")
async def get_website_schedule_status(
    website_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get schedule status for a specific website.
    """
    try:
        status_info = get_website_status(website_id)

        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Website schedule status not found: {website_id}"
            )

        return {
            "success": True,
            "data": {
                "website_id": status_info.website_id,
                "website_domain": status_info.website_domain,
                "frequency": status_info.frequency,
                "schedule_status": status_info.schedule_status.value,
                "last_crawl_time": status_info.last_crawl_time.isoformat() if status_info.last_crawl_time else None,
                "last_crawl_status": status_info.last_crawl_status.value if status_info.last_crawl_status else None,
                "next_scheduled_time": status_info.next_scheduled_time.isoformat() if status_info.next_scheduled_time else None,
                "user_timezone": status_info.user_timezone,
                "pages_crawled": status_info.pages_crawled,
                "success_rate": status_info.success_rate,
                "consecutive_failures": status_info.consecutive_failures,
                "created_at": status_info.created_at.isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get website schedule status for {website_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve website schedule status"
        )


@router.get("/websites")
async def get_all_website_schedule_statuses(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get schedule status for all websites, optionally filtered by user.
    """
    try:
        # If no user_id provided, use current user's ID for filtering
        filter_user_id = user_id or current_user.get('user_id')

        status_list = get_all_website_statuses(filter_user_id)

        return {
            "success": True,
            "data": {
                "websites": [
                    {
                        "website_id": status.website_id,
                        "website_domain": status.website_domain,
                        "frequency": status.frequency,
                        "schedule_status": status.schedule_status.value,
                        "last_crawl_time": status.last_crawl_time.isoformat() if status.last_crawl_time else None,
                        "last_crawl_status": status.last_crawl_status.value if status.last_crawl_status else None,
                        "next_scheduled_time": status.next_scheduled_time.isoformat() if status.next_scheduled_time else None,
                        "user_timezone": status.user_timezone,
                        "pages_crawled": status.pages_crawled,
                        "success_rate": status.success_rate,
                        "consecutive_failures": status.consecutive_failures,
                        "created_at": status.created_at.isoformat()
                    }
                    for status in status_list
                ],
                "total_websites": len(status_list)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get all website schedule statuses: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve website schedule statuses"
        )


@router.get("/health")
async def get_schedule_health_metrics(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get overall schedule health metrics.
    """
    try:
        health_metrics = get_system_health()

        return {
            "success": True,
            "data": {
                "total_schedules": health_metrics.total_schedules,
                "active_schedules": health_metrics.active_schedules,
                "failed_schedules": health_metrics.failed_schedules,
                "paused_schedules": health_metrics.paused_schedules,
                "average_success_rate": health_metrics.average_success_rate,
                "total_executions_24h": health_metrics.total_executions_24h,
                "failed_executions_24h": health_metrics.failed_executions_24h,
                "last_updated": health_metrics.last_updated.isoformat(),
                "health_score": _calculate_overall_health_score(health_metrics)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get schedule health metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve schedule health metrics"
        )


@router.put("/website/{website_id}/status")
async def update_website_schedule_status(
    website_id: str,
    request: ScheduleStatusUpdateRequest,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update schedule status for a specific website.
    """
    try:
        service = get_schedule_status_service()

        # Validate status value
        try:
            new_status = ScheduleStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status value: {request.status}"
            )

        success = service.update_schedule_status(
            website_id,
            new_status,
            request.reason
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to update schedule status"
            )

        return {
            "success": True,
            "data": {
                "website_id": website_id,
                "new_status": request.status,
                "reason": request.reason,
                "updated_at": datetime.utcnow().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update schedule status for {website_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update schedule status"
        )


@router.get("/website/{website_id}/history")
async def get_website_crawl_history(
    website_id: str,
    limit: int = Query(10, description="Number of history entries to return", ge=1, le=100),
    offset: int = Query(0, description="Number of entries to skip", ge=0),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get crawl history for a specific website with schedule status context.
    """
    try:
        service = get_schedule_status_service()

        # Get basic website status first
        status_info = get_website_status(website_id)
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Website not found: {website_id}"
            )

        # Get crawl history from database
        history_response = service.supabase.table('crawl_history').select(
            'id, status, completed_at, pages_crawled, trigger_type, error_message, duration_seconds'
        ).eq('website_id', website_id).order(
            'completed_at', desc=True
        ).range(offset, offset + limit - 1).execute()

        history_entries = []
        for entry in history_response.data:
            history_entries.append({
                "id": entry['id'],
                "status": entry['status'],
                "completed_at": entry['completed_at'],
                "pages_crawled": entry.get('pages_crawled', 0),
                "trigger_type": entry.get('trigger_type', 'unknown'),
                "error_message": entry.get('error_message'),
                "duration_seconds": entry.get('duration_seconds')
            })

        return {
            "success": True,
            "data": {
                "website_id": website_id,
                "website_domain": status_info.website_domain,
                "current_status": {
                    "schedule_status": status_info.schedule_status.value,
                    "last_crawl_status": status_info.last_crawl_status.value if status_info.last_crawl_status else None,
                    "next_scheduled_time": status_info.next_scheduled_time.isoformat() if status_info.next_scheduled_time else None,
                    "success_rate": status_info.success_rate
                },
                "history": history_entries,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total_entries": len(history_entries)
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl history for {website_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve crawl history"
        )


@router.get("/dashboard")
async def get_schedule_dashboard_data(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive schedule dashboard data including status and health metrics.
    """
    try:
        user_id = current_user.get('user_id')

        # Get all website statuses for user
        website_statuses = get_all_website_statuses(user_id)

        # Get system health metrics
        health_metrics = get_system_health()

        # Calculate user-specific metrics
        user_metrics = _calculate_user_metrics(website_statuses)

        return {
            "success": True,
            "data": {
                "user_metrics": user_metrics,
                "system_health": {
                    "total_schedules": health_metrics.total_schedules,
                    "active_schedules": health_metrics.active_schedules,
                    "failed_schedules": health_metrics.failed_schedules,
                    "average_success_rate": health_metrics.average_success_rate,
                    "total_executions_24h": health_metrics.total_executions_24h,
                    "health_score": _calculate_overall_health_score(health_metrics)
                },
                "websites": [
                    {
                        "website_id": status.website_id,
                        "website_domain": status.website_domain,
                        "frequency": status.frequency,
                        "schedule_status": status.schedule_status.value,
                        "last_crawl_status": status.last_crawl_status.value if status.last_crawl_status else None,
                        "next_scheduled_time": status.next_scheduled_time.isoformat() if status.next_scheduled_time else None,
                        "success_rate": status.success_rate,
                        "consecutive_failures": status.consecutive_failures,
                        "status_indicator": _get_status_indicator(status)
                    }
                    for status in website_statuses
                ]
            }
        }

    except Exception as e:
        logger.error(f"Failed to get schedule dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve schedule dashboard data"
        )


def _calculate_overall_health_score(health_metrics) -> float:
    """
    Calculate overall system health score (0-100).

    Args:
        health_metrics: ScheduleHealthMetrics object

    Returns:
        Health score from 0 (critical) to 100 (excellent)
    """
    try:
        if health_metrics.total_schedules == 0:
            return 100.0  # No schedules means no problems

        score = 100.0

        # Deduct points for failed schedules
        if health_metrics.total_schedules > 0:
            failed_ratio = health_metrics.failed_schedules / health_metrics.total_schedules
            score -= failed_ratio * 40

        # Deduct points for low success rate
        if health_metrics.average_success_rate < 90:
            score -= (90 - health_metrics.average_success_rate) * 0.5

        # Deduct points for high failure rate in last 24h
        if health_metrics.total_executions_24h > 0:
            failure_rate_24h = health_metrics.failed_executions_24h / health_metrics.total_executions_24h
            if failure_rate_24h > 0.1:  # More than 10% failures
                score -= (failure_rate_24h - 0.1) * 30

        return max(score, 0.0)

    except Exception:
        return 50.0  # Default neutral score if calculation fails


def _calculate_user_metrics(website_statuses) -> Dict[str, Any]:
    """Calculate user-specific metrics from website statuses."""
    if not website_statuses:
        return {
            "total_websites": 0,
            "active_websites": 0,
            "failed_websites": 0,
            "average_success_rate": 0.0,
            "total_pages_crawled": 0
        }

    total_websites = len(website_statuses)
    active_websites = sum(1 for s in website_statuses if s.schedule_status == ScheduleStatus.ACTIVE)
    failed_websites = sum(1 for s in website_statuses if s.schedule_status == ScheduleStatus.FAILED)

    total_success_rate = sum(s.success_rate for s in website_statuses)
    average_success_rate = total_success_rate / total_websites if total_websites > 0 else 0.0

    total_pages_crawled = sum(s.pages_crawled for s in website_statuses)

    return {
        "total_websites": total_websites,
        "active_websites": active_websites,
        "failed_websites": failed_websites,
        "average_success_rate": average_success_rate,
        "total_pages_crawled": total_pages_crawled
    }


def _get_status_indicator(status_info) -> Dict[str, str]:
    """Get status indicator information for UI display."""
    indicators = {
        ScheduleStatus.ACTIVE: {"color": "green", "icon": "check-circle", "label": "Active"},
        ScheduleStatus.FAILED: {"color": "red", "icon": "x-circle", "label": "Failed"},
        ScheduleStatus.PAUSED: {"color": "yellow", "icon": "pause-circle", "label": "Paused"},
        ScheduleStatus.IDLE: {"color": "gray", "icon": "minus-circle", "label": "Idle"},
        ScheduleStatus.PENDING: {"color": "blue", "icon": "clock", "label": "Pending"}
    }

    base_indicator = indicators.get(status_info.schedule_status, indicators[ScheduleStatus.IDLE])

    # Modify indicator based on additional context
    if status_info.consecutive_failures >= 3:
        base_indicator = {"color": "red", "icon": "alert-triangle", "label": "Critical"}
    elif status_info.success_rate < 50 and status_info.success_rate > 0:
        base_indicator = {"color": "orange", "icon": "alert-circle", "label": "Warning"}

    return base_indicator