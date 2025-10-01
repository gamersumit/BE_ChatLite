"""
Progress Tracking API Endpoints

This module provides REST API endpoints for accessing progress tracking data,
job status monitoring, and real-time progress updates.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.progress_tracking_service import (
    progress_service,
    JobProgressTracker,
    ProgressPhase,
    JobStatus,
    AlertLevel
)
from app.services.real_time_progress import (
    realtime_service,
    NotificationPreferences,
    DeliveryMethod
)

router = APIRouter(prefix="/api/v1/progress", tags=["Progress Tracking"])


# Pydantic models for API
class ProgressMetricsResponse(BaseModel):
    total_urls: int
    urls_discovered: int
    urls_queued: int
    urls_processing: int
    urls_completed: int
    urls_failed: int
    urls_skipped: int
    pages_crawled: int
    pages_processed: int
    content_chunks: int
    entities_extracted: int
    bytes_downloaded: int
    bytes_processed: int
    workers_active: int
    workers_idle: int
    workers_failed: int
    avg_page_time: float
    avg_processing_time: float
    current_rate_pages_per_min: float
    completion_percentage: float
    success_rate: float


class PhaseProgressResponse(BaseModel):
    phase: str
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    progress_percentage: float
    items_total: int
    items_completed: int
    current_item: Optional[str]
    error_count: int
    warnings: List[str]
    duration_seconds: Optional[float]
    estimated_remaining_seconds: Optional[float]


class JobProgressResponse(BaseModel):
    job_id: str
    snapshot_time: datetime
    status: str
    current_phase: str
    metrics: ProgressMetricsResponse
    phases: Dict[str, PhaseProgressResponse]
    error_messages: List[str]
    warnings: List[str]
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    total_duration_seconds: Optional[float]


class AlertResponse(BaseModel):
    alert_id: str
    job_id: str
    level: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    acknowledged: bool


class ProgressEventResponse(BaseModel):
    event_id: str
    job_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]


class SystemStatsResponse(BaseModel):
    jobs: Dict[str, int]
    alerts: Dict[str, int]
    processing: Dict[str, float]
    events: Dict[str, int]


class NotificationPreferencesRequest(BaseModel):
    job_completed: List[str] = Field(default=["websocket"])
    job_failed: List[str] = Field(default=["websocket", "email"])
    progress_milestones: List[str] = Field(default=["websocket"])
    alerts: List[str] = Field(default=["websocket"])
    system_status: List[str] = Field(default=["websocket"])


# Helper functions
def _convert_snapshot_to_response(snapshot) -> JobProgressResponse:
    """Convert progress snapshot to API response model."""
    metrics = ProgressMetricsResponse(
        total_urls=snapshot.metrics.total_urls,
        urls_discovered=snapshot.metrics.urls_discovered,
        urls_queued=snapshot.metrics.urls_queued,
        urls_processing=snapshot.metrics.urls_processing,
        urls_completed=snapshot.metrics.urls_completed,
        urls_failed=snapshot.metrics.urls_failed,
        urls_skipped=snapshot.metrics.urls_skipped,
        pages_crawled=snapshot.metrics.pages_crawled,
        pages_processed=snapshot.metrics.pages_processed,
        content_chunks=snapshot.metrics.content_chunks,
        entities_extracted=snapshot.metrics.entities_extracted,
        bytes_downloaded=snapshot.metrics.bytes_downloaded,
        bytes_processed=snapshot.metrics.bytes_processed,
        workers_active=snapshot.metrics.workers_active,
        workers_idle=snapshot.metrics.workers_idle,
        workers_failed=snapshot.metrics.workers_failed,
        avg_page_time=snapshot.metrics.avg_page_time,
        avg_processing_time=snapshot.metrics.avg_processing_time,
        current_rate_pages_per_min=snapshot.metrics.current_rate_pages_per_min,
        completion_percentage=snapshot.metrics.get_completion_percentage(),
        success_rate=snapshot.metrics.get_success_rate()
    )

    phases = {}
    for phase_enum, phase_progress in snapshot.phases.items():
        phases[phase_enum.value] = PhaseProgressResponse(
            phase=phase_progress.phase.value,
            status=phase_progress.status.value,
            start_time=phase_progress.start_time,
            end_time=phase_progress.end_time,
            progress_percentage=phase_progress.progress_percentage,
            items_total=phase_progress.items_total,
            items_completed=phase_progress.items_completed,
            current_item=phase_progress.current_item,
            error_count=phase_progress.error_count,
            warnings=phase_progress.warnings,
            duration_seconds=phase_progress.get_duration().total_seconds() if phase_progress.get_duration() else None,
            estimated_remaining_seconds=phase_progress.get_estimated_remaining().total_seconds() if phase_progress.get_estimated_remaining() else None
        )

    # Calculate total duration
    total_duration = None
    if snapshot.started_at:
        end_time = snapshot.estimated_completion or datetime.now(timezone.utc)
        total_duration = (end_time - snapshot.started_at).total_seconds()

    return JobProgressResponse(
        job_id=snapshot.job_id,
        snapshot_time=snapshot.snapshot_time,
        status=snapshot.status.value,
        current_phase=snapshot.current_phase.value,
        metrics=metrics,
        phases=phases,
        error_messages=snapshot.error_messages,
        warnings=snapshot.warnings,
        started_at=snapshot.started_at,
        estimated_completion=snapshot.estimated_completion,
        total_duration_seconds=total_duration
    )


# API Endpoints
@router.get("/jobs", response_model=List[JobProgressResponse])
async def get_all_jobs(
    active_only: bool = Query(False, description="Return only active jobs"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs to return")
):
    """Get progress information for all jobs."""
    try:
        if active_only:
            snapshots = await progress_service.get_active_jobs()
        else:
            snapshots = await progress_service.get_all_jobs_status()

        # Apply limit
        snapshots = snapshots[:limit]

        return [_convert_snapshot_to_response(snapshot) for snapshot in snapshots]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job progress: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobProgressResponse)
async def get_job_progress(
    job_id: str = Path(..., description="Job ID to get progress for")
):
    """Get progress information for a specific job."""
    try:
        snapshot = await progress_service.get_job_status(job_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return _convert_snapshot_to_response(snapshot)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job progress: {str(e)}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str = Path(..., description="Job ID to cancel")
):
    """Cancel a running job."""
    try:
        success = await progress_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


@router.get("/events", response_model=List[ProgressEventResponse])
async def get_progress_events(
    job_id: Optional[str] = Query(None, description="Filter events by job ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return")
):
    """Get recent progress events."""
    try:
        events = await progress_service.get_recent_events(job_id=job_id, limit=limit)

        return [
            ProgressEventResponse(
                event_id=event.event_id,
                job_id=event.job_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                data=event.data
            )
            for event in events
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching events: {str(e)}")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    job_id: Optional[str] = Query(None, description="Filter alerts by job ID"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return")
):
    """Get alerts."""
    try:
        alerts = await progress_service.get_alerts(job_id=job_id, acknowledged=acknowledged)

        # Apply limit
        alerts = alerts[:limit]

        return [
            AlertResponse(
                alert_id=alert.alert_id,
                job_id=alert.job_id,
                level=alert.level.value,
                message=alert.message,
                timestamp=alert.timestamp,
                details=alert.details,
                acknowledged=alert.acknowledged
            )
            for alert in alerts
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID to acknowledge")
):
    """Acknowledge an alert."""
    try:
        success = await progress_service.acknowledge_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return {"message": f"Alert {alert_id} acknowledged"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error acknowledging alert: {str(e)}")


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_statistics():
    """Get overall system statistics."""
    try:
        stats = await progress_service.get_system_statistics()
        return SystemStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system statistics: {str(e)}")


# Real-time endpoints
@router.websocket("/ws/{user_id}")
async def websocket_progress_updates(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    try:
        connection = await realtime_service.register_websocket(user_id, websocket)

        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_json()
                await _handle_websocket_message(user_id, connection, message)

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
    finally:
        await realtime_service.unregister_websocket(user_id, connection)


async def _handle_websocket_message(user_id: str, connection, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")

    if message_type == "subscribe_job":
        job_id = message.get("job_id")
        if job_id:
            realtime_service.subscribe_to_job(user_id, job_id)
            await connection.websocket.send_json({
                "type": "subscription_confirmed",
                "job_id": job_id
            })

    elif message_type == "unsubscribe_job":
        job_id = message.get("job_id")
        if job_id:
            realtime_service.unsubscribe_from_job(user_id, job_id)
            await connection.websocket.send_json({
                "type": "unsubscription_confirmed",
                "job_id": job_id
            })

    elif message_type == "get_job_status":
        job_id = message.get("job_id")
        if job_id:
            snapshot = await progress_service.get_job_status(job_id)
            if snapshot:
                await connection.websocket.send_json({
                    "type": "job_status",
                    "job_id": job_id,
                    "data": snapshot.to_dict()
                })
            else:
                await connection.websocket.send_json({
                    "type": "error",
                    "message": f"Job {job_id} not found"
                })

    elif message_type == "ping":
        await connection.websocket.send_json({
            "type": "pong",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@router.post("/notifications/preferences/{user_id}")
async def set_notification_preferences(
    user_id: str = Path(..., description="User ID"),
    preferences: NotificationPreferencesRequest = ...
):
    """Set notification preferences for a user."""
    try:
        # Convert string delivery methods to enums
        delivery_methods = {
            "job_completed": {DeliveryMethod(method) for method in preferences.job_completed},
            "job_failed": {DeliveryMethod(method) for method in preferences.job_failed},
            "progress_milestones": {DeliveryMethod(method) for method in preferences.progress_milestones},
            "alerts": {DeliveryMethod(method) for method in preferences.alerts},
            "system_status": {DeliveryMethod(method) for method in preferences.system_status}
        }

        notification_prefs = NotificationPreferences(
            user_id=user_id,
            **delivery_methods
        )

        realtime_service.set_notification_preferences(user_id, notification_prefs)

        return {"message": f"Notification preferences updated for user {user_id}"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid delivery method: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting preferences: {str(e)}")


@router.get("/connections/stats")
async def get_connection_stats():
    """Get real-time connection statistics."""
    try:
        stats = await realtime_service.get_connection_stats()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching connection stats: {str(e)}")


# Server-Sent Events endpoint for progress updates
@router.get("/stream/{job_id}")
async def stream_job_progress(
    job_id: str = Path(..., description="Job ID to stream progress for")
):
    """Stream job progress updates via Server-Sent Events."""

    async def generate_progress_stream():
        """Generate progress updates as Server-Sent Events."""
        yield f"data: {json.dumps({'type': 'connection_established', 'job_id': job_id})}\n\n"

        # Track the job and send initial status
        tracker = progress_service.get_job_tracker(job_id)
        if tracker:
            snapshot = tracker.get_snapshot()
            yield f"data: {json.dumps({'type': 'progress_update', 'data': snapshot.to_dict()})}\n\n"

        # Stream updates (this would be enhanced with actual event streaming)
        last_update = datetime.now(timezone.utc)
        while True:
            try:
                await asyncio.sleep(2)  # Update every 2 seconds

                # Get current status
                current_snapshot = await progress_service.get_job_status(job_id)
                if current_snapshot:
                    # Only send if there's been an update
                    if current_snapshot.snapshot_time > last_update:
                        yield f"data: {json.dumps({'type': 'progress_update', 'data': current_snapshot.to_dict()})}\n\n"
                        last_update = current_snapshot.snapshot_time

                    # Stop streaming if job is complete
                    if current_snapshot.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        yield f"data: {json.dumps({'type': 'stream_ended', 'reason': 'job_completed'})}\n\n"
                        break
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                    break

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )