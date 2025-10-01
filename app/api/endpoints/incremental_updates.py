"""
Incremental Update Detection API Endpoints

This module provides REST API endpoints for managing incremental update detection,
content versioning, smart re-crawling, and update scheduling.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Set, Any
import asyncio

from fastapi import APIRouter, HTTPException, Query, Path, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.incremental_update_service import (
    incremental_service,
    ChangeType,
    UpdatePriority,
    ContentChange,
    ContentFingerprint
)
from app.services.content_versioning import (
    versioning_service,
    ContentVersion,
    ContentDiff,
    DiffType,
    VersionHistory
)
from app.services.smart_recrawl_service import (
    smart_recrawl_service,
    initialize_smart_recrawl_service,
    CrawlStrategy,
    CrawlReason,
    CrawlRequest,
    UrlChangePattern
)
from app.services.update_scheduler import (
    update_scheduler,
    TriggerType,
    TriggerStatus,
    UpdateTrigger,
    ScheduledTask
)

router = APIRouter(prefix="/api/v1/incremental", tags=["Incremental Updates"])


# Pydantic models for API requests/responses
class FingerprintRequest(BaseModel):
    url: str
    content: str
    html: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None


class FingerprintResponse(BaseModel):
    url: str
    content_hash: str
    structure_hash: str
    metadata_hash: str
    links_hash: str
    title: Optional[str]
    content_length: int
    last_modified: Optional[datetime]
    etag: Optional[str]
    timestamp: datetime


class ChangeDetectionRequest(BaseModel):
    url: str
    fingerprint: FingerprintRequest


class ChangeResponse(BaseModel):
    url: str
    change_type: str
    confidence_score: float
    priority: str
    change_details: Dict[str, Any]
    detected_at: datetime
    old_fingerprint: Optional[FingerprintResponse]
    new_fingerprint: FingerprintResponse


class VersionResponse(BaseModel):
    url: str
    version_id: str
    created_at: datetime
    size_bytes: int
    compression_ratio: float
    fingerprint: FingerprintResponse


class DiffRequest(BaseModel):
    url: str
    old_version_id: str
    new_version_id: str
    diff_type: str = "text_diff"


class DiffResponse(BaseModel):
    url: str
    old_version_id: str
    new_version_id: str
    diff_type: str
    similarity_score: float
    change_magnitude: int
    changes_summary: Dict[str, Any]
    detailed_diff: str
    created_at: datetime


class ScheduleUpdateRequest(BaseModel):
    url: str
    frequency_hours: int = Field(ge=1, le=168)  # 1 hour to 1 week
    priority: str = "medium"


class CrawlStrategyRequest(BaseModel):
    url: str
    strategy: str


class TriggerCreateRequest(BaseModel):
    name: str
    trigger_type: str
    target_urls: List[str]
    schedule_config: Dict[str, Any]
    priority: str = "medium"
    conditions: Dict[str, Any] = Field(default_factory=dict)


class TriggerResponse(BaseModel):
    trigger_id: str
    name: str
    trigger_type: str
    target_urls: List[str]
    status: str
    schedule_config: Dict[str, Any]
    priority: str
    conditions: Dict[str, Any]
    created_at: datetime
    last_triggered: Optional[datetime]
    next_trigger: Optional[datetime]
    trigger_count: int
    error_count: int


class ChangePatternResponse(BaseModel):
    url: str
    change_frequency: float
    last_change_time: Optional[datetime]
    average_change_interval_hours: float
    change_predictability: float
    content_volatility: float
    crawl_success_rate: float
    last_significant_change: Optional[datetime]


# Helper functions
def _fingerprint_to_response(fp: ContentFingerprint) -> FingerprintResponse:
    """Convert ContentFingerprint to response model."""
    return FingerprintResponse(
        url=fp.url,
        content_hash=fp.content_hash,
        structure_hash=fp.structure_hash,
        metadata_hash=fp.metadata_hash,
        links_hash=fp.links_hash,
        title=fp.title,
        content_length=fp.content_length,
        last_modified=fp.last_modified,
        etag=fp.etag,
        timestamp=fp.timestamp
    )


def _change_to_response(change: ContentChange) -> ChangeResponse:
    """Convert ContentChange to response model."""
    return ChangeResponse(
        url=change.url,
        change_type=change.change_type.value,
        confidence_score=change.confidence_score,
        priority=change.priority.value,
        change_details=change.change_details,
        detected_at=change.detected_at,
        old_fingerprint=_fingerprint_to_response(change.old_fingerprint) if change.old_fingerprint else None,
        new_fingerprint=_fingerprint_to_response(change.new_fingerprint)
    )


# API Endpoints

# Content Fingerprinting
@router.post("/fingerprint", response_model=FingerprintResponse)
async def create_fingerprint(request: FingerprintRequest):
    """Create a content fingerprint for change detection."""
    try:
        fingerprint = await incremental_service.create_fingerprint(
            url=request.url,
            content=request.content,
            html=request.html,
            metadata=request.metadata,
            links=set(request.links),
            title=request.title,
            last_modified=request.last_modified,
            etag=request.etag
        )

        return _fingerprint_to_response(fingerprint)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating fingerprint: {str(e)}")


# Change Detection
@router.post("/detect-changes", response_model=ChangeResponse)
async def detect_content_changes(request: ChangeDetectionRequest):
    """Detect changes in content compared to previous version."""
    try:
        # Create fingerprint for new content
        new_fingerprint = await incremental_service.create_fingerprint(
            url=request.fingerprint.url,
            content=request.fingerprint.content,
            html=request.fingerprint.html,
            metadata=request.fingerprint.metadata,
            links=set(request.fingerprint.links),
            title=request.fingerprint.title,
            last_modified=request.fingerprint.last_modified,
            etag=request.fingerprint.etag
        )

        # Detect changes
        change = await incremental_service.detect_changes(request.url, new_fingerprint)

        return _change_to_response(change)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting changes: {str(e)}")


@router.get("/changes", response_model=List[ChangeResponse])
async def get_pending_changes(
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of changes to return")
):
    """Get pending content changes."""
    try:
        priority_enum = None
        if priority:
            try:
                priority_enum = UpdatePriority(priority.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

        changes = await incremental_service.get_pending_changes(priority=priority_enum, limit=limit)
        return [_change_to_response(change) for change in changes]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching changes: {str(e)}")


@router.post("/changes/clear")
async def clear_processed_changes(processed_urls: List[str] = Body(...)):
    """Mark changes as processed and remove from pending list."""
    try:
        await incremental_service.clear_processed_changes(processed_urls)
        return {"message": f"Cleared {len(processed_urls)} processed changes"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing changes: {str(e)}")


# Content Versioning
@router.post("/versions", response_model=VersionResponse)
async def store_content_version(request: FingerprintRequest):
    """Store a new version of content."""
    try:
        # Create fingerprint
        fingerprint = await incremental_service.create_fingerprint(
            url=request.url,
            content=request.content,
            html=request.html,
            metadata=request.metadata,
            links=set(request.links),
            title=request.title,
            last_modified=request.last_modified,
            etag=request.etag
        )

        # Store version
        version = await versioning_service.store_version(
            url=request.url,
            fingerprint=fingerprint,
            content=request.content,
            html=request.html,
            metadata=request.metadata,
            links=set(request.links)
        )

        return VersionResponse(
            url=version.url,
            version_id=version.version_id,
            created_at=version.created_at,
            size_bytes=version.size_bytes,
            compression_ratio=version.compression_ratio,
            fingerprint=_fingerprint_to_response(version.fingerprint)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing version: {str(e)}")


@router.get("/versions/{url:path}", response_model=List[VersionResponse])
async def get_content_versions(
    url: str = Path(..., description="URL to get versions for"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of versions to return")
):
    """Get version history for a URL."""
    try:
        history = await versioning_service.get_version_history(url)
        if not history:
            return []

        # Sort by creation time (newest first) and apply limit
        versions = sorted(history.versions, key=lambda v: v.created_at, reverse=True)[:limit]

        return [
            VersionResponse(
                url=v.url,
                version_id=v.version_id,
                created_at=v.created_at,
                size_bytes=v.size_bytes,
                compression_ratio=getattr(v, 'compression_ratio', 1.0),
                fingerprint=_fingerprint_to_response(v.fingerprint)
            )
            for v in versions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching versions: {str(e)}")


@router.post("/diff", response_model=DiffResponse)
async def generate_content_diff(request: DiffRequest):
    """Generate diff between two content versions."""
    try:
        diff_type = DiffType(request.diff_type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid diff type: {request.diff_type}")

    try:
        diff = await versioning_service.generate_diff(
            url=request.url,
            old_version_id=request.old_version_id,
            new_version_id=request.new_version_id,
            diff_type=diff_type
        )

        if not diff:
            raise HTTPException(status_code=404, detail="One or both versions not found")

        return DiffResponse(
            url=diff.url,
            old_version_id=diff.old_version_id,
            new_version_id=diff.new_version_id,
            diff_type=diff.diff_type.value,
            similarity_score=diff.similarity_score,
            change_magnitude=diff.change_magnitude,
            changes_summary=diff.changes_summary,
            detailed_diff=diff.detailed_diff,
            created_at=diff.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diff: {str(e)}")


# Update Scheduling
@router.post("/schedule")
async def schedule_update_check(request: ScheduleUpdateRequest):
    """Schedule regular update checks for a URL."""
    try:
        priority = UpdatePriority(request.priority.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")

    try:
        frequency = timedelta(hours=request.frequency_hours)
        await incremental_service.schedule_update_check(
            url=request.url,
            frequency=frequency,
            priority=priority
        )

        return {"message": f"Scheduled update checks for {request.url} every {request.frequency_hours} hours"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling updates: {str(e)}")


@router.get("/scheduled")
async def get_scheduled_updates():
    """Get URLs scheduled for update checks."""
    try:
        urls = await incremental_service.get_urls_for_update_check()
        return {"scheduled_urls": urls, "count": len(urls)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching scheduled updates: {str(e)}")


# Smart Re-crawling
@router.post("/crawl-strategy")
async def set_crawl_strategy(request: CrawlStrategyRequest):
    """Set crawling strategy for a URL."""
    if not smart_recrawl_service:
        # Initialize if not already done
        initialize_smart_recrawl_service(incremental_service, versioning_service)

    try:
        strategy = CrawlStrategy(request.strategy.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")

    try:
        smart_recrawl_service.set_crawl_strategy(request.url, strategy)
        return {"message": f"Set crawl strategy for {request.url}: {strategy.value}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting crawl strategy: {str(e)}")


@router.get("/change-patterns/{url:path}", response_model=ChangePatternResponse)
async def get_change_pattern(url: str = Path(..., description="URL to analyze")):
    """Get change patterns for a URL."""
    if not smart_recrawl_service:
        initialize_smart_recrawl_service(incremental_service, versioning_service)

    try:
        pattern = await smart_recrawl_service.analyze_change_patterns(url)

        return ChangePatternResponse(
            url=pattern.url,
            change_frequency=pattern.change_frequency,
            last_change_time=pattern.last_change_time,
            average_change_interval_hours=pattern.average_change_interval.total_seconds() / 3600,
            change_predictability=pattern.change_predictability,
            content_volatility=pattern.content_volatility,
            crawl_success_rate=pattern.crawl_success_rate,
            last_significant_change=pattern.last_significant_change
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing change patterns: {str(e)}")


@router.get("/crawl-queue")
async def get_crawl_queue_stats():
    """Get crawl queue statistics."""
    if not smart_recrawl_service:
        initialize_smart_recrawl_service(incremental_service, versioning_service)

    try:
        stats = await smart_recrawl_service.get_queue_statistics()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching queue stats: {str(e)}")


# Trigger Management
@router.post("/triggers", response_model=TriggerResponse)
async def create_update_trigger(request: TriggerCreateRequest):
    """Create a new update trigger."""
    try:
        trigger_type = TriggerType(request.trigger_type.lower())
        priority = UpdatePriority(request.priority.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")

    try:
        trigger = await update_scheduler.create_trigger(
            name=request.name,
            trigger_type=trigger_type,
            target_urls=set(request.target_urls),
            schedule_config=request.schedule_config,
            priority=priority,
            conditions=request.conditions
        )

        return TriggerResponse(
            trigger_id=trigger.trigger_id,
            name=trigger.name,
            trigger_type=trigger.trigger_type.value,
            target_urls=list(trigger.target_urls),
            status=trigger.status.value,
            schedule_config=trigger.schedule_config,
            priority=trigger.priority.value,
            conditions=trigger.conditions,
            created_at=trigger.created_at,
            last_triggered=trigger.last_triggered,
            next_trigger=trigger.next_trigger,
            trigger_count=trigger.trigger_count,
            error_count=trigger.error_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating trigger: {str(e)}")


@router.get("/triggers", response_model=List[TriggerResponse])
async def get_update_triggers():
    """Get all update triggers."""
    try:
        triggers = []
        for trigger in update_scheduler.triggers.values():
            triggers.append(TriggerResponse(
                trigger_id=trigger.trigger_id,
                name=trigger.name,
                trigger_type=trigger.trigger_type.value,
                target_urls=list(trigger.target_urls),
                status=trigger.status.value,
                schedule_config=trigger.schedule_config,
                priority=trigger.priority.value,
                conditions=trigger.conditions,
                created_at=trigger.created_at,
                last_triggered=trigger.last_triggered,
                next_trigger=trigger.next_trigger,
                trigger_count=trigger.trigger_count,
                error_count=trigger.error_count
            ))

        return triggers

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching triggers: {str(e)}")


@router.post("/triggers/{trigger_id}/pause")
async def pause_trigger(trigger_id: str = Path(..., description="Trigger ID to pause")):
    """Pause a trigger."""
    try:
        success = await update_scheduler.pause_trigger(trigger_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

        return {"message": f"Trigger {trigger_id} paused"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pausing trigger: {str(e)}")


@router.post("/triggers/{trigger_id}/resume")
async def resume_trigger(trigger_id: str = Path(..., description="Trigger ID to resume")):
    """Resume a paused trigger."""
    try:
        success = await update_scheduler.resume_trigger(trigger_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

        return {"message": f"Trigger {trigger_id} resumed"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming trigger: {str(e)}")


@router.delete("/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str = Path(..., description="Trigger ID to delete")):
    """Delete a trigger."""
    try:
        success = await update_scheduler.delete_trigger(trigger_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

        return {"message": f"Trigger {trigger_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting trigger: {str(e)}")


@router.post("/triggers/{trigger_id}/fire")
async def manual_fire_trigger(trigger_id: str = Path(..., description="Trigger ID to fire manually")):
    """Manually fire a trigger."""
    try:
        success = await update_scheduler.manual_trigger(trigger_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

        return {"message": f"Trigger {trigger_id} fired manually"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error firing trigger: {str(e)}")


# System Statistics
@router.get("/statistics")
async def get_system_statistics():
    """Get comprehensive system statistics."""
    try:
        stats = {}

        # Incremental service stats
        stats['incremental_updates'] = await incremental_service.get_statistics()

        # Versioning service stats
        stats['content_versioning'] = await versioning_service.get_statistics()

        # Smart recrawl stats
        if smart_recrawl_service:
            stats['smart_recrawl'] = await smart_recrawl_service.get_queue_statistics()

        # Scheduler stats
        stats['scheduler'] = await update_scheduler.get_statistics()

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")


# Background task to start scheduler
@router.on_event("startup")
async def startup_event():
    """Start the update scheduler on startup."""
    if not update_scheduler.running:
        await update_scheduler.start_scheduler()

    # Initialize smart recrawl service if not already done
    if not smart_recrawl_service:
        initialize_smart_recrawl_service(incremental_service, versioning_service)


@router.on_event("shutdown")
async def shutdown_event():
    """Stop the update scheduler on shutdown."""
    await update_scheduler.shutdown()