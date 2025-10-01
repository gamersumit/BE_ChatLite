"""
Progress Tracking and Status Monitoring Service

This module provides comprehensive progress tracking and status monitoring for
crawling operations, worker processes, and background jobs with real-time updates,
persistence, and notification capabilities.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status enumeration."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ProgressPhase(Enum):
    """Different phases of job execution."""
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    CRAWLING = "crawling"
    PROCESSING = "processing"
    STORAGE = "storage"
    FINALIZATION = "finalization"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ProgressMetrics:
    """Metrics for job progress tracking."""
    total_urls: int = 0
    urls_discovered: int = 0
    urls_queued: int = 0
    urls_processing: int = 0
    urls_completed: int = 0
    urls_failed: int = 0
    urls_skipped: int = 0

    pages_crawled: int = 0
    pages_processed: int = 0
    content_chunks: int = 0
    entities_extracted: int = 0

    bytes_downloaded: int = 0
    bytes_processed: int = 0

    workers_active: int = 0
    workers_idle: int = 0
    workers_failed: int = 0

    avg_page_time: float = 0.0
    avg_processing_time: float = 0.0
    current_rate_pages_per_min: float = 0.0

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_urls == 0:
            return 0.0
        return (self.urls_completed / self.total_urls) * 100.0

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total_attempted = self.urls_completed + self.urls_failed
        if total_attempted == 0:
            return 0.0
        return (self.urls_completed / total_attempted) * 100.0


@dataclass
class PhaseProgress:
    """Progress tracking for individual phases."""
    phase: ProgressPhase
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    items_total: int = 0
    items_completed: int = 0
    current_item: Optional[str] = None
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def get_duration(self) -> Optional[timedelta]:
        """Get phase duration."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now(timezone.utc)
        return end - self.start_time

    def get_estimated_remaining(self) -> Optional[timedelta]:
        """Estimate remaining time for phase."""
        if self.progress_percentage <= 0 or not self.start_time:
            return None

        elapsed = self.get_duration()
        if not elapsed:
            return None

        total_estimated = elapsed.total_seconds() / (self.progress_percentage / 100.0)
        remaining_seconds = total_estimated - elapsed.total_seconds()

        return timedelta(seconds=max(0, remaining_seconds))


@dataclass
class JobProgressSnapshot:
    """Complete snapshot of job progress at a point in time."""
    job_id: str
    snapshot_time: datetime
    status: JobStatus
    current_phase: ProgressPhase
    metrics: ProgressMetrics
    phases: Dict[ProgressPhase, PhaseProgress]
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "snapshot_time": self.snapshot_time.isoformat(),
            "status": self.status.value,
            "current_phase": self.current_phase.value,
            "metrics": asdict(self.metrics),
            "phases": {
                phase.value: {
                    "phase": phase_progress.phase.value,
                    "status": phase_progress.status.value,
                    "start_time": phase_progress.start_time.isoformat() if phase_progress.start_time else None,
                    "end_time": phase_progress.end_time.isoformat() if phase_progress.end_time else None,
                    "progress_percentage": phase_progress.progress_percentage,
                    "items_total": phase_progress.items_total,
                    "items_completed": phase_progress.items_completed,
                    "current_item": phase_progress.current_item,
                    "error_count": phase_progress.error_count,
                    "warnings": phase_progress.warnings
                }
                for phase, phase_progress in self.phases.items()
            },
            "error_messages": self.error_messages,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None
        }


@dataclass
class ProgressAlert:
    """Alert for progress monitoring."""
    alert_id: str
    job_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "job_id": self.job_id,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "acknowledged": self.acknowledged
        }


class ProgressEventType(Enum):
    """Types of progress events."""
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PROGRESS_UPDATE = "progress_update"
    WORKER_STATUS_CHANGE = "worker_status_change"
    ERROR_OCCURRED = "error_occurred"
    MILESTONE_REACHED = "milestone_reached"


@dataclass
class ProgressEvent:
    """Progress tracking event."""
    event_id: str
    job_id: str
    event_type: ProgressEventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


class JobProgressTracker:
    """Progress tracker for individual jobs."""

    def __init__(self, job_id: str, total_urls: int = 0):
        self.job_id = job_id
        self.metrics = ProgressMetrics(total_urls=total_urls)
        self.phases: Dict[ProgressPhase, PhaseProgress] = {}
        self.current_phase = ProgressPhase.INITIALIZATION
        self.status = JobStatus.PENDING

        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        self.error_messages: List[str] = []
        self.warnings: List[str] = []

        self.progress_history: deque = deque(maxlen=1000)
        self.event_callbacks: List[Callable] = []

        self._lock = threading.Lock()

        # Initialize all phases
        for phase in ProgressPhase:
            self.phases[phase] = PhaseProgress(phase=phase)

    def add_event_callback(self, callback: Callable[[ProgressEvent], None]):
        """Add callback for progress events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: ProgressEventType, data: Dict[str, Any] = None):
        """Emit progress event."""
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            job_id=self.job_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data or {}
        )

        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in progress event callback: {e}")

    def start_job(self):
        """Mark job as started."""
        with self._lock:
            self.status = JobStatus.INITIALIZING
            self.started_at = datetime.now(timezone.utc)
            self._emit_event(ProgressEventType.JOB_STARTED, {"started_at": self.started_at.isoformat()})

    def start_phase(self, phase: ProgressPhase, total_items: int = 0):
        """Start a new phase."""
        with self._lock:
            self.current_phase = phase
            phase_progress = self.phases[phase]
            phase_progress.status = JobStatus.RUNNING
            phase_progress.start_time = datetime.now(timezone.utc)
            phase_progress.items_total = total_items
            phase_progress.items_completed = 0
            phase_progress.progress_percentage = 0.0

            if self.status == JobStatus.INITIALIZING:
                self.status = JobStatus.RUNNING

            self._emit_event(ProgressEventType.PHASE_STARTED, {
                "phase": phase.value,
                "total_items": total_items
            })

    def update_phase_progress(self, phase: ProgressPhase, completed_items: int, current_item: str = None):
        """Update progress for a specific phase."""
        with self._lock:
            phase_progress = self.phases[phase]
            phase_progress.items_completed = completed_items
            phase_progress.current_item = current_item

            if phase_progress.items_total > 0:
                phase_progress.progress_percentage = (completed_items / phase_progress.items_total) * 100.0

            self._emit_event(ProgressEventType.PROGRESS_UPDATE, {
                "phase": phase.value,
                "completed_items": completed_items,
                "total_items": phase_progress.items_total,
                "progress_percentage": phase_progress.progress_percentage
            })

    def complete_phase(self, phase: ProgressPhase):
        """Mark phase as completed."""
        with self._lock:
            phase_progress = self.phases[phase]
            phase_progress.status = JobStatus.COMPLETED
            phase_progress.end_time = datetime.now(timezone.utc)
            phase_progress.progress_percentage = 100.0

            self._emit_event(ProgressEventType.PHASE_COMPLETED, {
                "phase": phase.value,
                "duration": phase_progress.get_duration().total_seconds() if phase_progress.get_duration() else 0
            })

    def update_metrics(self, **kwargs):
        """Update progress metrics."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)

            # Calculate derived metrics
            if self.metrics.urls_completed > 0:
                self.metrics.avg_page_time = self.get_total_duration().total_seconds() / self.metrics.urls_completed if self.get_total_duration() else 0.0

    def add_error(self, error_message: str, phase: ProgressPhase = None):
        """Add error message."""
        with self._lock:
            self.error_messages.append(f"{datetime.now(timezone.utc).isoformat()}: {error_message}")

            if phase:
                self.phases[phase].error_count += 1

            self._emit_event(ProgressEventType.ERROR_OCCURRED, {
                "error_message": error_message,
                "phase": phase.value if phase else None
            })

    def add_warning(self, warning_message: str, phase: ProgressPhase = None):
        """Add warning message."""
        with self._lock:
            self.warnings.append(f"{datetime.now(timezone.utc).isoformat()}: {warning_message}")

            if phase:
                self.phases[phase].warnings.append(warning_message)

    def complete_job(self, success: bool = True):
        """Mark job as completed."""
        with self._lock:
            self.status = JobStatus.COMPLETED if success else JobStatus.FAILED
            self.completed_at = datetime.now(timezone.utc)

            # Complete current phase if not already completed
            if self.current_phase and self.phases[self.current_phase].status == JobStatus.RUNNING:
                self.complete_phase(self.current_phase)

            event_type = ProgressEventType.JOB_COMPLETED if success else ProgressEventType.JOB_FAILED
            self._emit_event(event_type, {
                "completed_at": self.completed_at.isoformat(),
                "success": success,
                "total_duration": self.get_total_duration().total_seconds() if self.get_total_duration() else 0
            })

    def get_total_duration(self) -> Optional[timedelta]:
        """Get total job duration."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now(timezone.utc)
        return end_time - self.started_at

    def get_estimated_completion(self) -> Optional[datetime]:
        """Estimate job completion time."""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return self.completed_at

        if not self.started_at or self.metrics.total_urls == 0:
            return None

        completion_percentage = self.metrics.get_completion_percentage()
        if completion_percentage <= 0:
            return None

        elapsed = self.get_total_duration().total_seconds()
        total_estimated = elapsed / (completion_percentage / 100.0)
        remaining_seconds = total_estimated - elapsed

        return datetime.now(timezone.utc) + timedelta(seconds=max(0, remaining_seconds))

    def get_snapshot(self) -> JobProgressSnapshot:
        """Get current progress snapshot."""
        with self._lock:
            return JobProgressSnapshot(
                job_id=self.job_id,
                snapshot_time=datetime.now(timezone.utc),
                status=self.status,
                current_phase=self.current_phase,
                metrics=self.metrics,
                phases=self.phases.copy(),
                error_messages=self.error_messages.copy(),
                warnings=self.warnings.copy(),
                started_at=self.started_at,
                estimated_completion=self.get_estimated_completion()
            )


class ProgressTrackingService:
    """Main progress tracking and monitoring service."""

    def __init__(self):
        self.job_trackers: Dict[str, JobProgressTracker] = {}
        self.progress_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        self.alerts: Dict[str, ProgressAlert] = {}
        self.event_history: deque = deque(maxlen=10000)

        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Configuration
        self.alert_thresholds = {
            "error_rate_threshold": 0.1,  # 10% error rate
            "slow_progress_threshold": 300,  # 5 minutes without progress
            "worker_failure_threshold": 0.3,  # 30% worker failure rate
            "memory_usage_threshold": 0.9  # 90% memory usage
        }

    async def start_monitoring(self):
        """Start background monitoring."""
        if not self._monitoring_task or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Progress tracking monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._shutdown_event.set()
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Progress tracking monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_job_health()
                await self._check_progress_stalls()
                await self._cleanup_old_data()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in progress monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _check_job_health(self):
        """Check health of all active jobs."""
        current_time = datetime.now(timezone.utc)

        for job_id, tracker in self.job_trackers.items():
            if tracker.status not in [JobStatus.RUNNING, JobStatus.PAUSED]:
                continue

            # Check for high error rates
            if tracker.metrics.urls_failed > 0:
                error_rate = tracker.metrics.urls_failed / (tracker.metrics.urls_completed + tracker.metrics.urls_failed)
                if error_rate > self.alert_thresholds["error_rate_threshold"]:
                    await self._create_alert(
                        job_id,
                        AlertLevel.WARNING,
                        f"High error rate detected: {error_rate:.1%}",
                        {"error_rate": error_rate, "failed_urls": tracker.metrics.urls_failed}
                    )

            # Check for stalled progress
            if tracker.started_at:
                elapsed = (current_time - tracker.started_at).total_seconds()
                if elapsed > self.alert_thresholds["slow_progress_threshold"] and tracker.metrics.urls_completed == 0:
                    await self._create_alert(
                        job_id,
                        AlertLevel.WARNING,
                        f"Job appears stalled: no progress in {elapsed/60:.1f} minutes",
                        {"elapsed_minutes": elapsed/60}
                    )

    async def _check_progress_stalls(self):
        """Check for stalled progress in individual phases."""
        current_time = datetime.now(timezone.utc)

        for job_id, tracker in self.job_trackers.items():
            if tracker.status != JobStatus.RUNNING:
                continue

            current_phase_progress = tracker.phases.get(tracker.current_phase)
            if not current_phase_progress or not current_phase_progress.start_time:
                continue

            # Check if phase has been running too long without progress
            phase_duration = (current_time - current_phase_progress.start_time).total_seconds()
            if (phase_duration > 600 and  # 10 minutes
                current_phase_progress.progress_percentage < 10):  # Less than 10% progress

                await self._create_alert(
                    job_id,
                    AlertLevel.WARNING,
                    f"Phase {tracker.current_phase.value} appears stalled",
                    {
                        "phase": tracker.current_phase.value,
                        "duration_minutes": phase_duration / 60,
                        "progress_percentage": current_phase_progress.progress_percentage
                    }
                )

    async def _cleanup_old_data(self):
        """Cleanup old tracking data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        # Remove completed jobs older than 24 hours
        jobs_to_remove = []
        for job_id, tracker in self.job_trackers.items():
            if (tracker.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                tracker.completed_at and tracker.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.job_trackers[job_id]
            logger.info(f"Cleaned up old job tracking data: {job_id}")

        # Remove old alerts
        old_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.timestamp < cutoff_time
        ]
        for alert_id in old_alerts:
            del self.alerts[alert_id]

    def create_job_tracker(self, job_id: str, total_urls: int = 0) -> JobProgressTracker:
        """Create a new job progress tracker."""
        tracker = JobProgressTracker(job_id, total_urls)
        tracker.add_event_callback(self._handle_progress_event)
        self.job_trackers[job_id] = tracker

        logger.info(f"Created progress tracker for job: {job_id}")
        return tracker

    def get_job_tracker(self, job_id: str) -> Optional[JobProgressTracker]:
        """Get job progress tracker."""
        return self.job_trackers.get(job_id)

    def _handle_progress_event(self, event: ProgressEvent):
        """Handle progress events from job trackers."""
        self.event_history.append(event)

        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    async def _create_alert(self, job_id: str, level: AlertLevel, message: str, details: Dict[str, Any] = None):
        """Create and emit an alert."""
        alert = ProgressAlert(
            alert_id=str(uuid.uuid4()),
            job_id=job_id,
            level=level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            details=details or {}
        )

        self.alerts[alert.alert_id] = alert

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.info(f"Created alert [{level.value}] for job {job_id}: {message}")

    def add_progress_callback(self, callback: Callable[[ProgressEvent], None]):
        """Add callback for progress events."""
        self.progress_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[ProgressAlert], Any]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)

    async def get_job_status(self, job_id: str) -> Optional[JobProgressSnapshot]:
        """Get current job status."""
        tracker = self.job_trackers.get(job_id)
        return tracker.get_snapshot() if tracker else None

    async def get_all_jobs_status(self) -> List[JobProgressSnapshot]:
        """Get status of all tracked jobs."""
        return [tracker.get_snapshot() for tracker in self.job_trackers.values()]

    async def get_active_jobs(self) -> List[JobProgressSnapshot]:
        """Get status of active jobs only."""
        return [
            tracker.get_snapshot()
            for tracker in self.job_trackers.values()
            if tracker.status in [JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.INITIALIZING]
        ]

    async def get_recent_events(self, job_id: str = None, limit: int = 100) -> List[ProgressEvent]:
        """Get recent progress events."""
        events = list(self.event_history)

        if job_id:
            events = [e for e in events if e.job_id == job_id]

        return events[-limit:] if limit else events

    async def get_alerts(self, job_id: str = None, acknowledged: bool = None) -> List[ProgressAlert]:
        """Get alerts."""
        alerts = list(self.alerts.values())

        if job_id:
            alerts = [a for a in alerts if a.job_id == job_id]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            return True
        return False

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        tracker = self.job_trackers.get(job_id)
        if tracker and tracker.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
            tracker.status = JobStatus.CANCELLED
            tracker.completed_at = datetime.now(timezone.utc)
            tracker._emit_event(ProgressEventType.JOB_CANCELLED, {
                "cancelled_at": tracker.completed_at.isoformat()
            })
            return True
        return False

    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        total_jobs = len(self.job_trackers)
        active_jobs = len([t for t in self.job_trackers.values() if t.status == JobStatus.RUNNING])
        completed_jobs = len([t for t in self.job_trackers.values() if t.status == JobStatus.COMPLETED])
        failed_jobs = len([t for t in self.job_trackers.values() if t.status == JobStatus.FAILED])

        total_alerts = len(self.alerts)
        unacknowledged_alerts = len([a for a in self.alerts.values() if not a.acknowledged])

        # Calculate aggregate metrics
        total_urls_processed = sum(t.metrics.urls_completed for t in self.job_trackers.values())
        total_errors = sum(t.metrics.urls_failed for t in self.job_trackers.values())

        return {
            "jobs": {
                "total": total_jobs,
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs
            },
            "alerts": {
                "total": total_alerts,
                "unacknowledged": unacknowledged_alerts
            },
            "processing": {
                "total_urls_processed": total_urls_processed,
                "total_errors": total_errors,
                "overall_success_rate": (total_urls_processed / (total_urls_processed + total_errors)) if (total_urls_processed + total_errors) > 0 else 0.0
            },
            "events": {
                "total_events": len(self.event_history)
            }
        }


# Global progress tracking service instance
progress_service = ProgressTrackingService()