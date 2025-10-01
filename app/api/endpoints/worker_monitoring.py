"""
API endpoints for worker monitoring and management.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...core.auth_middleware import get_current_user
from ...services.worker_monitoring import WorkerMonitoringService

logger = logging.getLogger(__name__)
router = APIRouter()


class WorkerMetricsResponse(BaseModel):
    """Response model for worker metrics."""
    worker_name: str = Field(..., description="Worker name")
    status: str = Field(..., description="Worker status")
    active_tasks: int = Field(default=0, description="Number of active tasks")
    processed_tasks: int = Field(default=0, description="Total processed tasks")
    failed_tasks: int = Field(default=0, description="Total failed tasks")
    load_average: float = Field(default=0.0, description="Load average")
    memory_usage: float = Field(default=0.0, description="Memory usage percentage")
    last_heartbeat: str = Field(..., description="Last heartbeat timestamp")
    uptime: float = Field(default=0.0, description="Uptime in seconds")
    queue_length: int = Field(default=0, description="Queue length")


class QueueMetricsResponse(BaseModel):
    """Response model for queue metrics."""
    queue_name: str = Field(..., description="Queue name")
    status: str = Field(..., description="Queue status")
    pending_tasks: int = Field(default=0, description="Number of pending tasks")
    active_tasks: int = Field(default=0, description="Number of active tasks")
    processed_tasks_1h: int = Field(default=0, description="Tasks processed in last hour")
    failed_tasks_1h: int = Field(default=0, description="Tasks failed in last hour")
    average_processing_time: float = Field(default=0.0, description="Average processing time")
    throughput_per_minute: float = Field(default=0.0, description="Throughput per minute")


class SystemOverviewResponse(BaseModel):
    """Response model for system overview."""
    total_workers: int = Field(default=0, description="Total number of workers")
    active_workers: int = Field(default=0, description="Number of active workers")
    offline_workers: int = Field(default=0, description="Number of offline workers")
    total_queues: int = Field(default=0, description="Total number of queues")
    healthy_queues: int = Field(default=0, description="Number of healthy queues")
    total_pending_tasks: int = Field(default=0, description="Total pending tasks")
    total_active_tasks: int = Field(default=0, description="Total active tasks")
    system_load: float = Field(default=0.0, description="System load average")
    redis_status: str = Field(..., description="Redis connection status")
    last_updated: str = Field(..., description="Last update timestamp")


class PerformanceAlert(BaseModel):
    """Model for performance alerts."""
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    timestamp: str = Field(..., description="Alert timestamp")


class WorkerActionRequest(BaseModel):
    """Request model for worker actions."""
    action: str = Field(..., description="Action to perform (restart, stop, scale)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


@router.get("/overview")
async def get_system_overview(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get system overview with worker and queue metrics.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        overview = monitoring_service.get_system_overview()

        return {
            "success": True,
            "data": {
                "total_workers": overview.total_workers,
                "active_workers": overview.active_workers,
                "offline_workers": overview.offline_workers,
                "total_queues": overview.total_queues,
                "healthy_queues": overview.healthy_queues,
                "total_pending_tasks": overview.total_pending_tasks,
                "total_active_tasks": overview.total_active_tasks,
                "system_load": overview.system_load,
                "redis_status": overview.redis_status,
                "last_updated": overview.last_updated
            }
        }

    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system overview")


@router.get("/workers")
async def get_worker_metrics(
    worker_name: Optional[str] = Query(None, description="Specific worker name"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed metrics for all workers or a specific worker.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        worker_metrics = monitoring_service.get_worker_metrics(worker_name)

        return {
            "success": True,
            "data": {
                "workers": [
                    {
                        "worker_name": w.worker_name,
                        "status": w.status,
                        "active_tasks": w.active_tasks,
                        "processed_tasks": w.processed_tasks,
                        "failed_tasks": w.failed_tasks,
                        "load_average": w.load_average,
                        "memory_usage": w.memory_usage,
                        "last_heartbeat": w.last_heartbeat,
                        "uptime": w.uptime,
                        "queue_length": w.queue_length
                    }
                    for w in worker_metrics
                ],
                "total_workers": len(worker_metrics)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get worker metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get worker metrics")


@router.get("/workers/{worker_name}/details")
async def get_worker_details(
    worker_name: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific worker.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        worker_details = monitoring_service.get_worker_details(worker_name)

        if 'error' in worker_details:
            raise HTTPException(status_code=404, detail=worker_details['error'])

        return {
            "success": True,
            "data": worker_details
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worker details for {worker_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get worker details")


@router.get("/queues")
async def get_queue_metrics(
    queue_name: Optional[str] = Query(None, description="Specific queue name"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed metrics for all queues or a specific queue.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        queue_metrics = monitoring_service.get_queue_metrics(queue_name)

        return {
            "success": True,
            "data": {
                "queues": [
                    {
                        "queue_name": q.queue_name,
                        "status": q.status,
                        "pending_tasks": q.pending_tasks,
                        "active_tasks": q.active_tasks,
                        "processed_tasks_1h": q.processed_tasks_1h,
                        "failed_tasks_1h": q.failed_tasks_1h,
                        "average_processing_time": q.average_processing_time,
                        "throughput_per_minute": q.throughput_per_minute
                    }
                    for q in queue_metrics
                ],
                "total_queues": len(queue_metrics)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get queue metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue metrics")


@router.get("/alerts")
async def get_performance_alerts(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current performance alerts.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        alerts = monitoring_service.get_performance_alerts()

        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get('severity') == 'critical']),
                "warning_alerts": len([a for a in alerts if a.get('severity') == 'warning'])
            }
        }

    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance alerts")


@router.post("/workers/{worker_name}/restart")
async def restart_worker(
    worker_name: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Restart a specific worker.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        result = monitoring_service.restart_worker(worker_name)

        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Restart failed'))

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart worker {worker_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart worker")


@router.post("/queues/{queue_name}/scale")
async def scale_queue_workers(
    queue_name: str,
    target_workers: int = Query(..., description="Target number of workers", ge=0, le=20),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Scale workers for a specific queue.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        result = monitoring_service.scale_workers(queue_name, target_workers)

        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Scaling failed'))

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to scale queue {queue_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to scale queue")


@router.get("/metrics/export")
async def export_metrics(
    format_type: str = Query("json", regex="^(json|prometheus)$", description="Export format"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Export comprehensive metrics in specified format.
    """
    try:
        monitoring_service = WorkerMonitoringService()
        exported_data = monitoring_service.export_metrics(format_type)

        if 'error' in exported_data:
            raise HTTPException(status_code=400, detail=exported_data['error'])

        return {
            "success": True,
            "data": exported_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


@router.get("/dashboard")
async def get_monitoring_dashboard(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive monitoring dashboard data.
    """
    try:
        monitoring_service = WorkerMonitoringService()

        # Get all monitoring data
        system_overview = monitoring_service.get_system_overview()
        worker_metrics = monitoring_service.get_worker_metrics()
        queue_metrics = monitoring_service.get_queue_metrics()
        alerts = monitoring_service.get_performance_alerts()

        return {
            "success": True,
            "data": {
                "system_overview": {
                    "total_workers": system_overview.total_workers,
                    "active_workers": system_overview.active_workers,
                    "offline_workers": system_overview.offline_workers,
                    "total_queues": system_overview.total_queues,
                    "healthy_queues": system_overview.healthy_queues,
                    "total_pending_tasks": system_overview.total_pending_tasks,
                    "total_active_tasks": system_overview.total_active_tasks,
                    "system_load": system_overview.system_load,
                    "redis_status": system_overview.redis_status,
                    "last_updated": system_overview.last_updated
                },
                "workers": [
                    {
                        "worker_name": w.worker_name,
                        "status": w.status,
                        "active_tasks": w.active_tasks,
                        "processed_tasks": w.processed_tasks,
                        "failed_tasks": w.failed_tasks,
                        "uptime": w.uptime
                    }
                    for w in worker_metrics
                ],
                "queues": [
                    {
                        "queue_name": q.queue_name,
                        "status": q.status,
                        "pending_tasks": q.pending_tasks,
                        "active_tasks": q.active_tasks
                    }
                    for q in queue_metrics
                ],
                "alerts": alerts,
                "summary": {
                    "total_workers": len(worker_metrics),
                    "total_queues": len(queue_metrics),
                    "total_alerts": len(alerts),
                    "critical_alerts": len([a for a in alerts if a.get('severity') == 'critical']),
                    "health_score": _calculate_health_score(system_overview, alerts)
                }
            }
        }

    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring dashboard")


def _calculate_health_score(system_overview, alerts) -> float:
    """
    Calculate overall system health score (0-100).

    Args:
        system_overview: System metrics
        alerts: List of current alerts

    Returns:
        Health score from 0 (critical) to 100 (excellent)
    """
    try:
        score = 100.0

        # Deduct points for offline workers
        if system_overview.total_workers > 0:
            offline_ratio = system_overview.offline_workers / system_overview.total_workers
            score -= offline_ratio * 30

        # Deduct points for high system load
        if system_overview.system_load > 0.8:
            load_penalty = (system_overview.system_load - 0.8) * 50
            score -= min(load_penalty, 20)

        # Deduct points for Redis issues
        if system_overview.redis_status != 'healthy':
            score -= 25

        # Deduct points for alerts
        critical_alerts = len([a for a in alerts if a.get('severity') == 'critical'])
        warning_alerts = len([a for a in alerts if a.get('severity') == 'warning'])
        score -= critical_alerts * 15
        score -= warning_alerts * 5

        return max(score, 0.0)

    except Exception:
        return 50.0  # Default neutral score if calculation fails