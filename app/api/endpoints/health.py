"""
Health check endpoints for system monitoring.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...core.celery_config import get_celery_health_info, check_redis_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "ChatLite API"}


@router.get("/health/redis")
async def redis_health_check() -> Dict[str, Any]:
    """Check Redis connection health."""
    try:
        is_healthy = check_redis_connection()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "Redis",
            "connected": is_healthy
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        raise HTTPException(status_code=503, detail="Redis health check failed")


@router.get("/health/celery")
async def celery_health_check() -> Dict[str, Any]:
    """Check Celery workers and Redis broker health."""
    try:
        health_info = get_celery_health_info()
        status_code = 200 if health_info['overall_status'] == 'healthy' else 503

        if status_code == 503:
            raise HTTPException(
                status_code=status_code,
                detail=f"Celery unhealthy: {health_info}"
            )

        return health_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        raise HTTPException(status_code=503, detail="Celery health check failed")


@router.get("/health/workers")
async def workers_health_check() -> Dict[str, Any]:
    """Get detailed Celery workers status."""
    try:
        from ...core.celery_config import get_worker_health_status
        return get_worker_health_status()
    except Exception as e:
        logger.error(f"Workers health check failed: {e}")
        raise HTTPException(status_code=503, detail="Workers health check failed")


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Comprehensive health check for all system components."""
    try:
        redis_healthy = check_redis_connection()
        celery_info = get_celery_health_info()

        overall_healthy = (
            redis_healthy and
            celery_info['overall_status'] == 'healthy'
        )

        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "components": {
                "api": {"status": "healthy"},
                "redis": {
                    "status": "healthy" if redis_healthy else "unhealthy",
                    "connected": redis_healthy
                },
                "celery": celery_info
            },
            "timestamp": "2025-09-17T00:00:00Z"  # Will be updated with actual timestamp
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")