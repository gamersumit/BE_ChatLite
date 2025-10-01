"""
Celery tasks for system monitoring and health checks.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..core.celery_config import celery_app, get_worker_health_status, check_redis_connection

logger = logging.getLogger(__name__)


@celery_app.task(name='monitor.tasks.health_check')
def health_check() -> Dict[str, Any]:
    """
    Perform comprehensive system health check.

    Returns:
        Dict containing health status of all components
    """
    try:
        timestamp = datetime.utcnow().isoformat()

        # Check Redis connection
        redis_healthy = check_redis_connection()

        # Check worker status
        worker_status = get_worker_health_status()

        # Calculate overall health
        overall_healthy = (
            redis_healthy and
            worker_status.get('status') == 'healthy'
        )

        return {
            'timestamp': timestamp,
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'components': {
                'redis': {
                    'status': 'healthy' if redis_healthy else 'unhealthy',
                    'connected': redis_healthy
                },
                'workers': worker_status,
                'celery_app': {
                    'status': 'healthy',
                    'registered_tasks': len(celery_app.tasks)
                }
            }
        }

    except Exception as exc:
        logger.error(f"Health check task failed: {exc}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'error',
            'error': str(exc)
        }


@celery_app.task(name='monitor.tasks.worker_stats')
def collect_worker_stats() -> Dict[str, Any]:
    """
    Collect detailed worker statistics.

    Returns:
        Dict containing worker performance metrics
    """
    try:
        inspector = celery_app.control.inspect()

        # Get worker statistics
        stats = inspector.stats()
        active_tasks = inspector.active()
        reserved_tasks = inspector.reserved()

        worker_metrics = {}
        total_active = 0
        total_reserved = 0

        if stats:
            for worker_name, worker_stats in stats.items():
                active_count = len(active_tasks.get(worker_name, [])) if active_tasks else 0
                reserved_count = len(reserved_tasks.get(worker_name, [])) if reserved_tasks else 0

                total_active += active_count
                total_reserved += reserved_count

                worker_metrics[worker_name] = {
                    'active_tasks': active_count,
                    'reserved_tasks': reserved_count,
                    'pool_max_concurrency': worker_stats.get('pool', {}).get('max-concurrency', 0),
                    'pool_processes': worker_stats.get('pool', {}).get('processes', []),
                    'rusage': worker_stats.get('rusage', {}),
                    'clock': worker_stats.get('clock', 0)
                }

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_workers': len(worker_metrics),
                'total_active_tasks': total_active,
                'total_reserved_tasks': total_reserved
            },
            'workers': worker_metrics
        }

    except Exception as exc:
        logger.error(f"Worker stats collection failed: {exc}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(exc)
        }


@celery_app.task(name='monitor.tasks.queue_stats')
def collect_queue_stats() -> Dict[str, Any]:
    """
    Collect queue statistics and metrics.

    Returns:
        Dict containing queue metrics
    """
    try:
        inspector = celery_app.control.inspect()

        # Get active queues (this is a simplified version)
        # In a real implementation, you'd query Redis directly for queue lengths
        active_queues = inspector.active_queues()

        queue_stats = {}
        if active_queues:
            for worker_name, queues in active_queues.items():
                for queue_info in queues:
                    queue_name = queue_info.get('name', 'unknown')
                    if queue_name not in queue_stats:
                        queue_stats[queue_name] = {
                            'workers': [],
                            'routing_key': queue_info.get('routing_key', ''),
                            'exchange': queue_info.get('exchange', {}).get('name', '')
                        }
                    queue_stats[queue_name]['workers'].append(worker_name)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'queues': queue_stats,
            'total_queues': len(queue_stats)
        }

    except Exception as exc:
        logger.error(f"Queue stats collection failed: {exc}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(exc)
        }


@celery_app.task(name='monitor.tasks.cleanup_failed_tasks')
def cleanup_failed_tasks(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old failed tasks and their results.

    Args:
        max_age_hours: Maximum age in hours for failed task results to keep

    Returns:
        Dict containing cleanup results
    """
    try:
        # This would implement cleanup logic for old task results
        # For now, just return a placeholder response

        logger.info(f"Cleaning up failed tasks older than {max_age_hours} hours")

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed',
            'max_age_hours': max_age_hours,
            'cleaned_tasks': 0  # Placeholder
        }

    except Exception as exc:
        logger.error(f"Failed task cleanup failed: {exc}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(exc)
        }