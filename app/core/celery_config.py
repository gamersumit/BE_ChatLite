"""
Celery configuration with Redis broker for background crawling tasks.
"""

import logging
import ssl
from typing import Dict, Any, Optional
from celery import Celery
import redis

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_celery_config() -> Dict[str, Any]:
    """Get Celery configuration dictionary."""
    broker_url = getattr(settings, 'celery_broker_url', settings.redis_url)
    result_backend = getattr(settings, 'celery_result_backend', settings.redis_url)

    return {
        # Broker and Result Backend
        'broker_url': broker_url,
        'result_backend': result_backend,
        'result_backend_transport_options': {
            'master_name': 'mymaster'
        },

        # Task Routing and Queues
        'task_routes': {
            'crawler.tasks.crawl_url': {'queue': 'crawl_queue'},
            'crawler.tasks.process_data': {'queue': 'process_queue'},
            'crawler.tasks.schedule_crawl': {'queue': 'schedule_queue'},
            'crawler.tasks.generate_embeddings': {'queue': 'process_queue'},
            'crawler.tasks.update_knowledge_base': {'queue': 'process_queue'},
            'monitor.tasks.*': {'queue': 'monitor_queue'},
        },

        # Worker Configuration
        'worker_prefetch_multiplier': 1,
        'task_acks_late': True,
        'worker_disable_rate_limits': False,
        'worker_max_tasks_per_child': 1000,

        # Task Configuration
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'result_expires': 1800,  # 30 minutes (reduced from 1 hour)
        'timezone': 'UTC',
        'enable_utc': True,
        'task_ignore_result': False,  # Store results but expire quickly
        'task_track_started': True,
        'result_compression': 'gzip',  # Compress results to save Redis memory

        # Retry Configuration
        'task_default_retry_delay': 60,  # 1 minute
        'task_max_retries': 3,
        'task_soft_time_limit': 300,  # 5 minutes
        'task_time_limit': 600,  # 10 minutes

        # Redis Connection Pool
        'broker_transport_options': {
            'fanout_prefix': True,
            'fanout_patterns': True,
            'retry_on_timeout': True,
            'max_connections': 20,
        },

        # SSL Configuration for rediss:// URLs
        'broker_use_ssl': {
            'ssl_cert_reqs': ssl.CERT_NONE  # Disable SSL verification for managed Redis
        } if broker_url.startswith('rediss://') else None,
        'redis_backend_use_ssl': {
            'ssl_cert_reqs': ssl.CERT_NONE  # Disable SSL verification for managed Redis
        } if result_backend.startswith('rediss://') else None,

        # Task Discovery
        'include': [
            'app.tasks.crawler_tasks',
            'app.tasks.monitor_tasks',
        ],
    }


# Create Celery app instance
celery_app = Celery('app.core.celery_config')
celery_app.config_from_object(get_celery_config())


def check_redis_connection() -> bool:
    """Check if Redis connection is healthy."""
    try:
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
        r = redis.Redis.from_url(redis_url)
        r.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


def get_worker_health_status() -> Dict[str, Any]:
    """Get health status of Celery workers."""
    try:
        inspector = celery_app.control.inspect()

        # Get active tasks and worker stats
        active_tasks = inspector.active()
        worker_stats = inspector.stats()

        if not active_tasks and not worker_stats:
            return {
                'status': 'unhealthy',
                'workers': [],
                'message': 'No workers detected'
            }

        workers = []
        if worker_stats:
            for worker_name, stats in worker_stats.items():
                worker_info = {
                    'name': worker_name,
                    'status': 'active',
                    'pool_max_concurrency': stats.get('pool', {}).get('max-concurrency', 0),
                    'active_tasks': len(active_tasks.get(worker_name, [])) if active_tasks else 0
                }
                workers.append(worker_info)

        return {
            'status': 'healthy' if workers else 'unhealthy',
            'workers': workers,
            'total_workers': len(workers)
        }

    except Exception as e:
        logger.error(f"Failed to get worker health status: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'workers': []
        }


def route_task(name: str, args, kwargs, options) -> Dict[str, str]:
    """Custom task routing function."""
    routes = get_celery_config()['task_routes']

    # Check exact match first
    if name in routes:
        return routes[name]

    # Check pattern matches
    for pattern, route in routes.items():
        if pattern.endswith('*') and name.startswith(pattern[:-1]):
            return route

    # Default queue
    return {'queue': 'default'}


def check_worker_startup() -> bool:
    """Check if workers have started up successfully."""
    try:
        inspector = celery_app.control.inspect()
        registered_tasks = inspector.registered()

        if registered_tasks:
            logger.info(f"Workers detected: {list(registered_tasks.keys())}")
            return True
        else:
            logger.warning("No workers detected during startup check")
            return False

    except Exception as e:
        logger.error(f"Failed to check worker startup: {e}")
        return False


# Health check endpoint helper
def get_celery_health_info() -> Dict[str, Any]:
    """Get comprehensive Celery health information."""
    redis_healthy = check_redis_connection()
    worker_status = get_worker_health_status()

    return {
        'redis_connection': 'healthy' if redis_healthy else 'unhealthy',
        'workers': worker_status,
        'overall_status': (
            'healthy' if redis_healthy and worker_status['status'] == 'healthy'
            else 'unhealthy'
        )
    }