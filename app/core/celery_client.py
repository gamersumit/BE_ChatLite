"""
Celery client helper for sending tasks to workers.
"""
from .celery_config import celery_app


def get_celery_app():
    """Get the configured Celery app instance."""
    return celery_app
