"""
Celery Beat configuration for automated scheduling.

This module configures Celery Beat with periodic tasks for automated crawling
and maintenance operations.
"""

from celery.schedules import crontab
from typing import Dict, Any

from .celery_config import get_celery_config


def get_celery_beat_schedule() -> Dict[str, Any]:
    """
    Get the Celery Beat schedule configuration.

    Returns:
        Dict containing Celery Beat schedule
    """
    return {
        # Monitor schedule changes every 5 minutes
        'monitor-schedule-changes': {
            'task': 'crawler.tasks.monitor_schedule_changes',
            'schedule': crontab(minute='*/5'),
            'options': {
                'queue': 'schedule_queue',
                'routing_key': 'schedule_queue'
            }
        },

        # Clean up old crawl data daily at 1 AM
        'cleanup-old-crawl-data': {
            'task': 'crawler.tasks.cleanup_old_crawl_data',
            'schedule': crontab(hour=1, minute=0),
            'args': [30],  # Keep 30 days of data
            'options': {
                'queue': 'process_queue',
                'routing_key': 'process_queue'
            }
        },

        # Health check websites every 4 hours
        'health-check-websites': {
            'task': 'crawler.tasks.health_check_websites',
            'schedule': crontab(minute=0, hour='*/4'),
            'options': {
                'queue': 'monitor_queue',
                'routing_key': 'monitor_queue'
            }
        },
    }


def get_celery_beat_config() -> Dict[str, Any]:
    """
    Get complete Celery Beat configuration.

    Returns:
        Dict containing full Celery Beat configuration
    """
    base_config = get_celery_config()

    beat_config = {
        **base_config,
        'beat_schedule': get_celery_beat_schedule(),
        'timezone': 'UTC',
        'beat_scheduler': 'django_celery_beat.schedulers:DatabaseScheduler',  # For dynamic schedules
        'beat_max_loop_interval': 60,  # Check for new schedules every minute
    }

    return beat_config


def get_dynamic_schedule_template() -> Dict[str, Any]:
    """
    Get template for dynamic website crawl schedules.

    This template is used by the automated scheduler to create
    dynamic schedules for individual websites.

    Returns:
        Dict containing schedule template
    """
    return {
        'task': 'crawler.tasks.crawl_url',
        'options': {
            'queue': 'crawl_queue',
            'routing_key': 'crawl_queue',
            'expires': 3600,  # Task expires after 1 hour if not executed
        }
    }


# Example of how dynamic schedules would be structured
EXAMPLE_DYNAMIC_SCHEDULES = {
    # Daily crawl schedule
    'crawl_website_daily_example': {
        'task': 'crawler.tasks.crawl_url',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        'args': ['website-uuid', 'https://example.com', 100, 3],
        'options': {
            'queue': 'crawl_queue',
            'routing_key': 'crawl_queue'
        }
    },

    # Weekly crawl schedule
    'crawl_website_weekly_example': {
        'task': 'crawler.tasks.crawl_url',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),  # Sunday at 2 AM
        'args': ['website-uuid', 'https://example.com', 100, 3],
        'options': {
            'queue': 'crawl_queue',
            'routing_key': 'crawl_queue'
        }
    },

    # Monthly crawl schedule
    'crawl_website_monthly_example': {
        'task': 'crawler.tasks.crawl_url',
        'schedule': crontab(hour=2, minute=0, day_of_month=1),  # 1st of month at 2 AM
        'args': ['website-uuid', 'https://example.com', 100, 3],
        'options': {
            'queue': 'crawl_queue',
            'routing_key': 'crawl_queue'
        }
    }
}