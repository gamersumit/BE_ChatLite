"""
Crawl management service for handling manual and scheduled crawling operations.
"""

import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum

from ..tasks.crawler_tasks import crawl_url, process_crawled_content, schedule_crawl
from ..core.celery_config import celery_app, get_worker_health_status

logger = logging.getLogger(__name__)


class CrawlStatus(Enum):
    """Crawl status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CrawlManager:
    """Service for managing crawling operations and job lifecycle."""

    def __init__(self):
        self.celery_app = celery_app

    def trigger_manual_crawl(
        self,
        website_id: UUID,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Trigger a manual crawl for a website.

        Args:
            website_id: UUID of the website to crawl
            base_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum crawl depth

        Returns:
            Dict containing task information and status
        """
        try:
            # Check if workers are available
            worker_status = get_worker_health_status()
            if worker_status['status'] != 'healthy':
                return {
                    'success': False,
                    'error': 'No healthy workers available',
                    'worker_status': worker_status
                }

            # Submit crawl task
            task = crawl_url.delay(
                website_id=str(website_id),
                url=base_url,
                max_pages=max_pages,
                max_depth=max_depth
            )

            logger.info(f"Manual crawl triggered for website {website_id}, task: {task.id}")

            return {
                'success': True,
                'task_id': task.id,
                'website_id': str(website_id),
                'status': CrawlStatus.PENDING.value,
                'estimated_duration': self._estimate_crawl_duration(max_pages),
                'started_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to trigger manual crawl for website {website_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'website_id': str(website_id)
            }

    def get_crawl_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a crawl task.

        Args:
            task_id: Celery task ID

        Returns:
            Dict containing task status and progress information
        """
        try:
            task_result = self.celery_app.AsyncResult(task_id)

            if task_result.state == 'PENDING':
                return {
                    'status': CrawlStatus.PENDING.value,
                    'progress': 0,
                    'message': 'Task is waiting to be processed'
                }
            elif task_result.state == 'PROGRESS':
                return {
                    'status': CrawlStatus.IN_PROGRESS.value,
                    'progress': task_result.info.get('progress', 0),
                    'message': task_result.info.get('status', 'Processing...')
                }
            elif task_result.state == 'SUCCESS':
                return {
                    'status': CrawlStatus.COMPLETED.value,
                    'progress': 100,
                    'result': task_result.result,
                    'completed_at': datetime.utcnow().isoformat()
                }
            else:  # FAILURE
                return {
                    'status': CrawlStatus.FAILED.value,
                    'progress': 0,
                    'error': str(task_result.info),
                    'failed_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get crawl status for task {task_id}: {e}")
            return {
                'status': CrawlStatus.FAILED.value,
                'error': str(e)
            }

    def cancel_crawl(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a running crawl task.

        Args:
            task_id: Celery task ID to cancel

        Returns:
            Dict containing cancellation result
        """
        try:
            self.celery_app.control.revoke(task_id, terminate=True)

            logger.info(f"Crawl task {task_id} cancelled")

            return {
                'success': True,
                'task_id': task_id,
                'status': CrawlStatus.CANCELLED.value,
                'cancelled_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to cancel crawl task {task_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id
            }

    def get_active_crawls(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active crawl tasks.

        Returns:
            List of active crawl task information
        """
        try:
            inspector = self.celery_app.control.inspect()
            active_tasks = inspector.active()

            crawl_tasks = []
            if active_tasks:
                for worker_name, tasks in active_tasks.items():
                    for task in tasks:
                        if task['name'].startswith('crawler.tasks.'):
                            crawl_tasks.append({
                                'task_id': task['id'],
                                'name': task['name'],
                                'args': task['args'],
                                'worker': worker_name,
                                'time_start': task['time_start']
                            })

            return crawl_tasks

        except Exception as e:
            logger.error(f"Failed to get active crawls: {e}")
            return []

    def schedule_website_crawl(
        self,
        website_id: UUID,
        frequency: str,
        base_url: str
    ) -> Dict[str, Any]:
        """
        Schedule automatic crawling for a website based on registration frequency.

        Args:
            website_id: UUID of the website
            frequency: Crawl frequency (daily, weekly, monthly)
            base_url: Website base URL

        Returns:
            Dict containing schedule result
        """
        try:
            # Calculate next crawl time based on frequency
            next_crawl = self._calculate_next_crawl_time(frequency)

            # For now, just log the scheduling - in a full implementation,
            # this would integrate with Celery Beat or a custom scheduler
            logger.info(f"Scheduling {frequency} crawl for website {website_id}, next crawl: {next_crawl}")

            return {
                'success': True,
                'website_id': str(website_id),
                'frequency': frequency,
                'next_crawl': next_crawl.isoformat(),
                'scheduled_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to schedule crawl for website {website_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'website_id': str(website_id)
            }

    def get_crawl_statistics(self, website_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get crawling statistics for a website or all websites.

        Args:
            website_id: Optional website UUID to filter statistics

        Returns:
            Dict containing crawl statistics
        """
        try:
            # This would integrate with your database to get actual statistics
            # For now, return placeholder data

            return {
                'total_crawls': 0,
                'successful_crawls': 0,
                'failed_crawls': 0,
                'average_duration': 0,
                'last_crawl': None,
                'website_id': str(website_id) if website_id else None
            }

        except Exception as e:
            logger.error(f"Failed to get crawl statistics: {e}")
            return {'error': str(e)}

    def _estimate_crawl_duration(self, max_pages: int) -> int:
        """
        Estimate crawl duration in minutes based on page count.

        Args:
            max_pages: Maximum number of pages to crawl

        Returns:
            Estimated duration in minutes
        """
        # Rough estimate: 1-2 seconds per page
        base_seconds = max_pages * 1.5
        return max(1, int(base_seconds / 60))  # At least 1 minute

    def _calculate_next_crawl_time(self, frequency: str) -> datetime:
        """
        Calculate next crawl time based on frequency.

        Args:
            frequency: Crawl frequency setting

        Returns:
            Next crawl datetime
        """
        now = datetime.utcnow()

        if frequency == 'daily':
            return now + timedelta(days=1)
        elif frequency == 'weekly':
            return now + timedelta(weeks=1)
        elif frequency == 'monthly':
            return now + timedelta(days=30)
        else:
            # Default to daily
            return now + timedelta(days=1)