"""
Schedule status tracking and display service.

This service provides functionality to track and display schedule status
information including next execution times, health monitoring, and status indicators.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from supabase import Client

from app.core.database import get_supabase_admin_client
from app.services.timezone_scheduler import get_timezone_scheduler
from app.services.automated_scheduler import get_automated_scheduler

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    """Schedule status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    IDLE = "idle"
    PENDING = "pending"


class CrawlStatus(Enum):
    """Crawl execution status enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class ScheduleInfo:
    """Schedule information data class."""
    website_id: str
    website_domain: str
    frequency: str
    schedule_status: ScheduleStatus
    last_crawl_time: Optional[datetime]
    last_crawl_status: Optional[CrawlStatus]
    next_scheduled_time: Optional[datetime]
    user_timezone: str
    created_at: datetime
    pages_crawled: int = 0
    success_rate: float = 0.0
    consecutive_failures: int = 0


@dataclass
class ScheduleHealthMetrics:
    """Schedule health monitoring metrics."""
    total_schedules: int
    active_schedules: int
    failed_schedules: int
    paused_schedules: int
    average_success_rate: float
    total_executions_24h: int
    failed_executions_24h: int
    last_updated: datetime


class ScheduleStatusService:
    """
    Service for tracking and displaying schedule status information.

    This service:
    1. Tracks schedule execution status and health
    2. Provides next execution time calculations
    3. Monitors schedule health and performance
    4. Manages status indicators and notifications
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the schedule status service."""
        self.supabase = supabase_client or get_supabase_admin_client()
        self.timezone_scheduler = get_timezone_scheduler()
        self.automated_scheduler = get_automated_scheduler()

    def get_website_schedule_status(self, website_id: str) -> Optional[ScheduleInfo]:
        """
        Get comprehensive schedule status for a website.

        Args:
            website_id: Website identifier

        Returns:
            ScheduleInfo object with current status or None if not found
        """
        try:
            # Get website information
            website_response = self.supabase.table('websites').select(
                'id, domain, scraping_frequency, user_timezone, created_at'
            ).eq('id', website_id).execute()

            if not website_response.data:
                logger.warning(f"Website not found: {website_id}")
                return None

            website = website_response.data[0]

            # Get latest crawl history
            crawl_response = self.supabase.table('crawl_history').select(
                'status, completed_at, pages_crawled, trigger_type'
            ).eq('website_id', website_id).order(
                'completed_at', desc=True
            ).limit(1).execute()

            last_crawl_time = None
            last_crawl_status = None
            pages_crawled = 0

            if crawl_response.data:
                latest_crawl = crawl_response.data[0]
                last_crawl_time = datetime.fromisoformat(latest_crawl['completed_at']) if latest_crawl['completed_at'] else None
                last_crawl_status = CrawlStatus(latest_crawl['status']) if latest_crawl['status'] else None
                pages_crawled = latest_crawl.get('pages_crawled', 0)

            # Calculate schedule status
            schedule_status = self._determine_schedule_status(
                website['scraping_frequency'],
                last_crawl_time,
                last_crawl_status
            )

            # Calculate next scheduled time
            next_scheduled_time = self._calculate_next_scheduled_time(
                website['scraping_frequency'],
                last_crawl_time,
                website.get('user_timezone', 'UTC')
            )

            # Calculate success rate
            success_rate = self._calculate_success_rate(website_id)

            # Count consecutive failures
            consecutive_failures = self._count_consecutive_failures(website_id)

            return ScheduleInfo(
                website_id=website_id,
                website_domain=website['domain'],
                frequency=website['scraping_frequency'],
                schedule_status=schedule_status,
                last_crawl_time=last_crawl_time,
                last_crawl_status=last_crawl_status,
                next_scheduled_time=next_scheduled_time,
                user_timezone=website.get('user_timezone', 'UTC'),
                created_at=datetime.fromisoformat(website['created_at']),
                pages_crawled=pages_crawled,
                success_rate=success_rate,
                consecutive_failures=consecutive_failures
            )

        except Exception as e:
            logger.error(f"Failed to get schedule status for website {website_id}: {e}")
            return None

    def get_all_schedule_statuses(self, user_id: Optional[str] = None) -> List[ScheduleInfo]:
        """
        Get schedule status for all websites (optionally filtered by user).

        Args:
            user_id: Optional user ID to filter websites

        Returns:
            List of ScheduleInfo objects
        """
        try:
            query = self.supabase.table('websites').select(
                'id, domain, scraping_frequency, user_timezone, created_at, user_id'
            )

            if user_id:
                query = query.eq('user_id', user_id)

            websites_response = query.execute()

            if not websites_response.data:
                return []

            statuses = []
            for website in websites_response.data:
                status = self.get_website_schedule_status(website['id'])
                if status:
                    statuses.append(status)

            return statuses

        except Exception as e:
            logger.error(f"Failed to get all schedule statuses: {e}")
            return []

    def _determine_schedule_status(
        self,
        frequency: str,
        last_crawl_time: Optional[datetime],
        last_crawl_status: Optional[CrawlStatus]
    ) -> ScheduleStatus:
        """
        Determine the current schedule status based on configuration and history.

        Args:
            frequency: Crawling frequency setting
            last_crawl_time: Last crawl execution time
            last_crawl_status: Last crawl execution status

        Returns:
            Current schedule status
        """
        try:
            # If frequency is manual, consider it idle
            if frequency == 'manual':
                return ScheduleStatus.IDLE

            # If no crawl history, status is pending
            if not last_crawl_time:
                return ScheduleStatus.PENDING

            # Check if schedule should have run but didn't
            expected_interval = self._get_frequency_interval(frequency)
            time_since_last = datetime.now(timezone.utc) - last_crawl_time

            if time_since_last > expected_interval * 2:  # Double the expected interval
                return ScheduleStatus.FAILED

            # Check recent crawl status
            if last_crawl_status == CrawlStatus.FAILED:
                # Check if there have been multiple consecutive failures
                consecutive_failures = self._count_consecutive_failures_from_status(last_crawl_status)
                if consecutive_failures >= 3:
                    return ScheduleStatus.FAILED
                else:
                    return ScheduleStatus.ACTIVE  # Still trying

            if last_crawl_status == CrawlStatus.SUCCESS:
                return ScheduleStatus.ACTIVE

            return ScheduleStatus.ACTIVE

        except Exception as e:
            logger.error(f"Failed to determine schedule status: {e}")
            return ScheduleStatus.FAILED

    def _get_frequency_interval(self, frequency: str) -> timedelta:
        """Get expected interval for a frequency setting."""
        intervals = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30),  # Approximate
            'high': timedelta(minutes=15),
            'medium': timedelta(hours=2),
            'low': timedelta(days=1)
        }
        return intervals.get(frequency, timedelta(days=1))

    def _calculate_next_scheduled_time(
        self,
        frequency: str,
        last_crawl_time: Optional[datetime],
        user_timezone: str
    ) -> Optional[datetime]:
        """
        Calculate the next scheduled execution time.

        Args:
            frequency: Crawling frequency
            last_crawl_time: Last crawl execution time
            user_timezone: User's timezone

        Returns:
            Next scheduled execution time in user's timezone
        """
        try:
            if frequency == 'manual':
                return None

            # Use timezone scheduler to calculate next execution
            schedule = self.timezone_scheduler.convert_schedule_to_utc(
                frequency, 2, 0, user_timezone  # Default to 2 AM
            )

            if schedule:
                next_time = self.timezone_scheduler.get_next_execution_time(
                    schedule, user_timezone
                )
                return next_time

            return None

        except Exception as e:
            logger.error(f"Failed to calculate next scheduled time: {e}")
            return None

    def _calculate_success_rate(self, website_id: str, days: int = 30) -> float:
        """
        Calculate success rate over the last N days.

        Args:
            website_id: Website identifier
            days: Number of days to look back

        Returns:
            Success rate as percentage (0.0 to 100.0)
        """
        try:
            since_date = datetime.now(timezone.utc) - timedelta(days=days)

            total_response = self.supabase.table('crawl_history').select(
                'id', count='exact'
            ).eq('website_id', website_id).gte(
                'completed_at', since_date.isoformat()
            ).execute()

            success_response = self.supabase.table('crawl_history').select(
                'id', count='exact'
            ).eq('website_id', website_id).eq(
                'status', 'success'
            ).gte('completed_at', since_date.isoformat()).execute()

            total_count = total_response.count or 0
            success_count = success_response.count or 0

            if total_count == 0:
                return 0.0

            return (success_count / total_count) * 100.0

        except Exception as e:
            logger.error(f"Failed to calculate success rate for {website_id}: {e}")
            return 0.0

    def _count_consecutive_failures(self, website_id: str) -> int:
        """
        Count consecutive failures from the most recent crawl.

        Args:
            website_id: Website identifier

        Returns:
            Number of consecutive failures
        """
        try:
            response = self.supabase.table('crawl_history').select(
                'status'
            ).eq('website_id', website_id).order(
                'completed_at', desc=True
            ).limit(10).execute()

            if not response.data:
                return 0

            consecutive_failures = 0
            for crawl in response.data:
                if crawl['status'] == 'failed':
                    consecutive_failures += 1
                else:
                    break

            return consecutive_failures

        except Exception as e:
            logger.error(f"Failed to count consecutive failures for {website_id}: {e}")
            return 0

    def _count_consecutive_failures_from_status(self, last_status: CrawlStatus) -> int:
        """Helper to estimate consecutive failures from last status."""
        if last_status == CrawlStatus.FAILED:
            return 1  # At least one failure
        return 0

    def get_schedule_health_metrics(self) -> ScheduleHealthMetrics:
        """
        Get overall schedule health metrics.

        Returns:
            ScheduleHealthMetrics with system-wide health information
        """
        try:
            # Get all websites
            websites_response = self.supabase.table('websites').select(
                'id, scraping_frequency'
            ).execute()

            if not websites_response.data:
                return ScheduleHealthMetrics(
                    total_schedules=0,
                    active_schedules=0,
                    failed_schedules=0,
                    paused_schedules=0,
                    average_success_rate=0.0,
                    total_executions_24h=0,
                    failed_executions_24h=0,
                    last_updated=datetime.now(timezone.utc)
                )

            # Calculate status counts
            total_schedules = len(websites_response.data)
            active_schedules = 0
            failed_schedules = 0
            paused_schedules = 0
            total_success_rate = 0.0

            for website in websites_response.data:
                status = self.get_website_schedule_status(website['id'])
                if status:
                    if status.schedule_status == ScheduleStatus.ACTIVE:
                        active_schedules += 1
                    elif status.schedule_status == ScheduleStatus.FAILED:
                        failed_schedules += 1
                    elif status.schedule_status == ScheduleStatus.PAUSED:
                        paused_schedules += 1

                    total_success_rate += status.success_rate

            average_success_rate = total_success_rate / total_schedules if total_schedules > 0 else 0.0

            # Get 24h execution metrics
            since_24h = datetime.now(timezone.utc) - timedelta(hours=24)

            total_24h_response = self.supabase.table('crawl_history').select(
                'id', count='exact'
            ).gte('completed_at', since_24h.isoformat()).execute()

            failed_24h_response = self.supabase.table('crawl_history').select(
                'id', count='exact'
            ).eq('status', 'failed').gte(
                'completed_at', since_24h.isoformat()
            ).execute()

            total_executions_24h = total_24h_response.count or 0
            failed_executions_24h = failed_24h_response.count or 0

            return ScheduleHealthMetrics(
                total_schedules=total_schedules,
                active_schedules=active_schedules,
                failed_schedules=failed_schedules,
                paused_schedules=paused_schedules,
                average_success_rate=average_success_rate,
                total_executions_24h=total_executions_24h,
                failed_executions_24h=failed_executions_24h,
                last_updated=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Failed to get schedule health metrics: {e}")
            return ScheduleHealthMetrics(
                total_schedules=0,
                active_schedules=0,
                failed_schedules=0,
                paused_schedules=0,
                average_success_rate=0.0,
                total_executions_24h=0,
                failed_executions_24h=0,
                last_updated=datetime.now(timezone.utc)
            )

    def update_schedule_status(
        self,
        website_id: str,
        new_status: ScheduleStatus,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update schedule status for a website.

        Args:
            website_id: Website identifier
            new_status: New schedule status
            reason: Optional reason for status change

        Returns:
            True if updated successfully
        """
        try:
            # Log status change
            logger.info(f"Updating schedule status for {website_id} to {new_status.value}: {reason}")

            # In a real implementation, you might store this in a schedule_status table
            # For now, we'll just log it as the status is derived from crawl history

            return True

        except Exception as e:
            logger.error(f"Failed to update schedule status for {website_id}: {e}")
            return False


# Global service instance
_schedule_status_service = None


def get_schedule_status_service() -> ScheduleStatusService:
    """Get global schedule status service instance."""
    global _schedule_status_service
    if _schedule_status_service is None:
        _schedule_status_service = ScheduleStatusService()
    return _schedule_status_service


def get_website_status(website_id: str) -> Optional[ScheduleInfo]:
    """Get schedule status for a specific website."""
    return get_schedule_status_service().get_website_schedule_status(website_id)


def get_all_website_statuses(user_id: Optional[str] = None) -> List[ScheduleInfo]:
    """Get schedule status for all websites."""
    return get_schedule_status_service().get_all_schedule_statuses(user_id)


def get_system_health() -> ScheduleHealthMetrics:
    """Get system-wide schedule health metrics."""
    return get_schedule_status_service().get_schedule_health_metrics()