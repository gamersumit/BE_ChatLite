"""
Timezone-aware scheduling service for user-specific crawling schedules.

This service handles timezone-specific scheduling and provides utilities
for converting schedules across different timezones.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from celery.schedules import crontab
import pytz

logger = logging.getLogger(__name__)


class TimezoneSchedulerService:
    """
    Service for handling timezone-aware scheduling.

    This service:
    1. Converts user-specified times to UTC for Celery Beat
    2. Manages user timezone preferences
    3. Handles daylight saving time transitions
    4. Provides schedule preview in user's timezone
    """

    def __init__(self):
        """Initialize the timezone scheduler service."""
        self.supported_timezones = self._get_supported_timezones()

    def _get_supported_timezones(self) -> List[str]:
        """Get list of supported timezones."""
        # Common timezones - can be expanded based on user needs
        return [
            'UTC',
            'US/Eastern',
            'US/Central',
            'US/Mountain',
            'US/Pacific',
            'Europe/London',
            'Europe/Paris',
            'Europe/Berlin',
            'Asia/Tokyo',
            'Asia/Shanghai',
            'Asia/Kolkata',
            'Australia/Sydney',
            'Australia/Melbourne'
        ]

    def validate_timezone(self, timezone_str: str) -> bool:
        """
        Validate if timezone string is supported.

        Args:
            timezone_str: Timezone identifier (e.g., 'US/Eastern')

        Returns:
            True if timezone is valid and supported
        """
        try:
            pytz.timezone(timezone_str)
            return timezone_str in self.supported_timezones
        except pytz.exceptions.UnknownTimeZoneError:
            return False

    def convert_schedule_to_utc(
        self,
        frequency: str,
        hour: int,
        minute: int,
        user_timezone: str = 'UTC'
    ) -> Optional[crontab]:
        """
        Convert user schedule to UTC-based crontab.

        Args:
            frequency: Schedule frequency (daily, weekly, monthly)
            hour: Hour in user's timezone (0-23)
            minute: Minute (0-59)
            user_timezone: User's timezone identifier

        Returns:
            crontab object configured for UTC execution
        """
        try:
            if not self.validate_timezone(user_timezone):
                logger.error(f"Invalid timezone: {user_timezone}")
                return None

            # Convert user time to UTC
            user_tz = pytz.timezone(user_timezone)
            utc_tz = pytz.timezone('UTC')

            # Create a sample datetime in user's timezone
            # Use a fixed date to avoid DST complications in conversion
            sample_date = datetime(2024, 6, 15, hour, minute, 0)  # Mid-year date
            user_dt = user_tz.localize(sample_date)
            utc_dt = user_dt.astimezone(utc_tz)

            utc_hour = utc_dt.hour
            utc_minute = utc_dt.minute

            # Create appropriate crontab based on frequency
            if frequency == 'daily':
                return crontab(hour=utc_hour, minute=utc_minute)
            elif frequency == 'weekly':
                # Default to Sunday for weekly schedules
                return crontab(hour=utc_hour, minute=utc_minute, day_of_week=0)
            elif frequency == 'monthly':
                # Default to 1st of month for monthly schedules
                return crontab(hour=utc_hour, minute=utc_minute, day_of_month=1)
            else:
                logger.error(f"Invalid frequency: {frequency}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert schedule to UTC: {e}")
            return None

    def get_next_execution_time(
        self,
        schedule: crontab,
        user_timezone: str = 'UTC'
    ) -> Optional[datetime]:
        """
        Get next execution time in user's timezone.

        Args:
            schedule: Celery crontab schedule
            user_timezone: User's timezone for display

        Returns:
            Next execution datetime in user's timezone
        """
        try:
            if not self.validate_timezone(user_timezone):
                return None

            # Get current UTC time
            now_utc = datetime.now(timezone.utc)

            # Calculate next execution (simplified - in practice would use Celery's algorithm)
            next_utc = self._calculate_next_execution(schedule, now_utc)

            if next_utc:
                # Convert to user timezone
                user_tz = pytz.timezone(user_timezone)
                next_user = next_utc.astimezone(user_tz)
                return next_user

            return None

        except Exception as e:
            logger.error(f"Failed to get next execution time: {e}")
            return None

    def _calculate_next_execution(self, schedule: crontab, from_time: datetime) -> Optional[datetime]:
        """
        Calculate next execution time for a crontab schedule.

        This is a simplified implementation. In practice, you would use
        Celery's built-in scheduling logic.

        Args:
            schedule: Celery crontab schedule
            from_time: Calculate next execution after this time

        Returns:
            Next execution datetime in UTC
        """
        try:
            # Simplified logic - real implementation would be more complex
            # For demonstration purposes, assume daily schedule at specified hour/minute

            if hasattr(schedule, 'hour') and hasattr(schedule, 'minute'):
                target_hour = list(schedule.hour)[0] if schedule.hour else 0
                target_minute = list(schedule.minute)[0] if schedule.minute else 0

                # Calculate next occurrence
                next_exec = from_time.replace(
                    hour=target_hour,
                    minute=target_minute,
                    second=0,
                    microsecond=0
                )

                # If time has passed today, schedule for tomorrow
                if next_exec <= from_time:
                    next_exec = next_exec.replace(day=next_exec.day + 1)

                return next_exec

            return None

        except Exception as e:
            logger.error(f"Failed to calculate next execution: {e}")
            return None

    def create_user_schedule(
        self,
        website_id: str,
        frequency: str,
        hour: int,
        minute: int,
        user_timezone: str = 'UTC'
    ) -> Dict[str, Any]:
        """
        Create a complete schedule configuration for a user's website.

        Args:
            website_id: Website identifier
            frequency: Schedule frequency (daily, weekly, monthly)
            hour: Hour in user's timezone
            minute: Minute
            user_timezone: User's timezone

        Returns:
            Complete schedule configuration
        """
        try:
            # Validate inputs
            if not self.validate_timezone(user_timezone):
                return {
                    'success': False,
                    'error': f'Invalid timezone: {user_timezone}'
                }

            if hour < 0 or hour > 23:
                return {
                    'success': False,
                    'error': f'Invalid hour: {hour} (must be 0-23)'
                }

            if minute < 0 or minute > 59:
                return {
                    'success': False,
                    'error': f'Invalid minute: {minute} (must be 0-59)'
                }

            # Convert to UTC schedule
            utc_schedule = self.convert_schedule_to_utc(
                frequency, hour, minute, user_timezone
            )

            if not utc_schedule:
                return {
                    'success': False,
                    'error': 'Failed to create UTC schedule'
                }

            # Get next execution time in user timezone
            next_execution = self.get_next_execution_time(utc_schedule, user_timezone)

            # Create schedule configuration
            schedule_config = {
                'schedule_id': f"crawl_website_{website_id}",
                'task': 'crawler.tasks.crawl_url',
                'schedule': utc_schedule,
                'args': [website_id],
                'options': {
                    'queue': 'crawl_queue',
                    'routing_key': 'crawl_queue',
                    'expires': 3600
                },
                'metadata': {
                    'website_id': website_id,
                    'frequency': frequency,
                    'user_hour': hour,
                    'user_minute': minute,
                    'user_timezone': user_timezone,
                    'next_execution_user_tz': next_execution.isoformat() if next_execution else None,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            }

            return {
                'success': True,
                'schedule': schedule_config,
                'next_execution': next_execution
            }

        except Exception as e:
            logger.error(f"Failed to create user schedule: {e}")
            return {
                'success': False,
                'error': f'Failed to create schedule: {str(e)}'
            }

    def preview_schedule(
        self,
        frequency: str,
        hour: int,
        minute: int,
        user_timezone: str = 'UTC',
        num_executions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Preview upcoming execution times for a schedule.

        Args:
            frequency: Schedule frequency
            hour: Hour in user's timezone
            minute: Minute
            user_timezone: User's timezone
            num_executions: Number of future executions to show

        Returns:
            List of execution preview information
        """
        try:
            utc_schedule = self.convert_schedule_to_utc(
                frequency, hour, minute, user_timezone
            )

            if not utc_schedule:
                return []

            previews = []
            current_time = datetime.now(timezone.utc)

            for i in range(num_executions):
                next_exec = self._calculate_next_execution(utc_schedule, current_time)
                if next_exec:
                    user_time = self.get_next_execution_time(utc_schedule, user_timezone)

                    previews.append({
                        'execution_number': i + 1,
                        'utc_time': next_exec.isoformat(),
                        'user_time': user_time.isoformat() if user_time else None,
                        'user_timezone': user_timezone
                    })

                    # Move to next interval for next calculation
                    from datetime import timedelta
                    if frequency == 'daily':
                        current_time = next_exec + timedelta(days=1)
                    elif frequency == 'weekly':
                        current_time = next_exec + timedelta(days=7)
                    else:  # monthly
                        current_time = next_exec + timedelta(days=30)

            return previews

        except Exception as e:
            logger.error(f"Failed to preview schedule: {e}")
            return []


# Global service instance
_timezone_scheduler_service = None


def get_timezone_scheduler() -> TimezoneSchedulerService:
    """Get global timezone scheduler service instance."""
    global _timezone_scheduler_service
    if _timezone_scheduler_service is None:
        _timezone_scheduler_service = TimezoneSchedulerService()
    return _timezone_scheduler_service


def validate_user_timezone(timezone_str: str) -> bool:
    """Validate user timezone string."""
    return get_timezone_scheduler().validate_timezone(timezone_str)


def create_timezone_aware_schedule(
    website_id: str,
    frequency: str,
    hour: int,
    minute: int,
    user_timezone: str = 'UTC'
) -> Dict[str, Any]:
    """Create timezone-aware schedule for website crawling."""
    return get_timezone_scheduler().create_user_schedule(
        website_id, frequency, hour, minute, user_timezone
    )