"""
Automated Scheduling Backend System for ChatLite Crawling.

This service provides automated scheduling capabilities using Celery Beat and
integrates with the registration scheduler to manage periodic crawling tasks.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from celery import Celery
from celery.beat import PersistentScheduler
from celery.schedules import crontab
from supabase import Client

from app.core.database import get_supabase_admin_client
from app.core.celery_config import celery_app
from app.tasks.crawler_tasks import crawl_url, schedule_crawl
from app.services.registration_scheduler import get_registration_scheduler

logger = logging.getLogger(__name__)


class AutomatedSchedulerService:
    """
    Automated scheduling service for managing periodic crawl tasks.

    This service:
    1. Manages Celery Beat schedule entries for website crawling
    2. Monitors websites for schedule updates
    3. Handles automatic task scheduling based on website configurations
    4. Provides health monitoring and status reporting
    """

    def __init__(self, supabase_client: Optional[Client] = None, celery_app_instance: Optional[Celery] = None):
        """Initialize the automated scheduler service."""
        self.supabase = supabase_client or get_supabase_admin_client()
        self.celery_app = celery_app_instance or celery_app
        self.registration_scheduler = get_registration_scheduler()
        self.active_schedules: Dict[str, Dict[str, Any]] = {}

    def start_scheduler(self) -> Dict[str, Any]:
        """
        Start the automated scheduler system.

        Returns:
            Dict containing startup result
        """
        try:
            logger.info("Starting automated scheduler system...")

            # Load existing website schedules
            self.load_website_schedules()

            # Setup monitoring for schedule changes
            self.setup_schedule_monitoring()

            # Register periodic tasks
            self.register_periodic_tasks()

            logger.info("Automated scheduler system started successfully")
            return {
                'success': True,
                'message': 'Automated scheduler started',
                'active_schedules': len(self.active_schedules)
            }

        except Exception as e:
            logger.error(f"Failed to start automated scheduler: {e}")
            return {'success': False, 'error': str(e)}

    def load_website_schedules(self) -> None:
        """Load all website schedules from database and register with Celery Beat."""
        try:
            # Get all websites with automatic scheduling enabled
            websites_result = self.supabase.table('websites').select(
                'id, domain, name, scraping_frequency, scraping_enabled, max_pages, max_depth, next_scheduled_crawl'
            ).eq('scraping_enabled', True).neq('scraping_frequency', 'manual').execute()

            websites = websites_result.data or []
            logger.info(f"Loading schedules for {len(websites)} websites")

            for website in websites:
                self.register_website_schedule(website)

            logger.info(f"Loaded {len(self.active_schedules)} active schedules")

        except Exception as e:
            logger.error(f"Failed to load website schedules: {e}")

    def register_website_schedule(self, website: Dict[str, Any]) -> bool:
        """
        Register a website's crawl schedule with Celery Beat.

        Args:
            website: Website configuration dictionary

        Returns:
            True if registered successfully
        """
        try:
            if website is None:
                logger.error("Cannot register schedule: website data is None")
                return False

            website_id = website['id']
            domain = website['domain']
            frequency = website['scraping_frequency']

            # Create unique task name
            task_name = f"crawl_website_{website_id}"

            # Convert frequency to Celery schedule
            schedule = self.frequency_to_celery_schedule(frequency)

            if not schedule:
                logger.warning(f"Invalid frequency '{frequency}' for website {domain}")
                return False

            # Register with Celery Beat (conceptual - actual implementation would use celery beat)
            schedule_config = {
                'task': 'crawler.tasks.crawl_url',
                'schedule': schedule,
                'args': [website_id, f"https://{domain}", website.get('max_pages', 100), website.get('max_depth', 3)],
                'options': {
                    'queue': 'crawl_queue',
                    'routing_key': 'crawl_queue'
                }
            }

            # Store schedule configuration
            self.active_schedules[task_name] = {
                'website_id': website_id,
                'domain': domain,
                'frequency': frequency,
                'schedule_config': schedule_config,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'last_run': None,
                'next_run': website.get('next_scheduled_crawl')
            }

            logger.info(f"Registered schedule for {domain}: {frequency}")
            return True

        except Exception as e:
            website_id = website.get('id', 'unknown') if website else 'None'
            logger.error(f"Failed to register schedule for website {website_id}: {e}")
            return False

    def frequency_to_celery_schedule(self, frequency: str) -> Optional[crontab]:
        """
        Convert frequency string to Celery crontab schedule.

        Args:
            frequency: Frequency string (daily, weekly, monthly)

        Returns:
            Celery crontab schedule or None if invalid
        """
        if frequency == 'daily':
            # Run daily at 2 AM
            return crontab(hour=2, minute=0)
        elif frequency == 'weekly':
            # Run weekly on Sunday at 2 AM
            return crontab(hour=2, minute=0, day_of_week=0)
        elif frequency == 'monthly':
            # Run monthly on the 1st at 2 AM
            return crontab(hour=2, minute=0, day_of_month=1)
        else:
            return None

    def setup_schedule_monitoring(self) -> None:
        """Setup monitoring for schedule changes in the database."""
        logger.info("Setting up schedule monitoring...")

        # In a production system, this could use database triggers, webhooks,
        # or periodic polling to detect schedule changes
        # For now, we'll implement a periodic check mechanism

        self.register_schedule_monitor_task()

    def register_schedule_monitor_task(self) -> None:
        """Register a task to monitor for schedule changes."""
        monitor_schedule = crontab(minute='*/5')  # Check every 5 minutes

        monitor_config = {
            'task': 'crawler.tasks.monitor_schedule_changes',
            'schedule': monitor_schedule,
            'options': {
                'queue': 'schedule_queue'
            }
        }

        self.active_schedules['schedule_monitor'] = {
            'task_type': 'monitor',
            'schedule_config': monitor_config,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        logger.info("Schedule monitoring task registered")

    def register_periodic_tasks(self) -> None:
        """Register additional periodic maintenance tasks."""
        # Cleanup old crawl data
        cleanup_schedule = crontab(hour=1, minute=0)  # Daily at 1 AM

        cleanup_config = {
            'task': 'crawler.tasks.cleanup_old_crawl_data',
            'schedule': cleanup_schedule,
            'options': {
                'queue': 'process_queue'
            }
        }

        self.active_schedules['cleanup_task'] = {
            'task_type': 'maintenance',
            'schedule_config': cleanup_config,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        logger.info("Periodic maintenance tasks registered")

    def update_website_schedule(self, website_id: str, new_frequency: str) -> Dict[str, Any]:
        """
        Update a website's crawl schedule.

        Args:
            website_id: Website UUID
            new_frequency: New frequency setting

        Returns:
            Dict containing update result
        """
        try:
            task_name = f"crawl_website_{website_id}"

            # Remove existing schedule
            if task_name in self.active_schedules:
                del self.active_schedules[task_name]
                logger.info(f"Removed existing schedule for website {website_id}")

            # If new frequency is not manual, create new schedule
            if new_frequency != 'manual':
                # Get updated website data
                website_result = self.supabase.table('websites').select(
                    'id, domain, name, scraping_frequency, max_pages, max_depth'
                ).eq('id', website_id).single().execute()

                if website_result.data:
                    website = website_result.data
                    website['scraping_frequency'] = new_frequency  # Update with new frequency

                    if self.register_website_schedule(website):
                        logger.info(f"Updated schedule for website {website_id}: {new_frequency}")
                        return {
                            'success': True,
                            'website_id': website_id,
                            'frequency': new_frequency,
                            'message': 'Schedule updated successfully'
                        }
            else:
                logger.info(f"Disabled automatic scheduling for website {website_id}")
                return {
                    'success': True,
                    'website_id': website_id,
                    'frequency': new_frequency,
                    'message': 'Automatic scheduling disabled'
                }

            return {'success': False, 'error': 'Failed to update schedule'}

        except Exception as e:
            logger.error(f"Failed to update schedule for website {website_id}: {e}")
            return {'success': False, 'error': str(e)}

    def remove_website_schedule(self, website_id: str) -> Dict[str, Any]:
        """
        Remove a website's schedule from the automated system.

        Args:
            website_id: Website UUID

        Returns:
            Dict containing removal result
        """
        try:
            task_name = f"crawl_website_{website_id}"

            if task_name in self.active_schedules:
                del self.active_schedules[task_name]
                logger.info(f"Removed schedule for website {website_id}")
                return {
                    'success': True,
                    'website_id': website_id,
                    'message': 'Schedule removed successfully'
                }
            else:
                return {
                    'success': True,
                    'website_id': website_id,
                    'message': 'No active schedule found'
                }

        except Exception as e:
            logger.error(f"Failed to remove schedule for website {website_id}: {e}")
            return {'success': False, 'error': str(e)}

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current status of the automated scheduler.

        Returns:
            Dict containing scheduler status
        """
        try:
            # Count schedules by type
            website_schedules = len([s for s in self.active_schedules.values()
                                   if s.get('website_id')])
            monitor_tasks = len([s for s in self.active_schedules.values()
                               if s.get('task_type') == 'monitor'])
            maintenance_tasks = len([s for s in self.active_schedules.values()
                                   if s.get('task_type') == 'maintenance'])

            # Get next scheduled tasks
            upcoming_tasks = self.get_upcoming_tasks(limit=5)

            return {
                'success': True,
                'status': 'running',
                'active_schedules': {
                    'websites': website_schedules,
                    'monitoring': monitor_tasks,
                    'maintenance': maintenance_tasks,
                    'total': len(self.active_schedules)
                },
                'upcoming_tasks': upcoming_tasks,
                'scheduler_started': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            return {'success': False, 'error': str(e)}

    def get_upcoming_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get upcoming scheduled tasks.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of upcoming task information
        """
        try:
            upcoming = []

            for task_name, schedule_info in self.active_schedules.items():
                if schedule_info.get('next_run'):
                    upcoming.append({
                        'task_name': task_name,
                        'website_id': schedule_info.get('website_id'),
                        'domain': schedule_info.get('domain'),
                        'frequency': schedule_info.get('frequency'),
                        'next_run': schedule_info['next_run'],
                        'task_type': schedule_info.get('task_type', 'crawl')
                    })

            # Sort by next run time
            upcoming.sort(key=lambda x: x['next_run'])

            return upcoming[:limit]

        except Exception as e:
            logger.error(f"Failed to get upcoming tasks: {e}")
            return []

    def trigger_manual_schedule_check(self) -> Dict[str, Any]:
        """
        Manually trigger a schedule check and update.

        Returns:
            Dict containing check result
        """
        try:
            logger.info("Triggering manual schedule check...")

            # Get current website schedules from database
            websites_result = self.supabase.table('websites').select(
                'id, domain, scraping_frequency, scraping_enabled, next_scheduled_crawl'
            ).eq('scraping_enabled', True).execute()

            websites = websites_result.data or []
            updated_count = 0

            for website in websites:
                website_id = website['id']
                current_frequency = website['scraping_frequency']

                task_name = f"crawl_website_{website_id}"

                # Check if schedule needs updating
                if current_frequency == 'manual':
                    if task_name in self.active_schedules:
                        del self.active_schedules[task_name]
                        updated_count += 1
                else:
                    # Check if schedule exists and is correct
                    existing_schedule = self.active_schedules.get(task_name)

                    if not existing_schedule or existing_schedule.get('frequency') != current_frequency:
                        self.register_website_schedule(website)
                        updated_count += 1

            logger.info(f"Manual schedule check complete: {updated_count} schedules updated")

            return {
                'success': True,
                'message': f'Schedule check complete: {updated_count} updates',
                'total_websites': len(websites),
                'updates_made': updated_count
            }

        except Exception as e:
            logger.error(f"Failed manual schedule check: {e}")
            return {'success': False, 'error': str(e)}


# Global scheduler instance
_scheduler_instance: Optional[AutomatedSchedulerService] = None


def get_automated_scheduler() -> AutomatedSchedulerService:
    """Get the global automated scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutomatedSchedulerService()
    return _scheduler_instance


def start_automated_scheduler() -> Dict[str, Any]:
    """Start the global automated scheduler."""
    scheduler = get_automated_scheduler()
    return scheduler.start_scheduler()


def get_scheduler_status() -> Dict[str, Any]:
    """Get status of the automated scheduler."""
    scheduler = get_automated_scheduler()
    return scheduler.get_scheduler_status()