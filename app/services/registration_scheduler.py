"""
Registration Schedule Service for automatic crawl scheduling based on website registration parameters.

This service handles the integration between website registration data and the crawling system,
automatically setting up scheduled crawls based on the user's selected frequency during registration.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from supabase import Client

from app.core.database import get_supabase_admin_client
from app.services.crawl_manager import CrawlManager
from app.tasks.crawler_tasks import schedule_crawl

logger = logging.getLogger(__name__)


class RegistrationSchedulerService:
    """Service for managing automatic crawl scheduling based on registration parameters."""

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the registration scheduler service."""
        self.supabase = supabase_client or get_supabase_admin_client()
        self.crawl_manager = CrawlManager()

    def setup_website_crawl_schedule(self, website_id: str) -> Dict[str, Any]:
        """
        Set up automatic crawl scheduling for a website based on its registration parameters.

        This method:
        1. Retrieves the website's scraping frequency from registration data
        2. Creates automatic scheduled crawls if frequency is not 'manual'
        3. Updates the database with next scheduled crawl time

        Args:
            website_id: UUID of the website to schedule crawling for

        Returns:
            Dict containing the schedule setup result
        """
        try:
            # Get website data including scraping frequency
            website_result = self.supabase.table('websites').select(
                'id, domain, name, scraping_frequency, scraping_enabled, max_pages, max_depth'
            ).eq('id', website_id).single().execute()

            if not website_result.data:
                logger.error(f"Website {website_id} not found")
                return {'success': False, 'error': 'Website not found'}

            website = website_result.data
            scraping_frequency = website.get('scraping_frequency', 'daily')
            scraping_enabled = website.get('scraping_enabled', True)

            logger.info(f"Setting up crawl schedule for website {website['domain']} "
                       f"with frequency: {scraping_frequency}")

            # If scraping is disabled or frequency is manual, no automatic scheduling
            if not scraping_enabled or scraping_frequency == 'manual':
                logger.info(f"Skipping automatic scheduling for {website['domain']} "
                           f"(enabled: {scraping_enabled}, frequency: {scraping_frequency})")
                return {
                    'success': True,
                    'message': 'Manual crawling only - no automatic scheduling',
                    'frequency': scraping_frequency,
                    'next_crawl': None
                }

            # Calculate next crawl time based on frequency
            next_crawl_time = self._calculate_next_crawl_time(scraping_frequency)

            # Update website with next scheduled crawl time
            update_result = self.supabase.table('websites').update({
                'next_scheduled_crawl': next_crawl_time.isoformat(),
                'last_scheduled_crawl': datetime.now(timezone.utc).isoformat()
            }).eq('id', website_id).execute()

            if not update_result.data:
                logger.error(f"Failed to update next crawl time for website {website_id}")

            # Schedule the crawl task using Celery
            try:
                schedule_result = schedule_crawl.delay(website_id, scraping_frequency)
                logger.info(f"Scheduled Celery task {schedule_result.id} for website {website_id}")
            except Exception as e:
                logger.warning(f"Failed to schedule Celery task for website {website_id}: {e}")
                # Continue without failing - the scheduler service will pick it up

            return {
                'success': True,
                'website_id': website_id,
                'frequency': scraping_frequency,
                'next_crawl': next_crawl_time.isoformat(),
                'message': f'Scheduled {scraping_frequency} crawling'
            }

        except Exception as e:
            logger.error(f"Failed to setup crawl schedule for website {website_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_website_crawl_schedule(
        self,
        website_id: str,
        new_frequency: str,
        enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Update the crawl schedule for an existing website.

        Args:
            website_id: UUID of the website
            new_frequency: New crawling frequency (daily, weekly, monthly, manual)
            enabled: Whether to enable/disable scraping

        Returns:
            Dict containing the update result
        """
        try:
            # Build update data
            update_data = {'scraping_frequency': new_frequency}

            if enabled is not None:
                update_data['scraping_enabled'] = enabled

            # Calculate next crawl time if applicable
            if new_frequency != 'manual' and enabled != False:
                next_crawl_time = self._calculate_next_crawl_time(new_frequency)
                update_data['next_scheduled_crawl'] = next_crawl_time.isoformat()
                update_data['last_scheduled_crawl'] = datetime.now(timezone.utc).isoformat()
            else:
                update_data['next_scheduled_crawl'] = None

            # Update in database
            result = self.supabase.table('websites').update(update_data).eq(
                'id', website_id
            ).execute()

            if result.data:
                logger.info(f"Updated crawl schedule for website {website_id}: {new_frequency}")
                return {
                    'success': True,
                    'website_id': website_id,
                    'frequency': new_frequency,
                    'enabled': enabled,
                    'next_crawl': update_data.get('next_scheduled_crawl')
                }
            else:
                logger.error(f"Failed to update crawl schedule for website {website_id}")
                return {'success': False, 'error': 'Database update failed'}

        except Exception as e:
            logger.error(f"Failed to update crawl schedule for website {website_id}: {e}")
            return {'success': False, 'error': str(e)}

    def get_websites_for_scheduled_crawling(self) -> list:
        """
        Get websites that are due for scheduled crawling.

        Returns:
            List of websites that need to be crawled
        """
        try:
            current_time = datetime.now(timezone.utc)

            result = self.supabase.table('websites').select(
                'id, domain, name, scraping_frequency, next_scheduled_crawl, max_pages, max_depth'
            ).eq('scraping_enabled', True).neq(
                'scraping_frequency', 'manual'
            ).lte(
                'next_scheduled_crawl', current_time.isoformat()
            ).execute()

            websites = result.data or []
            logger.info(f"Found {len(websites)} websites due for scheduled crawling")

            return websites

        except Exception as e:
            logger.error(f"Failed to get websites for scheduled crawling: {e}")
            return []

    def trigger_initial_crawl(self, website_id: str) -> Dict[str, Any]:
        """
        Trigger an initial crawl for a newly registered website.

        This is called when a website is first verified and should be crawled immediately,
        regardless of the scheduled frequency.

        Args:
            website_id: UUID of the website to crawl

        Returns:
            Dict containing the crawl trigger result
        """
        try:
            # Get website configuration
            website_result = self.supabase.table('websites').select(
                'id, domain, max_pages, max_depth'
            ).eq('id', website_id).single().execute()

            if not website_result.data:
                return {'success': False, 'error': 'Website not found'}

            website = website_result.data
            max_pages = website.get('max_pages', 100)
            max_depth = website.get('max_depth', 3)

            # Trigger immediate crawl
            crawl_result = self.crawl_manager.trigger_manual_crawl(
                website_id=website_id,
                max_pages=max_pages,
                max_depth=max_depth
            )

            logger.info(f"Triggered initial crawl for website {website['domain']}: {crawl_result}")
            return crawl_result

        except Exception as e:
            logger.error(f"Failed to trigger initial crawl for website {website_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_next_crawl_time(self, frequency: str) -> datetime:
        """
        Calculate the next crawl time based on frequency.

        Args:
            frequency: Crawl frequency (daily, weekly, monthly)

        Returns:
            datetime object for next crawl
        """
        now = datetime.now(timezone.utc)

        if frequency == 'daily':
            return now + timedelta(days=1)
        elif frequency == 'weekly':
            return now + timedelta(weeks=1)
        elif frequency == 'monthly':
            return now + timedelta(days=30)  # Approximate month
        else:
            # Default to daily for unknown frequencies
            return now + timedelta(days=1)


# Global instance for easy access
_scheduler_instance: Optional[RegistrationSchedulerService] = None


def get_registration_scheduler() -> RegistrationSchedulerService:
    """Get the global registration scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = RegistrationSchedulerService()
    return _scheduler_instance


def setup_website_schedule(website_id: str) -> Dict[str, Any]:
    """Convenience function to setup website crawl schedule."""
    scheduler = get_registration_scheduler()
    return scheduler.setup_website_crawl_schedule(website_id)


def trigger_initial_website_crawl(website_id: str) -> Dict[str, Any]:
    """Convenience function to trigger initial website crawl."""
    scheduler = get_registration_scheduler()
    return scheduler.trigger_initial_crawl(website_id)