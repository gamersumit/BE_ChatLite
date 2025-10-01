"""
Scheduled Crawler Service for automatic website crawling.
Handles periodic crawling, incremental updates, and crawl scheduling.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID

from .crawler_service import CrawlerService
from ..core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class ScheduledCrawlerService:
    """Service for managing scheduled and automatic crawling."""
    
    def __init__(self):
        self.crawler_service = CrawlerService()
        self.supabase = SupabaseClient().client
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start_scheduler(self, check_interval: int = 300) -> None:
        """
        Start the crawler scheduler.
        
        Args:
            check_interval: How often to check for websites needing crawling (seconds)
        """
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        logger.info(f"Starting crawler scheduler with {check_interval}s check interval")
        
        while self._running:
            try:
                await self._check_and_crawl_websites()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in crawler scheduler: {e}")
                await asyncio.sleep(check_interval)
    
    def stop_scheduler(self) -> None:
        """Stop the crawler scheduler."""
        self._running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        self._tasks.clear()
        logger.info("Crawler scheduler stopped")
    
    async def _check_and_crawl_websites(self) -> None:
        """Check for websites that need crawling and start crawl tasks."""
        try:
            # Get websites that need crawling
            websites_to_crawl = await self._get_websites_needing_crawl()
            
            if not websites_to_crawl:
                logger.debug("No websites need crawling")
                return
            
            logger.info(f"Found {len(websites_to_crawl)} websites needing crawling")
            
            # Clean up completed tasks
            self._tasks = [task for task in self._tasks if not task.done()]
            
            # Start crawl tasks for websites
            for website in websites_to_crawl:
                if self._should_start_crawl(website):
                    task = asyncio.create_task(
                        self._crawl_website_background(website)
                    )
                    self._tasks.append(task)
                    
        except Exception as e:
            logger.error(f"Error checking websites for crawling: {e}")
    
    async def _get_websites_needing_crawl(self) -> List[Dict[str, Any]]:
        """Get websites that need crawling based on schedule and last crawl time."""
        try:
            # Query websites with scraping enabled
            result = self.supabase.table('websites').select(
                'id, name, url, domain, scraping_enabled, scraping_config'
            ).eq('scraping_enabled', True).execute()
            
            if not result.data:
                return []
            
            websites_needing_crawl = []
            current_time = datetime.now(timezone.utc)
            
            for website in result.data:
                website_id = website['id']
                scraping_config = website.get('scraping_config', {})
                
                # Get crawl schedule (default: daily)
                crawl_frequency_hours = scraping_config.get('crawl_frequency_hours', 24)
                
                # Check last crawl time
                last_crawl_result = self.supabase.table('scraped_websites').select(
                    'last_crawled_at, crawl_status'
                ).eq('website_id', website_id).order(
                    'created_at', desc=True
                ).limit(1).execute()
                
                should_crawl = False
                
                if not last_crawl_result.data:
                    # Never crawled before
                    should_crawl = True
                    logger.info(f"Website {website['domain']} never crawled - scheduling")
                else:
                    last_crawl = last_crawl_result.data[0]
                    crawl_status = last_crawl.get('crawl_status', 'unknown')
                    
                    # Don't start if already crawling
                    if crawl_status == 'crawling':
                        continue
                    
                    last_crawled_at = last_crawl.get('last_crawled_at')
                    if last_crawled_at:
                        last_crawl_time = datetime.fromisoformat(last_crawled_at.replace('Z', '+00:00'))
                        time_since_crawl = current_time - last_crawl_time
                        
                        if time_since_crawl >= timedelta(hours=crawl_frequency_hours):
                            should_crawl = True
                            logger.info(f"Website {website['domain']} due for crawl - {time_since_crawl} since last crawl")
                    else:
                        # Has record but no last crawl time (possibly failed)
                        should_crawl = True
                
                if should_crawl:
                    websites_needing_crawl.append(website)
            
            return websites_needing_crawl
            
        except Exception as e:
            logger.error(f"Error getting websites needing crawl: {e}")
            return []
    
    def _should_start_crawl(self, website: Dict[str, Any]) -> bool:
        """Check if we should start crawling this website now."""
        website_id = website['id']
        
        # Check if there's already a task running for this website
        for task in self._tasks:
            if not task.done() and hasattr(task, '_website_id') and task._website_id == website_id:
                logger.debug(f"Crawl already running for {website['domain']}")
                return False
        
        return True
    
    async def _crawl_website_background(self, website: Dict[str, Any]) -> None:
        """Crawl a website in the background."""
        website_id = UUID(website['id'])
        domain = website['domain']
        base_url = website['url']
        
        try:
            logger.info(f"Starting scheduled crawl for {domain}")
            
            # Store website_id on task for tracking
            current_task = asyncio.current_task()
            if current_task:
                current_task._website_id = website_id
            
            result = await self.crawler_service.start_crawl(
                website_id=website_id,
                base_url=base_url,
                domain=domain
            )
            
            logger.info(f"Scheduled crawl completed for {domain}: {result['pages_processed']} pages")
            
        except Exception as e:
            logger.error(f"Scheduled crawl failed for {domain}: {e}")
    
    async def schedule_immediate_crawl(
        self, 
        website_id: UUID, 
        force: bool = False
    ) -> bool:
        """
        Schedule an immediate crawl for a specific website.
        
        Args:
            website_id: UUID of the website to crawl
            force: If True, start crawl even if one is already running
            
        Returns:
            True if crawl was scheduled, False otherwise
        """
        try:
            # Get website details
            result = self.supabase.table('websites').select(
                'id, name, url, domain, scraping_enabled, scraping_config'
            ).eq('id', str(website_id)).execute()
            
            if not result.data:
                logger.error(f"Website {website_id} not found")
                return False
            
            website = result.data[0]
            
            if not website.get('scraping_enabled', False):
                logger.error(f"Scraping not enabled for website {website_id}")
                return False
            
            # Check if crawl is already running (unless forced)
            if not force:
                existing_crawl = self.supabase.table('scraped_websites').select(
                    'crawl_status'
                ).eq('website_id', str(website_id)).eq('crawl_status', 'crawling').execute()
                
                if existing_crawl.data:
                    logger.warning(f"Crawl already running for {website['domain']}")
                    return False
            
            # Start immediate crawl task
            task = asyncio.create_task(
                self._crawl_website_background(website)
            )
            self._tasks.append(task)
            
            logger.info(f"Immediate crawl scheduled for {website['domain']}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling immediate crawl: {e}")
            return False
    
    async def get_crawler_status(self) -> Dict[str, Any]:
        """Get current status of the crawler scheduler."""
        active_tasks = len([task for task in self._tasks if not task.done()])
        completed_tasks = len([task for task in self._tasks if task.done()])
        
        return {
            'running': self._running,
            'active_crawls': active_tasks,
            'completed_crawls': completed_tasks,
            'total_tasks': len(self._tasks)
        }
    
    async def update_crawl_schedule(
        self, 
        website_id: UUID, 
        crawl_frequency_hours: int
    ) -> bool:
        """
        Update the crawl schedule for a website.
        
        Args:
            website_id: UUID of the website
            crawl_frequency_hours: How often to crawl (in hours)
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Get current config
            result = self.supabase.table('websites').select(
                'scraping_config'
            ).eq('id', str(website_id)).execute()
            
            if not result.data:
                return False
            
            current_config = result.data[0].get('scraping_config', {})
            current_config['crawl_frequency_hours'] = crawl_frequency_hours
            
            # Update config
            update_result = self.supabase.table('websites').update({
                'scraping_config': current_config,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', str(website_id)).execute()
            
            return len(update_result.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating crawl schedule: {e}")
            return False


# Global scheduler instance
_scheduler_instance: Optional[ScheduledCrawlerService] = None


def get_scheduler() -> ScheduledCrawlerService:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ScheduledCrawlerService()
    return _scheduler_instance


async def start_global_scheduler():
    """Start the global crawler scheduler."""
    scheduler = get_scheduler()
    await scheduler.start_scheduler()


def stop_global_scheduler():
    """Stop the global crawler scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop_scheduler()
        _scheduler_instance = None