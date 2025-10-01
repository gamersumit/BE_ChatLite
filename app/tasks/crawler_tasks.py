"""
Celery tasks for website crawling and content processing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from celery import Task
from celery.exceptions import Retry
from datetime import datetime, timezone

from ..core.celery_config import celery_app
from ..services.crawler_service import CrawlerService
from ..services.spa_crawler import SPACrawler, crawl_spa_website
from ..services.vector_search_service import VectorSearchService

logger = logging.getLogger(__name__)


class CrawlerTask(Task):
    """Base class for crawler tasks with error handling and retries."""

    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {self.name} failed: {exc}", extra={
            'task_id': task_id,
            'args': args,
            'kwargs': kwargs,
            'exception': str(exc)
        })

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {self.name} retrying: {exc}", extra={
            'task_id': task_id,
            'retry_count': self.request.retries
        })

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {self.name} completed successfully", extra={
            'task_id': task_id,
            'result': str(retval)[:100]  # First 100 chars of result
        })


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.crawl_url')
def crawl_url(self, job_id: str = None, website_id: str = None, url: str = None, max_pages: int = 100, max_depth: int = 3) -> Dict[str, Any]:
    """
    Crawl a website starting from the given URL.

    Args:
        job_id: ID of the crawling job (optional for backward compatibility)
        website_id: UUID of the website to crawl
        url: Starting URL for crawling
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth

    Returns:
        Dict containing crawl results and statistics
    """
    try:
        # If job_id is provided, update the crawling_jobs table
        supabase = None
        if job_id:
            try:
                from ..core.supabase_client import get_supabase_admin
                supabase = get_supabase_admin()

                # Update job status to running
                supabase.table('crawling_jobs').update({
                    'status': 'running',
                    'started_at': datetime.now(timezone.utc).isoformat()
                }).eq('id', job_id).execute()
                logger.info(f"Updated job {job_id} status to running")
            except Exception as e:
                logger.warning(f"Failed to update job status to running: {e}")
                # Continue without failing the task

        # Only update Celery state if we have a task ID
        if hasattr(self, 'request') and getattr(self.request, 'id', None):
            self.update_state(state='PROGRESS', meta={'status': 'Starting crawl', 'progress': 0})

        # Update job progress in database
        if job_id and supabase:
            try:
                supabase.table('crawling_jobs').update({
                    'crawl_metrics': {
                        'status': 'Starting crawl',
                        'progress': 0,
                        'pages_found': 0,
                        'pages_processed': 0,
                        'estimated_total': max_pages
                    }
                }).eq('id', job_id).execute()
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

        # Detect if website is a SPA and choose appropriate crawler
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        config_override = {
            'max_pages': max_pages,
            'max_depth': max_depth,
            'user_agent': 'ChatLite-Crawler/1.0'
        }

        # Check if it's a SPA website
        is_spa = False
        try:
            async def check_spa():
                async with SPACrawler(config_override) as crawler:
                    return await crawler.is_spa_website(url)

            is_spa = asyncio.run(check_spa())
            logger.info(f"✅ SPA Detection: Website {url} detected as {'SPA' if is_spa else 'static HTML'}")
        except Exception as e:
            logger.error(f"❌ SPA Detection Failed: Could not detect if website is SPA, using default crawler: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

        # Use appropriate crawler based on detection
        if is_spa:
            logger.info(f"Using SPA crawler for {url}")
            # Use the SPA crawler for JavaScript-rendered sites
            result = asyncio.run(crawl_spa_website(website_id, url, config_override))
        else:
            logger.info(f"Using standard crawler for {url}")
            # Use standard crawler for static sites
            crawler_service = CrawlerService(use_admin_client=True)
            result = asyncio.run(crawler_service.start_crawl(
                website_id=UUID(website_id),
                base_url=url,
                domain=domain,
                config_override=config_override
            ))

        # Calculate real progress based on actual results
        pages_found = result.get('pages_found', 0)
        pages_processed = result.get('pages_processed', 0)

        # Calculate progress percentage based on pages processed vs max_pages
        if max_pages > 0:
            # Base progress on pages processed relative to intended max_pages
            crawl_progress = min(100, int((pages_processed / max_pages) * 100))
        else:
            crawl_progress = 100 if pages_processed > 0 else 0

        # Add bonus progress for storage phase
        storage_progress = min(100, crawl_progress + 10)  # Add 10% for storage processing

        # Only update Celery state if we have a task ID
        if hasattr(self, 'request') and getattr(self.request, 'id', None):
            self.update_state(state='PROGRESS', meta={'status': 'Processing content', 'progress': storage_progress})

        # Update job progress in database with actual crawl results
        if job_id and supabase:
            try:
                supabase.table('crawling_jobs').update({
                    'crawl_metrics': {
                        'status': 'Processing content',
                        'progress': storage_progress,
                        'pages_found': pages_found,
                        'pages_processed': pages_processed,
                        'estimated_total': max_pages,
                        'crawl_time': result.get('crawl_time', 0)
                    }
                }).eq('id', job_id).execute()
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

        # Content is already stored in Supabase by the crawler
        # No local storage needed - vector embeddings will be generated on-demand
        logger.info(f"✅ Crawl completed. Content stored in Supabase for website_id: {website_id}")

        # Don't pass large data through Redis - content is already in Supabase
        # Processing tasks should query Supabase directly instead of passing data through Redis
        pages_count = result.get('pages_processed', 0)
        if pages_count > 0:
            # Pass only the website_id, not the actual page data
            process_crawled_content.delay(website_id, pages_count)

        # Only update Celery state if we have a task ID
        if hasattr(self, 'request') and getattr(self.request, 'id', None):
            self.update_state(state='PROGRESS', meta={'status': 'Content stored', 'progress': 100})

        # Update job progress to completed state
        if job_id and supabase:
            try:
                final_metrics = {
                    'status': 'Content stored',
                    'progress': 100,
                    'pages_found': result.get('pages_found', 0),
                    'pages_processed': result.get('pages_processed', 0),
                    'estimated_total': max_pages,
                    'crawl_time': result.get('crawl_time', 0)
                }
                supabase.table('crawling_jobs').update({
                    'crawl_metrics': final_metrics
                }).eq('id', job_id).execute()
            except Exception as e:
                logger.warning(f"Failed to update final job progress: {e}")

        # Update job status to completed if job_id is provided
        if job_id and supabase:
            try:
                # Calculate actual crawl time
                from datetime import datetime, timezone
                if hasattr(self, 'request') and getattr(self.request, 'id', None):
                    # Get job start time from database
                    job_data = supabase.table('crawling_jobs').select('started_at').eq('id', job_id).single().execute()
                    if job_data.data and job_data.data.get('started_at'):
                        start_time = datetime.fromisoformat(job_data.data['started_at'].replace('Z', '+00:00'))
                        end_time = datetime.now(timezone.utc)
                        actual_crawl_time = round((end_time - start_time).total_seconds(), 1)
                    else:
                        actual_crawl_time = result.get('crawl_time', 0)
                else:
                    actual_crawl_time = result.get('crawl_time', 0)

                crawl_metrics = {
                    'pages_crawled': result.get('pages_processed', 0),
                    'pages_processed': result.get('pages_processed', 0),
                    'pages_found': result.get('pages_found', 0),
                    'crawl_time': actual_crawl_time
                }
                supabase.table('crawling_jobs').update({
                    'status': 'completed',
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'crawl_metrics': crawl_metrics
                }).eq('id', job_id).execute()
                logger.info(f"Updated job {job_id} status to completed")
            except Exception as e:
                logger.warning(f"Failed to update job status to completed: {e}")

        return {
            'status': 'completed',
            'website_id': website_id,
            'pages_crawled': result.get('pages_processed', 0),
            'pages_processed': result.get('pages_processed', 0),
            'crawl_time': result.get('crawl_time', 0),
            'errors': result.get('errors', []),
            'storage_enabled': True
        }

    except Exception as exc:
        logger.error(f"Crawl task failed for website {website_id}: {exc}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Update job status to failed if job_id is provided
        if job_id and supabase:
            try:
                supabase.table('crawling_jobs').update({
                    'status': 'failed',
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'error_message': str(exc)
                }).eq('id', job_id).execute()
                logger.info(f"Updated job {job_id} status to failed")
            except Exception as update_error:
                logger.error(f"Failed to update job status: {update_error}")

        # Only update Celery state if we have a task ID
        if hasattr(self, 'request') and getattr(self.request, 'id', None):
            self.update_state(
                state='FAILURE',
                meta={'error': str(exc), 'website_id': website_id}
            )
            raise self.retry(exc=exc)
        else:
            # When called directly (not from Celery), just re-raise
            raise


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.process_data')
def process_crawled_content(self, website_id: str, pages_count: int) -> Dict[str, Any]:
    """
    Process crawled content and generate embeddings for vector search.

    Note: This task queries Supabase directly instead of receiving data through Redis
    to avoid memory issues with large crawl results.

    Args:
        website_id: UUID of the website
        pages_count: Number of pages that were crawled

    Returns:
        Dict containing processing results
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Fetching pages from database', 'progress': 0})

        # Query Supabase for recently crawled pages instead of passing through Redis
        from ..core.supabase_client import get_supabase_admin
        supabase = get_supabase_admin()

        # First, get the scraped_website record for this website_id
        # website_id is from websites.id, we need scraped_websites.id
        scraped_website_result = supabase.table('scraped_websites').select('id').eq(
            'website_id', website_id
        ).order('created_at', desc=True).limit(1).execute()

        if not scraped_website_result.data:
            logger.warning(f"No scraped_website record found for website_id: {website_id}")
            return {'success': False, 'processed': 0, 'message': 'No scraped_website record found'}

        scraped_website_id = scraped_website_result.data[0]['id']
        logger.info(f"Found scraped_website_id: {scraped_website_id} for website_id: {website_id}")

        # Get recent pages for this scraped_website
        pages_result = supabase.table('scraped_pages').select('id').eq(
            'scraped_website_id', scraped_website_id
        ).order('scraped_at', desc=True).limit(pages_count).execute()

        if not pages_result.data:
            logger.warning(f"No pages found for scraped_website_id: {scraped_website_id}")
            return {'success': False, 'processed': 0, 'message': 'No pages to process'}

        page_ids = [page['id'] for page in pages_result.data]
        processed_count = 0
        total_pages = len(page_ids)

        logger.info(f"Processing {total_pages} pages for website_id: {website_id}")

        vector_service = VectorSearchService()

        for i, page_id in enumerate(page_ids):
            try:
                # Fetch page content from database
                page_data = supabase.table('scraped_pages').select('content_text, url, title').eq('id', page_id).single().execute()

                if not page_data.data or not page_data.data.get('content_text'):
                    logger.warning(f"No content found for page {page_id}")
                    continue

                content_text = page_data.data['content_text']
                logger.info(f"Processing page {page_data.data.get('url', 'unknown')}: {len(content_text)} chars")

                # Generate embeddings for page content (async function, needs asyncio.run)
                chunks = asyncio.run(vector_service.chunk_and_embed_content(
                    page_id=UUID(page_id),
                    content=content_text,
                    website_id=website_id,
                    chunk_size=1000
                ))
                processed_count += 1

                # Update progress
                progress = int((i + 1) / total_pages * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Processing page {i+1}/{total_pages}', 'progress': progress}
                )

            except Exception as e:
                logger.error(f"Failed to process page {page_id}: {e}")
                continue

        return {
            'status': 'completed',
            'website_id': website_id,
            'total_pages': total_pages,
            'processed_pages': processed_count,
            'failed_pages': total_pages - processed_count
        }

    except Exception as exc:
        logger.error(f"Content processing failed for website {website_id}: {exc}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'website_id': website_id}
        )
        raise self.retry(exc=exc)


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.schedule_crawl')
def schedule_crawl(self, website_id: str, crawl_frequency: str) -> Dict[str, Any]:
    """
    Schedule a crawl for a website based on its frequency setting.

    Args:
        website_id: UUID of the website
        crawl_frequency: Frequency setting (daily, weekly, monthly)

    Returns:
        Dict containing schedule result
    """
    try:
        # This will be expanded to handle scheduling logic
        logger.info(f"Scheduling crawl for website {website_id} with frequency {crawl_frequency}")

        return {
            'status': 'scheduled',
            'website_id': website_id,
            'frequency': crawl_frequency,
            'next_crawl': None  # Will be calculated based on frequency
        }

    except Exception as exc:
        logger.error(f"Schedule crawl failed for website {website_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.generate_embeddings')
def generate_embeddings(self, page_id: str, content: str) -> Dict[str, Any]:
    """
    Generate embeddings for a specific page content.

    Args:
        page_id: UUID of the page
        content: Text content to generate embeddings for

    Returns:
        Dict containing embedding generation result
    """
    try:
        vector_service = VectorSearchService()

        # Generate embedding
        embedding = vector_service.generate_embedding(content)

        # Store embedding (this would integrate with your database)
        # For now, just return success

        return {
            'status': 'completed',
            'page_id': page_id,
            'embedding_dimensions': len(embedding),
            'content_length': len(content)
        }

    except Exception as exc:
        logger.error(f"Embedding generation failed for page {page_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.update_knowledge_base')
def update_knowledge_base(self, website_id: str, updated_pages: List[str]) -> Dict[str, Any]:
    """
    Update the knowledge base with newly crawled content.

    Args:
        website_id: UUID of the website
        updated_pages: List of page IDs that were updated

    Returns:
        Dict containing knowledge base update result
    """
    try:
        # This task will handle updating the knowledge base
        # with new content and ensuring it's available for RAG

        logger.info(f"Updating knowledge base for website {website_id}")

        return {
            'status': 'completed',
            'website_id': website_id,
            'updated_pages': len(updated_pages),
            'knowledge_base_updated': True
        }

    except Exception as exc:
        logger.error(f"Knowledge base update failed for website {website_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.monitor_schedule_changes')
def monitor_schedule_changes(self) -> Dict[str, Any]:
    """
    Monitor for schedule changes in the database and update Celery Beat accordingly.

    This task runs periodically to detect changes in website scheduling configurations
    and updates the automated scheduler.

    Returns:
        Dict containing monitoring result
    """
    try:
        logger.info("Monitoring schedule changes...")

        from app.services.automated_scheduler import get_automated_scheduler

        # Get the automated scheduler
        scheduler = get_automated_scheduler()

        # Trigger a manual schedule check
        result = scheduler.trigger_manual_schedule_check()

        logger.info(f"Schedule monitoring complete: {result}")
        return result

    except Exception as exc:
        logger.error(f"Schedule monitoring failed: {exc}")
        raise self.retry(exc=exc, countdown=300, max_retries=2)  # Retry after 5 minutes


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.cleanup_old_crawl_data')
def cleanup_old_crawl_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old crawl data to maintain database performance.

    Args:
        days_to_keep: Number of days of data to retain

    Returns:
        Dict containing cleanup result
    """
    try:
        from datetime import datetime, timezone, timedelta
        from app.core.database import get_supabase_admin_client

        logger.info(f"Cleaning up crawl data older than {days_to_keep} days...")

        supabase = get_supabase_admin_client()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        # Clean up old scraped pages (keep recent ones for reference)
        pages_result = supabase.table('scraped_pages').delete().lt(
            'scraped_at', cutoff_date.isoformat()
        ).execute()

        pages_deleted = len(pages_result.data) if pages_result.data else 0

        # Clean up old content chunks that are no longer referenced
        # This would cascade from page deletions due to foreign key constraints

        logger.info(f"Cleanup complete: {pages_deleted} old pages removed")

        return {
            'status': 'completed',
            'pages_deleted': pages_deleted,
            'cutoff_date': cutoff_date.isoformat(),
            'cleanup_completed_at': datetime.now(timezone.utc).isoformat()
        }

    except Exception as exc:
        logger.error(f"Cleanup failed: {exc}")
        raise self.retry(exc=exc, countdown=3600, max_retries=1)  # Retry after 1 hour


@celery_app.task(bind=True, base=CrawlerTask, name='crawler.tasks.health_check_websites')
def health_check_websites(self) -> Dict[str, Any]:
    """
    Perform health checks on registered websites to ensure they're accessible.

    Returns:
        Dict containing health check results
    """
    try:
        import requests
        from datetime import datetime, timezone
        from app.core.database import get_supabase_admin_client

        logger.info("Performing website health checks...")

        supabase = get_supabase_admin_client()

        # Get active websites
        websites_result = supabase.table('websites').select(
            'id, domain, url, name'
        ).eq('is_active', True).execute()

        websites = websites_result.data or []
        health_results = []

        for website in websites:
            website_id = website['id']
            domain = website['domain']
            url = website.get('url', f"https://{domain}")

            try:
                # Perform basic HTTP check
                response = requests.head(url, timeout=10, allow_redirects=True)
                status_code = response.status_code
                is_healthy = 200 <= status_code < 400

                health_results.append({
                    'website_id': website_id,
                    'domain': domain,
                    'status_code': status_code,
                    'is_healthy': is_healthy,
                    'checked_at': datetime.now(timezone.utc).isoformat()
                })

                # Update website health status in database
                supabase.table('websites').update({
                    'last_health_check': datetime.now(timezone.utc).isoformat(),
                    'is_healthy': is_healthy
                }).eq('id', website_id).execute()

            except Exception as e:
                logger.warning(f"Health check failed for {domain}: {e}")
                health_results.append({
                    'website_id': website_id,
                    'domain': domain,
                    'status_code': None,
                    'is_healthy': False,
                    'error': str(e),
                    'checked_at': datetime.now(timezone.utc).isoformat()
                })

        healthy_count = sum(1 for r in health_results if r['is_healthy'])

        logger.info(f"Health check complete: {healthy_count}/{len(health_results)} websites healthy")

        return {
            'status': 'completed',
            'total_websites': len(health_results),
            'healthy_websites': healthy_count,
            'unhealthy_websites': len(health_results) - healthy_count,
            'results': health_results,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }

    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        raise self.retry(exc=exc, countdown=1800, max_retries=2)  # Retry after 30 minutes