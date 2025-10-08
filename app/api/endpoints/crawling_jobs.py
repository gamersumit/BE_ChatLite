"""
API endpoints for streaming crawling jobs with real-time progress tracking.
Implements the memory-optimized crawl architecture.
"""
import logging
from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...core.supabase_client import get_supabase_admin
from ...core.auth_middleware import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models

class CreateCrawlingJobRequest(BaseModel):
    """Request model for creating a new crawling job."""
    website_id: UUID
    base_url: str = Field(..., description="Base URL to start crawling from")
    max_pages: int = Field(default=100, description="Maximum pages to crawl")
    max_depth: int = Field(default=3, description="Maximum crawl depth")


class CreateCrawlingJobResponse(BaseModel):
    """Response model for created crawling job."""
    job_id: str
    status: str
    website_id: str
    base_url: str
    celery_task_id: Optional[str] = None


class CrawlingJobProgressResponse(BaseModel):
    """Response model for job progress."""
    job_id: str
    status: str
    pages_queued: int
    pages_processing: int
    pages_completed: int
    pages_failed: int
    total_pages: int
    total_discovered: int
    pages_processed: int  # completed + failed
    max_pages: int
    progress_percentage: float
    estimated_time_remaining: Optional[int] = None  # seconds
    current_page_url: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class StorePageRequest(BaseModel):
    """Request model for storing a page from worker."""
    scraped_website_id: str
    url: str
    title: Optional[str] = None
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    meta_description: Optional[str] = None
    word_count: int = 0
    depth_level: int = 0
    status_code: Optional[int] = None
    processing_status: str = 'completed'
    links_discovered: int = 0
    crawling_job_id: Optional[str] = None


class StorePageResponse(BaseModel):
    """Response model for stored page."""
    page_id: str
    url: str


class UpdateJobProgressRequest(BaseModel):
    """Request model for updating job progress."""
    job_id: str
    current_page_url: Optional[str] = None
    pages_completed: Optional[int] = None
    pages_queued: Optional[int] = None
    pages_processing: Optional[int] = None
    pages_failed: Optional[int] = None


class CancelJobResponse(BaseModel):
    """Response model for job cancellation."""
    job_id: str
    status: str
    message: str


# API Endpoints

@router.post("/crawling-jobs", response_model=CreateCrawlingJobResponse)
async def create_crawling_job(
    request: CreateCrawlingJobRequest
) -> CreateCrawlingJobResponse:
    """
    Create a new crawling job and enqueue it to Celery worker.

    This endpoint:
    1. Validates the website exists and scraping is enabled
    2. Creates a crawling_jobs record with status 'pending'
    3. Enqueues the streaming crawl task to Celery
    4. Returns the job ID and initial status
    """
    try:
        supabase = get_supabase_admin()

        # Validate website exists and scraping is enabled
        logger.info(f"Creating crawling job for website_id: {request.website_id}")

        website_result = supabase.table('websites').select(
            'id, name, domain, scraping_enabled'
        ).eq('id', str(request.website_id)).execute()

        if not website_result.data:
            raise HTTPException(status_code=404, detail="Website not found")

        website = website_result.data[0]
        if not website.get('scraping_enabled', False):
            raise HTTPException(
                status_code=400,
                detail="Scraping not enabled for this website"
            )

        # Check for existing active crawl
        existing_job = supabase.table('crawling_jobs').select(
            'id, status'
        ).eq('website_id', str(request.website_id)).in_(
            'status', ['pending', 'running']
        ).execute()

        if existing_job.data:
            raise HTTPException(
                status_code=409,
                detail="Active crawl already in progress for this website"
            )

        # Create crawling_jobs record
        job_data = {
            'website_id': str(request.website_id),
            'config': {
                'url': request.base_url,
                'max_pages': request.max_pages,
                'max_depth': request.max_depth
            },
            'max_pages': request.max_pages,
            'max_depth': request.max_depth,
            'status': 'pending',
            'pages_queued': 0,
            'pages_processing': 0,
            'pages_completed': 0,
            'pages_failed': 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table('crawling_jobs').insert(job_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create crawling job")

        job_id = result.data[0]['id']

        # Enqueue Celery task (streaming crawl)
        celery_task_id = None
        try:
            from ...core.celery_client import get_celery_app

            celery_app = get_celery_app()
            task = celery_app.send_task(
                'crawler.tasks.crawl_url_streaming',
                args=[job_id, str(request.website_id), request.base_url, request.max_pages, request.max_depth],
                queue='crawl_queue'
            )
            celery_task_id = task.id

            # Update job with Celery task ID
            supabase.table('crawling_jobs').update({
                'celery_task_id': celery_task_id
            }).eq('id', job_id).execute()

            logger.info(f"Enqueued streaming crawl task: {celery_task_id} for job: {job_id}")

        except Exception as e:
            logger.error(f"Failed to enqueue Celery task: {e}")
            # Update job status to failed
            supabase.table('crawling_jobs').update({
                'status': 'failed',
                'error_message': f'Failed to enqueue task: {str(e)}'
            }).eq('id', job_id).execute()

            raise HTTPException(
                status_code=500,
                detail=f"Failed to enqueue crawl task: {str(e)}"
            )

        return CreateCrawlingJobResponse(
            job_id=job_id,
            status='pending',
            website_id=str(request.website_id),
            base_url=request.base_url,
            celery_task_id=celery_task_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create crawling job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create crawling job: {str(e)}")


@router.get("/crawling-jobs/{job_id}/progress", response_model=CrawlingJobProgressResponse)
async def get_crawling_job_progress(job_id: UUID) -> CrawlingJobProgressResponse:
    """
    Get real-time progress for a crawling job.

    Returns:
    - Job status (pending, running, completed, failed, cancelled)
    - Progress metrics (pages queued, processing, completed, failed)
    - Progress percentage
    - Current page being crawled
    - Timestamps
    """
    try:
        supabase = get_supabase_admin()

        # Get job details with progress metrics
        job_result = supabase.table('crawling_jobs').select(
            'id, status, pages_queued, pages_processing, pages_completed, pages_failed, '
            'current_page_url, started_at, completed_at, error_message, max_pages'
        ).eq('id', str(job_id)).execute()

        if not job_result.data:
            raise HTTPException(status_code=404, detail="Crawling job not found")

        job = job_result.data[0]

        # Calculate totals and progress
        max_pages_limit = job.get('max_pages', 100)
        pages_completed = job.get('pages_completed', 0)
        pages_failed = job.get('pages_failed', 0)
        pages_processing = job.get('pages_processing', 0)
        pages_queued = job.get('pages_queued', 0)

        # Pages processed (finished)
        pages_processed = pages_completed + pages_failed

        # Total discovered (should not exceed max_pages in queue + processed)
        discovered_total = pages_queued + pages_processing + pages_processed

        # Determine total_pages for display
        # If discovered < max_pages, show discovered as total
        # If discovered >= max_pages, show max_pages as total
        if discovered_total <= max_pages_limit:
            total_pages = discovered_total if discovered_total > 0 else max_pages_limit
            progress_base = discovered_total
        else:
            total_pages = max_pages_limit
            progress_base = max_pages_limit

        # Progress percentage
        if progress_base > 0:
            progress_percentage = (pages_processed / progress_base) * 100
        else:
            progress_percentage = 0.0

        # Estimated time remaining
        estimated_time_remaining = None
        started_at = job.get('started_at')
        if started_at and pages_processed > 0 and job['status'] == 'running':
            from datetime import datetime, timezone
            import dateutil.parser
            start_time = dateutil.parser.parse(started_at)
            elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            avg_time_per_page = elapsed_seconds / pages_processed
            pages_remaining = progress_base - pages_processed
            estimated_time_remaining = int(avg_time_per_page * pages_remaining)

        return CrawlingJobProgressResponse(
            job_id=str(job_id),
            status=job['status'],
            pages_queued=pages_queued,
            pages_processing=pages_processing,
            pages_completed=pages_completed,
            pages_failed=pages_failed,
            total_pages=total_pages,
            total_discovered=discovered_total,
            pages_processed=pages_processed,
            max_pages=max_pages_limit,
            progress_percentage=round(progress_percentage, 2),
            estimated_time_remaining=estimated_time_remaining,
            current_page_url=job.get('current_page_url'),
            started_at=started_at,
            completed_at=job.get('completed_at'),
            error_message=job.get('error_message')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job progress: {str(e)}")


@router.post("/crawling-jobs/{job_id}/pages", response_model=StorePageResponse)
async def store_page_for_job(
    job_id: UUID,
    request: StorePageRequest
) -> StorePageResponse:
    """
    Store a page from the Celery worker during streaming crawl.

    This endpoint is called by the worker after extracting each page.
    It stores the page with processing status and links discovered.
    Database triggers will automatically update the crawling_job progress.
    """
    try:
        supabase = get_supabase_admin()

        # Verify job exists
        job_result = supabase.table('crawling_jobs').select('id').eq('id', str(job_id)).execute()
        if not job_result.data:
            raise HTTPException(status_code=404, detail="Crawling job not found")

        # Store page with new optimization fields
        page_data = {
            'scraped_website_id': request.scraped_website_id,
            'url': request.url,
            'title': request.title,
            'content_text': request.content_text,
            'content_html': request.content_html,
            'meta_description': request.meta_description,
            'word_count': request.word_count,
            'depth_level': request.depth_level,
            'status_code': request.status_code,
            'processing_status': request.processing_status,
            'links_discovered': request.links_discovered,
            'crawling_job_id': str(job_id),
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'scraped_at': datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table('scraped_pages').insert(page_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to store page")

        page_id = result.data[0]['id']

        logger.info(f"Stored page {request.url} for job {job_id}")

        return StorePageResponse(
            page_id=page_id,
            url=request.url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store page: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store page: {str(e)}")


@router.post("/crawling-jobs/update-progress")
async def update_job_progress(request: UpdateJobProgressRequest):
    """
    Update crawling job progress metrics.

    This endpoint is called by the worker to update progress metrics.
    Note: Database triggers also auto-update some metrics when pages are stored.
    """
    try:
        supabase = get_supabase_admin()

        update_data = {}

        if request.current_page_url is not None:
            update_data['current_page_url'] = request.current_page_url

        if request.pages_completed is not None:
            update_data['pages_completed'] = request.pages_completed

        if request.pages_queued is not None:
            update_data['pages_queued'] = request.pages_queued

        if request.pages_processing is not None:
            update_data['pages_processing'] = request.pages_processing

        if request.pages_failed is not None:
            update_data['pages_failed'] = request.pages_failed

        if not update_data:
            raise HTTPException(status_code=400, detail="No progress data provided")

        result = supabase.table('crawling_jobs').update(update_data).eq('id', request.job_id).execute()

        logger.debug(f"Updated job {request.job_id} progress")

        return {"success": True, "job_id": request.job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update job progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update job progress: {str(e)}")


@router.patch("/crawling-jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_crawling_job(job_id: UUID) -> CancelJobResponse:
    """
    Cancel a running crawling job.

    This endpoint:
    1. Updates the job status to 'cancelled'
    2. Attempts to revoke the Celery task if it's still running
    3. Returns success confirmation
    """
    try:
        supabase = get_supabase_admin()

        # Get job details
        job_result = supabase.table('crawling_jobs').select(
            'id, status, celery_task_id'
        ).eq('id', str(job_id)).execute()

        if not job_result.data:
            raise HTTPException(status_code=404, detail="Crawling job not found")

        job = job_result.data[0]

        # Check if job can be cancelled
        if job['status'] in ['completed', 'failed', 'cancelled']:
            return CancelJobResponse(
                job_id=str(job_id),
                status=job['status'],
                message=f"Job already {job['status']}"
            )

        # Update job status to cancelled
        update_data = {
            'status': 'cancelled',
            'completed_at': datetime.now(timezone.utc).isoformat()
        }

        supabase.table('crawling_jobs').update(update_data).eq('id', str(job_id)).execute()

        # Attempt to revoke Celery task
        celery_task_id = job.get('celery_task_id')
        if celery_task_id:
            try:
                from ...core.celery_client import get_celery_app

                celery_app = get_celery_app()
                celery_app.control.revoke(celery_task_id, terminate=True)

                logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {celery_task_id}: {e}")

        return CancelJobResponse(
            job_id=str(job_id),
            status='cancelled',
            message='Crawling job cancelled successfully'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")
