"""
API endpoints for manual crawling operations.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from ...core.auth_middleware import get_current_user
from ...core.config import settings
from ...api.deps.database import get_db
from ...services.crawl_manager import CrawlManager
from ...core.supabase_client import get_supabase_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class ManualCrawlRequest(BaseModel):
    """Request model for manual crawl trigger."""
    website_id: str = Field(..., description="Website ID to crawl")
    max_pages: int = Field(default=settings.default_max_pages, description="Maximum pages to crawl", ge=1, le=settings.max_pages_per_crawl)
    max_depth: int = Field(default=settings.default_crawl_depth, description="Maximum crawl depth", ge=1, le=10)


class CrawlStatusResponse(BaseModel):
    """Response model for crawl status."""
    status: str = Field(..., description="Crawl status")
    last_crawled: str = Field(None, description="Last crawled timestamp")
    pages_crawled: int = Field(default=0, description="Number of pages crawled")
    last_success: str = Field(None, description="Last successful crawl timestamp")
    next_scheduled: str = Field(None, description="Next scheduled crawl time")


class CrawlHistoryEntry(BaseModel):
    """Model for crawl history entry."""
    crawl_id: str = Field(..., description="Crawl ID")
    started_at: str = Field(..., description="Crawl start time")
    completed_at: str = Field(None, description="Crawl completion time")
    status: str = Field(..., description="Crawl status")
    pages_crawled: int = Field(default=0, description="Number of pages crawled")
    trigger_type: str = Field(..., description="manual or scheduled")
    error_message: str = Field(None, description="Error message if failed")


class CrawlHistoryResponse(BaseModel):
    """Response model for crawl history."""
    history: List[CrawlHistoryEntry] = Field(default_factory=list, description="Crawl history entries")
    total: int = Field(default=0, description="Total number of history entries")
    has_more: bool = Field(default=False, description="Whether there are more entries")


# ==================== Celery Worker API Endpoints ====================
# These endpoints are called by the celery worker to store crawl data

class InitScrapedWebsiteRequest(BaseModel):
    """Request to initialize a scraped_website record."""
    website_id: str
    domain: str
    base_url: str
    max_pages: int = 100
    crawl_depth: int = 3


class InitScrapedWebsiteResponse(BaseModel):
    """Response with scraped_website_id."""
    scraped_website_id: str
    website_id: str


class StoreCrawledPageRequest(BaseModel):
    """Request to store a crawled page."""
    scraped_website_id: str
    url: str
    title: str = None
    content_text: str = None
    content_html: str = None
    meta_description: str = None
    status_code: int = None
    depth_level: int = 0


class StoreCrawledPageResponse(BaseModel):
    """Response after storing page."""
    page_id: str
    scraped_website_id: str
    url: str


class UpdateJobStatusRequest(BaseModel):
    """Request to update job status."""
    job_id: str
    status: str
    crawl_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ProcessEmbeddingsRequest(BaseModel):
    """Request to process embeddings."""
    website_id: str
    pages_count: int


class ProcessEmbeddingsResponse(BaseModel):
    """Response after processing embeddings."""
    status: str
    website_id: str
    pages_processed: int = 0
    chunks_created: int = 0


@router.post("/trigger")
async def trigger_manual_crawl(
    request: ManualCrawlRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger a manual crawl for a website.
    """
    try:
        from datetime import datetime, timezone
        import uuid

        # Use service role client to bypass RLS
        supabase = get_supabase_admin()

        # Get the website (without ownership check as websites table doesn't have user_id)
        website_result = supabase.table('websites').select('*').eq(
            'id', request.website_id
        ).single().execute()

        if not website_result.data:
            raise HTTPException(status_code=404, detail="Website not found")

        website = website_result.data

        # Check if website is active
        if not website.get('is_active', True):
            raise HTTPException(
                status_code=403,
                detail="Cannot crawl inactive website. Please activate the website first."
            )

        base_url = website.get('website_url') or website.get('url')

        # Find or create corresponding scraped_website entry
        scraped_website_result = supabase.table('scraped_websites').select('*').eq(
            'website_id', request.website_id
        ).execute()

        if scraped_website_result.data:
            scraped_website_id = scraped_website_result.data[0]['id']
        else:
            # Create a new scraped_website entry
            from urllib.parse import urlparse
            domain = urlparse(base_url).netloc

            new_scraped_website = {
                "id": str(uuid.uuid4()),
                "website_id": request.website_id,
                "domain": domain,
                "base_url": base_url,
                "crawl_status": "pending",
                "total_pages_found": 0,
                "pages_processed": 0,
                "crawl_depth": request.max_depth,
                "max_pages": request.max_pages
            }

            scraped_result = supabase.table('scraped_websites').insert(new_scraped_website).execute()
            if scraped_result.data:
                scraped_website_id = scraped_result.data[0]['id']
            else:
                raise HTTPException(status_code=500, detail="Failed to create scraped website entry")

        # Check for existing running crawls to prevent concurrent crawls
        existing_jobs = supabase.table('crawling_jobs').select('id, status').eq(
            'website_id', request.website_id
        ).in_('status', ['pending', 'queued', 'running']).execute()

        if existing_jobs.data:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "A crawl is already in progress for this website",
                    "existing_job_id": existing_jobs.data[0]['id'],
                    "status": existing_jobs.data[0]['status']
                }
            )

        # Create crawl job in crawling_jobs table
        crawl_job_id = str(uuid.uuid4())
        crawl_job = {
            'id': crawl_job_id,
            'website_id': request.website_id,
            'user_id': current_user["id"],
            'job_type': 'manual_crawl',
            'status': 'pending',
            'config': {
                'max_pages': request.max_pages,
                'max_depth': request.max_depth,
                'url': base_url
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        # Insert crawl job
        job_result = supabase.table('crawling_jobs').insert(crawl_job).execute()
        if not job_result.data:
            raise HTTPException(status_code=500, detail="Failed to create crawl job")

        # Use Celery for reliable task execution
        from ...tasks.crawler_tasks import crawl_url

        # Dispatch to Celery worker
        task = crawl_url.delay(
            job_id=crawl_job_id,
            website_id=request.website_id,
            url=base_url,
            max_pages=request.max_pages,
            max_depth=request.max_depth
        )

        execution_method = "celery"
        task_id = task.id

        logger.info(f"Successfully dispatched crawl task {task_id} for job {crawl_job_id}")

        return {
            "success": True,
            "data": {
                "crawl_id": crawl_job_id,
                "task_id": task_id,
                "status": "running",
                "execution_method": execution_method,
                "estimated_duration": 5,
                "max_pages_limit": request.max_pages,  # The limit setting
                "pages_to_crawl": "TBD"  # Will be determined during crawl
            }
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Manual crawl trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{crawl_id}")
async def get_crawl_status(
    crawl_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the current status and progress of a crawl job.
    """
    try:
        # Use service role client to bypass RLS
        supabase = get_supabase_admin()

        # Get the crawl job
        job_result = supabase.table('crawling_jobs').select('*').eq(
            'id', crawl_id
        ).single().execute()

        if not job_result.data:
            raise HTTPException(status_code=404, detail="Crawl job not found")

        job = job_result.data
        crawl_metrics = job.get('crawl_metrics', {})

        # Build progress data
        progress_data = {
            "crawl_id": crawl_id,
            "status": job.get('status', 'unknown'),
            "progress_percentage": crawl_metrics.get('progress', 0),
            "pages_found": crawl_metrics.get('pages_found', 0),
            "pages_processed": crawl_metrics.get('pages_processed', 0),
            "pages_remaining": max(0, crawl_metrics.get('estimated_total', 0) - crawl_metrics.get('pages_processed', 0)),
            "estimated_total": crawl_metrics.get('estimated_total', 0),
            "crawl_time": crawl_metrics.get('crawl_time', 0),
            "current_activity": crawl_metrics.get('status', 'Unknown'),
            "started_at": job.get('started_at'),
            "completed_at": job.get('completed_at'),
            "error_message": job.get('error_message')
        }

        return {
            "success": True,
            "data": progress_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl status for job {crawl_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get crawl status")


@router.post("/cancel/{task_id}")
async def cancel_crawl(
    task_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Cancel a running crawl task.
    """
    try:
        crawl_manager = CrawlManager()
        result = crawl_manager.cancel_crawl(task_id)

        if result['success']:
            return {
                "success": True,
                "data": result
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Failed to cancel crawl')
            )

    except Exception as e:
        logger.error(f"Failed to cancel crawl task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel crawl")


@router.get("/website/{website_id}/status")
async def get_website_crawl_status(
    website_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get current crawl status and real-time progress for a specific website.
    """
    try:
        # Use service role client to bypass RLS
        supabase = get_supabase_admin()

        # Get the website (without ownership check as websites table doesn't have user_id)
        website_result = supabase.table('websites').select('*').eq(
            'id', website_id
        ).single().execute()

        if not website_result.data:
            raise HTTPException(status_code=404, detail="Website not found")

        # Check for active crawl jobs first
        active_jobs = supabase.table('crawling_jobs').select('*').eq(
            'website_id', website_id
        ).in_('status', ['pending', 'queued', 'running']).order('created_at', desc=True).limit(1).execute()

        if active_jobs.data:
            # There's an active crawl
            active_job = active_jobs.data[0]
            metrics = active_job.get('crawl_metrics', {})

            status_data = {
                "status": "crawling",
                "crawl_active": True,
                "active_crawl_id": active_job['id'],
                "progress_percentage": metrics.get('progress', 0),
                "current_activity": metrics.get('status', 'Starting crawl'),
                "pages_found": metrics.get('pages_found', 0),
                "pages_processed": metrics.get('pages_processed', 0),
                "pages_remaining": max(0, metrics.get('estimated_total', 0) - metrics.get('pages_processed', 0)),
                "estimated_total": metrics.get('estimated_total', 0),
                "crawl_time": metrics.get('crawl_time', 0),
                "started_at": active_job.get('started_at'),
                "job_type": active_job.get('job_type', 'unknown')
            }
        else:
            # No active crawl, get the latest completed/failed crawl
            latest_job = supabase.table('crawling_jobs').select('*').eq(
                'website_id', website_id
            ).order('created_at', desc=True).limit(1).execute()

            if latest_job.data:
                job = latest_job.data[0]
                metrics = job.get('crawl_metrics', {})

                # Determine status based on job status
                if job['status'] == 'completed':
                    last_status = "idle"
                elif job['status'] == 'failed':
                    last_status = "failed"
                else:
                    last_status = "idle"

                # Calculate actual crawl time if not in metrics
                actual_crawl_time = metrics.get('crawl_time', 0)
                if actual_crawl_time == 0 and job.get('started_at') and job.get('completed_at'):
                    from datetime import datetime
                    start = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
                    actual_crawl_time = round((end - start).total_seconds(), 1)

                status_data = {
                    "status": last_status,
                    "crawl_active": False,
                    "active_crawl_id": None,
                    "last_crawl_id": job['id'],
                    "last_crawled": job.get('completed_at') or job.get('started_at'),
                    "last_crawl_status": job['status'],
                    "pages_crawled": metrics.get('pages_processed', 0),
                    "last_crawl_time": actual_crawl_time,
                    "error_message": job.get('error_message') if job['status'] == 'failed' else None,
                    "job_type": job.get('job_type', 'unknown'),
                    "show_success": job['status'] == 'completed',  # For green button styling
                    "pages_found": metrics.get('pages_found', metrics.get('pages_processed', 0))  # Actual pages vs max_pages
                }
            else:
                # No crawl history at all
                status_data = {
                    "status": "idle",
                    "crawl_active": False,
                    "active_crawl_id": None,
                    "last_crawled": None,
                    "pages_crawled": 0,
                    "last_success": None,
                    "next_scheduled": None
                }

        return {
            "success": True,
            "data": status_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get website crawl status for {website_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get crawl status")


@router.get("/website/{website_id}/history")
async def get_website_crawl_history(
    website_id: str,
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get crawl history for a specific website.

    Since we don't have a dedicated crawling_jobs table yet, we'll return
    empty history or use scraped_pages data as a workaround.
    """
    try:
        # Use service role client to bypass RLS
        supabase = get_supabase_admin()

        # Get the website (without ownership check as websites table doesn't have user_id)
        website_result = supabase.table('websites').select('*').eq(
            'id', website_id
        ).single().execute()

        if not website_result.data:
            raise HTTPException(status_code=404, detail="Website not found")

        # Find the corresponding scraped_website entry
        scraped_website_result = supabase.table('scraped_websites').select('id').eq(
            'website_id', website_id
        ).execute()

        if not scraped_website_result.data:
            # No scraped_website entry means no crawl history
            return {
                "success": True,
                "data": {
                    "history": [],
                    "total": 0,
                    "has_more": False
                }
            }

        scraped_website_id = scraped_website_result.data[0]['id']

        # Get crawl history from scraped_pages table (crawl jobs are stored there)
        try:
            # Query crawling_jobs table directly
            history_result = supabase.table('crawling_jobs').select('*').eq(
                'website_id', website_id
            ).order('created_at', desc=True).range(offset, offset + limit - 1).execute()

            crawl_records = history_result.data if history_result.data else []

            # Get total count of crawl jobs
            count_result = supabase.table('crawling_jobs').select('id', count='exact').eq(
                'website_id', website_id
            ).execute()

            total = count_result.count if hasattr(count_result, 'count') else len(crawl_records)

            # Format the response
            history_data = {
                "history": [],
                "total": total,
                "has_more": (offset + limit) < total
            }

            for record in crawl_records:
                try:
                    history_entry = {
                        "crawl_id": record['id'],
                        "started_at": record.get('started_at') or record.get('created_at'),
                        "completed_at": record.get('completed_at'),
                        "status": record.get('status', 'unknown'),
                        "pages_crawled": record.get('crawl_metrics', {}).get('pages_processed', 0) if record.get('crawl_metrics') else 0,
                        "trigger_type": record.get('job_type', '').replace('_crawl', ''),
                        "error_message": record.get('error_message')
                    }

                    history_data["history"].append(history_entry)

                except KeyError as e:
                    logger.warning(f"Missing field in crawl job record: {e}")
                    continue

        except Exception as crawl_error:
            logger.warning(f"Failed to get crawl history from crawling_jobs table: {crawl_error}")
            # Fallback to empty history
            history_data = {
                "history": [],
                "total": 0,
                "has_more": False
            }

        return {
            "success": True,
            "data": history_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl history for website {website_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get crawl history")


@router.get("/active")
async def get_active_crawls(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get list of currently active crawl tasks.
    """
    try:
        crawl_manager = CrawlManager()
        active_crawls = crawl_manager.get_active_crawls()

        return {
            "success": True,
            "data": {
                "active_crawls": active_crawls,
                "total_active": len(active_crawls)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get active crawls: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active crawls")


# ==================== Celery Worker Endpoints (No Auth Required) ====================

@router.post("/init-scraped-website", response_model=InitScrapedWebsiteResponse)
async def init_scraped_website(request: InitScrapedWebsiteRequest):
    """
    Initialize a scraped_website record at the start of crawling.
    Called by celery worker - no auth required.
    """
    try:
        from datetime import datetime, timezone
        import uuid

        supabase = get_supabase_admin()

        scraped_website_data = {
            'id': str(uuid.uuid4()),
            'website_id': request.website_id,
            'domain': request.domain,
            'base_url': request.base_url,
            'crawl_status': 'crawling',
            'total_pages_found': 0,
            'pages_processed': 0,
            'max_pages': request.max_pages,
            'crawl_depth': request.crawl_depth,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table('scraped_websites').insert(scraped_website_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create scraped_website record")

        logger.info(f"Initialized scraped_website {result.data[0]['id']} for website {request.website_id}")

        return InitScrapedWebsiteResponse(
            scraped_website_id=result.data[0]['id'],
            website_id=request.website_id
        )

    except Exception as e:
        logger.error(f"Failed to initialize scraped_website: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store-page", response_model=StoreCrawledPageResponse)
async def store_crawled_page(request: StoreCrawledPageRequest):
    """
    Store a crawled page.
    Called by celery worker - no auth required.
    """
    try:
        from datetime import datetime, timezone
        import uuid

        supabase = get_supabase_admin()

        page_data = {
            'id': str(uuid.uuid4()),
            'scraped_website_id': request.scraped_website_id,
            'url': request.url,
            'title': request.title,
            'content_text': request.content_text,
            'content_html': request.content_html,
            'meta_description': request.meta_description,
            'status_code': request.status_code,
            'depth_level': request.depth_level,
            'scraped_at': datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table('scraped_pages').insert(page_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to store page")

        logger.info(f"Stored page: {request.url}")

        return StoreCrawledPageResponse(
            page_id=result.data[0]['id'],
            scraped_website_id=request.scraped_website_id,
            url=request.url
        )

    except Exception as e:
        logger.error(f"Failed to store page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-job-status")
async def update_job_status(request: UpdateJobStatusRequest):
    """
    Update crawling job status.
    Called by celery worker - no auth required.
    """
    try:
        from datetime import datetime, timezone

        supabase = get_supabase_admin()

        update_data = {
            'status': request.status
        }

        if request.status in ['completed', 'failed']:
            update_data['completed_at'] = datetime.now(timezone.utc).isoformat()

        if request.status == 'running':
            update_data['started_at'] = datetime.now(timezone.utc).isoformat()

        if request.crawl_metrics:
            update_data['crawl_metrics'] = request.crawl_metrics

        if request.error_message:
            update_data['error_message'] = request.error_message

        result = supabase.table('crawling_jobs').update(update_data).eq('id', request.job_id).execute()

        if not result.data:
            logger.warning(f"Job {request.job_id} not found for status update")

        logger.info(f"Updated job {request.job_id} status to {request.status}")

        return {"success": True, "job_id": request.job_id}

    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-embeddings", response_model=ProcessEmbeddingsResponse)
async def process_embeddings(request: ProcessEmbeddingsRequest):
    """
    Process embeddings for crawled pages.
    Called by celery worker - no auth required.
    """
    try:
        from ...services.vector_search_service import VectorSearchService
        from uuid import UUID
        import asyncio

        supabase = get_supabase_admin()
        vector_service = VectorSearchService()

        # Get the most recent scraped_website for this website
        scraped_website_result = supabase.table('scraped_websites').select('id').eq(
            'website_id', request.website_id
        ).order('created_at', desc=True).limit(1).execute()

        if not scraped_website_result.data:
            raise HTTPException(status_code=404, detail="No scraped_website found")

        scraped_website_id = scraped_website_result.data[0]['id']

        # Get all pages for this scraped_website
        pages_result = supabase.table('scraped_pages').select('id, content_text, url, title').eq(
            'scraped_website_id', scraped_website_id
        ).execute()

        pages = pages_result.data or []
        total_chunks = 0

        for page in pages:
            if not page.get('content_text'):
                continue

            # Generate embeddings for this page
            chunks = await vector_service.chunk_and_embed_content(
                page_id=UUID(page['id']),
                content=page['content_text'],
                website_id=request.website_id,
                chunk_size=1000
            )

            total_chunks += len(chunks)
            logger.info(f"Created {len(chunks)} chunks for page {page['url']}")

        logger.info(f"Embeddings processed: {len(pages)} pages, {total_chunks} chunks")

        return ProcessEmbeddingsResponse(
            status='completed',
            website_id=request.website_id,
            pages_processed=len(pages),
            chunks_created=total_chunks
        )

    except Exception as e:
        logger.error(f"Failed to process embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))