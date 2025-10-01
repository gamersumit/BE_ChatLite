"""
API endpoints for website crawler functionality.
"""
import logging
from typing import Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone

from ...services.vector_crawler_service import get_vector_crawler_service
from ...core.supabase_client import SupabaseClient
from ...core.auth_middleware import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class CrawlRequest(BaseModel):
    """Request model for starting a crawl."""
    website_id: UUID
    base_url: str = Field(..., description="Base URL to start crawling from")
    domain: str = Field(..., description="Domain to crawl")
    config_override: Optional[Dict[str, Any]] = Field(None, description="Optional config overrides")


class CrawlResponse(BaseModel):
    """Response model for crawl operations."""
    scraped_website_id: str
    status: str
    pages_processed: int
    total_pages_found: int
    message: str


class CrawlStatusResponse(BaseModel):
    """Response model for crawl status."""
    status: str
    pages_processed: int
    total_pages_found: int
    progress_percentage: float
    last_crawled_at: Optional[str]


class CreateTestWebsiteRequest(BaseModel):
    """Request model for creating a test website."""
    name: str = Field(..., description="Website name")
    url: str = Field(..., description="Website URL") 
    domain: str = Field(..., description="Website domain")


class CreateTestWebsiteResponse(BaseModel):
    """Response model for created test website."""
    id: str
    name: str
    url: str
    domain: str
    message: str


async def run_crawler_background(
    website_id: UUID,
    base_url: str,
    domain: str,
    config_override: Optional[Dict[str, Any]] = None
):
    """Run crawler in background task."""
    try:
        crawler_service = get_vector_crawler_service()
        result = await crawler_service.start_crawl(
            website_id=str(website_id),
            start_url=base_url,
            max_pages=50,  # Default max pages
            config_override=config_override
        )
        logger.info(f"Background crawl completed for {domain}: {result}")
    except Exception as e:
        logger.error(f"Background crawl failed for {domain}: {e}")


@router.post("/crawl", response_model=CrawlResponse)
async def start_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks
) -> CrawlResponse:
    """
    Start crawling a website.
    
    This endpoint starts the crawling process in the background and returns immediately
    with the initial status. Use the /status endpoint to check progress.
    """
    logger.info(f"🚀 Crawl request received for website_id: {request.website_id}")
    logger.info(f"📍 Request details - domain: {request.domain}, base_url: {request.base_url}")
    
    try:
        # Validate that website exists and scraping is enabled
        supabase = SupabaseClient().client
        logger.info(f"🔍 Looking up website in database: {request.website_id}")
        
        website_result = supabase.table('websites').select(
            'id, name, domain, scraping_enabled, business_description'
        ).eq('id', str(request.website_id)).execute()
        
        if not website_result.data:
            logger.error(f"❌ Website not found: {request.website_id}")
            raise HTTPException(status_code=404, detail="Website not found")
        
        website = website_result.data[0]
        logger.info(f"✅ Found website: {website['name']} ({website['domain']})")
        
        if not website.get('scraping_enabled', False):
            logger.error(f"❌ Scraping not enabled for website: {request.website_id}")
            raise HTTPException(
                status_code=400, 
                detail="Scraping not enabled for this website"
            )
        
        # Check if there's already a crawl in progress
        existing_crawl = supabase.table('scraped_websites').select(
            'id, crawl_status'
        ).eq('website_id', str(request.website_id)).eq(
            'crawl_status', 'crawling'
        ).execute()
        
        if existing_crawl.data:
            raise HTTPException(
                status_code=409, 
                detail="Crawl already in progress for this website"
            )
        
        # Create initial scraped_website record
        from ...models.scraper_schemas import ScrapedWebsiteCreate
        scraped_website_data = ScrapedWebsiteCreate(
            website_id=request.website_id,
            domain=request.domain,
            base_url=request.base_url,
            crawl_status="pending"
        )
        
        result = supabase.table('scraped_websites').insert(
            scraped_website_data.model_dump()
        ).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to initialize crawl")
        
        scraped_website_id = result.data[0]['id']
        
        # Start crawl in background
        background_tasks.add_task(
            run_crawler_background,
            request.website_id,
            request.base_url,
            request.domain,
            request.config_override
        )
        
        return CrawlResponse(
            scraped_website_id=scraped_website_id,
            status="started",
            pages_processed=0,
            total_pages_found=0,
            message=f"Crawl started for {request.domain}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start crawl: {str(e)}")


@router.get("/status/{website_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(website_id: UUID) -> CrawlStatusResponse:
    """
    Get the current crawl status for a website.
    """
    try:
        crawler_service = get_vector_crawler_service()
        status_data = await crawler_service.get_crawl_status(str(website_id))

        return CrawlStatusResponse(**status_data)
        
    except Exception as e:
        logger.error(f"Failed to get crawl status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get crawl status: {str(e)}")


@router.get("/pages/{website_id}")
async def get_scraped_pages(
    website_id: UUID,
    page_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get scraped pages for a website.
    """
    try:
        crawler_service = CrawlerService()
        pages_data = await crawler_service.get_scraped_pages(
            website_id=website_id,
            page_type=page_type,
            limit=limit,
            offset=offset
        )
        
        return pages_data
        
    except Exception as e:
        logger.error(f"Failed to get scraped pages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scraped pages: {str(e)}")


@router.post("/search/{website_id}")
async def search_content(
    website_id: UUID,
    query: str,
    page_type: Optional[str] = None,
    limit: int = 10
):
    """
    Search scraped content for a website.
    """
    try:
        crawler_service = CrawlerService()
        search_results = await crawler_service.search_content(
            website_id=website_id,
            query=query,
            page_type=page_type,
            limit=limit
        )
        
        return {"results": search_results}
        
    except Exception as e:
        logger.error(f"Failed to search content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search content: {str(e)}")


@router.put("/config/{website_id}")
async def update_scraping_config(
    website_id: UUID,
    config: Dict[str, Any]
):
    """
    Update scraping configuration for a website.
    """
    try:
        crawler_service = CrawlerService()
        success = await crawler_service.update_scraping_config(website_id, config)
        
        if success:
            return {"message": "Configuration updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/enable/{website_id}")
async def enable_scraping(website_id: UUID, enabled: bool = True):
    """
    Enable or disable scraping for a website.
    """
    try:
        crawler_service = CrawlerService()
        success = await crawler_service.enable_scraping(website_id, enabled)
        
        if success:
            action = "enabled" if enabled else "disabled"
            return {"message": f"Scraping {action} successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update scraping status")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update scraping status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update scraping status: {str(e)}")


@router.delete("/data/{website_id}")
async def delete_scraped_data(website_id: UUID):
    """
    Delete all scraped data for a website.
    """
    try:
        crawler_service = CrawlerService()
        success = await crawler_service.delete_scraped_data(website_id)
        
        if success:
            return {"message": "Scraped data deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete scraped data")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete scraped data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete scraped data: {str(e)}")


@router.get("/statistics/{website_id}")
async def get_crawl_statistics(website_id: UUID):
    """
    Get comprehensive crawl statistics for a website.
    """
    try:
        crawler_service = CrawlerService()
        stats = await crawler_service.get_crawl_statistics(website_id)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/create-test-website", response_model=CreateTestWebsiteResponse)
async def create_test_website(request: CreateTestWebsiteRequest) -> CreateTestWebsiteResponse:
    """
    Create a test website for crawler testing.
    This endpoint is for development/testing purposes only.
    """
    try:
        # Use regular client (RLS policies disabled)
        supabase = SupabaseClient().service_client
        
        # Check if domain already exists
        existing_website = supabase.table('websites').select(
            'id, name, domain, scraping_enabled'
        ).eq('domain', request.domain).execute()
        
        if existing_website.data:
            # Domain exists, return the existing website info
            existing = existing_website.data[0]
            return CreateTestWebsiteResponse(
                id=existing['id'],
                name=existing['name'],
                url=request.url,
                domain=existing['domain'],
                message=f"Website already exists for domain '{request.domain}'. Using existing website."
            )
        
        # Generate UUID for the new website
        website_id = str(uuid.uuid4())
        
        # Create website record with scraping configuration
        # Using correct field names that match the Supabase table schema
        website_data = {
            'id': website_id,
            'name': request.name,
            'url': request.url,  # This field exists in the table
            'domain': request.domain,
            'widget_id': f'widget-{website_id[:8]}',  # Generate a widget ID
            'is_active': True,
            'scraping_enabled': True,
            'verification_status': 'verified',
            'settings': {
                'welcome_message': 'Hello! How can I help you?',
                'placeholder_text': 'Ask me anything about this website...',
                'widget_position': 'bottom-right',
                'widget_color': '#0066CC',
                'widget_theme': 'light'
            },
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        result = supabase.table('websites').insert(website_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create test website")
        
        return CreateTestWebsiteResponse(
            id=website_id,
            name=request.name,
            url=request.url,  # Return the URL from request even though it's stored in config
            domain=request.domain,
            message="Test website created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create test website: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test website: {str(e)}")


@router.get("/test-websites")
async def list_test_websites():
    """
    List all websites available for testing.
    """
    try:
        supabase = SupabaseClient().client
        result = supabase.table('websites').select('id, name, domain, scraping_enabled, scraping_config').execute()
        
        return {
            "websites": result.data or [],
            "count": len(result.data) if result.data else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to list test websites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test websites: {str(e)}")


class ProcessEmbeddingsRequest(BaseModel):
    """Request model for processing embeddings."""
    website_id: str = Field(..., description="Website ID")
    pages_count: int = Field(..., description="Number of pages to process")


class ProcessEmbeddingsResponse(BaseModel):
    """Response model for embeddings processing."""
    status: str
    website_id: str
    total_pages: int
    processed_pages: int
    failed_pages: int


@router.post("/process-embeddings", response_model=ProcessEmbeddingsResponse)
async def process_embeddings(
    request: ProcessEmbeddingsRequest
) -> ProcessEmbeddingsResponse:
    """
    Process embeddings for crawled pages.
    This endpoint is called by Celery workers to generate embeddings.
    """
    try:
        import asyncio
        from ...services.vector_search_service import VectorSearchService
        from ...core.supabase_client import get_supabase_admin

        supabase = get_supabase_admin()

        # Get the latest scraped_website record for this website_id
        scraped_website_result = supabase.table('scraped_websites').select('id').eq(
            'website_id', request.website_id
        ).order('created_at', desc=True).limit(1).execute()

        if not scraped_website_result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No scraped_website record found for website_id: {request.website_id}"
            )

        scraped_website_id = scraped_website_result.data[0]['id']
        logger.info(f"Processing embeddings for scraped_website_id: {scraped_website_id}")

        # Get recent pages for this scraped_website
        pages_result = supabase.table('scraped_pages').select('id, content_text, url, title').eq(
            'scraped_website_id', scraped_website_id
        ).order('scraped_at', desc=True).limit(request.pages_count).execute()

        if not pages_result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No pages found for scraped_website_id: {scraped_website_id}"
            )

        vector_service = VectorSearchService()
        processed_count = 0
        failed_count = 0
        total_pages = len(pages_result.data)

        logger.info(f"Processing {total_pages} pages for website_id: {request.website_id}")

        for page in pages_result.data:
            page_id = page['id']
            content_text = page.get('content_text')

            if not content_text:
                logger.warning(f"No content found for page {page_id}")
                failed_count += 1
                continue

            try:
                logger.info(f"Processing page {page.get('url', 'unknown')}: {len(content_text)} chars")

                # Generate embeddings (async function)
                chunks = await vector_service.chunk_and_embed_content(
                    page_id=UUID(page_id),
                    content=content_text,
                    website_id=request.website_id,
                    chunk_size=1000
                )

                if chunks:
                    processed_count += 1
                    logger.info(f"Created {len(chunks)} chunks for page {page_id}")
                else:
                    failed_count += 1
                    logger.warning(f"No chunks created for page {page_id}")

            except Exception as e:
                logger.error(f"Failed to process page {page_id}: {e}")
                failed_count += 1
                continue

        return ProcessEmbeddingsResponse(
            status='completed',
            website_id=request.website_id,
            total_pages=total_pages,
            processed_pages=processed_count,
            failed_pages=failed_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process embeddings: {str(e)}")


class InitScrapedWebsiteRequest(BaseModel):
    """Request model for initializing a scraped website record."""
    website_id: str
    domain: str
    base_url: str
    max_pages: int = 100
    crawl_depth: int = 3


class InitScrapedWebsiteResponse(BaseModel):
    """Response model for initialized scraped website."""
    scraped_website_id: str
    website_id: str


@router.post("/init-scraped-website", response_model=InitScrapedWebsiteResponse)
async def init_scraped_website(
    request: InitScrapedWebsiteRequest
) -> InitScrapedWebsiteResponse:
    """
    Initialize a scraped_website record.
    This endpoint is called by Celery workers at the start of crawling.
    """
    try:
        from ...core.supabase_client import get_supabase_admin

        supabase = get_supabase_admin()

        scraped_website_data = {
            'website_id': request.website_id,
            'domain': request.domain,
            'base_url': request.base_url,
            'crawl_status': 'crawling',
            'max_pages': request.max_pages,
            'crawl_depth': request.crawl_depth,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'total_pages_found': 0,
            'pages_processed': 0
        }

        result = supabase.table('scraped_websites').insert(scraped_website_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create scraped_website record")

        return InitScrapedWebsiteResponse(
            scraped_website_id=result.data[0]['id'],
            website_id=request.website_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize scraped_website: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize scraped_website: {str(e)}")


class StoreCrawledPageRequest(BaseModel):
    """Request model for storing a crawled page."""
    scraped_website_id: str
    url: str
    title: Optional[str] = None
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    meta_description: Optional[str] = None
    status_code: Optional[int] = None
    depth_level: int = 0


class StoreCrawledPageResponse(BaseModel):
    """Response model for stored page."""
    page_id: str
    scraped_website_id: str
    url: str


@router.post("/store-page", response_model=StoreCrawledPageResponse)
async def store_crawled_page(
    request: StoreCrawledPageRequest
) -> StoreCrawledPageResponse:
    """
    Store a crawled page.
    This endpoint is called by Celery workers during crawling.
    """
    try:
        from ...core.supabase_client import get_supabase_admin

        supabase = get_supabase_admin()

        page_data = {
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

        return StoreCrawledPageResponse(
            page_id=result.data[0]['id'],
            scraped_website_id=request.scraped_website_id,
            url=request.url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store crawled page: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store page: {str(e)}")


class UpdateJobStatusRequest(BaseModel):
    """Request model for updating job status."""
    job_id: str
    status: str
    crawl_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@router.post("/update-job-status")
async def update_job_status(request: UpdateJobStatusRequest):
    """
    Update crawling job status.
    This endpoint is called by Celery workers to update job progress.
    """
    try:
        from ...core.supabase_client import get_supabase_admin

        supabase = get_supabase_admin()

        update_data = {
            'status': request.status,
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

        return {"success": True, "job_id": request.job_id}

    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update job status: {str(e)}")