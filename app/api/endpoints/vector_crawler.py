"""
Vector Crawler API Endpoints
Cloud-ready crawler using Supabase vector storage
"""

import logging
from typing import Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ...services.vector_crawler_service import get_vector_crawler_service
from ...core.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector-crawler", tags=["vector-crawler"])


# Request/Response Models
class VectorCrawlRequest(BaseModel):
    """Request model for starting a vector crawl"""
    website_id: str = Field(..., description="Website ID to crawl")
    start_url: str = Field(..., description="Start URL to begin crawling from")
    max_pages: int = Field(default=50, description="Maximum pages to crawl")
    config_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration overrides"
    )


class VectorCrawlResponse(BaseModel):
    """Response model for vector crawl operations"""
    job_id: str = Field(..., description="Crawl job identifier")
    status: str = Field(..., description="Crawl status")
    pages_processed: int = Field(..., description="Number of pages processed")
    pages_found: int = Field(..., description="Total pages found")
    pages_with_errors: int = Field(..., description="Pages that failed to process")
    domain: str = Field(..., description="Crawled domain")
    base_url: str = Field(..., description="Base URL")
    storage_type: str = Field(..., description="Storage type used")


class CrawlStatusResponse(BaseModel):
    """Response model for crawl status"""
    status: str = Field(..., description="Current crawl status")
    job_id: Optional[str] = Field(default=None, description="Job identifier")
    pages_found: int = Field(default=0, description="Total pages found")
    pages_processed: int = Field(default=0, description="Pages processed")
    pages_with_errors: int = Field(default=0, description="Pages with errors")
    started_at: Optional[str] = Field(default=None, description="Crawl start time")
    completed_at: Optional[str] = Field(default=None, description="Crawl completion time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    content_stats: Optional[Dict[str, Any]] = Field(default=None, description="Content statistics")


# API Endpoints
@router.post("/crawl", response_model=Dict[str, str])
async def start_background_crawl(
    request: VectorCrawlRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Start crawling a website with vector storage in background
    """
    try:
        logger.info(f"Starting background vector crawl for website {request.website_id}")

        service = get_vector_crawler_service()

        # Extract domain from start_url
        from urllib.parse import urlparse
        parsed_url = urlparse(request.start_url)
        domain = parsed_url.netloc

        # Add crawl to background tasks
        background_tasks.add_task(
            service.start_crawl,
            website_id=request.website_id,
            start_url=request.start_url,
            max_pages=request.max_pages,
            config_override=request.config_override
        )

        return {
            "status": "accepted",
            "message": f"Vector crawl started in background for {domain}",
            "website_id": str(request.website_id)
        }

    except Exception as e:
        logger.error(f"Error starting background vector crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start background crawl: {str(e)}")


@router.get("/status/{website_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get crawl status for a website
    """
    try:
        service = get_vector_crawler_service()
        status = await service.get_crawl_status(website_id)

        return CrawlStatusResponse(**status)

    except Exception as e:
        logger.error(f"Error getting crawl status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get crawl status: {str(e)}")


@router.delete("/content/{website_id}")
async def delete_website_content(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete all crawled content for a website
    """
    try:
        logger.info(f"Deleting content for website {website_id} by user {current_user.get('id')}")

        service = get_vector_crawler_service()
        success = await service.delete_website_content(website_id)

        if success:
            return {
                "status": "success",
                "message": f"Deleted all content for website {website_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete content")

    except Exception as e:
        logger.error(f"Error deleting website content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete content: {str(e)}")


@router.get("/stats/{website_id}")
async def get_content_stats(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get content statistics for a website
    """
    try:
        service = get_vector_crawler_service()
        vector_service = service.vector_service

        stats = await vector_service.get_content_stats(website_id)

        return {
            "status": "success",
            "website_id": website_id,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get content stats: {str(e)}")


@router.post("/crawl-url")
async def crawl_single_url(
    website_id: str,
    url: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Crawl a single URL and store in vector database
    """
    try:
        from urllib.parse import urlparse
        import aiohttp
        from bs4 import BeautifulSoup

        logger.info(f"Crawling single URL {url} for website {website_id}")

        # Parse domain from URL
        parsed = urlparse(url)
        domain = parsed.netloc

        service = get_vector_crawler_service()

        # Create HTTP session for single request
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {'User-Agent': 'ChatLite-Crawler/2.0 (+https://chatlite.com/bot)'}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            result = await service._crawl_single_page(
                session=session,
                website_id=website_id,
                url=url,
                domain=domain,
                crawled_urls=set(),
                failed_urls=set()
            )

        if result:
            return {
                "status": "success",
                "message": f"Successfully crawled {url}",
                "result": result
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to crawl {url}")

    except Exception as e:
        logger.error(f"Error crawling single URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl URL: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check for vector crawler service
    """
    try:
        service = get_vector_crawler_service()

        # Test basic functionality
        test_url = "https://example.com"
        is_valid = service._is_valid_url(test_url, "example.com")

        return {
            "status": "healthy",
            "service": "vector_crawler",
            "url_validation": "working" if is_valid else "error",
            "timestamp": "2025-09-30T00:00:00Z"
        }

    except Exception as e:
        logger.error(f"Vector crawler health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/config")
async def get_crawler_config():
    """
    Get current crawler configuration
    """
    try:
        service = get_vector_crawler_service()

        return {
            "status": "success",
            "config": {
                "max_pages_default": service.max_pages_default,
                "max_concurrent_requests": service.max_concurrent_requests,
                "request_timeout": service.request_timeout,
                "min_content_length": service.min_content_length,
                "max_content_length": service.max_content_length
            }
        }

    except Exception as e:
        logger.error(f"Error getting crawler config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@router.put("/config")
async def update_crawler_config(
    max_pages: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    timeout: Optional[int] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update crawler configuration (admin only)
    """
    try:
        # Check if user is admin (you may need to implement admin check)
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")

        service = get_vector_crawler_service()

        # Update configuration
        if max_pages is not None:
            service.max_pages_default = max_pages
        if max_concurrent is not None:
            service.max_concurrent_requests = max_concurrent
        if timeout is not None:
            service.request_timeout = timeout

        return {
            "status": "success",
            "message": "Crawler configuration updated",
            "config": {
                "max_pages_default": service.max_pages_default,
                "max_concurrent_requests": service.max_concurrent_requests,
                "request_timeout": service.request_timeout
            }
        }

    except Exception as e:
        logger.error(f"Error updating crawler config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/test")
async def test_crawl(
    url: str = "https://example.com",
    current_user: Dict = Depends(get_current_user)
):
    """
    Test crawl functionality with a simple URL
    """
    try:
        from urllib.parse import urlparse
        import uuid

        logger.info(f"Testing crawl with URL: {url}")

        parsed = urlparse(url)
        domain = parsed.netloc
        test_website_id = str(uuid.uuid4())

        service = get_vector_crawler_service()

        # Mock website validation for testing
        original_validate = service._validate_website
        service._validate_website = lambda *args: {
            'id': test_website_id,
            'name': 'Test Website',
            'domain': domain,
            'scraping_enabled': True
        }

        try:
            result = await service.start_crawl(
                website_id=uuid.UUID(test_website_id),
                base_url=url,
                domain=domain,
                config_override={'max_pages': 1}  # Limit to 1 page for testing
            )

            return {
                "status": "success",
                "message": "Test crawl completed",
                "result": result
            }

        finally:
            # Restore original validation
            service._validate_website = original_validate

            # Clean up test data
            try:
                await service.delete_website_content(test_website_id)
            except:
                pass

    except Exception as e:
        logger.error(f"Test crawl failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test crawl failed: {str(e)}")