"""
Vector Storage API Endpoints
Provides HTTP interface for vector database operations
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from ...services.supabase_vector_service import get_vector_service
from ...services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector", tags=["vector-storage"])


# Request/Response Models
class StoreContentRequest(BaseModel):
    website_id: str = Field(..., description="Website identifier")
    url: str = Field(..., description="Page URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Page content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    website_id: str = Field(..., description="Website identifier")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum results")


class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    use_cache: bool = Field(default=True, description="Use embedding cache")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    use_cache: bool = Field(default=True, description="Use embedding cache")


class UpdateContentRequest(BaseModel):
    website_id: str = Field(..., description="Website identifier")
    url: str = Field(..., description="Page URL")
    content: str = Field(..., description="New content")
    title: Optional[str] = Field(default=None, description="New title")


class DeleteContentRequest(BaseModel):
    website_id: str = Field(..., description="Website identifier")
    url: Optional[str] = Field(default=None, description="Specific URL to delete")


class CrawlJobRequest(BaseModel):
    website_id: str = Field(..., description="Website identifier")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Crawl configuration")


# API Endpoints
@router.post("/store")
async def store_content(request: StoreContentRequest):
    """
    Store content with vector embeddings
    """
    try:
        service = get_vector_service()
        result = await service.store_content(
            website_id=request.website_id,
            url=request.url,
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )

        if result['success']:
            return {
                "status": "success",
                "message": f"Stored {result['chunks_created']} chunks",
                "chunks_created": result['chunks_created'],
                "document_ids": result['document_ids']
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store content: {result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error storing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_similar(request: SearchRequest):
    """
    Search for similar content using vector similarity
    """
    try:
        service = get_vector_service()
        results = await service.search_similar(
            query=request.query,
            website_id=request.website_id,
            limit=request.limit
        )

        return {
            "status": "success",
            "query": request.query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/hybrid")
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search (vector + keyword matching)
    """
    try:
        service = get_vector_service()
        results = await service.hybrid_search(
            query=request.query,
            website_id=request.website_id,
            limit=request.limit
        )

        return {
            "status": "success",
            "query": request.query,
            "results": results,
            "count": len(results),
            "search_type": "hybrid"
        }

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate embedding for text
    """
    try:
        service = get_embedding_service()
        embedding = await service.generate_embedding(request.text)

        return {
            "status": "success",
            "embedding": embedding,
            "dimensions": len(embedding),
            "token_count": service.count_tokens(request.text)
        }

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed/batch")
async def batch_generate_embeddings(request: BatchEmbeddingRequest):
    """
    Generate embeddings for multiple texts
    """
    try:
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 texts allowed per batch"
            )

        service = get_embedding_service()
        embeddings = await service.batch_generate_embeddings(
            texts=request.texts,
            use_cache=request.use_cache
        )

        return {
            "status": "success",
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0
        }

    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update")
async def update_content(request: UpdateContentRequest):
    """
    Update existing content with new embeddings
    """
    try:
        service = get_vector_service()
        success = await service.update_content(
            website_id=request.website_id,
            url=request.url,
            new_content=request.content,
            new_title=request.title
        )

        if success:
            return {
                "status": "success",
                "message": "Content updated successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update content"
            )

    except Exception as e:
        logger.error(f"Error updating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_content(request: DeleteContentRequest):
    """
    Delete content from vector database
    """
    try:
        service = get_vector_service()
        success = await service.delete_content(
            website_id=request.website_id,
            url=request.url
        )

        if success:
            return {
                "status": "success",
                "message": "Content deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete content"
            )

    except Exception as e:
        logger.error(f"Error deleting content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{website_id}")
async def get_content_stats(website_id: str):
    """
    Get content statistics for a website
    """
    try:
        service = get_vector_service()
        stats = await service.get_content_stats(website_id)

        return {
            "status": "success",
            "website_id": website_id,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawl/job")
async def create_crawl_job(request: CrawlJobRequest):
    """
    Create a new crawl job
    """
    try:
        service = get_vector_service()
        job_id = await service.create_crawl_job(
            website_id=request.website_id,
            config=request.config
        )

        if job_id:
            return {
                "status": "success",
                "job_id": job_id,
                "message": "Crawl job created successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create crawl job"
            )

    except Exception as e:
        logger.error(f"Error creating crawl job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/crawl/job/{job_id}")
async def update_crawl_job_status(
    job_id: str,
    status: Optional[str] = None,
    pages_found: Optional[int] = None,
    pages_processed: Optional[int] = None,
    error_message: Optional[str] = None
):
    """
    Update crawl job status
    """
    try:
        service = get_vector_service()
        success = await service.update_crawl_job(
            job_id=job_id,
            status=status,
            pages_found=pages_found,
            pages_processed=pages_processed,
            error_message=error_message
        )

        if success:
            return {
                "status": "success",
                "message": "Crawl job updated successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update crawl job"
            )

    except Exception as e:
        logger.error(f"Error updating crawl job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get embedding cache statistics
    """
    try:
        service = get_embedding_service()
        stats = await service.get_cache_stats()

        return {
            "status": "success",
            "cache_stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear")
async def clear_cache():
    """
    Clear embedding cache
    """
    try:
        service = get_embedding_service()
        cleared_count = await service.clear_cache()

        return {
            "status": "success",
            "message": f"Cleared {cleared_count} cache entries"
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for vector storage services
    """
    try:
        # Test vector service
        vector_service = get_vector_service()
        # Simple connection test
        stats = await vector_service.get_content_stats("health-check")

        # Test embedding service
        embedding_service = get_embedding_service()
        cache_stats = await embedding_service.get_cache_stats()

        return {
            "status": "healthy",
            "services": {
                "vector_storage": "healthy",
                "embedding_service": "healthy",
                "cache_enabled": cache_stats.get("cache_enabled", False)
            },
            "timestamp": "2025-09-30T00:00:00Z"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )