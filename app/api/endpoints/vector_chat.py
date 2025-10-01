"""
Vector Chat API Endpoints
Cloud-ready chat using vector RAG for context-aware responses
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json

from ...services.vector_enhanced_chat_service import get_vector_enhanced_chat_service
from ...core.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector-chat", tags=["vector-chat"])


# Request/Response Models
class VectorChatRequest(BaseModel):
    """Request model for vector chat"""
    message: str = Field(..., description="User message")
    website_id: str = Field(..., description="Website identifier for context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation messages"
    )
    use_rag: bool = Field(default=True, description="Use RAG for context")
    use_hybrid_search: bool = Field(default=True, description="Use hybrid search")
    max_tokens: int = Field(default=500, ge=50, le=1000, description="Maximum response tokens")


class VectorChatResponse(BaseModel):
    """Response model for vector chat"""
    response: str = Field(..., description="AI response")
    context_used: bool = Field(..., description="Whether context was used")
    context_chunks: int = Field(default=0, description="Number of context chunks used")
    context_sources: List[Dict[str, str]] = Field(default=[], description="Context source information")
    service: str = Field(..., description="Service used for response")
    search_type: Optional[str] = Field(default=None, description="Type of search used")
    response_time: float = Field(..., description="Response time in seconds")


class ChatSuggestionsResponse(BaseModel):
    """Response model for chat suggestions"""
    suggestions: List[str] = Field(..., description="Suggested questions")
    website_id: str = Field(..., description="Website identifier")


class StreamChatRequest(BaseModel):
    """Request model for streaming chat"""
    message: str = Field(..., description="User message")
    website_id: str = Field(..., description="Website identifier")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None)
    use_rag: bool = Field(default=True, description="Use RAG for context")
    use_hybrid_search: bool = Field(default=True, description="Use hybrid search")


# API Endpoints
@router.post("/message", response_model=VectorChatResponse)
async def send_chat_message(
    request: VectorChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Send a chat message and get AI response with vector RAG
    """
    try:
        logger.info(f"Processing chat message for website {request.website_id}")

        service = get_vector_enhanced_chat_service()

        response = await service.generate_chat_response(
            user_message=request.message,
            website_id=request.website_id,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
            use_rag=request.use_rag,
            use_hybrid_search=request.use_hybrid_search
        )

        return VectorChatResponse(**response)

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.post("/stream")
async def stream_chat_message(
    request: StreamChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Stream chat response with vector RAG
    """
    try:
        logger.info(f"Starting streaming chat for website {request.website_id}")

        service = get_vector_enhanced_chat_service()

        async def generate_stream():
            try:
                async for chunk in service.stream_chat_response(
                    user_message=request.message,
                    website_id=request.website_id,
                    conversation_history=request.conversation_history,
                    use_rag=request.use_rag,
                    use_hybrid_search=request.use_hybrid_search
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Send final event
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            }
        )

    except Exception as e:
        logger.error(f"Error setting up chat stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")


@router.get("/suggestions/{website_id}", response_model=ChatSuggestionsResponse)
async def get_chat_suggestions(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get suggested chat questions based on website content
    """
    try:
        service = get_vector_enhanced_chat_service()
        suggestions = await service.get_chat_suggestions(website_id)

        return ChatSuggestionsResponse(
            suggestions=suggestions,
            website_id=website_id
        )

    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/content-summary/{website_id}")
async def get_content_summary(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get content summary for RAG readiness
    """
    try:
        service = get_vector_enhanced_chat_service()
        summary = await service.vector_rag_service.get_content_summary(website_id)

        return {
            "status": "success",
            "website_id": website_id,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error getting content summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.post("/test-functionality/{website_id}")
async def test_chat_functionality(
    website_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Test chat functionality for a website
    """
    try:
        logger.info(f"Testing chat functionality for website {website_id}")

        service = get_vector_enhanced_chat_service()
        test_results = await service.test_chat_functionality(website_id)

        return {
            "status": "success",
            "test_results": test_results
        }

    except Exception as e:
        logger.error(f"Error testing chat functionality: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@router.post("/quick-test")
async def quick_chat_test(
    message: str = "Hello, can you help me?",
    website_id: str = "test-website",
    current_user: Dict = Depends(get_current_user)
):
    """
    Quick test of chat functionality with default parameters
    """
    try:
        service = get_vector_enhanced_chat_service()

        # Test basic response
        response = await service.generate_chat_response(
            user_message=message,
            website_id=website_id,
            use_rag=True,
            use_hybrid_search=True
        )

        return {
            "status": "success",
            "test_message": message,
            "website_id": website_id,
            "response": response
        }

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick test failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check for vector chat service
    """
    try:
        service = get_vector_enhanced_chat_service()

        # Test basic functionality
        test_response = await service.generate_chat_response(
            user_message="test",
            website_id="health-check",
            use_rag=False  # Don't require content for health check
        )

        return {
            "status": "healthy",
            "service": "vector_chat",
            "response_generated": bool(test_response.get("response")),
            "openai_available": service.use_openai,
            "timestamp": "2025-09-30T00:00:00Z"
        }

    except Exception as e:
        logger.error(f"Vector chat health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/config")
async def get_chat_config():
    """
    Get current chat service configuration
    """
    try:
        service = get_vector_enhanced_chat_service()

        return {
            "status": "success",
            "config": {
                "openai_available": service.use_openai,
                "max_context_length": service.vector_rag_service.max_context_length,
                "context_chunk_limit": service.vector_rag_service.context_chunk_limit,
                "similarity_threshold": service.vector_rag_service.similarity_threshold
            }
        }

    except Exception as e:
        logger.error(f"Error getting chat config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


# Compatibility endpoint for existing chat systems
@router.post("/legacy-compatible")
async def legacy_chat_endpoint(
    message: str,
    website_id: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Legacy-compatible chat endpoint for existing integrations
    """
    try:
        service = get_vector_enhanced_chat_service()

        response = await service.generate_chat_response(
            user_message=message,
            website_id=website_id,
            conversation_history=conversation_history,
            use_rag=True,
            use_hybrid_search=True
        )

        # Return in legacy format
        return {
            "response": response.get("response", ""),
            "success": True,
            "context_used": response.get("context_used", False),
            "service": response.get("service", "vector_chat")
        }

    except Exception as e:
        logger.error(f"Legacy chat endpoint error: {e}")
        return {
            "response": "I apologize, but I'm having trouble processing your request.",
            "success": False,
            "error": str(e)
        }