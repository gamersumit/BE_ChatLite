"""
Session-aware chat API endpoints with persistent conversation history.
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from supabase import Client

from ...core.database import get_supabase_client
from ...services.enhanced_session_chat_service import EnhancedSessionChatService

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class SessionChatRequest(BaseModel):
    """Request model for session-aware chat."""
    message: str = Field(..., description="User message")
    website_id: UUID = Field(..., description="Website UUID")
    visitor_id: str = Field(..., description="Unique visitor identifier")
    session_token: Optional[str] = Field(None, description="Existing session token")
    page_url: Optional[str] = Field(None, description="Current page URL")
    page_title: Optional[str] = Field(None, description="Current page title")
    user_id: Optional[str] = Field(None, description="Authenticated user ID")


class SessionChatResponse(BaseModel):
    """Response model for session-aware chat."""
    response: str = Field(..., description="AI response")
    session_token: str = Field(..., description="Session token for future requests")
    session_expires_at: str = Field(..., description="Session expiration timestamp")
    is_new_session: bool = Field(..., description="Whether this is a new session")
    processing_time_ms: int = Field(..., description="Response processing time")
    conversation_id: str = Field(..., description="Conversation UUID")
    context_used: bool = Field(..., description="Whether website context was used")
    messages_in_context: int = Field(..., description="Number of messages in context window")
    total_context_tokens: int = Field(..., description="Total tokens used for context")


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
    session_info: Optional[Dict[str, Any]] = Field(..., description="Session metadata")
    total_messages: int = Field(..., description="Total number of messages")


class SessionValidationResponse(BaseModel):
    """Response model for session validation."""
    valid: bool = Field(..., description="Whether session is valid")
    session_token: str = Field(..., description="Session token")
    expires_at: Optional[str] = Field(None, description="Session expiration")
    visitor_id: Optional[str] = Field(None, description="Visitor ID")
    website_id: Optional[str] = Field(None, description="Website UUID")


class SessionExtensionRequest(BaseModel):
    """Request model for session extension."""
    extension_days: Optional[int] = Field(7, description="Days to extend session")


# Initialize service
chat_service = EnhancedSessionChatService()


@router.post("/message", response_model=SessionChatResponse)
async def send_message_with_session(
    request: SessionChatRequest,
    http_request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Send a message with session-aware context and history management.
    
    This endpoint provides continuous conversation experience by:
    - Managing persistent sessions across browser refreshes
    - Maintaining conversation context and history
    - Optimizing token usage with intelligent context window management
    - Providing RAG-enhanced responses using website content
    """
    try:
        # Extract client information
        user_agent = http_request.headers.get("user-agent")
        client_ip = http_request.client.host if http_request.client else None
        referrer = http_request.headers.get("referer")
        
        # Process chat message with session management
        result = await chat_service.process_chat_with_session(
            message=request.message,
            website_id=request.website_id,
            visitor_id=request.visitor_id,
            session_token=request.session_token,
            page_url=request.page_url,
            page_title=request.page_title,
            user_agent=user_agent,
            ip_address=client_ip,
            referrer=referrer,
            user_id=request.user_id
        )
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return SessionChatResponse(
            response=result['response'],
            session_token=result['session_token'],
            session_expires_at=result['session_expires_at'],
            is_new_session=result['is_new_session'],
            processing_time_ms=result['processing_time_ms'],
            conversation_id=result['conversation_id'],
            context_used=result.get('context_used', False),
            messages_in_context=result.get('messages_in_context', 0),
            total_context_tokens=result.get('total_context_tokens', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/sessions/{session_token}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_token: str,
    limit: Optional[int] = 20,
    include_metadata: bool = False,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get conversation history for a specific session.
    
    Args:
        session_token: Session identifier
        limit: Maximum number of messages to return (default: 20)
        include_metadata: Whether to include message metadata like tokens, processing time, etc.
    """
    try:
        result = await chat_service.get_session_history(
            session_token=session_token,
            limit=limit,
            include_metadata=include_metadata
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return SessionHistoryResponse(
            messages=result['messages'],
            session_info=result['session_info'],
            total_messages=result['total_messages']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")


@router.get("/sessions/{session_token}/validate", response_model=SessionValidationResponse)
async def validate_session(
    session_token: str,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Validate if a session is active and not expired.
    
    Args:
        session_token: Session identifier to validate
    """
    try:
        result = await chat_service.validate_session(session_token)
        
        return SessionValidationResponse(
            valid=result['valid'],
            session_token=session_token,
            expires_at=result.get('expires_at'),
            visitor_id=result.get('visitor_id'),
            website_id=result.get('website_id')
        )
        
    except Exception as e:
        logger.error(f"❌ API: Error validating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate session: {str(e)}")


@router.put("/sessions/{session_token}/extend")
async def extend_session(
    session_token: str,
    request: SessionExtensionRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Extend session expiration time.
    
    Args:
        session_token: Session to extend
        request: Extension parameters
    """
    try:
        result = await chat_service.extend_session_duration(
            session_token=session_token,
            extension_days=request.extension_days
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to extend session'))
        
        return {
            "success": True,
            "message": result['message'],
            "session_token": session_token,
            "expires_at": result.get('expires_at')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Error extending session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extend session: {str(e)}")


@router.delete("/sessions/{session_token}")
async def end_session(
    session_token: str,
    reason: str = "manual",
    supabase: Client = Depends(get_supabase_client)
):
    """
    End a chat session.
    
    Args:
        session_token: Session to end
        reason: Reason for ending (manual, timeout, error, etc.)
    """
    try:
        result = await chat_service.end_session(session_token, reason)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to end session'))
        
        return {
            "success": True,
            "message": result['message'],
            "session_token": session_token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Error ending session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.get("/analytics/websites/{website_id}/sessions")
async def get_website_session_analytics(
    website_id: UUID,
    days: int = 7,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get session analytics for a website.
    
    Args:
        website_id: Website UUID
        days: Number of days to analyze (default: 7)
    """
    try:
        result = await chat_service.get_website_session_analytics(website_id, days)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to get analytics'))
        
        return {
            "success": True,
            "website_id": str(website_id),
            "period_days": days,
            "analytics": result['analytics']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Error getting session analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session analytics: {str(e)}")


@router.post("/admin/cleanup-expired")
async def cleanup_expired_sessions(
    supabase: Client = Depends(get_supabase_client)
):
    """
    Clean up expired sessions (admin endpoint).
    
    This endpoint should be called periodically to clean up expired sessions
    and maintain database performance.
    """
    try:
        result = await chat_service.cleanup_expired_sessions()
        
        return {
            "success": result['success'],
            "message": result['message'],
            "cleaned_sessions": result['cleaned_sessions']
        }
        
    except Exception as e:
        logger.error(f"❌ API: Error cleaning up expired sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup expired sessions: {str(e)}")


# Health check endpoint
@router.get("/health")
async def session_service_health():
    """
    Health check for session service.
    """
    return {
        "status": "healthy",
        "service": "session_chat",
        "features": [
            "session_management",
            "context_optimization", 
            "conversation_history",
            "rag_integration",
            "analytics"
        ]
    }