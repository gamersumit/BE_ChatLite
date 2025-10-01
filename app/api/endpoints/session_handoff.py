"""
Session Handoff API endpoints for secure session sharing and transfers.
Provides controlled access to session data for customer service and collaboration.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, EmailStr

from ...services.session_handoff_service import SessionHandoffService
from ...services.session_service import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/handoff", tags=["session-handoff"])


class TransferRequest(BaseModel):
    """Request model for creating session transfers."""
    session_token: str
    transfer_type: str = 'share'  # 'share', 'handoff', 'export'
    recipient_email: Optional[EmailStr] = None
    access_level: str = 'read'  # 'read', 'write', 'full'
    expires_in_hours: Optional[int] = 24
    notes: Optional[str] = None


class AcceptTransferRequest(BaseModel):
    """Request model for accepting session transfers."""
    transfer_token: str
    recipient_visitor_id: str
    recipient_metadata: Optional[Dict[str, Any]] = None


@router.post("/create-transfer")
async def create_session_transfer(
    request: TransferRequest
) -> Dict[str, Any]:
    """
    Create a secure transfer link for session sharing or handoff.
    
    Args:
        request: Transfer request details
        
    Returns:
        Transfer information including secure link
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Validate transfer type
        if request.transfer_type not in ['share', 'handoff', 'export']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transfer type: {request.transfer_type}"
            )
        
        # Validate access level
        if request.access_level not in ['read', 'write', 'full']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid access level: {request.access_level}"
            )
        
        # Create transfer
        transfer_data = await handoff_service.create_session_transfer(
            session_token=request.session_token,
            transfer_type=request.transfer_type,
            recipient_email=request.recipient_email,
            access_level=request.access_level,
            expires_in_hours=request.expires_in_hours,
            notes=request.notes
        )
        
        logger.info(f"✅ HANDOFF_API: Created {request.transfer_type} transfer for session {request.session_token[:8]}...")
        return transfer_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error creating transfer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create transfer: {str(e)}")


@router.post("/accept-transfer")
async def accept_session_transfer(
    request: AcceptTransferRequest
) -> Dict[str, Any]:
    """
    Accept and process a session transfer using the transfer token.
    
    Args:
        request: Transfer acceptance details
        
    Returns:
        New session information for the recipient
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Accept transfer
        result = await handoff_service.accept_session_transfer(
            transfer_token=request.transfer_token,
            recipient_visitor_id=request.recipient_visitor_id,
            recipient_metadata=request.recipient_metadata
        )
        
        logger.info(f"✅ HANDOFF_API: Transfer {request.transfer_token[:8]}... accepted by {request.recipient_visitor_id}")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error accepting transfer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to accept transfer: {str(e)}")


@router.post("/revoke-transfer")
async def revoke_session_transfer(
    transfer_token: str = Body(..., description="Transfer token to revoke"),
    reason: Optional[str] = Body(None, description="Reason for revocation")
) -> Dict[str, Any]:
    """
    Revoke a session transfer link to prevent further use.
    
    Args:
        transfer_token: Transfer token to revoke
        reason: Optional reason for revocation
        
    Returns:
        Revocation confirmation
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Revoke transfer
        result = await handoff_service.revoke_session_transfer(
            transfer_token=transfer_token,
            reason=reason
        )
        
        logger.info(f"✅ HANDOFF_API: Transfer {transfer_token[:8]}... revoked")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error revoking transfer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to revoke transfer: {str(e)}")


@router.get("/transfer-status/{transfer_token}")
async def get_transfer_status(
    transfer_token: str
) -> Dict[str, Any]:
    """
    Get the current status of a session transfer.
    
    Args:
        transfer_token: Transfer token to check
        
    Returns:
        Transfer status and details
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Get transfer status
        status = await handoff_service.get_transfer_status(transfer_token)
        
        if status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Transfer not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error getting transfer status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get transfer status: {str(e)}")


@router.get("/list-transfers")
async def list_session_transfers(
    session_token: Optional[str] = Query(None, description="Filter by session token"),
    visitor_id: Optional[str] = Query(None, description="Filter by visitor ID"),
    status: Optional[str] = Query(None, description="Filter by transfer status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
) -> Dict[str, Any]:
    """
    List session transfers based on filters.
    
    Args:
        session_token: Optional filter by original session token
        visitor_id: Optional filter by visitor ID
        status: Optional filter by transfer status
        limit: Maximum number of results
        
    Returns:
        List of transfer records
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Get transfers
        transfers = await handoff_service.list_session_transfers(
            session_token=session_token,
            visitor_id=visitor_id,
            status=status,
            limit=limit
        )
        
        return {
            'transfers': transfers,
            'count': len(transfers),
            'filters': {
                'session_token': session_token,
                'visitor_id': visitor_id,
                'status': status
            }
        }
        
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error listing transfers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list transfers: {str(e)}")


@router.post("/customer-service-handoff")
async def create_customer_service_handoff(
    session_token: str = Body(..., description="Session to handoff"),
    agent_id: str = Body(..., description="Customer service agent ID"),
    priority: str = Body('normal', description="Handoff priority"),
    issue_description: Optional[str] = Body(None, description="Description of the issue")
) -> Dict[str, Any]:
    """
    Create a special handoff for customer service agents with priority routing.
    
    Args:
        session_token: Session to handoff to customer service
        agent_id: ID of the customer service agent
        priority: Priority level (low, normal, high, urgent)
        issue_description: Optional description of the customer's issue
        
    Returns:
        Handoff details and agent access information
    """
    try:
        handoff_service = SessionHandoffService()
        session_service = SessionService()
        
        # Validate priority
        if priority not in ['low', 'normal', 'high', 'urgent']:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Get session information
        session_data = await session_service.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create customer service handoff with full access
        transfer_data = await handoff_service.create_session_transfer(
            session_token=session_token,
            transfer_type='handoff',
            access_level='full',
            expires_in_hours=72,  # 3 days for CS handoffs
            notes=f"Customer Service Handoff - Agent: {agent_id}, Priority: {priority}, Issue: {issue_description}"
        )
        
        # Add CS-specific metadata
        cs_handoff_data = {
            **transfer_data,
            'customer_service': {
                'agent_id': agent_id,
                'priority': priority,
                'issue_description': issue_description,
                'handoff_time': datetime.now().isoformat(),
                'session_info': {
                    'visitor_id': session_data.get('visitor_id'),
                    'website_id': session_data.get('website_id'),
                    'conversation_count': session_data.get('conversation_count', 0)
                }
            }
        }
        
        logger.info(f"✅ HANDOFF_API: Created CS handoff for session {session_token[:8]}... to agent {agent_id}")
        return cs_handoff_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error creating CS handoff: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create customer service handoff: {str(e)}")


@router.get("/validate-transfer/{transfer_token}")
async def validate_transfer_link(
    transfer_token: str
) -> Dict[str, Any]:
    """
    Validate if a transfer link is valid and can be used.
    
    Args:
        transfer_token: Transfer token to validate
        
    Returns:
        Validation result with transfer details if valid
    """
    try:
        handoff_service = SessionHandoffService()
        
        # Get transfer status
        status = await handoff_service.get_transfer_status(transfer_token)
        
        if status.get('status') == 'not_found':
            return {
                'valid': False,
                'reason': 'Transfer link not found'
            }
        
        if status.get('status') == 'completed':
            return {
                'valid': False,
                'reason': 'Transfer already completed'
            }
        
        if status.get('status') == 'revoked':
            return {
                'valid': False,
                'reason': 'Transfer has been revoked',
                'revoked_at': status.get('revoked_at')
            }
        
        if status.get('is_expired'):
            return {
                'valid': False,
                'reason': 'Transfer link has expired',
                'expired_at': status.get('expires_at')
            }
        
        attempts_remaining = status.get('max_attempts', 3) - status.get('attempts', 0)
        if attempts_remaining <= 0:
            return {
                'valid': False,
                'reason': 'Maximum transfer attempts exceeded'
            }
        
        return {
            'valid': True,
            'transfer_type': status.get('transfer_type'),
            'access_level': status.get('access_level'),
            'expires_at': status.get('expires_at'),
            'attempts_remaining': attempts_remaining
        }
        
    except Exception as e:
        logger.error(f"❌ HANDOFF_API: Error validating transfer: {e}")
        return {
            'valid': False,
            'reason': 'Error validating transfer link'
        }