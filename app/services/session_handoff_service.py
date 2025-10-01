"""
Session Handoff Service for transferring sessions between visitors/users.
Provides secure session sharing and customer service handoff capabilities.
"""

import secrets
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
import json
import hashlib

from ..core.config import get_settings
from ..core.database import get_supabase_client

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionHandoffService:
    """
    Service for managing session transfers and handoffs between users.
    """
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.transfer_expiry_hours = 24  # Transfer links expire after 24 hours
        self.max_transfer_attempts = 3  # Maximum attempts to use a transfer link
    
    async def create_session_transfer(
        self,
        session_token: str,
        transfer_type: str = 'share',  # 'share', 'handoff', 'export'
        recipient_email: Optional[str] = None,
        access_level: str = 'read',  # 'read', 'write', 'full'
        expires_in_hours: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a secure transfer link for session sharing.
        
        Args:
            session_token: Session to transfer
            transfer_type: Type of transfer (share, handoff, export)
            recipient_email: Optional email of recipient for notifications
            access_level: Level of access to grant
            expires_in_hours: Custom expiry time in hours
            notes: Optional notes about the transfer
            
        Returns:
            Transfer information including secure link
        """
        try:
            # Validate session exists
            session_result = self.supabase.table('conversations')\
                .select('id, visitor_id, website_id')\
                .eq('session_token', session_token)\
                .limit(1)\
                .execute()
            
            if not session_result.data:
                raise ValueError(f"Session not found: {session_token}")
            
            session_data = session_result.data[0]
            
            # Generate secure transfer token
            transfer_token = f"transfer_{secrets.token_urlsafe(32)}"
            transfer_id = str(uuid4())
            
            # Calculate expiry
            expiry_hours = expires_in_hours or self.transfer_expiry_hours
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
            
            # Create transfer record in database
            transfer_data = {
                'id': transfer_id,
                'transfer_token': transfer_token,
                'session_token': session_token,
                'transfer_type': transfer_type,
                'access_level': access_level,
                'recipient_email': recipient_email,
                'expires_at': expires_at.isoformat(),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'visitor_id': session_data['visitor_id'],
                'website_id': session_data['website_id'],
                'notes': notes,
                'status': 'pending',
                'attempts': 0,
                'max_attempts': self.max_transfer_attempts
            }
            
            # Store transfer data (using conversations table with special type)
            self.supabase.table('conversations').insert({
                'id': transfer_id,
                'session_token': transfer_token,  # Store transfer token as session token
                'visitor_id': f"transfer_{session_data['visitor_id']}",
                'website_id': session_data['website_id'],
                'metadata': json.dumps(transfer_data),
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            # Generate secure transfer link
            base_url = settings.app_base_url if hasattr(settings, 'app_base_url') else 'http://localhost:3000'
            transfer_link = f"{base_url}/session/transfer/{transfer_token}"
            
            # Log transfer creation
            logger.info(f"✅ SESSION_TRANSFER: Created {transfer_type} transfer for session {session_token[:8]}...")
            
            return {
                'transfer_id': transfer_id,
                'transfer_token': transfer_token,
                'transfer_link': transfer_link,
                'transfer_type': transfer_type,
                'access_level': access_level,
                'expires_at': expires_at.isoformat(),
                'recipient_email': recipient_email,
                'notes': notes,
                'session_info': {
                    'session_token': session_token,
                    'visitor_id': session_data['visitor_id'],
                    'website_id': session_data['website_id']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION_TRANSFER: Error creating transfer: {e}")
            raise
    
    async def accept_session_transfer(
        self,
        transfer_token: str,
        recipient_visitor_id: str,
        recipient_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Accept and process a session transfer.
        
        Args:
            transfer_token: Transfer token from the link
            recipient_visitor_id: ID of the recipient accepting the transfer
            recipient_metadata: Optional metadata about the recipient
            
        Returns:
            New session information for the recipient
        """
        try:
            # Get transfer record
            transfer_result = self.supabase.table('conversations')\
                .select('*')\
                .eq('session_token', transfer_token)\
                .limit(1)\
                .execute()
            
            if not transfer_result.data:
                raise ValueError("Invalid or expired transfer token")
            
            transfer_record = transfer_result.data[0]
            transfer_data = json.loads(transfer_record.get('metadata', '{}'))
            
            # Validate transfer
            if transfer_data.get('status') != 'pending':
                raise ValueError(f"Transfer already {transfer_data.get('status')}")
            
            # Check expiry
            expires_at = datetime.fromisoformat(transfer_data['expires_at'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_at:
                raise ValueError("Transfer link has expired")
            
            # Check attempts
            attempts = transfer_data.get('attempts', 0)
            if attempts >= transfer_data.get('max_attempts', self.max_transfer_attempts):
                raise ValueError("Maximum transfer attempts exceeded")
            
            # Get original session data
            original_session = transfer_data.get('session_token')
            original_convs = self.supabase.table('conversations')\
                .select('*')\
                .eq('session_token', original_session)\
                .execute()
            
            # Create new session for recipient
            new_session_token = f"session_{secrets.token_urlsafe(32)}"
            new_session_id = str(uuid4())
            
            # Handle different transfer types
            if transfer_data['transfer_type'] == 'handoff':
                # Complete handoff - transfer ownership
                for conv in original_convs.data:
                    # Update conversation to new session
                    self.supabase.table('conversations').update({
                        'session_token': new_session_token,
                        'visitor_id': recipient_visitor_id,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }).eq('id', conv['id']).execute()
                
                # Mark original session as transferred
                self.supabase.table('conversations').update({
                    'metadata': json.dumps({
                        **json.loads(conv.get('metadata', '{}')),
                        'transferred_to': new_session_token,
                        'transferred_at': datetime.now(timezone.utc).isoformat()
                    })
                }).eq('session_token', original_session).execute()
                
            elif transfer_data['transfer_type'] == 'share':
                # Share session - create linked copy
                for conv in original_convs.data:
                    # Create new conversation linked to original
                    new_conv = {
                        'id': str(uuid4()),
                        'session_token': new_session_token,
                        'visitor_id': recipient_visitor_id,
                        'website_id': conv['website_id'],
                        'parent_conversation_id': conv['id'],
                        'metadata': json.dumps({
                            'shared_from': original_session,
                            'access_level': transfer_data['access_level'],
                            'shared_at': datetime.now(timezone.utc).isoformat()
                        }),
                        'created_at': datetime.now(timezone.utc).isoformat()
                    }
                    self.supabase.table('conversations').insert(new_conv).execute()
                    
                    # Copy messages based on access level
                    if transfer_data['access_level'] in ['read', 'write', 'full']:
                        messages = self.supabase.table('messages')\
                            .select('*')\
                            .eq('conversation_id', conv['id'])\
                            .execute()
                        
                        for msg in messages.data:
                            new_msg = {
                                'id': str(uuid4()),
                                'conversation_id': new_conv['id'],
                                'content': msg['content'],
                                'message_type': msg['message_type'],
                                'created_at': msg['created_at']
                            }
                            self.supabase.table('messages').insert(new_msg).execute()
            
            # Update transfer record
            transfer_data['status'] = 'completed'
            transfer_data['attempts'] = attempts + 1
            transfer_data['accepted_by'] = recipient_visitor_id
            transfer_data['accepted_at'] = datetime.now(timezone.utc).isoformat()
            
            self.supabase.table('conversations').update({
                'metadata': json.dumps(transfer_data)
            }).eq('id', transfer_record['id']).execute()
            
            logger.info(f"✅ SESSION_TRANSFER: Accepted transfer {transfer_token[:8]}... by {recipient_visitor_id}")
            
            return {
                'success': True,
                'new_session_token': new_session_token,
                'transfer_type': transfer_data['transfer_type'],
                'access_level': transfer_data['access_level'],
                'original_session': original_session,
                'recipient_visitor_id': recipient_visitor_id,
                'message': f"Successfully accepted {transfer_data['transfer_type']} transfer"
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION_TRANSFER: Error accepting transfer: {e}")
            raise
    
    async def revoke_session_transfer(
        self,
        transfer_token: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Revoke a session transfer link.
        
        Args:
            transfer_token: Transfer token to revoke
            reason: Optional reason for revocation
            
        Returns:
            Revocation confirmation
        """
        try:
            # Get transfer record
            transfer_result = self.supabase.table('conversations')\
                .select('*')\
                .eq('session_token', transfer_token)\
                .limit(1)\
                .execute()
            
            if not transfer_result.data:
                raise ValueError("Transfer token not found")
            
            transfer_record = transfer_result.data[0]
            transfer_data = json.loads(transfer_record.get('metadata', '{}'))
            
            # Update status to revoked
            transfer_data['status'] = 'revoked'
            transfer_data['revoked_at'] = datetime.now(timezone.utc).isoformat()
            transfer_data['revocation_reason'] = reason
            
            self.supabase.table('conversations').update({
                'metadata': json.dumps(transfer_data)
            }).eq('id', transfer_record['id']).execute()
            
            logger.info(f"✅ SESSION_TRANSFER: Revoked transfer {transfer_token[:8]}...")
            
            return {
                'success': True,
                'transfer_token': transfer_token,
                'status': 'revoked',
                'revoked_at': transfer_data['revoked_at'],
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION_TRANSFER: Error revoking transfer: {e}")
            raise
    
    async def get_transfer_status(
        self,
        transfer_token: str
    ) -> Dict[str, Any]:
        """
        Get the status of a session transfer.
        
        Args:
            transfer_token: Transfer token to check
            
        Returns:
            Transfer status and details
        """
        try:
            # Get transfer record
            transfer_result = self.supabase.table('conversations')\
                .select('*')\
                .eq('session_token', transfer_token)\
                .limit(1)\
                .execute()
            
            if not transfer_result.data:
                return {
                    'status': 'not_found',
                    'message': 'Transfer token not found'
                }
            
            transfer_record = transfer_result.data[0]
            transfer_data = json.loads(transfer_record.get('metadata', '{}'))
            
            # Check expiry
            expires_at = datetime.fromisoformat(transfer_data['expires_at'].replace('Z', '+00:00'))
            is_expired = datetime.now(timezone.utc) > expires_at
            
            return {
                'transfer_token': transfer_token,
                'status': transfer_data.get('status', 'unknown'),
                'transfer_type': transfer_data.get('transfer_type'),
                'access_level': transfer_data.get('access_level'),
                'created_at': transfer_data.get('created_at'),
                'expires_at': transfer_data.get('expires_at'),
                'is_expired': is_expired,
                'attempts': transfer_data.get('attempts', 0),
                'max_attempts': transfer_data.get('max_attempts', self.max_transfer_attempts),
                'accepted_by': transfer_data.get('accepted_by'),
                'accepted_at': transfer_data.get('accepted_at'),
                'revoked_at': transfer_data.get('revoked_at'),
                'revocation_reason': transfer_data.get('revocation_reason')
            }
            
        except Exception as e:
            logger.error(f"❌ SESSION_TRANSFER: Error getting transfer status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def list_session_transfers(
        self,
        session_token: Optional[str] = None,
        visitor_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List session transfers based on filters.
        
        Args:
            session_token: Filter by original session token
            visitor_id: Filter by visitor ID
            status: Filter by transfer status
            limit: Maximum number of results
            
        Returns:
            List of transfer records
        """
        try:
            # Build query
            query = self.supabase.table('conversations').select('*')
            query = query.like('visitor_id', 'transfer_%')
            query = query.limit(limit)
            
            result = query.execute()
            
            transfers = []
            for record in result.data:
                metadata = json.loads(record.get('metadata', '{}'))
                if not metadata.get('transfer_token'):
                    continue
                
                # Apply filters
                if session_token and metadata.get('session_token') != session_token:
                    continue
                if visitor_id and metadata.get('visitor_id') != visitor_id:
                    continue
                if status and metadata.get('status') != status:
                    continue
                
                transfers.append({
                    'transfer_id': record['id'],
                    'transfer_token': metadata.get('transfer_token'),
                    'session_token': metadata.get('session_token'),
                    'transfer_type': metadata.get('transfer_type'),
                    'status': metadata.get('status'),
                    'created_at': metadata.get('created_at'),
                    'expires_at': metadata.get('expires_at'),
                    'access_level': metadata.get('access_level')
                })
            
            return transfers
            
        except Exception as e:
            logger.error(f"❌ SESSION_TRANSFER: Error listing transfers: {e}")
            return []