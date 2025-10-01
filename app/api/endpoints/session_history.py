"""
Session History API endpoints for retrieving and managing conversation session data.
Provides detailed history access with pagination, filtering, and export capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID
import json
import csv
import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from ...core.config import get_settings
from ...core.database import get_supabase_client
from ...core.auth_middleware import get_current_user
from ...services.session_service import SessionService
from ...services.context_service import ContextService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["session-history"])
settings = get_settings()


@router.get("/{session_token}/history")
async def get_session_history(
    session_token: str,
    current_user: dict = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    include_messages: bool = Query(True, description="Include message details"),
    include_context: bool = Query(False, description="Include context information"),
    start_date: Optional[datetime] = Query(None, description="Filter start date"),
    end_date: Optional[datetime] = Query(None, description="Filter end date")
) -> Dict[str, Any]:
    """
    Get paginated conversation history for a specific session.
    
    Args:
        session_token: Session token to retrieve history for
        page: Page number (1-indexed)
        page_size: Number of items per page
        include_messages: Whether to include full message details
        include_context: Whether to include context optimization information
        start_date: Optional filter for conversations after this date
        end_date: Optional filter for conversations before this date
        
    Returns:
        Paginated session history with conversation and message data
    """
    try:
        supabase = get_supabase_client()
        session_service = SessionService()
        
        # Validate session token
        session_data = await session_service.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if session is expired
        if session_data.get('expired'):
            raise HTTPException(status_code=403, detail="Session has expired")
        
        # Build query for conversations
        query = supabase.table('conversations').select('*')
        query = query.eq('session_token', session_token)
        
        # Apply date filters
        if start_date:
            query = query.gte('created_at', start_date.isoformat())
        if end_date:
            query = query.lte('created_at', end_date.isoformat())
        
        # Order by creation date descending
        query = query.order('created_at', desc=True)
        
        # Calculate pagination
        offset = (page - 1) * page_size
        query = query.range(offset, offset + page_size - 1)
        
        # Execute query
        result = query.execute()
        conversations = result.data if result.data else []
        
        # Get total count for pagination
        count_query = supabase.table('conversations').select('id', count='exact')
        count_query = count_query.eq('session_token', session_token)
        if start_date:
            count_query = count_query.gte('created_at', start_date.isoformat())
        if end_date:
            count_query = count_query.lte('created_at', end_date.isoformat())
        
        count_result = count_query.execute()
        total_count = count_result.count if count_result.count else 0
        
        # Process conversations
        history_items = []
        for conv in conversations:
            item = {
                'conversation_id': conv['id'],
                'created_at': conv['created_at'],
                'updated_at': conv.get('updated_at'),
                'visitor_id': conv.get('visitor_id'),
                'website_id': conv.get('website_id'),
                'message_count': 0,
                'context_tokens_used': conv.get('context_tokens_used', 0)
            }
            
            # Include messages if requested
            if include_messages:
                msg_result = supabase.table('messages')\
                    .select('*')\
                    .eq('conversation_id', conv['id'])\
                    .order('created_at', desc=False)\
                    .execute()
                
                messages = msg_result.data if msg_result.data else []
                item['messages'] = [
                    {
                        'id': msg['id'],
                        'content': msg['content'],
                        'message_type': msg['message_type'],
                        'created_at': msg['created_at'],
                        'context_importance_score': msg.get('context_importance_score'),
                        'summarized_in_context': msg.get('summarized_in_context', False)
                    }
                    for msg in messages
                ]
                item['message_count'] = len(messages)
            else:
                # Just get message count
                msg_count = supabase.table('messages')\
                    .select('id', count='exact')\
                    .eq('conversation_id', conv['id'])\
                    .execute()
                item['message_count'] = msg_count.count if msg_count.count else 0
            
            # Include context information if requested
            if include_context and conv.get('context_summary'):
                item['context_info'] = {
                    'has_summary': bool(conv.get('context_summary')),
                    'summary_preview': conv.get('context_summary', '')[:200] + '...' if conv.get('context_summary') else None,
                    'tokens_used': conv.get('context_tokens_used', 0),
                    'last_context_update': conv.get('last_context_update')
                }
            
            history_items.append(item)
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        
        response_data = {
            'session_token': session_token,
            'session_info': {
                'visitor_id': session_data['visitor_id'],
                'website_id': session_data['website_id'],
                'created_at': session_data['created_at'],
                'expires_at': session_data.get('session_expires_at'),
                'is_active': not session_data.get('expired', False)
            },
            'history': history_items,
            'pagination': {
                'current_page': page,
                'page_size': page_size,
                'total_items': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_previous': page > 1
            },
            'filters_applied': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'include_messages': include_messages,
                'include_context': include_context
            }
        }
        
        logger.info(f"✅ SESSION_HISTORY: Retrieved history for session {session_token[:8]}... (page {page}/{total_pages})")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SESSION_HISTORY: Error retrieving history for session {session_token}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session history: {str(e)}")


@router.get("/{session_token}/export")
async def export_session_data(
    session_token: str,
    format: str = Query("json", regex="^(json|csv|txt)$", description="Export format"),
    include_metadata: bool = Query(True, description="Include metadata in export")
) -> Response:
    """
    Export session data in various formats.
    
    Args:
        session_token: Session token to export data for
        format: Export format (json, csv, txt)
        include_metadata: Whether to include metadata in the export
        
    Returns:
        Downloadable file with session data
    """
    try:
        supabase = get_supabase_client()
        session_service = SessionService()
        
        # Validate session
        session_data = await session_service.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all conversations for the session
        conv_result = supabase.table('conversations')\
            .select('*')\
            .eq('session_token', session_token)\
            .order('created_at', desc=False)\
            .execute()
        
        conversations = conv_result.data if conv_result.data else []
        
        # Get all messages for these conversations
        all_messages = []
        for conv in conversations:
            msg_result = supabase.table('messages')\
                .select('*')\
                .eq('conversation_id', conv['id'])\
                .order('created_at', desc=False)\
                .execute()
            
            messages = msg_result.data if msg_result.data else []
            for msg in messages:
                msg['conversation_id'] = conv['id']
                msg['visitor_id'] = conv.get('visitor_id')
                all_messages.append(msg)
        
        # Format based on requested type
        if format == "json":
            export_data = {
                'export_info': {
                    'session_token': session_token,
                    'export_date': datetime.now().isoformat(),
                    'total_conversations': len(conversations),
                    'total_messages': len(all_messages)
                }
            }
            
            if include_metadata:
                export_data['session_metadata'] = {
                    'visitor_id': session_data['visitor_id'],
                    'website_id': session_data['website_id'],
                    'created_at': session_data['created_at'],
                    'expires_at': session_data.get('session_expires_at')
                }
            
            export_data['conversations'] = []
            for conv in conversations:
                conv_messages = [m for m in all_messages if m['conversation_id'] == conv['id']]
                export_data['conversations'].append({
                    'id': conv['id'],
                    'created_at': conv['created_at'],
                    'message_count': len(conv_messages),
                    'messages': conv_messages
                })
            
            content = json.dumps(export_data, indent=2, default=str)
            media_type = "application/json"
            filename = f"session_{session_token[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        elif format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            headers = ['timestamp', 'conversation_id', 'message_type', 'content']
            if include_metadata:
                headers.extend(['visitor_id', 'importance_score'])
            writer.writerow(headers)
            
            # Write messages
            for msg in all_messages:
                row = [
                    msg['created_at'],
                    msg['conversation_id'],
                    msg['message_type'],
                    msg['content']
                ]
                if include_metadata:
                    row.extend([
                        msg.get('visitor_id', ''),
                        msg.get('context_importance_score', '')
                    ])
                writer.writerow(row)
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"session_{session_token[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        else:  # txt format
            lines = []
            lines.append(f"Session History Export - {session_token[:16]}...")
            lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Total Conversations: {len(conversations)}")
            lines.append(f"Total Messages: {len(all_messages)}")
            lines.append("=" * 80)
            
            for conv in conversations:
                lines.append(f"\nConversation: {conv['id']}")
                lines.append(f"Started: {conv['created_at']}")
                lines.append("-" * 40)
                
                conv_messages = [m for m in all_messages if m['conversation_id'] == conv['id']]
                for msg in conv_messages:
                    timestamp = datetime.fromisoformat(msg['created_at'].replace('Z', '+00:00'))
                    lines.append(f"[{timestamp.strftime('%H:%M:%S')}] {msg['message_type'].upper()}: {msg['content']}")
                
                lines.append("")
            
            content = "\n".join(lines)
            media_type = "text/plain"
            filename = f"session_{session_token[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logger.info(f"✅ SESSION_EXPORT: Exported session {session_token[:8]}... as {format}")
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SESSION_EXPORT: Error exporting session {session_token}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export session data: {str(e)}")


@router.get("/{session_token}/summary")
async def get_session_summary(
    session_token: str
) -> Dict[str, Any]:
    """
    Get a summary of the session including key metrics and insights.
    
    Args:
        session_token: Session token to get summary for
        
    Returns:
        Session summary with metrics and key information
    """
    try:
        supabase = get_supabase_client()
        session_service = SessionService()
        context_service = ContextService()
        
        # Validate session
        session_data = await session_service.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get conversation statistics
        conv_result = supabase.table('conversations')\
            .select('id, created_at, updated_at, context_tokens_used')\
            .eq('session_token', session_token)\
            .execute()
        
        conversations = conv_result.data if conv_result.data else []
        
        # Calculate session duration
        if conversations:
            first_conv = min(conversations, key=lambda x: x['created_at'])
            last_conv = max(conversations, key=lambda x: x.get('updated_at', x['created_at']))
            
            start_time = datetime.fromisoformat(first_conv['created_at'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(last_conv.get('updated_at', last_conv['created_at']).replace('Z', '+00:00'))
            duration = end_time - start_time
        else:
            duration = timedelta(0)
        
        # Get message statistics
        total_messages = 0
        user_messages = 0
        assistant_messages = 0
        
        for conv in conversations:
            msg_result = supabase.table('messages')\
                .select('message_type', count='exact')\
                .eq('conversation_id', conv['id'])\
                .execute()
            
            if msg_result.data:
                for msg in msg_result.data:
                    total_messages += 1
                    if msg['message_type'] == 'user':
                        user_messages += 1
                    elif msg['message_type'] == 'assistant':
                        assistant_messages += 1
        
        # Calculate context usage
        total_context_tokens = sum(conv.get('context_tokens_used', 0) for conv in conversations)
        avg_context_tokens = total_context_tokens / len(conversations) if conversations else 0
        
        # Build summary
        summary = {
            'session_token': session_token,
            'session_info': {
                'visitor_id': session_data['visitor_id'],
                'website_id': session_data['website_id'],
                'created_at': session_data['created_at'],
                'expires_at': session_data.get('session_expires_at'),
                'is_active': not session_data.get('expired', False),
                'duration_minutes': int(duration.total_seconds() / 60)
            },
            'conversation_metrics': {
                'total_conversations': len(conversations),
                'total_messages': total_messages,
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'avg_messages_per_conversation': total_messages / len(conversations) if conversations else 0
            },
            'context_metrics': {
                'total_tokens_used': total_context_tokens,
                'avg_tokens_per_conversation': avg_context_tokens,
                'optimization_applied': total_context_tokens > 0
            },
            'engagement_indicators': {
                'session_active_time_minutes': int(duration.total_seconds() / 60),
                'conversation_completion_rate': assistant_messages / user_messages if user_messages > 0 else 0,
                'user_engagement_score': min(5.0, (total_messages / max(1, len(conversations))) / 10 * 5)
            }
        }
        
        # Add key topics if we have messages
        if total_messages > 5:
            # This would normally use NLP to extract topics
            summary['key_topics'] = ['General inquiry', 'Product information', 'Support']
        
        logger.info(f"✅ SESSION_SUMMARY: Generated summary for session {session_token[:8]}...")
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SESSION_SUMMARY: Error generating summary for session {session_token}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate session summary: {str(e)}")


@router.delete("/{session_token}")
async def delete_session_data(
    session_token: str,
    confirm: bool = Query(False, description="Confirm deletion of all session data")
) -> Dict[str, Any]:
    """
    Delete all data associated with a session (requires confirmation).
    
    Args:
        session_token: Session token to delete data for
        confirm: Must be true to proceed with deletion
        
    Returns:
        Deletion confirmation with statistics
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Deletion not confirmed. Set confirm=true to proceed."
            )
        
        supabase = get_supabase_client()
        session_service = SessionService()
        
        # Validate session
        session_data = await session_service.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all conversations for the session
        conv_result = supabase.table('conversations')\
            .select('id')\
            .eq('session_token', session_token)\
            .execute()
        
        conversation_ids = [conv['id'] for conv in (conv_result.data or [])]
        
        # Delete messages for all conversations
        deleted_messages = 0
        for conv_id in conversation_ids:
            msg_result = supabase.table('messages')\
                .delete()\
                .eq('conversation_id', conv_id)\
                .execute()
            deleted_messages += len(msg_result.data) if msg_result.data else 0
        
        # Delete conversations
        conv_delete = supabase.table('conversations')\
            .delete()\
            .eq('session_token', session_token)\
            .execute()
        deleted_conversations = len(conv_delete.data) if conv_delete.data else 0
        
        # Clear session token from conversations table (soft delete)
        supabase.table('conversations')\
            .update({'session_token': None})\
            .eq('session_token', session_token)\
            .execute()
        
        response_data = {
            'success': True,
            'session_token': session_token,
            'deleted': {
                'conversations': deleted_conversations,
                'messages': deleted_messages,
                'deletion_timestamp': datetime.now().isoformat()
            },
            'message': f"Successfully deleted all data for session {session_token[:8]}..."
        }
        
        logger.info(f"✅ SESSION_DELETE: Deleted session {session_token[:8]}... ({deleted_conversations} conversations, {deleted_messages} messages)")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SESSION_DELETE: Error deleting session {session_token}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session data: {str(e)}")