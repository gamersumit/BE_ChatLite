"""
Supabase service for database operations.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from supabase import Client
from ..core.supabase_client import get_supabase, get_supabase_admin


class SupabaseService:
    """Service for Supabase database operations."""
    
    def __init__(self, use_admin: bool = False):
        self.client: Client = get_supabase_admin() if use_admin else get_supabase()
    
    # Website operations
    def create_website(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new website."""
        website_data = {
            "id": str(uuid.uuid4()),
            "domain": data.get("domain"),
            "widget_id": data.get("widget_id", str(uuid.uuid4())),
            "name": data.get("name"),
            "description": data.get("description"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self.client.table("websites").insert(website_data).execute()
        return result.data[0] if result.data else {}
    
    def get_website_by_widget_id(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get website by widget ID."""
        result = self.client.table("websites").select("*").eq("widget_id", widget_id).execute()
        return result.data[0] if result.data else None
    
    def get_website_by_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get website by domain."""
        result = self.client.table("websites").select("*").eq("domain", domain).execute()
        return result.data[0] if result.data else None
    
    # Conversation operations
    def create_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new conversation."""
        conversation_data = {
            "id": str(uuid.uuid4()),
            "session_id": data.get("session_id"),
            "website_id": data.get("website_id"),
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
            "user_email": data.get("user_email"),
            "is_active": True,
            "is_resolved": False,
            "status": "active",
            "page_url": data.get("page_url"),
            "page_title": data.get("page_title"),
            "user_agent": data.get("user_agent"),
            "ip_address": data.get("ip_address"),
            "referrer": data.get("referrer"),
            "total_messages": 0,
            "user_messages": 0,
            "ai_messages": 0,
            "started_at": datetime.utcnow().isoformat(),
            "last_activity_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "lead_captured": False,
            "lead_data": None,
            "satisfaction_rating": None,
            "feedback": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self.client.table("conversations").insert(conversation_data).execute()
        return result.data[0] if result.data else {}
    
    def get_conversation_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by session ID."""
        result = self.client.table("conversations").select("*").eq("session_id", session_id).execute()
        return result.data[0] if result.data else None
    
    # Message operations
    def create_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new message."""
        # Get sequence number (simplified - in production you'd want to handle this properly)
        sequence_number = 1
        try:
            existing = self.client.table("messages").select("sequence_number").eq("conversation_id", data.get("conversation_id")).order("sequence_number", desc=True).limit(1).execute()
            if existing.data:
                sequence_number = existing.data[0]["sequence_number"] + 1
        except:
            pass
        
        metadata = data.get("metadata", {})
        content = data.get("content", "")
        
        message_data = {
            "id": str(uuid.uuid4()),
            "conversation_id": data.get("conversation_id"),
            "content": content,
            "message_type": data.get("role", "user"),  # Using message_type instead of role
            "sequence_number": sequence_number,
            "word_count": len(content.split()) if content else 0,
            "character_count": len(content) if content else 0,
            "processing_time_ms": metadata.get("processing_time_ms"),
            "model_used": metadata.get("model_used"),
            "tokens_used": metadata.get("tokens_used"),
            "cost_usd": metadata.get("cost_usd"),
            "context_sources": metadata.get("context_sources"),
            "confidence_score": metadata.get("confidence_score"),
            "was_helpful": None,
            "user_reaction": None,
            "status": "sent",
            "delivery_attempts": 0,
            "error_message": None,
            "sent_at": datetime.utcnow().isoformat(),
            "delivered_at": None,
            "read_at": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self.client.table("messages").insert(message_data).execute()
        return result.data[0] if result.data else {}
    
    def get_messages_by_conversation(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        result = (self.client.table("messages")
                 .select("*")
                 .eq("conversation_id", conversation_id)
                 .order("created_at", desc=False)
                 .limit(limit)
                 .execute())
        return result.data or []
    
    def get_recent_messages_by_session(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for a session."""
        # First get the conversation
        conversation = self.get_conversation_by_session(session_id)
        if not conversation:
            return []
        
        result = (self.client.table("messages")
                 .select("*")
                 .eq("conversation_id", conversation["id"])
                 .order("created_at", desc=True)
                 .limit(limit)
                 .execute())
        
        # Reverse to get chronological order
        return list(reversed(result.data or []))
    
    # Analytics operations
    def create_analytics_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an analytics event."""
        event_data = {
            "id": str(uuid.uuid4()),
            "website_id": data.get("website_id"),
            "session_id": data.get("session_id"),
            "event_type": data.get("event_type"),
            "event_data": data.get("event_data", {}),
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = self.client.table("analytics_events").insert(event_data).execute()
        return result.data[0] if result.data else {}
    
    # Utility methods
    def test_connection(self) -> bool:
        """Test Supabase connection."""
        try:
            result = self.client.table("websites").select("id").limit(1).execute()
            return True
        except Exception as e:
            print(f"Supabase connection test failed: {e}")
            return False