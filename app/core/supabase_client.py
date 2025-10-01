"""
Supabase client configuration and utilities.
"""
from typing import Optional
from supabase import create_client, Client
from .config import settings


class SupabaseClient:
    """Supabase client wrapper with connection management."""
    
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls) -> 'SupabaseClient':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._client = None
            self._initialized = True
    
    @property
    def client(self) -> Client:
        """Get Supabase client instance."""
        if self._client is None:
            if not settings.supabase_url or not settings.supabase_anon_key:
                raise ValueError("Supabase URL and anon key must be configured")

            self._client = create_client(
                settings.supabase_url,
                settings.supabase_anon_key
            )
        
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get Supabase client with service role key for admin operations."""
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise ValueError("Supabase URL and service role key must be configured")
        
        return create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )


# Global instance
supabase_client = SupabaseClient()


def get_supabase() -> Client:
    """Get Supabase client instance."""
    return supabase_client.client


def get_supabase_admin() -> Client:
    """Get Supabase admin client instance."""
    return supabase_client.service_client