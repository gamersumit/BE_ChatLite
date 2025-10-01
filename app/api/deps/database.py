from supabase import Client
from ...core.database import get_supabase_client


async def get_db():
    """
    Dependency to provide Supabase client to FastAPI routes.
    """
    return get_supabase_client()