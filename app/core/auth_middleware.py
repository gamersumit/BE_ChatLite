"""
Authentication middleware and dependencies.
Handles JWT token validation and user authentication for protected routes.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.services.auth_service import verify_token

# Security scheme for JWT tokens
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """
    Dependency to get current authenticated user from JWT token (basic implementation).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify JWT token
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise credentials_exception
    
    # Extract user ID from token
    user_id: str = payload.get("user_id")
    if user_id is None:
        raise credentials_exception
    
    # Get user from Supabase
    try:
        result = supabase.table('users').select('*').eq('id', user_id).execute()
        if not result.data:
            raise credentials_exception
            
        user = result.data[0]
        if not user['is_active']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        return user
    except Exception:
        raise credentials_exception


async def get_current_verified_user(
    current_user: dict = Depends(get_current_user)
):
    """
    Dependency to get current authenticated user with verified email.
    """
    if not current_user['email_verified']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    supabase: Client = Depends(get_supabase_admin_client)
) -> Optional[dict]:
    """
    Dependency to get current user if authenticated, None otherwise.
    Does not raise exceptions for missing/invalid tokens.
    """
    if credentials is None:
        return None
    
    try:
        # Verify JWT token
        token = credentials.credentials
        payload = verify_token(token)
        
        if payload is None:
            return None
        
        # Extract user ID from token
        user_id: str = payload.get("user_id")
        if user_id is None:
            return None
        
        # Get user from Supabase
        try:
            result = supabase.table('users').select('*').eq('id', user_id).execute()
            if not result.data:
                return None
                
            user = result.data[0]
            if not user['is_active']:
                return None
            
            return user
        except Exception:
            return None
        
    except Exception:
        # Return None for any authentication errors
        return None


class RequireRoles:
    """
    Dependency class to require specific user roles.
    Usage: Depends(RequireRoles(["admin", "moderator"]))
    """
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: dict = Depends(get_current_verified_user)) -> dict:
        """Check if user has required role."""
        # For now, all verified users have "user" role
        # This can be extended when user roles are implemented
        user_role = "user"
        
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user


class RequireWebsiteOwnership:
    """
    Dependency class to require website ownership.
    Usage: Depends(RequireWebsiteOwnership())
    """
    
    def __init__(self):
        pass
    
    async def __call__(
        self, 
        website_id: str,
        current_user: dict = Depends(get_current_verified_user),
        supabase: Client = Depends(get_supabase_admin_client)
    ) -> dict:
        """Check if user owns the specified website."""
        from uuid import UUID
        
        try:
            website_uuid = UUID(website_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid website ID format"
            )
        
        # Check if user has access to website
        try:
            result = supabase.table('user_websites').select('*').eq('user_id', current_user['id']).eq('website_id', str(website_uuid)).execute()
            if not result.data:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have permission to access this website"
                )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to access this website"
            )
        
        return current_user


# Rate limiting dependency (basic implementation)
class RateLimit:
    """
    Basic rate limiting dependency.
    Usage: Depends(RateLimit(max_requests=100, window_seconds=3600))
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # In production, use Redis or similar
    
    async def __call__(self, current_user: dict = Depends(get_current_user)) -> dict:
        """Check rate limit for current user."""
        import time
        
        user_id = str(current_user['id'])
        current_time = time.time()
        
        # Clean old entries (basic cleanup)
        if user_id in self.requests:
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[user_id] = []
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        self.requests[user_id].append(current_time)
        
        return current_user