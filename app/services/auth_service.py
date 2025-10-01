"""
Authentication service for user management.
Handles password hashing, JWT tokens, and user verification.
"""

import secrets
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
import bcrypt
import jwt
from supabase import Client

from app.core.config import get_settings
from app.core.database import get_supabase, get_supabase_admin

settings = get_settings()

# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT refresh token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except jwt.PyJWTError:
        return None


class AuthService:
    """Service for handling user authentication operations."""

    def __init__(self):
        self.settings = settings
        self.supabase = get_supabase_admin()

    # Password Management
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        password_bytes = password.encode('utf-8')
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            password_bytes = plain_password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False

    def validate_password_strength(self, password: str) -> bool:
        """
        Validate password meets security requirements.
        Requirements:
        - At least 8 characters
        - Contains uppercase letter
        - Contains lowercase letter  
        - Contains digit
        - Contains special character
        """
        if len(password) < 8:
            return False
        
        # Check for required character types
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        
        return all([has_upper, has_lower, has_digit, has_special])

    # Token Management
    def generate_verification_token(self) -> str:
        """Generate a secure random token for email verification."""
        return secrets.token_urlsafe(32)

    def generate_reset_token(self) -> str:
        """Generate a secure random token for password reset."""
        return secrets.token_urlsafe(32)

    # User Operations
    async def create_user(
        self, 
        email: str, 
        password: str, 
        name: str
    ) -> Optional[dict]:
        """
        Create a new user with hashed password and verification token.
        Returns None if user already exists.
        """
        try:
            # Check if user already exists
            result = self.supabase.table('users').select('*').eq('email', email.lower()).execute()
            if result.data:
                return None

            # Validate password strength
            if not self.validate_password_strength(password):
                raise ValueError("Password does not meet security requirements")

            # Create new user
            hashed_password = self.hash_password(password)
            verification_token = self.generate_verification_token()
            
            user_data = {
                'id': str(uuid4()),
                'email': email.lower().strip(),
                'password_hash': hashed_password,
                'name': name.strip(),
                'email_verified': True,  # Skip email verification
                'verification_token': verification_token,
                'is_active': True,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.supabase.table('users').insert(user_data).execute()
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error creating user: {e}")
            return None

    async def authenticate_user(
        self, 
        email: str, 
        password: str
    ) -> Optional[dict]:
        """
        Authenticate user with email and password.
        Returns None if authentication fails.
        """
        try:
            result = self.supabase.table('users').select('*').eq('email', email.lower()).execute()
            if not result.data:
                return None
                
            user = result.data[0]
            
            if not user['is_active']:
                return None
                
            if not self.verify_password(password, user['password_hash']):
                return None
                
            # Update last login
            self.supabase.table('users').update({
                'last_login': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', user['id']).execute()
            
            return user
            
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None

    async def get_user_by_id(self, user_id: UUID) -> Optional[dict]:
        """Get user by ID."""
        try:
            result = self.supabase.table('users').select('*').eq('id', str(user_id)).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email."""
        try:
            result = self.supabase.table('users').select('*').eq('email', email.lower()).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None

    async def verify_user_email(self, verification_token: str) -> Optional[dict]:
        """Verify user email with token."""
        try:
            result = self.supabase.table('users').select('*').eq('verification_token', verification_token).execute()
            if not result.data:
                return None
                
            user = result.data[0]
            
            # Update user as verified
            update_result = self.supabase.table('users').update({
                'email_verified': True,
                'verification_token': None,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', user['id']).execute()
            
            return update_result.data[0] if update_result.data else None
            
        except Exception as e:
            print(f"Error verifying user email: {e}")
            return None

    async def initiate_password_reset(self, email: str) -> Optional[str]:
        """Initiate password reset process."""
        try:
            result = self.supabase.table('users').select('*').eq('email', email.lower()).execute()
            if not result.data:
                return None
                
            user = result.data[0]
            reset_token = self.generate_reset_token()
            reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
            
            self.supabase.table('users').update({
                'reset_token': reset_token,
                'reset_token_expires': reset_expires.isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', user['id']).execute()
            
            return reset_token
            
        except Exception as e:
            print(f"Error initiating password reset: {e}")
            return None

    async def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password using reset token."""
        try:
            # Validate password strength
            if not self.validate_password_strength(new_password):
                raise ValueError("Password does not meet security requirements")
            
            result = self.supabase.table('users').select('*').eq('reset_token', reset_token).execute()
            if not result.data:
                return False
                
            user = result.data[0]
            
            # Check if token has expired
            if user['reset_token_expires']:
                expires_at = datetime.fromisoformat(user['reset_token_expires'].replace('Z', '+00:00'))
                if expires_at < datetime.now(timezone.utc):
                    return False
            
            # Update password and clear reset token
            hashed_password = self.hash_password(new_password)
            self.supabase.table('users').update({
                'password_hash': hashed_password,
                'reset_token': None,
                'reset_token_expires': None,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', user['id']).execute()
            
            return True
            
        except Exception as e:
            print(f"Error resetting password: {e}")
            return False