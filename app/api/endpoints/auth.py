"""
Authentication API endpoints.
Handles user registration, login, email verification, and password reset.
"""

from typing import Optional
from datetime import timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field
from supabase import Client

from app.core.database import get_supabase_client
from app.services.auth_service import AuthService, create_access_token, create_refresh_token
from app.services.email_service import send_verification_email, send_password_reset_email, send_welcome_email
from app.core.config import get_settings
from app.core.auth_middleware import get_current_user

router = APIRouter()
settings = get_settings()


# Request/Response Models
class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)


class UserRegistrationResponse(BaseModel):
    """Response model for user registration."""
    message: str
    user_id: str


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    email: EmailStr
    password: str


class UserData(BaseModel):
    """User data for API responses."""
    id: str
    email: str
    name: str
    email_verified: bool


class UserLoginResponse(BaseModel):
    """Response model for user login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds
    user: UserData


class EmailVerificationRequest(BaseModel):
    """Request model for email verification."""
    token: str


class EmailVerificationResponse(BaseModel):
    """Response model for email verification."""
    message: str


class PasswordResetRequest(BaseModel):
    """Request model for password reset."""
    email: EmailStr


class PasswordResetResponse(BaseModel):
    """Response model for password reset."""
    message: str


class PasswordResetConfirmRequest(BaseModel):
    """Request model for password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)


class PasswordResetConfirmResponse(BaseModel):
    """Response model for password reset confirmation."""
    message: str


class TokenRefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str

class TokenRefreshResponse(BaseModel):
    """Response model for token refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# API Endpoints
@router.post("/register", response_model=UserRegistrationResponse)
async def register_user(
    request: UserRegistrationRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Register a new user account.
    User can login immediately after registration.
    """
    auth_service = AuthService()
    
    try:
        # Create user
        user = await auth_service.create_user(
            email=request.email,
            password=request.password,
            name=request.name
        )
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email address is already registered"
            )
        
        # Email verification skipped - users can login immediately
        
        return UserRegistrationResponse(
            message="Registration successful! You can now login with your credentials.",
            user_id=str(user['id'])
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    request: UserLoginRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Authenticate user and return JWT token.
    No email verification required.
    """
    auth_service = AuthService()
    
    # Authenticate user
    user = await auth_service.authenticate_user(
        email=request.email,
        password=request.password
    )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Email verification check removed - users can login immediately after signup
    
    # Create JWT tokens
    token_data = {"user_id": user['id'], "email": user['email']}
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    user_data = UserData(
        id=str(user['id']),
        email=user['email'],
        name=user['name'],
        email_verified=user['email_verified']
    )

    return UserLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,  # Convert to seconds
        user=user_data
    )


@router.post("/verify-email", response_model=EmailVerificationResponse)
async def verify_email(
    request: EmailVerificationRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Verify user email address with verification token.
    Sends welcome email upon successful verification.
    """
    auth_service = AuthService()
    
    user = await auth_service.verify_user_email(token=request.token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    # Send welcome email
    try:
        await send_welcome_email(user)
    except Exception as e:
        import logging
        logging.warning(f"Failed to send welcome email: {e}")
    
    return EmailVerificationResponse(
        message="Email verified successfully. Welcome to LiteChat!"
    )


@router.post("/request-reset", response_model=PasswordResetResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Request password reset for user account.
    Sends reset email if account exists and is verified.
    """
    auth_service = AuthService()
    
    reset_token = await auth_service.initiate_password_reset(email=request.email)
    
    # Always return success to prevent email enumeration
    # But only send email if user actually exists
    if reset_token:
        user = await auth_service.get_user_by_email(request.email)
        if user:
            reset_url = f"{settings.frontend_url}/reset-password?token={reset_token}"
            try:
                await send_password_reset_email(user, reset_url)
            except Exception as e:
                import logging
                logging.warning(f"Failed to send password reset email: {e}")
    
    return PasswordResetResponse(
        message="If an account with that email exists, password reset instructions have been sent."
    )


@router.post("/reset-password", response_model=PasswordResetConfirmResponse)
async def reset_password(
    request: PasswordResetConfirmRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Reset user password with valid reset token.
    """
    auth_service = AuthService()
    
    try:
        success = await auth_service.reset_password(
            reset_token=request.token,
            new_password=request.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        return PasswordResetConfirmResponse(
            message="Password reset successful. You can now login with your new password."
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/me", response_model=UserData)
async def get_me(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    Requires valid JWT token.
    """
    user_data = UserData(
        id=str(current_user['id']),
        email=current_user['email'],
        name=current_user['name'],
        email_verified=current_user['email_verified']
    )

    return user_data


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Refresh JWT access token using refresh token.
    Returns new access and refresh tokens.
    """
    try:
        from app.services.auth_service import verify_token

        # Verify refresh token
        payload = verify_token(request.refresh_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Check if it's actually a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        # Get user info
        auth_service = AuthService()
        user = await auth_service.get_user_by_id(payload.get("user_id"))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        # Create new tokens
        token_data = {"user_id": user['id'], "email": user['email']}
        new_access_token = create_access_token(data=token_data)
        new_refresh_token = create_refresh_token(data=token_data)

        return TokenRefreshResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60  # Convert to seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )


