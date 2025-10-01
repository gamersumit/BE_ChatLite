"""
Website Onboarding Schema definitions.
Pydantic models for website registration and verification API.
"""

from typing import Optional, List
from pydantic import BaseModel, HttpUrl, Field


class WebsiteRegistrationRequest(BaseModel):
    """Request model for website registration."""
    url: HttpUrl
    name: str = Field(..., min_length=1, max_length=255)
    business_name: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    business_description: Optional[str] = Field(None, max_length=1000)
    widget_color: Optional[str] = Field("#0066CC", pattern=r"^#[0-9A-Fa-f]{6}$")
    widget_position: Optional[str] = Field("bottom-right", pattern=r"^(bottom-right|bottom-left|top-right|top-left)$")
    welcome_message: Optional[str] = Field(None, max_length=500)


class WebsiteResponse(BaseModel):
    """Response model for website data."""
    id: str
    name: str
    url: str
    domain: str
    widget_id: str
    verification_status: str
    scraping_status: str
    widget_status: str
    business_name: Optional[str] = None
    contact_email: Optional[str] = None
    business_description: Optional[str] = None
    widget_color: str
    widget_position: str
    welcome_message: Optional[str] = None
    is_active: bool
    created_at: Optional[str] = None
    verified_at: Optional[str] = None
    last_crawled: Optional[str] = None


class WebsiteListResponse(BaseModel):
    """Response model for website list."""
    websites: List[WebsiteResponse]
    total_count: int


class WebsiteDetailResponse(BaseModel):
    """Response model for detailed website information."""
    id: str
    name: str
    url: str
    domain: str
    widget_id: str
    verification_status: str
    verification_method: Optional[str] = None
    verified_at: Optional[str] = None
    scraping_status: str
    widget_status: str
    last_crawled: Optional[str] = None
    business_name: Optional[str] = None
    contact_email: Optional[str] = None
    business_description: Optional[str] = None
    widget_color: str
    widget_position: str
    welcome_message: Optional[str] = None
    total_conversations: int
    total_messages: int
    monthly_message_limit: int
    role: str
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WebsiteUpdateRequest(BaseModel):
    """Request model for website updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    business_name: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    business_description: Optional[str] = Field(None, max_length=1000)
    widget_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    widget_position: Optional[str] = Field(None, pattern=r"^(bottom-right|bottom-left|top-right|top-left)$")
    welcome_message: Optional[str] = Field(None, max_length=500)


class VerificationMethodsResponse(BaseModel):
    """Response model for verification methods."""
    verification_token: str
    methods: dict


class WebsiteVerificationRequest(BaseModel):
    """Request model for website verification."""
    method: str = Field(..., pattern=r"^(html_tag|dns_record)$")


class WebsiteVerificationResponse(BaseModel):
    """Response model for website verification result."""
    verified: bool
    verification_status: str
    verification_method: str
    verified_at: str
    message: str
    scraping_initiated: Optional[bool] = None


class WebsiteStatusResponse(BaseModel):
    """Response model for website status."""
    website_id: str
    verification_status: str
    verification_method: Optional[str] = None
    verified_at: Optional[str] = None
    scraping_status: str
    widget_status: str
    last_crawled: Optional[str] = None
    is_verified: bool
    next_steps: List[str]