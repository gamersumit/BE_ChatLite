from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from uuid import UUID


class WebsiteCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Website name")
    url: HttpUrl = Field(..., description="Website URL")
    business_name: Optional[str] = Field(None, max_length=255, description="Business name")
    contact_email: Optional[str] = Field(None, description="Contact email")
    business_description: Optional[str] = Field(None, description="Business description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Example Company Website",
                "url": "https://example.com",
                "business_name": "Example Company",
                "contact_email": "support@example.com",
                "business_description": "We provide excellent products and services"
            }
        }


class WebsiteUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    business_name: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None)
    business_description: Optional[str] = Field(None)
    widget_color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')
    widget_position: Optional[str] = Field(None)
    welcome_message: Optional[str] = Field(None)
    is_active: Optional[bool] = Field(None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "widget_color": "#FF6B35",
                "welcome_message": "Welcome! How can we help you today?",
                "is_active": True
            }
        }


class WebsiteResponse(BaseModel):
    id: UUID
    name: str
    url: str
    domain: str
    widget_id: str
    is_active: bool
    widget_color: Optional[str] = None
    widget_position: str
    welcome_message: Optional[str] = None
    business_name: Optional[str] = None
    contact_email: Optional[str] = None
    total_conversations: int
    total_messages: int
    monthly_message_limit: int
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Example Company Website",
                "url": "https://example.com",
                "domain": "example.com",
                "widget_id": "widget_abc123",
                "is_active": True,
                "widget_color": "#0066CC",
                "widget_position": "bottom-right",
                "welcome_message": "Hi! How can I help you?",
                "business_name": "Example Company",
                "contact_email": "support@example.com",
                "total_conversations": 150,
                "total_messages": 450,
                "monthly_message_limit": 1000
            }
        }