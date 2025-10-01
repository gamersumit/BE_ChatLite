from typing import Optional
from pydantic import BaseModel, Field


class WidgetInit(BaseModel):
    widget_id: str = Field(..., description="Unique widget identifier")
    page_url: str = Field(..., description="Current page URL")
    page_title: Optional[str] = Field(None, description="Current page title")
    user_agent: Optional[str] = Field(None, description="User agent string")
    referrer: Optional[str] = Field(None, description="Referrer URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "widget_id": "widget_abc123",
                "page_url": "https://example.com/products",
                "page_title": "Products - Example Company",
                "user_agent": "Mozilla/5.0...",
                "referrer": "https://google.com/search?q=example"
            }
        }


class WidgetConfig(BaseModel):
    widget_id: str
    session_id: str
    website_name: str
    widget_color: str = "#0066CC"
    widget_position: str = "bottom-right"
    welcome_message: Optional[str] = None
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "widget_id": "widget_abc123",
                "session_id": "sess_123456789",
                "website_name": "Example Company",
                "widget_color": "#FF6B35",
                "widget_position": "bottom-left",
                "welcome_message": "Hi! How can I help you today?",
                "is_active": True
            }
        }


class WidgetStatus(BaseModel):
    widget_id: str
    is_active: bool
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "widget_id": "widget_abc123",
                "is_active": True,
                "message": "Widget is active and ready"
            }
        }