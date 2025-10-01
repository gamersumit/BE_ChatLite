"""
Pydantic schemas for User Preferences API endpoints.
Part of Enhanced User Experience specification.
Handles personalization settings, accessibility preferences, and adaptive theming.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class UserPreferencesBase(BaseModel):
    """Base schema for user preferences."""
    theme: str = Field("auto", description="Theme preference: light, dark, or auto")
    language: str = Field("en", description="Language preference (ISO 639-1 code)")
    notifications: bool = Field(True, description="Enable notifications")
    accessibility_mode: bool = Field(False, description="Enable accessibility mode")
    font_size: str = Field("medium", description="Font size: small, medium, large, xlarge")
    high_contrast: bool = Field(False, description="Enable high contrast mode")
    reduced_motion: bool = Field(False, description="Reduce animations and motion")
    conversation_style: str = Field("auto", description="Conversation style: formal, casual, auto")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional custom settings")


class UserPreferencesCreate(UserPreferencesBase):
    """Schema for creating user preferences."""
    user_id: str = Field(..., max_length=255, description="Unique identifier for the user")


class UserPreferencesUpdate(BaseModel):
    """Schema for updating user preferences (partial updates allowed)."""
    theme: Optional[str] = Field(None, description="Theme preference")
    language: Optional[str] = Field(None, description="Language preference")
    notifications: Optional[bool] = Field(None, description="Enable notifications")
    accessibility_mode: Optional[bool] = Field(None, description="Enable accessibility mode")
    font_size: Optional[str] = Field(None, description="Font size preference")
    high_contrast: Optional[bool] = Field(None, description="High contrast mode")
    reduced_motion: Optional[bool] = Field(None, description="Reduce motion preference")
    conversation_style: Optional[str] = Field(None, description="Conversation style")
    custom_settings: Optional[Dict[str, Any]] = Field(None, description="Custom settings")


class UserPreferencesResponse(UserPreferencesBase):
    """Schema for user preferences response."""
    id: UUID = Field(..., description="Unique preference record ID")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="When preferences were created")
    updated_at: datetime = Field(..., description="When preferences were last updated")

    class Config:
        from_attributes = True


class UserPreferencesAnalytics(BaseModel):
    """Schema for user preferences analytics."""
    total_users: int = Field(..., description="Total users with preferences")
    theme_distribution: Dict[str, int] = Field(default_factory=dict, description="Theme preference distribution")
    language_distribution: Dict[str, int] = Field(default_factory=dict, description="Language distribution")
    accessibility_usage: int = Field(..., description="Users with accessibility features enabled")
    font_size_distribution: Dict[str, int] = Field(default_factory=dict, description="Font size preferences")
    conversation_style_distribution: Dict[str, int] = Field(default_factory=dict, description="Conversation style distribution")


class UserPreferencesRecommendation(BaseModel):
    """Schema for AI-powered preference recommendations."""
    user_id: str = Field(..., description="User identifier")
    recommended_theme: Optional[str] = Field(None, description="Recommended theme based on usage patterns")
    recommended_font_size: Optional[str] = Field(None, description="Recommended font size")
    recommended_conversation_style: Optional[str] = Field(None, description="Recommended conversation style")
    confidence_score: float = Field(..., description="Confidence in recommendations (0-1)")
    reasoning: Dict[str, str] = Field(default_factory=dict, description="Explanation for recommendations")


class BulkPreferencesOperation(BaseModel):
    """Schema for bulk preference operations."""
    user_ids: list[str] = Field(..., description="List of user IDs")
    preferences: UserPreferencesUpdate = Field(..., description="Preferences to apply")


class BulkPreferencesResponse(BaseModel):
    """Schema for bulk preference operation results."""
    success_count: int = Field(..., description="Number of successful operations")
    error_count: int = Field(..., description="Number of failed operations")
    errors: Dict[str, str] = Field(default_factory=dict, description="Error details by user_id")
    updated_users: list[str] = Field(default_factory=list, description="Successfully updated user IDs")