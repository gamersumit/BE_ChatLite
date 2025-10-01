"""
Pydantic schemas for Script Generation API endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class WidgetConfiguration(BaseModel):
    """Widget configuration model for inline customization."""
    widget_color: Optional[str] = Field(default="#0066CC", description="Widget primary color (hex)")
    widget_position: str = Field(default="bottom-right", description="Widget position on page")
    widget_size: str = Field(default="medium", description="Widget size (small, medium, large)")
    widget_theme: str = Field(default="light", description="Widget theme (light, dark, auto)")

    # Widget behavior settings
    show_avatar: bool = Field(default=True, description="Whether to show avatar")
    enable_sound: bool = Field(default=True, description="Whether to enable sound notifications")
    auto_open_delay: Optional[int] = Field(default=None, description="Auto-open delay in seconds")
    show_online_status: bool = Field(default=True, description="Whether to show online status")

    # Message customization
    welcome_message: Optional[str] = Field(default="Hi! How can I help you today?", description="Welcome message")
    placeholder_text: str = Field(default="Type your message...", description="Input placeholder text")
    offline_message: str = Field(default="We're currently offline. We'll get back to you soon!", description="Offline message")
    thanks_message: str = Field(default="Thank you for your message!", description="Thanks message")

    # Branding customization
    show_branding: bool = Field(default=True, description="Whether to show ChatLite branding")
    custom_logo_url: Optional[str] = Field(default=None, description="Custom logo URL")
    company_name: Optional[str] = Field(default="Support", description="Company name")
    support_email: Optional[str] = Field(default=None, description="Support email")

    # Advanced styling
    custom_css: Optional[str] = Field(default=None, description="Custom CSS styles")
    font_family: Optional[str] = Field(default=None, description="Custom font family")
    border_radius: int = Field(default=12, description="Border radius in pixels")

    @field_validator("widget_color")
    @classmethod
    def validate_widget_color(cls, v):
        """Validate hex color format."""
        if v and not v.startswith('#') or len(v) != 7:
            raise ValueError("Widget color must be a valid hex color (e.g., #0066CC)")
        return v

    @field_validator("widget_position")
    @classmethod
    def validate_widget_position(cls, v):
        """Validate widget position."""
        valid_positions = ["bottom-right", "bottom-left", "top-right", "top-left"]
        if v not in valid_positions:
            raise ValueError(f"Widget position must be one of: {', '.join(valid_positions)}")
        return v

    @field_validator("widget_size")
    @classmethod
    def validate_widget_size(cls, v):
        """Validate widget size."""
        valid_sizes = ["small", "medium", "large"]
        if v not in valid_sizes:
            raise ValueError(f"Widget size must be one of: {', '.join(valid_sizes)}")
        return v

    @field_validator("widget_theme")
    @classmethod
    def validate_widget_theme(cls, v):
        """Validate widget theme."""
        valid_themes = ["light", "dark", "auto"]
        if v not in valid_themes:
            raise ValueError(f"Widget theme must be one of: {', '.join(valid_themes)}")
        return v


class ScriptGenerationPostRequest(BaseModel):
    """Request model for POST script generation with configuration."""
    widget_id: str = Field(..., description="Widget ID to generate script for")
    configuration: Optional[WidgetConfiguration] = Field(default_factory=WidgetConfiguration, description="Widget configuration")


class ScriptGenerationPostResponse(BaseModel):
    """Response model for POST script generation."""
    script: str = Field(..., description="Generated widget script")
    installation_instructions: Dict[str, str] = Field(..., description="Step-by-step installation instructions")
    widget_config: WidgetConfiguration = Field(..., description="Applied widget configuration")
    widget_id: str = Field(..., description="Widget ID")
    website_name: str = Field(..., description="Website name")
    generated_at: datetime = Field(..., description="When script was generated")


class ScriptGenerationRequest(BaseModel):
    """Request model for script generation."""
    prepare_for_cdn: bool = Field(default=True, description="Whether to prepare script for CDN upload")
    include_content: bool = Field(default=False, description="Whether to include script content in response")
    force_regenerate: bool = Field(default=False, description="Force regeneration even if recent version exists")


class ScriptGenerationResponse(BaseModel):
    """Response model for script generation."""
    script_version_id: str = Field(..., description="Generated script version ID")
    website_id: str = Field(..., description="Website ID")
    version_number: int = Field(..., description="Script version number")
    script_url: str = Field(..., description="CDN URL for the script")
    script_content: Optional[str] = Field(None, description="Generated script content")
    script_size: int = Field(..., description="Script size in bytes")
    version_hash: str = Field(..., description="SHA-256 hash of script content")
    is_active: bool = Field(..., description="Whether this version is active")
    generation_time: float = Field(..., description="Time taken to generate script in seconds")
    cdn_prepared: bool = Field(..., description="Whether script is prepared for CDN")
    cdn_metadata: Optional[Dict[str, Any]] = Field(None, description="CDN metadata")
    created_at: datetime = Field(..., description="When the script version was created")

    class Config:
        from_attributes = True


class ScriptVersionResponse(BaseModel):
    """Response model for script version information."""
    script_version_id: str = Field(..., description="Script version ID")
    website_id: str = Field(..., description="Website ID")
    version_number: int = Field(..., description="Version number")
    script_url: str = Field(..., description="CDN URL for the script")
    script_size: int = Field(..., description="Script size in bytes")
    version_hash: str = Field(..., description="SHA-256 hash of script content")
    is_active: bool = Field(..., description="Whether this version is active")
    is_published: bool = Field(..., description="Whether this version is published")
    generated_by: Optional[str] = Field(None, description="Who generated this version")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    cdn_uploaded: bool = Field(..., description="Whether uploaded to CDN")
    cache_key: Optional[str] = Field(None, description="Cache key for CDN")
    expires_at: Optional[datetime] = Field(None, description="When the cached version expires")
    created_at: datetime = Field(..., description="When the version was created")
    
    @field_validator("version_number")
    @classmethod
    def validate_version_number(cls, v):
        """Validate version number is positive."""
        if v < 1:
            raise ValueError("Version number must be positive")
        return v

    class Config:
        from_attributes = True


class ScriptVersionListResponse(BaseModel):
    """Response model for paginated script versions list."""
    versions: List[ScriptVersionResponse] = Field(..., description="List of script versions")
    total_count: int = Field(..., description="Total number of versions")
    limit: int = Field(..., description="Number of items requested")
    offset: int = Field(..., description="Number of items skipped")

    class Config:
        from_attributes = True


class ScriptActivationRequest(BaseModel):
    """Request model for activating a script version."""
    script_version_id: str = Field(..., description="Script version ID to activate")


class ScriptInstallationResponse(BaseModel):
    """Response model for script installation information."""
    website_id: str = Field(..., description="Website ID")
    widget_id: str = Field(..., description="Widget ID")
    installation_code: str = Field(..., description="HTML installation code")
    installation_instructions: List[str] = Field(..., description="Step-by-step installation instructions")

    class Config:
        from_attributes = True


class ScriptStatusResponse(BaseModel):
    """Response model for script status information."""
    website_id: str = Field(..., description="Website ID")
    widget_id: str = Field(..., description="Widget ID")
    has_script: bool = Field(..., description="Whether website has a generated script")
    active_version: Optional[int] = Field(None, description="Active script version number")
    script_url: Optional[str] = Field(None, description="URL of active script")
    installation_status: str = Field(..., description="Installation status")
    script_generated_at: Optional[datetime] = Field(None, description="When script was last generated")
    script_last_checked: Optional[datetime] = Field(None, description="When installation was last checked")
    installation_details: Optional[Dict[str, Any]] = Field(None, description="Detailed installation status")

    class Config:
        from_attributes = True


class InstallationDetectionResult(BaseModel):
    """Model for script installation detection results."""
    installed: bool = Field(..., description="Whether script is detected as installed")
    script_found: bool = Field(default=False, description="Whether script tag was found")
    widget_id_match: bool = Field(default=False, description="Whether widget ID matches")
    domain_authorized: bool = Field(default=True, description="Whether domain is authorized")
    last_checked: str = Field(..., description="When the check was performed")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    http_status: Optional[int] = Field(None, description="HTTP status code of the checked page")

    class Config:
        from_attributes = True


class CDNUploadRequest(BaseModel):
    """Request model for CDN upload operations."""
    script_version_id: str = Field(..., description="Script version ID to upload")
    cache_duration: int = Field(default=86400, description="Cache duration in seconds")
    compression: bool = Field(default=True, description="Whether to compress the script")


class CDNUploadResponse(BaseModel):
    """Response model for CDN upload operations."""
    script_version_id: str = Field(..., description="Script version ID")
    cdn_url: str = Field(..., description="CDN URL where script is available")
    upload_status: str = Field(..., description="Upload status")
    cache_key: str = Field(..., description="Cache key")
    expires_at: datetime = Field(..., description="When the cached version expires")
    compressed_size: Optional[int] = Field(None, description="Compressed size if compression was used")

    class Config:
        from_attributes = True


class ScriptVersionSummary(BaseModel):
    """Summary model for script version with minimal data."""
    version_number: int = Field(..., description="Version number")
    is_active: bool = Field(..., description="Whether this version is active")
    created_at: datetime = Field(..., description="When the version was created")
    script_size: int = Field(..., description="Script size in bytes")
    cdn_uploaded: bool = Field(..., description="Whether uploaded to CDN")

    class Config:
        from_attributes = True


class WebsiteScriptSummary(BaseModel):
    """Summary of script information for a website."""
    website_id: str = Field(..., description="Website ID")
    widget_id: str = Field(..., description="Widget ID")
    website_name: str = Field(..., description="Website name")
    total_versions: int = Field(..., description="Total number of script versions")
    active_version: Optional[ScriptVersionSummary] = Field(None, description="Currently active version")
    last_generated: Optional[datetime] = Field(None, description="When script was last generated")
    installation_status: str = Field(..., description="Installation status")

    class Config:
        from_attributes = True


# Additional validation can be added here as needed