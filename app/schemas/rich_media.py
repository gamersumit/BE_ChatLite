"""
Pydantic schemas for Rich Media File API endpoints.
Part of Enhanced User Experience specification.
Handles file uploads, media processing, and accessibility features.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from uuid import UUID
from enum import Enum


class MediaType(str, Enum):
    """Supported media types."""
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    AUDIO = "audio"
    OTHER = "other"


class UploadStatus(str, Enum):
    """File upload processing status."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ThumbnailSize(str, Enum):
    """Available thumbnail sizes."""
    SMALL = "small"      # 150x150
    MEDIUM = "medium"    # 300x300
    LARGE = "large"      # 600x600


class RichMediaFileBase(BaseModel):
    """Base schema for rich media files."""
    file_name: str = Field(..., max_length=500, description="Original filename")
    file_type: MediaType = Field(..., description="Type of media file")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    mime_type: str = Field(..., max_length=200, description="MIME type of the file")
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")
    file_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")


class RichMediaFileCreate(RichMediaFileBase):
    """Schema for creating a rich media file record."""
    conversation_id: UUID = Field(..., description="Associated conversation ID")
    message_id: Optional[UUID] = Field(None, description="Associated message ID")
    file_url: str = Field(..., description="URL where file is stored")
    thumbnail_url: Optional[str] = Field(None, description="URL for thumbnail image")


class RichMediaFileUpdate(BaseModel):
    """Schema for updating rich media file metadata."""
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")
    file_metadata: Optional[Dict[str, Any]] = Field(None, description="File metadata")
    upload_status: Optional[UploadStatus] = Field(None, description="Processing status")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")


class RichMediaFileResponse(RichMediaFileBase):
    """Schema for rich media file response."""
    id: UUID = Field(..., description="Unique file ID")
    conversation_id: UUID = Field(..., description="Associated conversation")
    message_id: Optional[UUID] = Field(None, description="Associated message")
    file_url: str = Field(..., description="File access URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    upload_status: UploadStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class MediaUploadRequest(BaseModel):
    """Schema for initiating file upload."""
    conversation_id: UUID = Field(..., description="Target conversation")
    message_id: Optional[UUID] = Field(None, description="Associated message")
    file_type: MediaType = Field(..., description="Type of media being uploaded")
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")


class MediaUploadResponse(BaseModel):
    """Schema for file upload response."""
    file_id: UUID = Field(..., description="Generated file ID")
    upload_url: str = Field(..., description="Pre-signed URL for file upload")
    upload_fields: Dict[str, str] = Field(default_factory=dict, description="Additional upload fields")
    max_file_size: int = Field(..., description="Maximum allowed file size in bytes")
    allowed_mime_types: List[str] = Field(default_factory=list, description="Allowed MIME types")
    expires_at: datetime = Field(..., description="Upload URL expiration time")


class MediaProcessingStatus(BaseModel):
    """Schema for media processing status."""
    file_id: UUID = Field(..., description="File identifier")
    status: UploadStatus = Field(..., description="Current processing status")
    progress: float = Field(0, ge=0, le=1, description="Processing progress (0-1)")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    thumbnail_generated: bool = Field(False, description="Whether thumbnail was created")
    accessibility_processed: bool = Field(False, description="Whether accessibility features were processed")


class MediaValidationResult(BaseModel):
    """Schema for file validation results."""
    is_valid: bool = Field(..., description="Whether file passes validation")
    file_size: int = Field(..., description="Actual file size")
    mime_type: str = Field(..., description="Detected MIME type")
    security_scan_passed: bool = Field(..., description="Security scan result")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class MediaAnalytics(BaseModel):
    """Schema for media usage analytics."""
    total_files: int = Field(..., description="Total files uploaded")
    files_by_type: Dict[str, int] = Field(default_factory=dict, description="File count by type")
    total_storage_used: int = Field(..., description="Total storage used in bytes")
    average_file_size: float = Field(..., description="Average file size")
    processing_success_rate: float = Field(..., description="Success rate of file processing")
    most_common_mime_types: List[Dict[str, Any]] = Field(default_factory=list, description="Popular MIME types")


class BulkMediaOperation(BaseModel):
    """Schema for bulk media operations."""
    file_ids: List[UUID] = Field(..., description="List of file IDs")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class BulkMediaResponse(BaseModel):
    """Schema for bulk media operation results."""
    success_count: int = Field(..., description="Successful operations")
    error_count: int = Field(..., description="Failed operations")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed results")
    errors: Dict[str, str] = Field(default_factory=dict, description="Error details by file ID")


class MediaSearchRequest(BaseModel):
    """Schema for media search and filtering."""
    conversation_id: Optional[UUID] = Field(None, description="Filter by conversation")
    file_type: Optional[MediaType] = Field(None, description="Filter by file type")
    mime_type: Optional[str] = Field(None, description="Filter by MIME type")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    has_alt_text: Optional[bool] = Field(None, description="Filter files with/without alt text")
    upload_status: Optional[UploadStatus] = Field(None, description="Filter by upload status")
    search_query: Optional[str] = Field(None, description="Search in file names and metadata")


class MediaSearchResponse(BaseModel):
    """Schema for media search results."""
    files: List[RichMediaFileResponse] = Field(default_factory=list, description="Matching files")
    total_count: int = Field(..., description="Total matching files")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Items per page")
    has_more: bool = Field(False, description="Whether more results exist")


class MediaAccessLog(BaseModel):
    """Schema for media access logging."""
    file_id: UUID = Field(..., description="File accessed")
    user_id: Optional[str] = Field(None, description="User who accessed file")
    access_type: str = Field(..., description="Type of access (view, download, etc.)")
    user_agent: Optional[str] = Field(None, description="User agent string")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    accessed_at: datetime = Field(..., description="Access timestamp")


class MediaConfiguration(BaseModel):
    """Schema for media service configuration."""
    max_file_size: int = Field(10 * 1024 * 1024, description="Maximum file size (default 10MB)")
    allowed_extensions: List[str] = Field(default_factory=list, description="Allowed file extensions")
    thumbnail_sizes: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Thumbnail configurations")
    storage_provider: str = Field("local", description="Storage provider (local, s3, etc.)")
    cdn_enabled: bool = Field(False, description="Whether CDN is enabled")
    virus_scanning: bool = Field(True, description="Enable virus scanning")
    watermark_enabled: bool = Field(False, description="Enable watermarking")