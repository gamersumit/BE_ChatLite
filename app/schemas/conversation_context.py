"""
Pydantic schemas for Conversation Context API endpoints.
Part of Enhanced User Experience specification.
Handles conversation threading, context analysis, and smart suggestions.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID
from decimal import Decimal


class ConversationContextBase(BaseModel):
    """Base schema for conversation context."""
    user_intent: Optional[str] = Field(None, max_length=50, description="Detected user intent")
    conversation_summary: Optional[str] = Field(None, description="AI-generated conversation summary")
    user_sentiment: Optional[str] = Field(None, max_length=20, description="User sentiment analysis")
    suggested_responses: List[str] = Field(default_factory=list, description="AI-suggested responses")
    context_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    confidence_score: Optional[Decimal] = Field(None, description="Confidence in analysis (0-1)")


class ConversationContextCreate(ConversationContextBase):
    """Schema for creating conversation context."""
    conversation_id: UUID = Field(..., description="Associated conversation ID")


class ConversationContextUpdate(BaseModel):
    """Schema for updating conversation context."""
    user_intent: Optional[str] = Field(None, description="Updated user intent")
    conversation_summary: Optional[str] = Field(None, description="Updated summary")
    user_sentiment: Optional[str] = Field(None, description="Updated sentiment")
    suggested_responses: Optional[List[str]] = Field(None, description="Updated suggestions")
    context_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    confidence_score: Optional[Decimal] = Field(None, description="Updated confidence")


class ConversationContextResponse(ConversationContextBase):
    """Schema for conversation context response."""
    id: UUID = Field(..., description="Unique context ID")
    conversation_id: UUID = Field(..., description="Associated conversation")
    created_at: datetime = Field(..., description="Context creation time")
    updated_at: datetime = Field(..., description="Last update time")

    class Config:
        from_attributes = True


class ContextAnalysisRequest(BaseModel):
    """Schema for requesting context analysis."""
    conversation_id: UUID = Field(..., description="Conversation to analyze")
    force_refresh: bool = Field(False, description="Force re-analysis even if context exists")
    analysis_depth: str = Field("standard", description="Analysis depth: basic, standard, detailed")


class ContextAnalysisResponse(BaseModel):
    """Schema for context analysis results."""
    conversation_id: UUID = Field(..., description="Analyzed conversation")
    analysis_completed: bool = Field(..., description="Whether analysis completed successfully")
    context: ConversationContextResponse = Field(..., description="Generated context")
    processing_time_ms: int = Field(..., description="Analysis processing time")
    tokens_used: Optional[int] = Field(None, description="AI tokens consumed")


class SmartSuggestion(BaseModel):
    """Schema for AI-generated smart suggestions."""
    id: UUID = Field(..., description="Suggestion ID")
    suggestion_text: str = Field(..., description="Suggested response text")
    suggestion_type: str = Field(..., description="Type of suggestion")
    confidence_score: float = Field(..., description="Confidence in suggestion")
    context_tags: List[str] = Field(default_factory=list, description="Relevant context tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SuggestionFeedback(BaseModel):
    """Schema for suggestion feedback."""
    suggestion_id: UUID = Field(..., description="Suggestion being rated")
    feedback_type: str = Field(..., description="Feedback type: used, helpful, not_helpful, irrelevant")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    user_id: Optional[str] = Field(None, description="User providing feedback")


class ConversationThread(BaseModel):
    """Schema for conversation threading."""
    thread_id: UUID = Field(..., description="Thread identifier")
    conversation_id: UUID = Field(..., description="Parent conversation")
    topic: str = Field(..., description="Thread topic/subject")
    start_message_id: UUID = Field(..., description="First message in thread")
    end_message_id: Optional[UUID] = Field(None, description="Last message in thread")
    status: str = Field(..., description="Thread status")
    priority: int = Field(1, description="Thread priority")
    tags: List[str] = Field(default_factory=list, description="Thread tags")
    participant_count: int = Field(1, description="Number of participants")
    created_at: datetime = Field(..., description="Thread creation time")
    updated_at: datetime = Field(..., description="Last thread activity")


class ContextAnalytics(BaseModel):
    """Schema for context analysis analytics."""
    total_contexts: int = Field(..., description="Total contexts analyzed")
    intent_distribution: Dict[str, int] = Field(default_factory=dict, description="Intent analysis distribution")
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict, description="Sentiment distribution")
    average_confidence_score: float = Field(..., description="Average confidence score")
    suggestion_usage_rate: float = Field(..., description="Rate of suggestion usage")
    context_accuracy_rate: float = Field(..., description="Context prediction accuracy")
    processing_times: Dict[str, float] = Field(default_factory=dict, description="Average processing times")


class ContextSearchRequest(BaseModel):
    """Schema for searching conversation contexts."""
    user_intent: Optional[str] = Field(None, description="Filter by user intent")
    user_sentiment: Optional[str] = Field(None, description="Filter by sentiment")
    confidence_min: Optional[float] = Field(None, description="Minimum confidence score")
    confidence_max: Optional[float] = Field(None, description="Maximum confidence score")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    search_query: Optional[str] = Field(None, description="Search in summaries")
    has_suggestions: Optional[bool] = Field(None, description="Filter contexts with suggestions")


class ContextSearchResponse(BaseModel):
    """Schema for context search results."""
    contexts: List[ConversationContextResponse] = Field(default_factory=list, description="Matching contexts")
    total_count: int = Field(..., description="Total matching contexts")
    page: int = Field(1, description="Current page")
    page_size: int = Field(20, description="Items per page")
    has_more: bool = Field(False, description="Whether more results exist")


class ContextExportRequest(BaseModel):
    """Schema for exporting context data."""
    conversation_ids: Optional[List[UUID]] = Field(None, description="Specific conversations to export")
    date_from: Optional[datetime] = Field(None, description="Export data from date")
    date_to: Optional[datetime] = Field(None, description="Export data to date")
    export_format: str = Field("json", description="Export format: json, csv, xlsx")
    include_metadata: bool = Field(True, description="Include context metadata")
    include_suggestions: bool = Field(True, description="Include suggestions")


class ContextExportResponse(BaseModel):
    """Schema for context export results."""
    export_id: UUID = Field(..., description="Export job ID")
    status: str = Field(..., description="Export status")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    records_count: int = Field(..., description="Number of records to export")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    created_at: datetime = Field(..., description="Export request time")


class ContextModelTraining(BaseModel):
    """Schema for context model training data."""
    model_version: str = Field(..., description="Model version identifier")
    training_data_size: int = Field(..., description="Training dataset size")
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict, description="Model accuracy metrics")
    last_trained: datetime = Field(..., description="Last training timestamp")
    next_training: Optional[datetime] = Field(None, description="Next scheduled training")
    performance_benchmarks: Dict[str, Any] = Field(default_factory=dict, description="Performance benchmarks")