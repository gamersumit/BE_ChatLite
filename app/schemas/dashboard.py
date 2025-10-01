"""
Pydantic schemas for Dashboard API endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class WebsiteStats(BaseModel):
    """Statistics for a single website."""
    total_conversations: int = Field(..., description="Total number of conversations")
    total_messages: int = Field(..., description="Total number of messages")
    unique_visitors: int = Field(..., description="Number of unique visitors")
    active_conversations: int = Field(..., description="Active conversations in last 24h")

    class Config:
        from_attributes = True


class WebsiteSummary(BaseModel):
    """Summary information for a website in dashboard."""
    id: str = Field(..., description="Website ID")
    name: str = Field(..., description="Website name")
    url: str = Field(..., description="Website URL")
    domain: str = Field(..., description="Website domain")
    widget_id: str = Field(..., description="Widget ID")
    is_active: bool = Field(..., description="Whether website is active")
    verification_status: str = Field(..., description="Verification status")
    widget_status: str = Field(..., description="Widget configuration status")
    scraping_status: str = Field(..., description="Content scraping status")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    last_crawled: Optional[str] = Field(None, description="Last crawl timestamp")
    stats: WebsiteStats = Field(..., description="Website statistics")

    class Config:
        from_attributes = True


class DashboardSummary(BaseModel):
    """Overall dashboard summary statistics."""
    total_websites: int = Field(..., description="Total number of websites")
    active_websites: int = Field(..., description="Number of active websites")
    verified_websites: int = Field(..., description="Number of verified websites")
    total_conversations: int = Field(..., description="Total conversations across all websites")
    total_messages: int = Field(..., description="Total messages across all websites")
    avg_response_time: float = Field(..., description="Average response time in seconds")

    class Config:
        from_attributes = True


class DashboardWebsitesSummaryResponse(BaseModel):
    """Response model for dashboard websites summary."""
    websites: List[WebsiteSummary] = Field(..., description="List of website summaries")
    total_count: int = Field(..., description="Total number of websites")
    limit: int = Field(..., description="Number of items requested")
    offset: int = Field(..., description="Number of items skipped")
    summary: DashboardSummary = Field(..., description="Overall summary statistics")

    class Config:
        from_attributes = True


class WebsiteMetrics(BaseModel):
    """Detailed metrics for a website."""
    total_conversations: int = Field(..., description="Total conversations")
    total_messages: int = Field(..., description="Total messages")
    user_messages: int = Field(..., description="Messages from users")
    ai_messages: int = Field(..., description="Messages from AI")
    unique_visitors: int = Field(..., description="Unique visitors")
    unique_sessions: int = Field(..., description="Unique sessions")
    avg_conversation_length: float = Field(..., description="Average messages per conversation")
    response_rate: float = Field(..., description="Percentage of conversations with AI responses")
    satisfaction_score: float = Field(..., description="Average satisfaction rating (1-5)")
    total_satisfaction_ratings: int = Field(..., description="Number of satisfaction ratings")
    leads_captured: int = Field(..., description="Number of leads captured")
    lead_conversion_rate: float = Field(..., description="Lead conversion percentage")

    class Config:
        from_attributes = True


class TrendDataPoint(BaseModel):
    """Single data point for trend charts."""
    date: Optional[str] = Field(None, description="Date for the data point")
    count: int = Field(..., description="Count value for the data point")

    class Config:
        from_attributes = True


class HourlyActivityPoint(BaseModel):
    """Hourly activity data point."""
    hour: int = Field(..., description="Hour of day (0-23)")
    conversations: int = Field(..., description="Number of conversations in that hour")

    class Config:
        from_attributes = True


class WebsiteTrends(BaseModel):
    """Trend data for charts and graphs."""
    daily_conversations: List[TrendDataPoint] = Field(..., description="Daily conversation counts")
    daily_messages: List[TrendDataPoint] = Field(..., description="Daily message counts")
    daily_visitors: List[TrendDataPoint] = Field(..., description="Daily unique visitor counts")
    hourly_activity: List[HourlyActivityPoint] = Field(..., description="Hourly activity pattern")

    class Config:
        from_attributes = True


class TopPage(BaseModel):
    """Top page analytics data."""
    page_url: str = Field(..., description="Page URL")
    visits: int = Field(..., description="Number of visits")
    unique_visitors: int = Field(..., description="Number of unique visitors")

    class Config:
        from_attributes = True


class TrafficSource(BaseModel):
    """Traffic source data."""
    source: str = Field(..., description="Traffic source domain")
    visits: int = Field(..., description="Number of visits from this source")

    class Config:
        from_attributes = True


class VisitorAnalytics(BaseModel):
    """Visitor behavior analytics."""
    unique_visitors: int = Field(..., description="Total unique visitors")
    returning_visitors: int = Field(..., description="Number of returning visitors")
    new_visitors: int = Field(..., description="Number of new visitors")
    avg_session_duration: float = Field(..., description="Average session duration in minutes")
    bounce_rate: float = Field(..., description="Bounce rate percentage")
    top_pages: List[TopPage] = Field(..., description="Top visited pages")
    traffic_sources: List[TrafficSource] = Field(..., description="Traffic source breakdown")

    class Config:
        from_attributes = True


class TopicFrequency(BaseModel):
    """Topic frequency data."""
    topic: str = Field(..., description="Topic or keyword")
    frequency: int = Field(..., description="Frequency count")

    class Config:
        from_attributes = True


class PopularQuestions(BaseModel):
    """Popular questions analysis."""
    top_questions: List[str] = Field(..., description="Most common question-like messages")
    trending_topics: List[TopicFrequency] = Field(..., description="Trending topics and keywords")
    question_categories: Dict[str, int] = Field(..., description="Question categorization counts")
    unanswered_questions: int = Field(..., description="Number of unanswered questions")
    total_analyzed_messages: int = Field(..., description="Total messages analyzed")

    class Config:
        from_attributes = True


class WebsiteAnalyticsResponse(BaseModel):
    """Response model for website analytics."""
    website_id: str = Field(..., description="Website ID")
    website_name: str = Field(..., description="Website name")
    time_range: str = Field(..., description="Time range for the analytics")
    start_date: str = Field(..., description="Start date of the analytics period")
    end_date: str = Field(..., description="End date of the analytics period")
    metrics: WebsiteMetrics = Field(..., description="Core website metrics")
    trends: Optional[WebsiteTrends] = Field(None, description="Trend data for charts")
    visitor_analytics: Optional[VisitorAnalytics] = Field(None, description="Visitor behavior data")
    popular_questions: Optional[PopularQuestions] = Field(None, description="Question analysis data")

    class Config:
        from_attributes = True


class AnalyticsExportResponse(BaseModel):
    """Response model for analytics export."""
    export_date: str = Field(..., description="When the export was generated")
    format: str = Field(..., description="Export format (json/csv)")
    website_id: str = Field(..., description="Website ID")
    time_range: str = Field(..., description="Time range for the export")
    analytics_data: Dict[str, Any] = Field(..., description="Full analytics data")
    summary: Dict[str, Union[int, float]] = Field(..., description="Summary statistics")

    class Config:
        from_attributes = True


class QuickStatsResponse(BaseModel):
    """Response model for quick website statistics."""
    website_id: str = Field(..., description="Website ID")
    website_name: str = Field(..., description="Website name")
    domain: str = Field(..., description="Website domain")
    is_active: bool = Field(..., description="Whether website is active")
    verification_status: str = Field(..., description="Verification status")
    widget_status: str = Field(..., description="Widget status")
    stats: WebsiteStats = Field(..., description="Quick statistics")
    last_updated: str = Field(..., description="When stats were last updated")

    class Config:
        from_attributes = True


class DashboardOverviewResponse(BaseModel):
    """Response model for dashboard overview."""
    user_id: str = Field(..., description="User ID")
    user_name: str = Field(..., description="User name")
    total_websites: int = Field(..., description="Total websites")
    active_websites: int = Field(..., description="Active websites")
    verified_websites: int = Field(..., description="Verified websites")
    total_conversations: int = Field(..., description="Total conversations")
    total_messages: int = Field(..., description="Total messages")
    avg_response_time: float = Field(..., description="Average response time")
    recent_conversations: int = Field(..., description="Recent conversations (7 days)")
    activation_rate: float = Field(..., description="Website activation rate percentage")
    verification_rate: float = Field(..., description="Website verification rate percentage")
    generated_at: str = Field(..., description="When overview was generated")

    class Config:
        from_attributes = True


class AnalyticsRefreshResponse(BaseModel):
    """Response model for analytics refresh request."""
    message: str = Field(..., description="Success message")
    website_id: str = Field(..., description="Website ID")
    refresh_requested_at: str = Field(..., description="When refresh was requested")
    estimated_completion: float = Field(..., description="Estimated completion timestamp")

    class Config:
        from_attributes = True


# Request models for filtering and pagination
class DashboardWebsitesRequest(BaseModel):
    """Request model for dashboard websites query."""
    limit: Optional[int] = Field(10, ge=1, le=100, description="Number of websites to return")
    offset: Optional[int] = Field(0, ge=0, description="Number of websites to skip")
    status: Optional[str] = Field(None, description="Filter by verification status")
    sort_by: Optional[str] = Field("created_at", description="Field to sort by")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")

    class Config:
        from_attributes = True


class AnalyticsRequest(BaseModel):
    """Request model for analytics query."""
    time_range: Optional[str] = Field("7d", description="Time range: 1d, 7d, 30d, 90d, 1y")
    include_trends: Optional[bool] = Field(False, description="Include trend data")
    include_visitors: Optional[bool] = Field(False, description="Include visitor analytics")
    include_questions: Optional[bool] = Field(False, description="Include question analysis")

    class Config:
        from_attributes = True


class ExportRequest(BaseModel):
    """Request model for analytics export."""
    format: Optional[str] = Field("json", description="Export format: json, csv")
    time_range: Optional[str] = Field("30d", description="Time range for export")

    class Config:
        from_attributes = True