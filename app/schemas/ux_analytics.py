"""
Pydantic schemas for UX Analytics API endpoints.
Part of Enhanced User Experience specification.
Handles performance metrics, user behavior tracking, and UX insights.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date
from pydantic import BaseModel, Field
from uuid import UUID
from decimal import Decimal


class UXAnalyticsBase(BaseModel):
    """Base schema for UX analytics."""
    widget_id: str = Field(..., max_length=255, description="Widget identifier")
    metric_type: str = Field(..., max_length=100, description="Type of metric being recorded")
    metric_value: Decimal = Field(..., description="Numeric value of the metric")
    metric_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metric context")


class UXAnalyticsCreate(UXAnalyticsBase):
    """Schema for creating UX analytics entry."""
    conversation_id: Optional[UUID] = Field(None, description="Associated conversation ID")
    recorded_at: Optional[datetime] = Field(None, description="When metric was recorded (defaults to now)")


class UXAnalyticsUpdate(BaseModel):
    """Schema for updating UX analytics (primarily metadata)."""
    metric_value: Optional[Decimal] = Field(None, description="Updated metric value")
    metric_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class UXAnalyticsResponse(UXAnalyticsBase):
    """Schema for UX analytics response."""
    id: UUID = Field(..., description="Unique analytics ID")
    conversation_id: Optional[UUID] = Field(None, description="Associated conversation")
    recorded_at: datetime = Field(..., description="Recording timestamp")
    date_partition: Optional[date] = Field(None, description="Date partition for queries")
    created_at: datetime = Field(..., description="Record creation time")
    updated_at: datetime = Field(..., description="Last update time")

    class Config:
        from_attributes = True


class UXMetricsBatch(BaseModel):
    """Schema for batch UX metrics creation."""
    metrics: List[UXAnalyticsCreate] = Field(..., description="List of metrics to create")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")


class UXAnalyticsQuery(BaseModel):
    """Schema for querying UX analytics data."""
    widget_id: Optional[str] = Field(None, description="Filter by widget ID")
    conversation_id: Optional[UUID] = Field(None, description="Filter by conversation")
    metric_types: Optional[List[str]] = Field(None, description="Filter by metric types")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    value_min: Optional[Decimal] = Field(None, description="Minimum metric value")
    value_max: Optional[Decimal] = Field(None, description="Maximum metric value")
    aggregate_by: Optional[str] = Field(None, description="Aggregation type: date, metric_type, widget")


class UXAnalyticsQueryResponse(BaseModel):
    """Schema for UX analytics query results."""
    metrics: List[UXAnalyticsResponse] = Field(default_factory=list, description="Individual metrics")
    aggregated_data: List[Dict[str, Any]] = Field(default_factory=list, description="Aggregated results")
    total_count: int = Field(..., description="Total matching records")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Items per page")
    has_more: bool = Field(False, description="Whether more results exist")


class PerformanceMetrics(BaseModel):
    """Schema for performance-related metrics."""
    average_load_time: float = Field(..., description="Average page load time in seconds")
    p95_load_time: float = Field(..., description="95th percentile load time")
    total_page_loads: int = Field(..., description="Total page loads in period")
    error_rate: float = Field(..., description="Error rate as percentage (0-1)")
    uptime_percentage: float = Field(..., description="Uptime percentage")


class UserBehaviorMetrics(BaseModel):
    """Schema for user behavior metrics."""
    unique_sessions: int = Field(..., description="Unique user sessions")
    average_session_duration: float = Field(..., description="Average session duration in minutes")
    total_interactions: int = Field(..., description="Total user interactions")
    bounce_rate: float = Field(..., description="Bounce rate as percentage (0-1)")
    pages_per_session: float = Field(..., description="Average pages viewed per session")


class ConversionMetrics(BaseModel):
    """Schema for conversion-related metrics."""
    conversion_rate: float = Field(..., description="Conversion rate as percentage (0-1)")
    total_conversions: int = Field(..., description="Total conversions in period")
    revenue_impact: float = Field(..., description="Total revenue from conversions")
    cost_per_conversion: float = Field(..., description="Average cost per conversion")
    lifetime_value: float = Field(..., description="Average customer lifetime value")


class AccessibilityMetrics(BaseModel):
    """Schema for accessibility usage metrics."""
    wcag_compliance_score: float = Field(..., description="WCAG compliance score (0-1)")
    keyboard_usage_rate: float = Field(..., description="Percentage of keyboard-only users")
    screen_reader_usage_rate: float = Field(..., description="Percentage of screen reader users")
    high_contrast_usage_rate: float = Field(..., description="High contrast mode usage rate")
    font_size_adjustments: float = Field(..., description="Font size adjustment usage rate")


class MetricsTrend(BaseModel):
    """Schema for metrics trend analysis."""
    metric_name: str = Field(..., description="Name of the metric")
    trend_direction: str = Field(..., description="Trend direction: up, down, stable")
    percentage_change: float = Field(..., description="Percentage change over period")
    period_comparison: str = Field(..., description="Comparison period type")


class UXDashboardData(BaseModel):
    """Schema for comprehensive UX dashboard data."""
    widget_id: str = Field(..., description="Widget identifier")
    date_range_days: int = Field(..., description="Date range for metrics")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance data")
    user_behavior_metrics: UserBehaviorMetrics = Field(..., description="User behavior data")
    conversion_metrics: ConversionMetrics = Field(..., description="Conversion data")
    accessibility_metrics: AccessibilityMetrics = Field(..., description="Accessibility data")
    trends: List[MetricsTrend] = Field(default_factory=list, description="Metrics trends")
    last_updated: datetime = Field(..., description="Last data update time")


class UXReportRequest(BaseModel):
    """Schema for UX report generation request."""
    widget_ids: List[str] = Field(..., description="Widgets to include in report")
    date_from: datetime = Field(..., description="Report start date")
    date_to: datetime = Field(..., description="Report end date")
    report_format: str = Field("pdf", description="Report format: pdf, excel, json")
    metrics_to_include: List[str] = Field(default_factory=list, description="Specific metrics to include")
    include_recommendations: bool = Field(True, description="Include UX recommendations")
    include_benchmarks: bool = Field(True, description="Include industry benchmarks")


class UXReportResponse(BaseModel):
    """Schema for UX report generation response."""
    report_id: UUID = Field(..., description="Generated report ID")
    status: str = Field(..., description="Report generation status")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    metrics_included: List[str] = Field(default_factory=list, description="Metrics included")
    date_range: Dict[str, str] = Field(default_factory=dict, description="Report date range")
    generated_at: datetime = Field(..., description="Report generation time")
    expires_at: datetime = Field(..., description="Download link expiration")


class UXBenchmarks(BaseModel):
    """Schema for UX benchmarks data."""
    performance_benchmarks: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance benchmarks")
    conversion_benchmarks: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Conversion benchmarks")
    accessibility_benchmarks: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Accessibility benchmarks")
    industry: str = Field(..., description="Industry category")
    last_updated: datetime = Field(..., description="Benchmarks last update")


class UXInsight(BaseModel):
    """Schema for AI-generated UX insights."""
    insight_id: UUID = Field(..., description="Insight identifier")
    widget_id: str = Field(..., description="Related widget")
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    impact_score: float = Field(..., description="Potential impact score (0-1)")
    confidence_level: float = Field(..., description="Confidence in insight (0-1)")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")
    generated_at: datetime = Field(..., description="Insight generation time")


class UXOptimizationSuggestion(BaseModel):
    """Schema for UX optimization suggestions."""
    suggestion_id: UUID = Field(..., description="Suggestion identifier")
    widget_id: str = Field(..., description="Target widget")
    optimization_type: str = Field(..., description="Type of optimization")
    current_metric_value: float = Field(..., description="Current metric value")
    projected_improvement: float = Field(..., description="Projected improvement percentage")
    implementation_effort: str = Field(..., description="Implementation effort: low, medium, high")
    priority_score: float = Field(..., description="Priority score (0-1)")
    detailed_steps: List[str] = Field(default_factory=list, description="Implementation steps")
    estimated_timeline: str = Field(..., description="Estimated implementation time")


class UXABTestMetrics(BaseModel):
    """Schema for A/B testing metrics."""
    test_id: UUID = Field(..., description="A/B test identifier")
    variant_id: str = Field(..., description="Test variant identifier")
    widget_id: str = Field(..., description="Widget being tested")
    metric_name: str = Field(..., description="Metric being measured")
    variant_value: float = Field(..., description="Metric value for this variant")
    control_value: float = Field(..., description="Metric value for control")
    improvement_percentage: float = Field(..., description="Improvement over control")
    statistical_significance: float = Field(..., description="Statistical significance level")
    sample_size: int = Field(..., description="Sample size for variant")
    test_duration_days: int = Field(..., description="Test duration in days")


class UXRealTimeMetrics(BaseModel):
    """Schema for real-time UX metrics."""
    widget_id: str = Field(..., description="Widget identifier")
    current_active_users: int = Field(..., description="Currently active users")
    live_conversations: int = Field(..., description="Active conversations")
    real_time_performance: PerformanceMetrics = Field(..., description="Real-time performance data")
    recent_events: List[Dict[str, Any]] = Field(default_factory=list, description="Recent UX events")
    alerts: List[str] = Field(default_factory=list, description="Current alerts or issues")
    last_updated: datetime = Field(..., description="Last metrics update")