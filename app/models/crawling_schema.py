"""
Optimized crawling data schema for enhanced crawling system.
Provides comprehensive job management, performance tracking, and analytics.
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4
from sqlalchemy import (
    String, Text, Boolean, Integer, JSON, DateTime, ForeignKey,
    DECIMAL, Index, UniqueConstraint, CheckConstraint, text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.dialects.sqlite import CHAR
from sqlalchemy import TypeDecorator
from .base import Base


# Universal UUID type that works with both PostgreSQL and SQLite
class UniversalUUID(TypeDecorator):
    """Platform-independent UUID type."""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgresUUID(as_uuid=False))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, str):
                return str(value)
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return str(value)


class CrawlingJob(Base):
    """
    Core model for managing crawling jobs with comprehensive tracking.
    Supports manual, scheduled, and automated crawling operations.
    """
    __tablename__ = "crawling_jobs"

    # Core job identification
    website_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("websites.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Job configuration
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type: manual_crawl, scheduled_crawl, verification_crawl, full_site_crawl"
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        index=True,
        comment="Priority 1-10, higher numbers = higher priority"
    )

    # Job status and lifecycle
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
        comment="Status: pending, queued, running, completed, failed, cancelled"
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When job execution started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When job execution completed"
    )

    # Job configuration and metadata
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        default=lambda: {},
        comment="Job-specific configuration (max_pages, depth_limit, etc.)"
    )

    # Crawl metrics stored as JSONB
    crawl_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Error message if job failed
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    website = relationship("Website", foreign_keys=[website_id])
    user = relationship("User", foreign_keys=[user_id])

    # Table constraints
    __table_args__ = (
        CheckConstraint('priority >= 1 AND priority <= 10', name='valid_priority'),
        CheckConstraint("status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled')", name='valid_status'),
        CheckConstraint("job_type IN ('manual_crawl', 'scheduled_crawl', 'verification_crawl', 'full_site_crawl', 'incremental_crawl')", name='valid_job_type'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingJob {self.id} {self.job_type} {self.status}>"


class CrawlingJobResult(Base):
    """
    Stores detailed results and outcomes of crawling job execution.
    """
    __tablename__ = "crawling_job_results"

    # Foreign key to crawling job
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Result summary
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Result status: success, partial_success, failed"
    )

    # Crawling metrics
    pages_discovered: Mapped[int] = mapped_column(Integer, default=0)
    pages_crawled: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, default=0)
    data_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)

    # Execution timing
    execution_time_seconds: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 3))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Result data and metadata
    result_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Crawled data, URLs, content hashes, etc."
    )
    error_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Summary of errors encountered during crawling"
    )
    quality_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Data quality metrics and validation results"
    )

    # Relationships
    job = relationship("CrawlingJob", back_populates="results")

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("status IN ('success', 'partial_success', 'failed')", name='valid_result_status'),
        CheckConstraint('pages_crawled <= pages_discovered', name='valid_page_counts'),
        Index('idx_crawling_results_job_status', 'job_id', 'status'),
        Index('idx_crawling_results_timing', 'started_at', 'completed_at'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingJobResult {self.job_id} {self.status} {self.pages_crawled}p>"


class CrawlingJobMetadata(Base):
    """
    Flexible metadata storage for crawling jobs with key-value structure.
    Supports custom metadata, configuration overrides, and runtime data.
    """
    __tablename__ = "crawling_job_metadata"

    # Foreign key to crawling job
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Metadata structure
    metadata_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metadata_value: Mapped[Optional[str]] = mapped_column(Text)
    metadata_type: Mapped[str] = mapped_column(
        String(20),
        default="string",
        comment="Type: string, integer, float, boolean, json"
    )

    # Metadata categorization
    category: Mapped[Optional[str]] = mapped_column(
        String(50),
        index=True,
        comment="Category: config, runtime, result, error, performance"
    )
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    job = relationship("CrawlingJob", back_populates="metadata_entries")

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint('job_id', 'metadata_key', name='unique_job_metadata_key'),
        CheckConstraint("metadata_type IN ('string', 'integer', 'float', 'boolean', 'json')", name='valid_metadata_type'),
        Index('idx_job_metadata_key_category', 'metadata_key', 'category'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingJobMetadata {self.job_id}:{self.metadata_key}>"


class CrawlingSchedule(Base):
    """
    Manages automated crawling schedules for websites.
    Supports cron expressions, interval-based scheduling, and timezone handling.
    """
    __tablename__ = "crawling_schedules"

    # Associated website and user
    website_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("websites.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Schedule configuration
    schedule_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Type: cron, interval, once"
    )
    cron_expression: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Cron expression for scheduled execution"
    )
    interval_seconds: Mapped[Optional[int]] = mapped_column(
        Integer,
        comment="Interval in seconds for interval-based scheduling"
    )

    # Schedule status and control
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Execution tracking
    next_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        index=True,
        comment="When the next scheduled execution should occur"
    )
    last_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="When the last scheduled execution occurred"
    )
    last_job_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        comment="ID of the last job created by this schedule"
    )

    # Job configuration for scheduled runs
    job_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        default=lambda: {},
        comment="Default configuration for scheduled jobs"
    )

    # Failure handling
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    max_failures: Mapped[int] = mapped_column(Integer, default=5)
    failure_action: Mapped[str] = mapped_column(
        String(20),
        default="disable",
        comment="Action on max failures: disable, alert, continue"
    )

    # Relationships
    website = relationship("Website", foreign_keys=[website_id])
    user = relationship("User", foreign_keys=[user_id])

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("schedule_type IN ('cron', 'interval', 'once')", name='valid_schedule_type'),
        CheckConstraint("failure_action IN ('disable', 'alert', 'continue')", name='valid_failure_action'),
        CheckConstraint('consecutive_failures >= 0', name='valid_failure_count'),
        UniqueConstraint('website_id', name='unique_website_schedule'),
        Index('idx_schedules_next_run_active', 'next_run_at', 'is_active'),
        Index('idx_schedules_website_active', 'website_id', 'is_active'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingSchedule {self.website_id} {self.schedule_type} active={self.is_active}>"


class CrawlingPerformanceMetrics(Base):
    """
    Stores detailed performance metrics for crawling jobs.
    Enables performance monitoring, optimization, and analytics.
    """
    __tablename__ = "crawling_performance_metrics"

    # Foreign key to crawling job
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Metric identification
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(DECIMAL(15, 6), nullable=False)
    metric_unit: Mapped[Optional[str]] = mapped_column(String(20))

    # Measurement context
    measurement_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    measurement_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Additional context for the measurement"
    )

    # Metric categorization
    metric_category: Mapped[Optional[str]] = mapped_column(
        String(50),
        index=True,
        comment="Category: performance, resource, network, quality, error"
    )
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    job = relationship("CrawlingJob", back_populates="performance_metrics")

    # Table constraints and indexes
    __table_args__ = (
        Index('idx_perf_metrics_job_name_time', 'job_id', 'metric_name', 'measurement_time'),
        Index('idx_perf_metrics_category_time', 'metric_category', 'measurement_time'),
        Index('idx_perf_metrics_anomaly', 'is_anomaly', 'measurement_time'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingPerformanceMetrics {self.job_id}:{self.metric_name}={self.metric_value}>"


class CrawlingError(Base):
    """
    Comprehensive error tracking for crawling operations.
    Supports error categorization, recovery tracking, and debugging.
    """
    __tablename__ = "crawling_errors"

    # Foreign key to crawling job
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Error classification
    error_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type: network_error, parsing_error, javascript_error, timeout_error, etc."
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)

    # Error context
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    stack_trace: Mapped[Optional[str]] = mapped_column(Text)
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Additional context about the error"
    )

    # Error timing and frequency
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    is_recoverable: Mapped[bool] = mapped_column(Boolean, default=True)

    # Error resolution
    resolution_status: Mapped[str] = mapped_column(
        String(20),
        default="unresolved",
        comment="Status: unresolved, resolved, ignored, escalated"
    )
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    job = relationship("CrawlingJob", back_populates="errors")

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("resolution_status IN ('unresolved', 'resolved', 'ignored', 'escalated')", name='valid_resolution_status'),
        Index('idx_crawling_errors_type_time', 'error_type', 'occurred_at'),
        Index('idx_crawling_errors_job_type', 'job_id', 'error_type'),
        Index('idx_crawling_errors_recoverable', 'is_recoverable', 'retry_count'),
        Index('idx_crawling_errors_resolution', 'resolution_status', 'occurred_at'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingError {self.job_id}:{self.error_type}>"


class CrawlingJobDependency(Base):
    """
    Manages dependencies between crawling jobs.
    Supports complex workflow orchestration and job chaining.
    """
    __tablename__ = "crawling_job_dependencies"

    # Job relationship
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    depends_on_job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crawling_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Dependency configuration
    dependency_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: sequential, verification_required, data_required, conditional"
    )
    is_blocking: Mapped[bool] = mapped_column(Boolean, default=True)

    # Condition and status
    condition_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Conditions that must be met for dependency resolution"
    )
    is_satisfied: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    satisfied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    job = relationship("CrawlingJob", foreign_keys=[job_id], back_populates="dependencies")
    depends_on_job = relationship("CrawlingJob", foreign_keys=[depends_on_job_id], back_populates="dependents")

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("dependency_type IN ('sequential', 'verification_required', 'data_required', 'conditional')", name='valid_dependency_type'),
        CheckConstraint('job_id != depends_on_job_id', name='no_self_dependency'),
        UniqueConstraint('job_id', 'depends_on_job_id', name='unique_job_dependency'),
        Index('idx_job_dependencies_satisfied', 'is_satisfied', 'job_id'),
        Index('idx_job_dependencies_blocking', 'is_blocking', 'depends_on_job_id'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingJobDependency {self.job_id} depends_on {self.depends_on_job_id}>"


class CrawlingAnalytics(Base):
    """
    Aggregated analytics and reporting data for crawling operations.
    Provides insights into crawling performance, costs, and trends.
    """
    __tablename__ = "crawling_analytics"

    # Time-based aggregation
    period_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="Aggregation period: daily, weekly, monthly"
    )
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Scope of analytics
    website_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("websites.id", ondelete="CASCADE"),
        index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True
    )

    # Job statistics
    total_jobs: Mapped[int] = mapped_column(Integer, default=0)
    successful_jobs: Mapped[int] = mapped_column(Integer, default=0)
    failed_jobs: Mapped[int] = mapped_column(Integer, default=0)

    # Performance metrics
    total_pages_crawled: Mapped[int] = mapped_column(Integer, default=0)
    total_data_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    average_execution_time_seconds: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 3))

    # Cost and resource metrics
    total_compute_time_seconds: Mapped[Optional[float]] = mapped_column(DECIMAL(15, 3))
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 4))

    # Analytics data
    analytics_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Detailed analytics and trend data"
    )

    # Relationships
    website = relationship("Website", foreign_keys=[website_id])
    user = relationship("User", foreign_keys=[user_id])

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("period_type IN ('daily', 'weekly', 'monthly')", name='valid_period_type'),
        CheckConstraint('period_start < period_end', name='valid_period_range'),
        CheckConstraint('successful_jobs + failed_jobs <= total_jobs', name='valid_job_counts'),
        UniqueConstraint('period_type', 'period_start', 'website_id', 'user_id', name='unique_analytics_period'),
        Index('idx_analytics_website_period', 'website_id', 'period_type', 'period_start'),
        Index('idx_analytics_user_period', 'user_id', 'period_type', 'period_start'),
    )

    def __repr__(self) -> str:
        return f"<CrawlingAnalytics {self.period_type} {self.period_start} jobs={self.total_jobs}>"