"""
Database optimization service for managing performance optimizations.
Handles indexing, query optimization, connection pooling, and monitoring.
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import asyncio
import time
import json
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text, Index, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select

from app.models.base import Base
from app.models.crawling_schema import (
    CrawlingJob, CrawlingJobResult, CrawlingPerformanceMetrics,
    CrawlingError, CrawlingSchedule, CrawlingAnalytics
)


class OptimizationType(Enum):
    """Types of database optimizations."""
    INDEX_CREATION = "index_creation"
    QUERY_OPTIMIZATION = "query_optimization"
    CONNECTION_POOLING = "connection_pooling"
    CACHING = "caching"
    MONITORING = "monitoring"


@dataclass
class OptimizationResult:
    """Result of a database optimization operation."""
    optimization_type: OptimizationType
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for database queries."""
    query_name: str
    execution_time: float
    row_count: int
    cache_hit: bool = False
    index_used: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class DatabaseIndexManager:
    """Manages database indexes for optimal query performance."""

    def __init__(self):
        self.created_indexes: List[str] = []

    async def create_crawling_job_indexes(self, session: AsyncSession) -> List[OptimizationResult]:
        """Create optimized indexes for crawling job queries."""
        results = []

        # Index definitions for common query patterns
        index_definitions = [
            # 1. Status + Priority composite index for job queue processing
            {
                "name": "idx_crawling_jobs_status_priority_optimized",
                "table": "crawling_jobs",
                "columns": ["status", "priority DESC", "created_at"],
                "where": "status IN ('pending', 'running')",
                "rationale": "Optimizes job queue processing with priority ordering"
            },

            # 2. Website + Status for dashboard queries
            {
                "name": "idx_crawling_jobs_website_status_optimized",
                "table": "crawling_jobs",
                "columns": ["website_id", "status", "updated_at DESC"],
                "rationale": "Optimizes website-specific job status queries"
            },

            # 3. User + Date range for user dashboard
            {
                "name": "idx_crawling_jobs_user_date_optimized",
                "table": "crawling_jobs",
                "columns": ["user_id", "created_at DESC", "status"],
                "rationale": "Optimizes user dashboard date range queries"
            },

            # 4. Scheduled jobs index
            {
                "name": "idx_crawling_jobs_scheduled_optimized",
                "table": "crawling_jobs",
                "columns": ["scheduled_at", "status"],
                "where": "scheduled_at IS NOT NULL AND status = 'pending'",
                "rationale": "Optimizes scheduled job pickup queries"
            },

            # 5. Worker assignment index
            {
                "name": "idx_crawling_jobs_worker_optimized",
                "table": "crawling_jobs",
                "columns": ["worker_id", "status", "started_at"],
                "where": "worker_id IS NOT NULL",
                "rationale": "Optimizes worker job tracking queries"
            }
        ]

        for index_def in index_definitions:
            result = await self._create_index(session, index_def)
            results.append(result)

        return results

    async def create_performance_metrics_indexes(self, session: AsyncSession) -> List[OptimizationResult]:
        """Create indexes for performance metrics queries."""
        results = []

        index_definitions = [
            # 1. Job + Metric + Time for time-series queries
            {
                "name": "idx_perf_metrics_job_metric_time_optimized",
                "table": "crawling_performance_metrics",
                "columns": ["job_id", "metric_name", "measurement_time DESC"],
                "rationale": "Optimizes time-series performance metric queries"
            },

            # 2. Metric category + Time for analytics
            {
                "name": "idx_perf_metrics_category_time_optimized",
                "table": "crawling_performance_metrics",
                "columns": ["metric_category", "measurement_time DESC", "metric_value"],
                "rationale": "Optimizes performance analytics by category"
            },

            # 3. Anomaly detection index
            {
                "name": "idx_perf_metrics_anomaly_optimized",
                "table": "crawling_performance_metrics",
                "columns": ["is_anomaly", "measurement_time DESC", "metric_name"],
                "where": "is_anomaly = true",
                "rationale": "Optimizes anomaly detection queries"
            }
        ]

        for index_def in index_definitions:
            result = await self._create_index(session, index_def)
            results.append(result)

        return results

    async def create_analytics_indexes(self, session: AsyncSession) -> List[OptimizationResult]:
        """Create indexes for analytics and reporting queries."""
        results = []

        index_definitions = [
            # 1. Website + Period for analytics dashboards
            {
                "name": "idx_analytics_website_period_optimized",
                "table": "crawling_analytics",
                "columns": ["website_id", "period_type", "period_start DESC"],
                "rationale": "Optimizes website analytics dashboard queries"
            },

            # 2. User + Period for user analytics
            {
                "name": "idx_analytics_user_period_optimized",
                "table": "crawling_analytics",
                "columns": ["user_id", "period_type", "period_start DESC"],
                "rationale": "Optimizes user analytics queries"
            },

            # 3. Global analytics by period
            {
                "name": "idx_analytics_global_period_optimized",
                "table": "crawling_analytics",
                "columns": ["period_type", "period_start DESC"],
                "where": "website_id IS NULL AND user_id IS NULL",
                "rationale": "Optimizes global analytics queries"
            }
        ]

        for index_def in index_definitions:
            result = await self._create_index(session, index_def)
            results.append(result)

        return results

    async def create_error_tracking_indexes(self, session: AsyncSession) -> List[OptimizationResult]:
        """Create indexes for error tracking and analysis."""
        results = []

        index_definitions = [
            # 1. Error type + Time for error analysis
            {
                "name": "idx_errors_type_time_optimized",
                "table": "crawling_errors",
                "columns": ["error_type", "occurred_at DESC", "resolution_status"],
                "rationale": "Optimizes error analysis by type and time"
            },

            # 2. Job errors for troubleshooting
            {
                "name": "idx_errors_job_resolution_optimized",
                "table": "crawling_errors",
                "columns": ["job_id", "resolution_status", "occurred_at DESC"],
                "rationale": "Optimizes job-specific error troubleshooting"
            },

            # 3. Unresolved errors index
            {
                "name": "idx_errors_unresolved_optimized",
                "table": "crawling_errors",
                "columns": ["resolution_status", "error_type", "occurred_at DESC"],
                "where": "resolution_status = 'unresolved'",
                "rationale": "Optimizes unresolved error tracking"
            }
        ]

        for index_def in index_definitions:
            result = await self._create_index(session, index_def)
            results.append(result)

        return results

    async def _create_index(self, session: AsyncSession, index_def: Dict[str, Any]) -> OptimizationResult:
        """Create a single database index."""
        start_time = time.time()

        try:
            # Build CREATE INDEX statement
            columns_str = ", ".join(index_def["columns"])
            sql = f"CREATE INDEX IF NOT EXISTS {index_def['name']} ON {index_def['table']} ({columns_str})"

            if "where" in index_def:
                sql += f" WHERE {index_def['where']}"

            # Execute index creation
            await session.execute(text(sql))
            await session.commit()

            execution_time = time.time() - start_time
            self.created_indexes.append(index_def["name"])

            return OptimizationResult(
                optimization_type=OptimizationType.INDEX_CREATION,
                success=True,
                execution_time=execution_time,
                details={
                    "index_name": index_def["name"],
                    "table": index_def["table"],
                    "columns": index_def["columns"],
                    "rationale": index_def["rationale"],
                    "partial": "where" in index_def
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return OptimizationResult(
                optimization_type=OptimizationType.INDEX_CREATION,
                success=False,
                execution_time=execution_time,
                details={"index_name": index_def["name"], "table": index_def["table"]},
                error_message=str(e)
            )

    async def analyze_index_usage(self, session: AsyncSession) -> Dict[str, Any]:
        """Analyze index usage statistics."""
        try:
            # PostgreSQL-specific query for index usage
            usage_query = """
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
            """

            result = await session.execute(text(usage_query))
            usage_stats = result.all()

            return {
                "total_indexes": len(usage_stats),
                "frequently_used": [
                    {
                        "index_name": row.indexname,
                        "table": row.tablename,
                        "scans": row.index_scans,
                        "tuples_read": row.tuples_read
                    }
                    for row in usage_stats[:10]  # Top 10
                ],
                "unused_indexes": [
                    {
                        "index_name": row.indexname,
                        "table": row.tablename
                    }
                    for row in usage_stats if row.index_scans == 0
                ]
            }

        except Exception as e:
            # Fallback for non-PostgreSQL databases
            return {
                "total_indexes": len(self.created_indexes),
                "created_indexes": self.created_indexes,
                "analysis_error": str(e)
            }


class QueryOptimizer:
    """Optimizes database queries for better performance."""

    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.query_stats: Dict[str, List[float]] = {}

    async def optimize_job_queue_query(self, session: AsyncSession, limit: int = 10) -> Tuple[List[CrawlingJob], QueryPerformanceMetrics]:
        """Optimized query for job queue processing."""
        start_time = time.time()

        # Optimized query using proper indexing
        query = (
            session.query(CrawlingJob)
            .filter(CrawlingJob.status == "pending")
            .order_by(CrawlingJob.priority.desc(), CrawlingJob.created_at.asc())
            .limit(limit)
        )

        result = await query.all()
        execution_time = time.time() - start_time

        metrics = QueryPerformanceMetrics(
            query_name="job_queue_optimized",
            execution_time=execution_time,
            row_count=len(result),
            index_used=True
        )

        self._record_query_stats("job_queue_optimized", execution_time)
        return result, metrics

    async def optimize_dashboard_query(self, session: AsyncSession, website_id: str) -> Tuple[Dict[str, Any], QueryPerformanceMetrics]:
        """Optimized query for website dashboard data."""
        start_time = time.time()

        # Use cache key
        cache_key = f"dashboard:{website_id}"

        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_result["timestamp"]).seconds < 300:  # 5 min cache
                execution_time = time.time() - start_time
                metrics = QueryPerformanceMetrics(
                    query_name="dashboard_optimized",
                    execution_time=execution_time,
                    row_count=cached_result["data"]["total_jobs"],
                    cache_hit=True
                )
                return cached_result["data"], metrics

        # Optimized dashboard query
        job_counts = await session.execute(
            text("""
                SELECT
                    status,
                    COUNT(*) as count,
                    AVG(priority) as avg_priority
                FROM crawling_jobs
                WHERE website_id = :website_id
                GROUP BY status
            """),
            {"website_id": website_id}
        )

        recent_jobs = await session.execute(
            text("""
                SELECT id, job_type, status, created_at, priority
                FROM crawling_jobs
                WHERE website_id = :website_id
                ORDER BY created_at DESC
                LIMIT 5
            """),
            {"website_id": website_id}
        )

        # Aggregate results
        status_counts = {row.status: {"count": row.count, "avg_priority": row.avg_priority} for row in job_counts}
        recent_job_list = [
            {
                "id": row.id,
                "job_type": row.job_type,
                "status": row.status,
                "created_at": row.created_at,
                "priority": row.priority
            }
            for row in recent_jobs
        ]

        dashboard_data = {
            "total_jobs": sum(data["count"] for data in status_counts.values()),
            "status_breakdown": status_counts,
            "recent_jobs": recent_job_list
        }

        # Cache result
        self.query_cache[cache_key] = {
            "data": dashboard_data,
            "timestamp": datetime.now(timezone.utc)
        }

        execution_time = time.time() - start_time
        metrics = QueryPerformanceMetrics(
            query_name="dashboard_optimized",
            execution_time=execution_time,
            row_count=dashboard_data["total_jobs"],
            cache_hit=False
        )

        self._record_query_stats("dashboard_optimized", execution_time)
        return dashboard_data, metrics

    async def optimize_analytics_query(self, session: AsyncSession, period_type: str = "daily", days: int = 30) -> Tuple[List[Dict], QueryPerformanceMetrics]:
        """Optimized query for analytics data."""
        start_time = time.time()

        # Optimized analytics query with date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        analytics_query = await session.execute(
            text("""
                SELECT
                    period_start,
                    period_type,
                    SUM(total_jobs) as total_jobs,
                    SUM(successful_jobs) as successful_jobs,
                    SUM(failed_jobs) as failed_jobs,
                    AVG(average_execution_time_seconds) as avg_execution_time
                FROM crawling_analytics
                WHERE period_type = :period_type
                    AND period_start >= :start_date
                    AND period_start <= :end_date
                GROUP BY period_start, period_type
                ORDER BY period_start DESC
            """),
            {
                "period_type": period_type,
                "start_date": start_date,
                "end_date": end_date
            }
        )

        analytics_data = [
            {
                "period_start": row.period_start,
                "period_type": row.period_type,
                "total_jobs": row.total_jobs,
                "successful_jobs": row.successful_jobs,
                "failed_jobs": row.failed_jobs,
                "success_rate": row.successful_jobs / row.total_jobs if row.total_jobs > 0 else 0,
                "avg_execution_time": float(row.avg_execution_time) if row.avg_execution_time else 0
            }
            for row in analytics_query
        ]

        execution_time = time.time() - start_time
        metrics = QueryPerformanceMetrics(
            query_name="analytics_optimized",
            execution_time=execution_time,
            row_count=len(analytics_data),
            index_used=True
        )

        self._record_query_stats("analytics_optimized", execution_time)
        return analytics_data, metrics

    def _record_query_stats(self, query_name: str, execution_time: float):
        """Record query execution statistics."""
        if query_name not in self.query_stats:
            self.query_stats[query_name] = []

        self.query_stats[query_name].append(execution_time)

        # Keep only last 100 measurements
        if len(self.query_stats[query_name]) > 100:
            self.query_stats[query_name] = self.query_stats[query_name][-100:]

    def get_query_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of query performance statistics."""
        summary = {}

        for query_name, times in self.query_stats.items():
            if times:
                summary[query_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "recent_avg": sum(times[-10:]) / min(len(times), 10)  # Last 10 queries
                }

        return summary


class ConnectionPoolManager:
    """Manages database connection pooling for optimal performance."""

    def __init__(self):
        self.engines: Dict[str, Any] = {}
        self.connection_stats: Dict[str, Dict[str, int]] = {}

    def create_optimized_engine(self, database_url: str, pool_config: Optional[Dict[str, Any]] = None) -> Any:
        """Create optimized database engine with connection pooling."""
        default_config = {
            "poolclass": QueuePool,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # 1 hour
            "echo": False
        }

        if pool_config:
            default_config.update(pool_config)

        engine = create_async_engine(database_url, **default_config)

        # Set up connection event listeners for monitoring
        self._setup_connection_monitoring(engine)

        return engine

    def _setup_connection_monitoring(self, engine: Any):
        """Set up connection monitoring for the engine."""
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Track connection creation."""
            engine_id = id(engine)
            if engine_id not in self.connection_stats:
                self.connection_stats[engine_id] = {"created": 0, "closed": 0, "active": 0}

            self.connection_stats[engine_id]["created"] += 1
            self.connection_stats[engine_id]["active"] += 1

        @event.listens_for(engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            """Track connection closure."""
            engine_id = id(engine)
            if engine_id in self.connection_stats:
                self.connection_stats[engine_id]["closed"] += 1
                self.connection_stats[engine_id]["active"] -= 1

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        total_stats = {
            "total_created": 0,
            "total_closed": 0,
            "total_active": 0,
            "engines": len(self.connection_stats)
        }

        for engine_id, stats in self.connection_stats.items():
            total_stats["total_created"] += stats["created"]
            total_stats["total_closed"] += stats["closed"]
            total_stats["total_active"] += stats["active"]

        return total_stats


class DatabasePerformanceOptimizer:
    """Main service for database performance optimization."""

    def __init__(self):
        self.index_manager = DatabaseIndexManager()
        self.query_optimizer = QueryOptimizer()
        self.connection_manager = ConnectionPoolManager()
        self.monitoring_enabled = True

    async def initialize_optimizations(self, session: AsyncSession) -> List[OptimizationResult]:
        """Initialize all database optimizations."""
        results = []

        # Create all optimized indexes
        results.extend(await self.index_manager.create_crawling_job_indexes(session))
        results.extend(await self.index_manager.create_performance_metrics_indexes(session))
        results.extend(await self.index_manager.create_analytics_indexes(session))
        results.extend(await self.index_manager.create_error_tracking_indexes(session))

        return results

    async def get_optimization_report(self, session: AsyncSession) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Index usage analysis
        index_analysis = await self.index_manager.analyze_index_usage(session)

        # Query performance summary
        query_performance = self.query_optimizer.get_query_performance_summary()

        # Connection pool statistics
        connection_stats = self.connection_manager.get_connection_stats()

        return {
            "timestamp": datetime.now(timezone.utc),
            "index_analysis": index_analysis,
            "query_performance": query_performance,
            "connection_statistics": connection_stats,
            "total_optimizations": len(self.index_manager.created_indexes),
            "cache_entries": len(self.query_optimizer.query_cache)
        }

    async def validate_optimization_effectiveness(self, session: AsyncSession) -> Dict[str, Any]:
        """Validate the effectiveness of applied optimizations."""
        validation_results = {
            "timestamp": datetime.now(timezone.utc),
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_improvements": {},
            "recommendations": []
        }

        # Test 1: Job queue query performance
        try:
            jobs, metrics = await self.query_optimizer.optimize_job_queue_query(session, limit=10)
            if metrics.execution_time < 0.1:  # Should be very fast
                validation_results["tests_passed"] += 1
                validation_results["performance_improvements"]["job_queue"] = "EXCELLENT"
            else:
                validation_results["tests_failed"] += 1
                validation_results["recommendations"].append("Consider reviewing job queue indexes")
        except Exception as e:
            validation_results["tests_failed"] += 1
            validation_results["recommendations"].append(f"Job queue query failed: {str(e)}")

        # Test 2: Dashboard query performance
        try:
            dashboard_data, metrics = await self.query_optimizer.optimize_dashboard_query(session, str(uuid4()))
            if metrics.execution_time < 0.5:  # Should be reasonably fast
                validation_results["tests_passed"] += 1
                validation_results["performance_improvements"]["dashboard"] = "GOOD"
            else:
                validation_results["tests_failed"] += 1
                validation_results["recommendations"].append("Dashboard queries may need optimization")
        except Exception as e:
            validation_results["tests_failed"] += 1
            validation_results["recommendations"].append(f"Dashboard query failed: {str(e)}")

        # Test 3: Analytics query performance
        try:
            analytics_data, metrics = await self.query_optimizer.optimize_analytics_query(session)
            if metrics.execution_time < 1.0:  # Should complete within 1 second
                validation_results["tests_passed"] += 1
                validation_results["performance_improvements"]["analytics"] = "GOOD"
            else:
                validation_results["tests_failed"] += 1
                validation_results["recommendations"].append("Analytics queries may need optimization")
        except Exception as e:
            validation_results["tests_failed"] += 1
            validation_results["recommendations"].append(f"Analytics query failed: {str(e)}")

        # Overall assessment
        total_tests = validation_results["tests_passed"] + validation_results["tests_failed"]
        validation_results["success_rate"] = validation_results["tests_passed"] / total_tests if total_tests > 0 else 0
        validation_results["overall_status"] = "HEALTHY" if validation_results["success_rate"] >= 0.8 else "NEEDS_ATTENTION"

        return validation_results