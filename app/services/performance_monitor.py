"""
Performance Monitor for comprehensive crawling performance monitoring and analytics.
Task 5.1: Implement crawling performance monitoring
"""

import asyncio
import logging
import psutil
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
from threading import Lock

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CrawlMetrics:
    """Individual crawl performance metrics."""
    url: str
    start_time: datetime
    end_time: datetime
    success: bool
    response_time: float
    status_code: Optional[int] = None
    content_size: int = 0
    page_load_time: Optional[float] = None
    dom_ready_time: Optional[float] = None
    first_paint_time: Optional[float] = None
    resource_count: int = 0
    javascript_errors: int = 0
    network_requests: int = 0
    error_message: Optional[str] = None
    user_agent: Optional[str] = None
    browser_engine: Optional[str] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    active_connections: int
    load_average: Optional[float] = None
    disk_io_bytes: Optional[int] = None


@dataclass
class PerformanceAlert:
    """Performance alert."""
    metric_type: MetricType
    level: AlertLevel
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    url: Optional[str] = None
    resolved: bool = False


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    total_crawls: int
    successful_crawls: int
    failed_crawls: int
    crawls_per_second: float
    crawls_per_minute: float
    crawls_per_hour: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float


class PerformanceDashboard:
    """Real-time performance dashboard."""

    def __init__(self, performance_monitor):
        """Initialize dashboard."""
        self.performance_monitor = performance_monitor

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current dashboard metrics."""
        stats = self.performance_monitor.get_performance_statistics()
        system_metrics = self.performance_monitor.get_current_system_metrics()

        return {
            "current_throughput": stats.get("current_throughput", 0),
            "success_rate": stats.get("success_rate", 0),
            "average_response_time": stats.get("average_response_time", 0),
            "active_crawls": stats.get("active_crawls", 0),
            "total_crawls": stats.get("total_crawls", 0),
            "system_metrics": {
                "cpu_usage": system_metrics.cpu_usage,
                "memory_usage": system_metrics.memory_usage_percent,
                "network_io": system_metrics.network_io_bytes,
                "active_connections": system_metrics.active_connections
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitor for crawling operations.

    Features:
    - Real-time performance metrics collection
    - Crawling speed and throughput monitoring
    - Resource utilization tracking (CPU, memory, bandwidth)
    - Performance baseline establishment
    - Alerting and notification system
    - Performance optimization recommendations
    - Real-time dashboard
    """

    def __init__(
        self,
        metrics_retention_hours: int = 24,
        alert_check_interval: int = 60,
        system_monitoring_interval: int = 30
    ):
        """Initialize performance monitor."""
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_check_interval = alert_check_interval
        self.system_monitoring_interval = system_monitoring_interval

        # Metrics storage
        self.crawl_metrics: deque = deque(maxlen=10000)  # Store last 10k crawls
        self.system_metrics: deque = deque(maxlen=2880)  # 24 hours at 30s intervals

        # Performance statistics
        self.performance_baseline: Optional[Dict[str, Any]] = None
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_config: Dict[str, float] = {
            "response_time_threshold": 5.0,
            "success_rate_threshold": 95.0,
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "error_rate_threshold": 5.0
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.metrics_lock = Lock()

        # Performance tracking
        self.start_time = datetime.now(timezone.utc)
        self.active_crawls = 0

        # Dashboard
        self.dashboard = PerformanceDashboard(self)

    async def initialize(self) -> None:
        """Initialize performance monitor."""
        try:
            self.is_monitoring = True

            # Start background monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._system_monitoring_loop()),
                asyncio.create_task(self._alert_checking_loop()),
                asyncio.create_task(self._metrics_cleanup_loop())
            ]

            logger.info("Performance monitor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize performance monitor: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown performance monitor."""
        logger.info("Shutting down performance monitor...")

        self.is_monitoring = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.monitoring_tasks.clear()
        logger.info("Performance monitor shutdown complete")

    def record_crawl_metrics(self, metrics: CrawlMetrics) -> None:
        """Record crawl performance metrics."""
        with self.metrics_lock:
            self.crawl_metrics.append(metrics)

        # Update active crawl count
        if metrics.success:
            self.active_crawls = max(0, self.active_crawls - 1)

        # Check for alerts
        self._check_crawl_metrics_alerts(metrics)

        logger.debug(f"Recorded metrics for {metrics.url}: {metrics.response_time:.2f}s")

    def start_crawl_tracking(self, url: str) -> None:
        """Start tracking a crawl operation."""
        self.active_crawls += 1
        logger.debug(f"Started tracking crawl for {url}")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.metrics_lock:
            if not self.crawl_metrics:
                return {
                    "total_crawls": 0,
                    "successful_crawls": 0,
                    "failed_crawls": 0,
                    "success_rate": 0.0,
                    "average_response_time": 0.0,
                    "active_crawls": self.active_crawls
                }

            crawl_list = list(self.crawl_metrics)

        # Basic statistics
        total_crawls = len(crawl_list)
        successful_crawls = sum(1 for m in crawl_list if m.success)
        failed_crawls = total_crawls - successful_crawls
        success_rate = (successful_crawls / total_crawls) * 100 if total_crawls > 0 else 0.0

        # Response time statistics
        response_times = [m.response_time for m in crawl_list if m.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        median_response_time = statistics.median(response_times) if response_times else 0.0

        # Calculate percentiles
        p95_response_time = 0.0
        p99_response_time = 0.0
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
            p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

        return {
            "total_crawls": total_crawls,
            "successful_crawls": successful_crawls,
            "failed_crawls": failed_crawls,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "active_crawls": self.active_crawls,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
        }

    def calculate_throughput_stats(self) -> ThroughputStats:
        """Calculate crawling throughput statistics."""
        with self.metrics_lock:
            crawl_list = list(self.crawl_metrics)

        if not crawl_list:
            return ThroughputStats(
                total_crawls=0,
                successful_crawls=0,
                failed_crawls=0,
                crawls_per_second=0.0,
                crawls_per_minute=0.0,
                crawls_per_hour=0.0,
                average_response_time=0.0,
                median_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0
            )

        # Time-based calculations
        now = datetime.now(timezone.utc)
        time_window = timedelta(hours=1)  # Calculate for last hour
        recent_crawls = [
            m for m in crawl_list
            if m.start_time > now - time_window
        ]

        if recent_crawls:
            time_span_seconds = (now - recent_crawls[0].start_time).total_seconds()
            crawls_per_second = len(recent_crawls) / max(time_span_seconds, 1)
        else:
            crawls_per_second = 0.0

        # Response time calculations
        response_times = [m.response_time for m in crawl_list if m.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        median_response_time = statistics.median(response_times) if response_times else 0.0

        # Percentiles
        p95_response_time = 0.0
        p99_response_time = 0.0
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
            p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

        return ThroughputStats(
            total_crawls=len(crawl_list),
            successful_crawls=sum(1 for m in crawl_list if m.success),
            failed_crawls=sum(1 for m in crawl_list if not m.success),
            crawls_per_second=crawls_per_second,
            crawls_per_minute=crawls_per_second * 60,
            crawls_per_hour=crawls_per_second * 3600,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time
        )

    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        try:
            process = psutil.Process()

            # CPU metrics
            cpu_usage = process.cpu_percent()

            # Memory metrics
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # System memory
            system_memory = psutil.virtual_memory()
            memory_percent = system_memory.percent

            # Disk usage
            disk_usage = psutil.disk_usage('/').percent

            # Network I/O
            network_io = psutil.net_io_counters()
            network_bytes = network_io.bytes_sent + network_io.bytes_recv

            # Connection count
            try:
                connections = len(psutil.net_connections())
            except:
                connections = 0

            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_mb,
                memory_usage_percent=memory_percent,
                disk_usage_percent=disk_usage,
                network_io_bytes=network_bytes,
                active_connections=connections
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=0.0,
                memory_usage_mb=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_io_bytes=0,
                active_connections=0
            )

    async def start_system_monitoring(self) -> None:
        """Start system resource monitoring."""
        self.is_monitoring = True
        logger.info("System monitoring started")

    async def start_resource_monitoring(self) -> None:
        """Start resource monitoring (alias for system monitoring)."""
        await self.start_system_monitoring()

    async def start_network_monitoring(self) -> None:
        """Start network monitoring."""
        # Network monitoring is part of system monitoring
        await self.start_system_monitoring()

    def get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU usage metrics."""
        try:
            current_usage = psutil.cpu_percent(interval=0.1)

            # Calculate average from recent system metrics
            recent_metrics = list(self.system_metrics)[-10:]  # Last 10 measurements
            if recent_metrics:
                average_usage = statistics.mean([m.cpu_usage for m in recent_metrics])
                peak_usage = max([m.cpu_usage for m in recent_metrics])
            else:
                average_usage = current_usage
                peak_usage = current_usage

            return {
                "current_usage": current_usage,
                "average_usage": average_usage,
                "peak_usage": peak_usage
            }
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            return {"current_usage": 0.0, "average_usage": 0.0, "peak_usage": 0.0}

    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            current_mb = memory_info.rss / 1024 / 1024
            system_memory = psutil.virtual_memory()
            current_percent = system_memory.percent

            # Calculate peak from recent metrics
            recent_metrics = list(self.system_metrics)[-10:]
            if recent_metrics:
                peak_mb = max([m.memory_usage_mb for m in recent_metrics])
            else:
                peak_mb = current_mb

            return {
                "current_usage_mb": current_mb,
                "current_usage_percent": current_percent,
                "peak_usage_mb": peak_mb
            }
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            return {"current_usage_mb": 0.0, "current_usage_percent": 0.0, "peak_usage_mb": 0.0}

    def get_network_metrics(self) -> Dict[str, int]:
        """Get network I/O metrics."""
        try:
            network_io = psutil.net_io_counters()

            return {
                "bytes_sent": network_io.bytes_sent,
                "bytes_received": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_received": network_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Error getting network metrics: {e}")
            return {"bytes_sent": 0, "bytes_received": 0, "packets_sent": 0, "packets_received": 0}

    def get_bandwidth_utilization(self) -> Dict[str, float]:
        """Get bandwidth utilization metrics."""
        return self._calculate_bandwidth_usage()

    def _calculate_bandwidth_usage(self) -> Dict[str, float]:
        """Calculate bandwidth usage (mock implementation)."""
        # This would need real network interface monitoring
        return {
            "current_bandwidth_mbps": 10.0,
            "peak_bandwidth_mbps": 50.0,
            "average_bandwidth_mbps": 15.0,
            "utilization_percent": 20.0
        }

    def establish_performance_baseline(self) -> Dict[str, Any]:
        """Establish performance baseline from historical data."""
        with self.metrics_lock:
            crawl_list = list(self.crawl_metrics)

        if len(crawl_list) < 10:  # Need minimum data
            return {"error": "Insufficient data for baseline establishment"}

        # Calculate baseline metrics
        successful_crawls = [m for m in crawl_list if m.success]
        if not successful_crawls:
            return {"error": "No successful crawls for baseline"}

        response_times = [m.response_time for m in successful_crawls]

        baseline = {
            "total_samples": len(successful_crawls),
            "average_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
            "success_rate": len(successful_crawls) / len(crawl_list) * 100,
            "baseline_established_at": datetime.now(timezone.utc).isoformat()
        }

        self.performance_baseline = baseline
        logger.info("Performance baseline established")

        return baseline

    async def measure_monitoring_overhead(self) -> Dict[str, float]:
        """Measure monitoring overhead."""
        start_time = time.time()

        # Measure metrics collection time
        metrics = self.get_current_system_metrics()
        collection_time = (time.time() - start_time) * 1000  # Convert to ms

        # Estimate storage overhead
        metrics_count = len(self.crawl_metrics) + len(self.system_metrics)
        storage_mb = metrics_count * 0.001  # Rough estimate

        # CPU overhead is minimal for monitoring
        cpu_overhead = 0.1  # Rough estimate

        return {
            "metrics_collection_time_ms": collection_time,
            "storage_overhead_mb": storage_mb,
            "cpu_overhead_percent": cpu_overhead
        }

    def configure_alerts(self, alert_config: Dict[str, float]) -> None:
        """Configure alert thresholds."""
        self.alert_config.update(alert_config)
        logger.info(f"Alert configuration updated: {alert_config}")

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        return [alert for alert in self.active_alerts if not alert.resolved]

    def _check_crawl_metrics_alerts(self, metrics: CrawlMetrics) -> None:
        """Check crawl metrics against alert thresholds."""
        # Response time alert
        if metrics.response_time > self.alert_config["response_time_threshold"]:
            alert = PerformanceAlert(
                metric_type=MetricType.RESPONSE_TIME,
                level=AlertLevel.WARNING if metrics.response_time < self.alert_config["response_time_threshold"] * 2 else AlertLevel.CRITICAL,
                message=f"High response time: {metrics.response_time:.2f}s",
                current_value=metrics.response_time,
                threshold_value=self.alert_config["response_time_threshold"],
                timestamp=datetime.now(timezone.utc),
                url=metrics.url
            )
            self.active_alerts.append(alert)

    def get_dashboard(self) -> PerformanceDashboard:
        """Get dashboard instance."""
        return self.dashboard

    def get_performance_charts_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get data for performance charts."""
        with self.metrics_lock:
            crawl_list = list(self.crawl_metrics)

        # Generate time-series data for charts
        now = datetime.now(timezone.utc)
        time_buckets = []

        # Create 12 time buckets (5 minutes each for last hour)
        for i in range(12):
            bucket_start = now - timedelta(minutes=(i+1)*5)
            bucket_end = now - timedelta(minutes=i*5)
            time_buckets.append((bucket_start, bucket_end))

        response_time_trend = []
        throughput_trend = []
        success_rate_trend = []

        for bucket_start, bucket_end in reversed(time_buckets):
            bucket_metrics = [
                m for m in crawl_list
                if bucket_start <= m.start_time < bucket_end
            ]

            if bucket_metrics:
                avg_response = statistics.mean([m.response_time for m in bucket_metrics])
                throughput = len(bucket_metrics)
                success_rate = sum(1 for m in bucket_metrics if m.success) / len(bucket_metrics) * 100
            else:
                avg_response = 0
                throughput = 0
                success_rate = 0

            timestamp = bucket_end.strftime("%H:%M")

            response_time_trend.append({"time": timestamp, "value": avg_response})
            throughput_trend.append({"time": timestamp, "value": throughput})
            success_rate_trend.append({"time": timestamp, "value": success_rate})

        return {
            "response_time_trend": response_time_trend,
            "throughput_trend": throughput_trend,
            "success_rate_trend": success_rate_trend,
            "system_resource_trend": self._get_system_resource_trend()
        }

    def _get_system_resource_trend(self) -> List[Dict[str, Any]]:
        """Get system resource trend data."""
        recent_metrics = list(self.system_metrics)[-12:]  # Last 12 measurements

        trend_data = []
        for metrics in recent_metrics:
            trend_data.append({
                "time": metrics.timestamp.strftime("%H:%M"),
                "cpu": metrics.cpu_usage,
                "memory": metrics.memory_usage_percent
            })

        return trend_data

    def identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        stats = self.get_performance_statistics()

        # Response time bottleneck
        if stats["average_response_time"] > 3.0:
            bottlenecks.append({
                "type": "high_response_time",
                "severity": "high" if stats["average_response_time"] > 5.0 else "medium",
                "description": f"Average response time is {stats['average_response_time']:.2f}s",
                "recommendation": "Consider optimizing crawling logic or increasing timeout values"
            })

        # Success rate bottleneck
        if stats["success_rate"] < 90.0:
            bottlenecks.append({
                "type": "low_success_rate",
                "severity": "high" if stats["success_rate"] < 75.0 else "medium",
                "description": f"Success rate is {stats['success_rate']:.1f}%",
                "recommendation": "Review error patterns and implement better error handling"
            })

        return bottlenecks

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current performance."""
        recommendations = []

        system_metrics = self.get_current_system_metrics()
        stats = self.get_performance_statistics()

        # CPU optimization
        if system_metrics.cpu_usage > 80.0:
            recommendations.append({
                "category": "cpu_optimization",
                "priority": "high",
                "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations",
                "current_value": system_metrics.cpu_usage,
                "impact": "high"
            })

        # Memory optimization
        if system_metrics.memory_usage_percent > 85.0:
            recommendations.append({
                "category": "memory_optimization",
                "priority": "high",
                "recommendation": "Implement memory cleanup or increase available memory",
                "current_value": system_metrics.memory_usage_percent,
                "impact": "medium"
            })

        # Response time optimization
        if stats["average_response_time"] > 3.0:
            recommendations.append({
                "category": "response_time_optimization",
                "priority": "medium",
                "recommendation": "Optimize network requests and implement request pooling",
                "current_value": stats["average_response_time"],
                "impact": "high"
            })

        return recommendations

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        stats = self.get_performance_statistics()

        if stats["total_crawls"] == 0:
            return 0.0

        # Score components (0-100 each)
        success_score = min(stats["success_rate"], 100.0)

        # Response time score (inverse relationship)
        avg_response = stats["average_response_time"]
        if avg_response <= 1.0:
            response_score = 100.0
        elif avg_response <= 3.0:
            response_score = 100.0 - (avg_response - 1.0) * 25.0  # 25 points per second
        else:
            response_score = max(25.0 - (avg_response - 3.0) * 5.0, 0.0)  # 5 points per second

        # System resource score
        system_metrics = self.get_current_system_metrics()
        cpu_score = max(100.0 - system_metrics.cpu_usage, 0.0)
        memory_score = max(100.0 - system_metrics.memory_usage_percent, 0.0)
        resource_score = (cpu_score + memory_score) / 2

        # Weighted overall score
        overall_score = (
            success_score * 0.4 +
            response_score * 0.4 +
            resource_score * 0.2
        )

        return round(overall_score, 2)

    def get_capacity_planning_recommendations(self) -> List[Dict[str, Any]]:
        """Get capacity planning recommendations."""
        recommendations = []

        system_metrics = self.get_current_system_metrics()
        throughput_stats = self.calculate_throughput_stats()

        # CPU capacity
        if system_metrics.cpu_usage > 70.0:
            recommendations.append({
                "resource": "cpu",
                "current_usage": system_metrics.cpu_usage,
                "recommendation": "Consider scaling up CPU capacity or distributing load",
                "urgency": "high" if system_metrics.cpu_usage > 85.0 else "medium",
                "scaling_factor": 1.5 if system_metrics.cpu_usage > 85.0 else 1.2
            })

        # Memory capacity
        if system_metrics.memory_usage_percent > 70.0:
            recommendations.append({
                "resource": "memory",
                "current_usage": system_metrics.memory_usage_percent,
                "recommendation": "Scale up memory capacity or optimize memory usage",
                "urgency": "high" if system_metrics.memory_usage_percent > 85.0 else "medium",
                "scaling_factor": 1.3 if system_metrics.memory_usage_percent > 85.0 else 1.1
            })

        # Throughput capacity
        if throughput_stats.crawls_per_minute > 500:  # High load threshold
            recommendations.append({
                "resource": "throughput",
                "current_throughput": throughput_stats.crawls_per_minute,
                "recommendation": "Consider horizontal scaling to handle increased load",
                "urgency": "medium",
                "scaling_factor": 2.0
            })

        return recommendations

    async def _system_monitoring_loop(self) -> None:
        """Background task for system monitoring."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_system_metrics()
                self.system_metrics.append(metrics)

                # Check system alerts
                self._check_system_alerts(metrics)

                await asyncio.sleep(self.system_monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(self.system_monitoring_interval)

    async def _alert_checking_loop(self) -> None:
        """Background task for alert checking."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.alert_check_interval)

                # Clean up resolved alerts
                self._cleanup_resolved_alerts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")

    async def _metrics_cleanup_loop(self) -> None:
        """Background task for metrics cleanup."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Clean up old metrics
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)

                with self.metrics_lock:
                    # Remove old crawl metrics
                    self.crawl_metrics = deque([
                        m for m in self.crawl_metrics
                        if m.start_time > cutoff_time
                    ], maxlen=self.crawl_metrics.maxlen)

                # Remove old system metrics
                self.system_metrics = deque([
                    m for m in self.system_metrics
                    if m.timestamp > cutoff_time
                ], maxlen=self.system_metrics.maxlen)

                logger.debug("Cleaned up old metrics")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup loop: {e}")

    def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check system metrics against alert thresholds."""
        # CPU alert
        if metrics.cpu_usage > self.alert_config["cpu_threshold"]:
            alert = PerformanceAlert(
                metric_type=MetricType.CPU_USAGE,
                level=AlertLevel.CRITICAL if metrics.cpu_usage > 90.0 else AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                current_value=metrics.cpu_usage,
                threshold_value=self.alert_config["cpu_threshold"],
                timestamp=metrics.timestamp
            )
            self.active_alerts.append(alert)

        # Memory alert
        if metrics.memory_usage_percent > self.alert_config["memory_threshold"]:
            alert = PerformanceAlert(
                metric_type=MetricType.MEMORY_USAGE,
                level=AlertLevel.CRITICAL if metrics.memory_usage_percent > 95.0 else AlertLevel.WARNING,
                message=f"High memory usage: {metrics.memory_usage_percent:.1f}%",
                current_value=metrics.memory_usage_percent,
                threshold_value=self.alert_config["memory_threshold"],
                timestamp=metrics.timestamp
            )
            self.active_alerts.append(alert)

    def _cleanup_resolved_alerts(self) -> None:
        """Clean up resolved alerts."""
        # Remove alerts older than 1 hour
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff_time
        ]


# Global instance
_performance_monitor_instance: Optional[PerformanceMonitor] = None

async def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor_instance

    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor()
        await _performance_monitor_instance.initialize()

    return _performance_monitor_instance

async def shutdown_performance_monitor() -> None:
    """Shutdown global performance monitor."""
    global _performance_monitor_instance

    if _performance_monitor_instance:
        await _performance_monitor_instance.shutdown()
        _performance_monitor_instance = None