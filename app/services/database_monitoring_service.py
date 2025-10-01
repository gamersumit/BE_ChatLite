"""
Database monitoring and performance tuning service.
Provides comprehensive monitoring, alerting, and automated performance tuning.
"""
import asyncio
import time
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from statistics import mean, median

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, event
from sqlalchemy.engine import Engine


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being monitored."""
    QUERY_PERFORMANCE = "query_performance"
    CONNECTION_POOL = "connection_pool"
    SYSTEM_RESOURCE = "system_resource"
    DATABASE_HEALTH = "database_health"
    CACHE_PERFORMANCE = "cache_performance"


@dataclass
class Alert:
    """Database performance alert."""
    id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class QueryMetrics:
    """Comprehensive query performance metrics."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    rows_affected: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    last_executed: Optional[datetime] = None
    recent_execution_times: List[float] = field(default_factory=list)

    def add_execution(self, execution_time: float, rows: int = 0, cached: bool = False, error: bool = False):
        """Add execution data to metrics."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.avg_execution_time = self.total_execution_time / self.execution_count
        self.rows_affected += rows
        self.last_executed = datetime.now(timezone.utc)

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.error_count += 1

        # Keep recent execution times for trend analysis
        self.recent_execution_times.append(execution_time)
        if len(self.recent_execution_times) > 100:
            self.recent_execution_times = self.recent_execution_times[-100:]


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    disk_usage_percent: float


class PerformanceThresholds:
    """Configurable performance thresholds."""

    def __init__(self):
        self.query_thresholds = {
            "slow_query_time": 1.0,  # seconds
            "very_slow_query_time": 5.0,  # seconds
            "max_query_error_rate": 0.05,  # 5%
            "max_avg_execution_time": 0.5  # seconds
        }

        self.system_thresholds = {
            "max_cpu_percent": 80.0,
            "max_memory_percent": 85.0,
            "min_memory_available_mb": 1024.0,
            "max_disk_usage_percent": 90.0,
            "max_disk_io_mb_per_sec": 100.0
        }

        self.connection_thresholds = {
            "max_pool_utilization": 0.8,  # 80%
            "max_checkout_time": 1.0,  # seconds
            "max_connection_errors": 10  # per hour
        }

        self.cache_thresholds = {
            "min_hit_rate": 0.7,  # 70%
            "max_memory_usage": 0.9  # 90%
        }


class DatabaseMonitor:
    """Comprehensive database monitoring service."""

    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        self.thresholds = thresholds or PerformanceThresholds()
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.alerts: List[Alert] = []
        self.monitoring_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[Callable] = []

        # Performance history
        self.performance_history: Dict[str, List[float]] = {}
        self.baseline_metrics: Dict[str, float] = {}

    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(
                self._monitoring_loop(interval_seconds)
            )

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()

    async def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._collect_system_metrics()
                await self._analyze_performance_trends()
                await self._check_alert_conditions()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval_seconds)

    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')

            # Network I/O
            network_io = psutil.net_io_counters()

            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_io_read_mb=disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                network_bytes_sent=network_io.bytes_sent if network_io else 0,
                network_bytes_recv=network_io.bytes_recv if network_io else 0,
                active_connections=0,  # Would be populated by connection pool
                disk_usage_percent=disk_usage.percent
            )

            self.system_metrics.append(metrics)

            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self.system_metrics = [
                m for m in self.system_metrics
                if m.timestamp > cutoff_time
            ]

        except Exception as e:
            logging.error(f"System metrics collection error: {e}")

    async def _analyze_performance_trends(self):
        """Analyze performance trends and establish baselines."""
        # Analyze query performance trends
        for query_hash, metrics in self.query_metrics.items():
            if len(metrics.recent_execution_times) >= 10:
                recent_avg = mean(metrics.recent_execution_times[-10:])

                # Store in performance history
                if query_hash not in self.performance_history:
                    self.performance_history[query_hash] = []

                self.performance_history[query_hash].append(recent_avg)

                # Keep only recent history
                if len(self.performance_history[query_hash]) > 100:
                    self.performance_history[query_hash] = self.performance_history[query_hash][-100:]

                # Update baseline if we have enough data
                if len(self.performance_history[query_hash]) >= 20:
                    self.baseline_metrics[f"query_{query_hash}_avg"] = median(
                        self.performance_history[query_hash][-20:]
                    )

        # Analyze system metrics trends
        if len(self.system_metrics) >= 10:
            recent_cpu = [m.cpu_percent for m in self.system_metrics[-10:]]
            recent_memory = [m.memory_percent for m in self.system_metrics[-10:]]

            self.baseline_metrics["cpu_avg"] = mean(recent_cpu)
            self.baseline_metrics["memory_avg"] = mean(recent_memory)

    async def _check_alert_conditions(self):
        """Check for alert conditions and generate alerts."""
        current_time = datetime.now(timezone.utc)

        # Check query performance alerts
        for query_hash, metrics in self.query_metrics.items():
            if metrics.execution_count > 0:
                # Slow query alert
                if metrics.avg_execution_time > self.thresholds.query_thresholds["slow_query_time"]:
                    await self._create_alert(
                        AlertSeverity.WARNING,
                        MetricType.QUERY_PERFORMANCE,
                        f"Slow query detected: {metrics.avg_execution_time:.3f}s average",
                        metrics.avg_execution_time,
                        self.thresholds.query_thresholds["slow_query_time"]
                    )

                # High error rate alert
                error_rate = metrics.error_count / metrics.execution_count
                if error_rate > self.thresholds.query_thresholds["max_query_error_rate"]:
                    await self._create_alert(
                        AlertSeverity.CRITICAL,
                        MetricType.QUERY_PERFORMANCE,
                        f"High query error rate: {error_rate:.2%}",
                        error_rate,
                        self.thresholds.query_thresholds["max_query_error_rate"]
                    )

        # Check system resource alerts
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]

            # CPU usage alert
            if latest_metrics.cpu_percent > self.thresholds.system_thresholds["max_cpu_percent"]:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.SYSTEM_RESOURCE,
                    f"High CPU usage: {latest_metrics.cpu_percent:.1f}%",
                    latest_metrics.cpu_percent,
                    self.thresholds.system_thresholds["max_cpu_percent"]
                )

            # Memory usage alert
            if latest_metrics.memory_percent > self.thresholds.system_thresholds["max_memory_percent"]:
                await self._create_alert(
                    AlertSeverity.CRITICAL,
                    MetricType.SYSTEM_RESOURCE,
                    f"High memory usage: {latest_metrics.memory_percent:.1f}%",
                    latest_metrics.memory_percent,
                    self.thresholds.system_thresholds["max_memory_percent"]
                )

            # Disk usage alert
            if latest_metrics.disk_usage_percent > self.thresholds.system_thresholds["max_disk_usage_percent"]:
                await self._create_alert(
                    AlertSeverity.EMERGENCY,
                    MetricType.SYSTEM_RESOURCE,
                    f"Critical disk usage: {latest_metrics.disk_usage_percent:.1f}%",
                    latest_metrics.disk_usage_percent,
                    self.thresholds.system_thresholds["max_disk_usage_percent"]
                )

    async def _create_alert(self, severity: AlertSeverity, metric_type: MetricType, message: str, value: float, threshold: float):
        """Create and process a new alert."""
        alert = Alert(
            id=f"{metric_type.value}_{int(time.time())}",
            severity=severity,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold
        )

        self.alerts.append(alert)

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logging.error(f"Alert callback error: {e}")

        # Log alert
        logging.log(
            logging.CRITICAL if severity == AlertSeverity.EMERGENCY else
            logging.ERROR if severity == AlertSeverity.CRITICAL else
            logging.WARNING if severity == AlertSeverity.WARNING else
            logging.INFO,
            f"Database Alert [{severity.value.upper()}]: {message}"
        )

    def add_alert_callback(self, callback: Callable):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def record_query_execution(self, query: str, execution_time: float, rows: int = 0, cached: bool = False, error: bool = False):
        """Record query execution metrics."""
        query_hash = self._hash_query(query)

        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_text=query[:200] + "..." if len(query) > 200 else query
            )

        self.query_metrics[query_hash].add_execution(execution_time, rows, cached, error)

    def _hash_query(self, query: str) -> str:
        """Generate hash for query normalization."""
        # Normalize query by removing parameters and extra whitespace
        normalized = " ".join(query.split())
        import hashlib
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        current_time = datetime.now(timezone.utc)

        # Query performance summary
        query_summary = {
            "total_queries": len(self.query_metrics),
            "total_executions": sum(m.execution_count for m in self.query_metrics.values()),
            "avg_execution_time": mean([m.avg_execution_time for m in self.query_metrics.values()]) if self.query_metrics else 0,
            "slow_queries": len([m for m in self.query_metrics.values()
                               if m.avg_execution_time > self.thresholds.query_thresholds["slow_query_time"]]),
            "error_rate": sum(m.error_count for m in self.query_metrics.values()) /
                         max(sum(m.execution_count for m in self.query_metrics.values()), 1)
        }

        # System metrics summary
        system_summary = {}
        if self.system_metrics:
            latest = self.system_metrics[-1]
            system_summary = {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_available_mb": latest.memory_available_mb,
                "disk_usage_percent": latest.disk_usage_percent,
                "active_connections": latest.active_connections
            }

        # Alert summary
        recent_alerts = [a for a in self.alerts if (current_time - a.timestamp).total_seconds() < 3600]  # Last hour
        alert_summary = {
            "total_alerts": len(self.alerts),
            "recent_alerts": len(recent_alerts),
            "unresolved_alerts": len([a for a in self.alerts if not a.resolved]),
            "critical_alerts": len([a for a in recent_alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]])
        }

        return {
            "timestamp": current_time,
            "monitoring_status": "active" if self.monitoring_enabled else "inactive",
            "query_performance": query_summary,
            "system_metrics": system_summary,
            "alerts": alert_summary,
            "performance_baselines": self.baseline_metrics
        }

    async def get_top_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top slow queries by average execution time."""
        sorted_queries = sorted(
            self.query_metrics.values(),
            key=lambda q: q.avg_execution_time,
            reverse=True
        )

        return [
            {
                "query_hash": q.query_hash,
                "query_text": q.query_text,
                "execution_count": q.execution_count,
                "avg_execution_time": q.avg_execution_time,
                "max_execution_time": q.max_execution_time,
                "error_count": q.error_count,
                "error_rate": q.error_count / q.execution_count if q.execution_count > 0 else 0,
                "last_executed": q.last_executed
            }
            for q in sorted_queries[:limit]
        ]

    async def get_recent_alerts(self, hours: int = 24, severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Get recent alerts with optional severity filter."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered_alerts = [
            a for a in self.alerts
            if a.timestamp > cutoff_time and (severity is None or a.severity == severity)
        ]

        return [
            {
                "id": a.id,
                "severity": a.severity.value,
                "metric_type": a.metric_type.value,
                "message": a.message,
                "value": a.value,
                "threshold": a.threshold,
                "timestamp": a.timestamp,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at
            }
            for a in sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
        ]

    async def get_system_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered_metrics = [
            m for m in self.system_metrics
            if m.timestamp > cutoff_time
        ]

        return [
            {
                "timestamp": m.timestamp,
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "memory_available_mb": m.memory_available_mb,
                "disk_usage_percent": m.disk_usage_percent,
                "network_bytes_sent": m.network_bytes_sent,
                "network_bytes_recv": m.network_bytes_recv
            }
            for m in filtered_metrics
        ]

    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                return True
        return False

    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        """Update performance thresholds."""
        if "query" in new_thresholds:
            self.thresholds.query_thresholds.update(new_thresholds["query"])

        if "system" in new_thresholds:
            self.thresholds.system_thresholds.update(new_thresholds["system"])

        if "connection" in new_thresholds:
            self.thresholds.connection_thresholds.update(new_thresholds["connection"])

        if "cache" in new_thresholds:
            self.thresholds.cache_thresholds.update(new_thresholds["cache"])


class AutoTuner:
    """Automatic database performance tuning service."""

    def __init__(self, monitor: DatabaseMonitor):
        self.monitor = monitor
        self.tuning_rules: List[Callable] = []
        self.tuning_history: List[Dict[str, Any]] = []

    def add_tuning_rule(self, rule: Callable):
        """Add automatic tuning rule."""
        self.tuning_rules.append(rule)

    async def analyze_and_tune(self, session: AsyncSession) -> List[Dict[str, Any]]:
        """Analyze performance and apply automatic tuning."""
        tuning_actions = []

        for rule in self.tuning_rules:
            try:
                action = await rule(self.monitor, session)
                if action:
                    tuning_actions.append(action)
                    self.tuning_history.append({
                        "timestamp": datetime.now(timezone.utc),
                        "action": action,
                        "rule": rule.__name__
                    })
            except Exception as e:
                logging.error(f"Auto-tuning rule error: {e}")

        return tuning_actions

    async def get_tuning_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance tuning recommendations."""
        recommendations = []

        # Analyze slow queries
        slow_queries = await self.monitor.get_top_slow_queries(5)
        for query in slow_queries:
            if query["avg_execution_time"] > 1.0:
                recommendations.append({
                    "type": "query_optimization",
                    "priority": "high",
                    "description": f"Query taking {query['avg_execution_time']:.3f}s on average",
                    "suggestion": "Consider adding indexes or optimizing WHERE clauses",
                    "query_hash": query["query_hash"]
                })

        # Analyze system resources
        dashboard = await self.monitor.get_monitoring_dashboard()
        system_metrics = dashboard.get("system_metrics", {})

        if system_metrics.get("memory_percent", 0) > 80:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "medium",
                "description": f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                "suggestion": "Consider increasing memory or optimizing query cache size"
            })

        if system_metrics.get("cpu_percent", 0) > 70:
            recommendations.append({
                "type": "cpu_optimization",
                "priority": "medium",
                "description": f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                "suggestion": "Consider optimizing queries or scaling database resources"
            })

        return recommendations


# Global database monitor instance
database_monitor = DatabaseMonitor()
auto_tuner = AutoTuner(database_monitor)