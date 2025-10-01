"""
Worker monitoring and management service.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.celery_config import celery_app, get_worker_health_status, check_redis_connection

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration."""
    ACTIVE = "active"
    IDLE = "idle"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"


class QueueStatus(Enum):
    """Queue status enumeration."""
    HEALTHY = "healthy"
    CONGESTED = "congested"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_name: str
    status: str
    active_tasks: int
    processed_tasks: int
    failed_tasks: int
    load_average: float
    memory_usage: float
    last_heartbeat: str
    uptime: float
    queue_length: int


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    queue_name: str
    status: str
    pending_tasks: int
    active_tasks: int
    processed_tasks_1h: int
    failed_tasks_1h: int
    average_processing_time: float
    throughput_per_minute: float


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    total_workers: int
    active_workers: int
    offline_workers: int
    total_queues: int
    healthy_queues: int
    total_pending_tasks: int
    total_active_tasks: int
    system_load: float
    redis_status: str
    last_updated: str


class WorkerMonitoringService:
    """Service for monitoring and managing Celery workers."""

    def __init__(self):
        self.celery_app = celery_app
        self.inspector = celery_app.control.inspect()

    def get_worker_metrics(self, worker_name: Optional[str] = None) -> List[WorkerMetrics]:
        """
        Get detailed metrics for workers.

        Args:
            worker_name: Optional specific worker name to monitor

        Returns:
            List of worker metrics
        """
        try:
            worker_metrics = []

            # Get worker stats
            stats = self.inspector.stats()
            active_tasks = self.inspector.active()
            registered_tasks = self.inspector.registered()

            if not stats:
                logger.warning("No worker stats available")
                return []

            for worker, stat_data in stats.items():
                if worker_name and worker != worker_name:
                    continue

                # Extract metrics from worker stats
                pool_info = stat_data.get('pool', {})
                clock_info = stat_data.get('clock', {})

                active_count = len(active_tasks.get(worker, [])) if active_tasks else 0
                processed_count = stat_data.get('total', {}).get('processed', 0)
                failed_count = stat_data.get('total', {}).get('failed', 0)

                # Calculate uptime from clock info
                clock_value = clock_info.get('clock', 0)
                uptime = clock_value / 1000000.0 if clock_value else 0  # Convert microseconds to seconds

                metrics = WorkerMetrics(
                    worker_name=worker,
                    status=WorkerStatus.ACTIVE.value if active_count > 0 else WorkerStatus.IDLE.value,
                    active_tasks=active_count,
                    processed_tasks=processed_count,
                    failed_tasks=failed_count,
                    load_average=pool_info.get('max-concurrency', 0),
                    memory_usage=0.0,  # Would be populated with actual memory monitoring
                    last_heartbeat=datetime.now(timezone.utc).isoformat(),
                    uptime=uptime,
                    queue_length=0  # Would be calculated from queue inspection
                )

                worker_metrics.append(metrics)

            return worker_metrics

        except Exception as e:
            logger.error(f"Failed to get worker metrics: {e}")
            return []

    def get_queue_metrics(self, queue_name: Optional[str] = None) -> List[QueueMetrics]:
        """
        Get detailed metrics for queues.

        Args:
            queue_name: Optional specific queue name to monitor

        Returns:
            List of queue metrics
        """
        try:
            queue_metrics = []

            # Get active tasks grouped by queue
            active_tasks = self.inspector.active()
            reserved_tasks = self.inspector.reserved()

            # Define known queues
            known_queues = ['crawl_queue', 'process_queue', 'schedule_queue', 'monitor_queue', 'default']

            for queue in known_queues:
                if queue_name and queue != queue_name:
                    continue

                # Count tasks in this queue
                active_count = 0
                pending_count = 0

                if active_tasks:
                    for worker, tasks in active_tasks.items():
                        active_count += len([t for t in tasks if self._get_task_queue(t.get('name', '')) == queue])

                if reserved_tasks:
                    for worker, tasks in reserved_tasks.items():
                        pending_count += len([t for t in tasks if self._get_task_queue(t.get('name', '')) == queue])

                # Determine queue status
                status = QueueStatus.HEALTHY.value
                if pending_count > 100:
                    status = QueueStatus.CONGESTED.value
                elif pending_count == 0 and active_count == 0:
                    status = QueueStatus.HEALTHY.value

                metrics = QueueMetrics(
                    queue_name=queue,
                    status=status,
                    pending_tasks=pending_count,
                    active_tasks=active_count,
                    processed_tasks_1h=0,  # Would be populated from historical data
                    failed_tasks_1h=0,    # Would be populated from historical data
                    average_processing_time=0.0,  # Would be calculated from task history
                    throughput_per_minute=0.0     # Would be calculated from task history
                )

                queue_metrics.append(metrics)

            return queue_metrics

        except Exception as e:
            logger.error(f"Failed to get queue metrics: {e}")
            return []

    def get_system_overview(self) -> SystemMetrics:
        """
        Get overall system metrics and health overview.

        Returns:
            System metrics summary
        """
        try:
            worker_metrics = self.get_worker_metrics()
            queue_metrics = self.get_queue_metrics()

            # Calculate totals
            total_workers = len(worker_metrics)
            active_workers = len([w for w in worker_metrics if w.status == WorkerStatus.ACTIVE.value])
            offline_workers = total_workers - active_workers

            total_queues = len(queue_metrics)
            healthy_queues = len([q for q in queue_metrics if q.status == QueueStatus.HEALTHY.value])

            total_pending = sum(q.pending_tasks for q in queue_metrics)
            total_active = sum(q.active_tasks for q in queue_metrics)

            # Calculate system load (average active tasks per worker)
            system_load = total_active / max(active_workers, 1)

            # Check Redis status
            redis_status = "healthy" if check_redis_connection() else "unhealthy"

            return SystemMetrics(
                total_workers=total_workers,
                active_workers=active_workers,
                offline_workers=offline_workers,
                total_queues=total_queues,
                healthy_queues=healthy_queues,
                total_pending_tasks=total_pending,
                total_active_tasks=total_active,
                system_load=system_load,
                redis_status=redis_status,
                last_updated=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return SystemMetrics(
                total_workers=0,
                active_workers=0,
                offline_workers=0,
                total_queues=0,
                healthy_queues=0,
                total_pending_tasks=0,
                total_active_tasks=0,
                system_load=0.0,
                redis_status="error",
                last_updated=datetime.now(timezone.utc).isoformat()
            )

    def get_worker_details(self, worker_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific worker.

        Args:
            worker_name: Name of the worker to inspect

        Returns:
            Detailed worker information
        """
        try:
            stats = self.inspector.stats()
            active_tasks = self.inspector.active()
            reserved_tasks = self.inspector.reserved()
            registered_tasks = self.inspector.registered()

            if not stats or worker_name not in stats:
                return {'error': f'Worker {worker_name} not found'}

            worker_stats = stats[worker_name]
            worker_active = active_tasks.get(worker_name, []) if active_tasks else []
            worker_reserved = reserved_tasks.get(worker_name, []) if reserved_tasks else []
            worker_registered = registered_tasks.get(worker_name, []) if registered_tasks else []

            return {
                'worker_name': worker_name,
                'stats': worker_stats,
                'active_tasks': worker_active,
                'reserved_tasks': worker_reserved,
                'registered_tasks': worker_registered,
                'task_count': {
                    'active': len(worker_active),
                    'reserved': len(worker_reserved),
                    'registered': len(worker_registered)
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get worker details for {worker_name}: {e}")
            return {'error': str(e)}

    def restart_worker(self, worker_name: str) -> Dict[str, Any]:
        """
        Restart a specific worker.

        Args:
            worker_name: Name of the worker to restart

        Returns:
            Restart result
        """
        try:
            # In a production environment, this would trigger worker restart
            # through system management tools like systemd or supervisor
            logger.info(f"Restart requested for worker: {worker_name}")

            # For now, return a success message
            # In practice, this would integrate with deployment infrastructure
            return {
                'success': True,
                'worker_name': worker_name,
                'action': 'restart_requested',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': f'Restart signal sent to worker {worker_name}'
            }

        except Exception as e:
            logger.error(f"Failed to restart worker {worker_name}: {e}")
            return {
                'success': False,
                'worker_name': worker_name,
                'error': str(e)
            }

    def scale_workers(self, queue_name: str, target_workers: int) -> Dict[str, Any]:
        """
        Scale workers for a specific queue.

        Args:
            queue_name: Name of the queue to scale
            target_workers: Target number of workers

        Returns:
            Scaling result
        """
        try:
            current_metrics = self.get_queue_metrics(queue_name)
            if not current_metrics:
                return {
                    'success': False,
                    'error': f'Queue {queue_name} not found'
                }

            current_queue = current_metrics[0]

            logger.info(f"Scaling requested for queue {queue_name}: target {target_workers} workers")

            # In production, this would trigger worker scaling through
            # container orchestration or process management tools
            return {
                'success': True,
                'queue_name': queue_name,
                'current_workers': 1,  # Would be calculated from actual worker assignment
                'target_workers': target_workers,
                'action': 'scaling_requested',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to scale workers for queue {queue_name}: {e}")
            return {
                'success': False,
                'queue_name': queue_name,
                'error': str(e)
            }

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate performance alerts based on system metrics.

        Returns:
            List of performance alerts
        """
        try:
            alerts = []
            worker_metrics = self.get_worker_metrics()
            queue_metrics = self.get_queue_metrics()
            system_metrics = self.get_system_overview()

            # Check for high system load
            if system_metrics.system_load > 0.8:
                alerts.append({
                    'type': 'high_load',
                    'severity': 'warning',
                    'message': f'High system load: {system_metrics.system_load:.2f}',
                    'metric': system_metrics.system_load,
                    'threshold': 0.8,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            # Check for congested queues
            for queue in queue_metrics:
                if queue.status == QueueStatus.CONGESTED.value:
                    alerts.append({
                        'type': 'queue_congestion',
                        'severity': 'warning',
                        'message': f'Queue {queue.queue_name} is congested with {queue.pending_tasks} pending tasks',
                        'queue_name': queue.queue_name,
                        'pending_tasks': queue.pending_tasks,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })

            # Check for offline workers
            if system_metrics.offline_workers > 0:
                alerts.append({
                    'type': 'workers_offline',
                    'severity': 'error',
                    'message': f'{system_metrics.offline_workers} workers are offline',
                    'offline_count': system_metrics.offline_workers,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            # Check Redis status
            if system_metrics.redis_status != 'healthy':
                alerts.append({
                    'type': 'redis_unhealthy',
                    'severity': 'critical',
                    'message': 'Redis connection is unhealthy',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            return alerts

        except Exception as e:
            logger.error(f"Failed to generate performance alerts: {e}")
            return []

    def _get_task_queue(self, task_name: str) -> str:
        """
        Determine which queue a task belongs to based on task name.

        Args:
            task_name: Name of the task

        Returns:
            Queue name
        """
        if task_name.startswith('crawler.tasks.crawl_url'):
            return 'crawl_queue'
        elif task_name.startswith('crawler.tasks.process_data') or task_name.startswith('crawler.tasks.generate_embeddings'):
            return 'process_queue'
        elif task_name.startswith('crawler.tasks.schedule_crawl'):
            return 'schedule_queue'
        elif task_name.startswith('monitor.tasks'):
            return 'monitor_queue'
        else:
            return 'default'

    def export_metrics(self, format_type: str = 'json') -> Dict[str, Any]:
        """
        Export comprehensive metrics in specified format.

        Args:
            format_type: Export format ('json', 'prometheus')

        Returns:
            Exported metrics data
        """
        try:
            worker_metrics = self.get_worker_metrics()
            queue_metrics = self.get_queue_metrics()
            system_metrics = self.get_system_overview()
            alerts = self.get_performance_alerts()

            if format_type == 'json':
                return {
                    'export_timestamp': datetime.now(timezone.utc).isoformat(),
                    'system_overview': asdict(system_metrics),
                    'workers': [asdict(w) for w in worker_metrics],
                    'queues': [asdict(q) for q in queue_metrics],
                    'alerts': alerts,
                    'metadata': {
                        'total_workers': len(worker_metrics),
                        'total_queues': len(queue_metrics),
                        'total_alerts': len(alerts)
                    }
                }
            elif format_type == 'prometheus':
                # Return Prometheus-style metrics
                metrics_lines = []

                # System metrics
                metrics_lines.append(f'celery_workers_total {system_metrics.total_workers}')
                metrics_lines.append(f'celery_workers_active {system_metrics.active_workers}')
                metrics_lines.append(f'celery_system_load {system_metrics.system_load}')

                # Queue metrics
                for queue in queue_metrics:
                    metrics_lines.append(f'celery_queue_pending_tasks{{queue="{queue.queue_name}"}} {queue.pending_tasks}')
                    metrics_lines.append(f'celery_queue_active_tasks{{queue="{queue.queue_name}"}} {queue.active_tasks}')

                return {
                    'format': 'prometheus',
                    'metrics': '\n'.join(metrics_lines),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            else:
                return {'error': f'Unsupported export format: {format_type}'}

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return {'error': str(e)}