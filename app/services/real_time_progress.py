"""
Real-time Progress Updates and Notification System

This module provides real-time progress updates using WebSockets, Server-Sent Events,
and push notifications for crawling operations and background jobs.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import weakref
from contextlib import asynccontextmanager

from app.services.progress_tracking_service import (
    ProgressEvent,
    ProgressAlert,
    JobProgressSnapshot,
    ProgressEventType,
    AlertLevel
)

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    PROGRESS_UPDATE = "progress_update"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    ALERT_CREATED = "alert_created"
    MILESTONE_REACHED = "milestone_reached"
    SYSTEM_STATUS = "system_status"


class DeliveryMethod(Enum):
    """Notification delivery methods."""
    WEBSOCKET = "websocket"
    SERVER_SENT_EVENTS = "server_sent_events"
    WEBHOOK = "webhook"
    EMAIL = "email"
    PUSH_NOTIFICATION = "push_notification"


@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: str
    job_completed: Set[DeliveryMethod] = None
    job_failed: Set[DeliveryMethod] = None
    progress_milestones: Set[DeliveryMethod] = None
    alerts: Set[DeliveryMethod] = None
    system_status: Set[DeliveryMethod] = None

    def __post_init__(self):
        if self.job_completed is None:
            self.job_completed = {DeliveryMethod.WEBSOCKET}
        if self.job_failed is None:
            self.job_failed = {DeliveryMethod.WEBSOCKET, DeliveryMethod.EMAIL}
        if self.progress_milestones is None:
            self.progress_milestones = {DeliveryMethod.WEBSOCKET}
        if self.alerts is None:
            self.alerts = {DeliveryMethod.WEBSOCKET}
        if self.system_status is None:
            self.system_status = {DeliveryMethod.WEBSOCKET}


@dataclass
class RealTimeNotification:
    """Real-time notification message."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    delivery_methods: Set[DeliveryMethod]
    priority: int = 5  # 1-10 scale, 10 being highest

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "notification_type": self.notification_type.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "delivery_methods": [method.value for method in self.delivery_methods],
            "priority": self.priority
        }


class WebSocketConnection:
    """WebSocket connection wrapper."""

    def __init__(self, user_id: str, websocket):
        self.user_id = user_id
        self.websocket = websocket
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = self.connected_at
        self.subscriptions: Set[str] = set()  # Job IDs user is subscribed to

    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        try:
            await self.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message to user {self.user_id}: {e}")
            return False

    async def ping(self):
        """Send ping to check connection health."""
        try:
            await self.websocket.ping()
            self.last_ping = datetime.now(timezone.utc)
            return True
        except Exception:
            return False

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale."""
        return (datetime.now(timezone.utc) - self.last_ping).total_seconds() > timeout_seconds


class RealTimeProgressService:
    """Real-time progress update service."""

    def __init__(self):
        self.websocket_connections: Dict[str, List[WebSocketConnection]] = {}
        self.sse_connections: Dict[str, List[Any]] = {}  # Server-Sent Events connections
        self.webhook_endpoints: Dict[str, str] = {}
        self.notification_preferences: Dict[str, NotificationPreferences] = {}

        self.notification_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.notification_history: List[RealTimeNotification] = []

        self._delivery_tasks: List[asyncio.Task] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}  # user_id -> timestamps
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 100  # messages per window

    async def start_service(self):
        """Start the real-time service."""
        # Start delivery workers
        for i in range(3):  # 3 concurrent delivery workers
            task = asyncio.create_task(self._delivery_worker(f"worker_{i}"))
            self._delivery_tasks.append(task)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_connections())

        logger.info("Real-time progress service started")

    async def stop_service(self):
        """Stop the real-time service."""
        self._shutdown_event.set()

        # Cancel delivery workers
        for task in self._delivery_tasks:
            task.cancel()

        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._delivery_tasks, self._cleanup_task, return_exceptions=True)

        # Close all WebSocket connections
        for connections in self.websocket_connections.values():
            for conn in connections:
                try:
                    await conn.websocket.close()
                except Exception:
                    pass

        logger.info("Real-time progress service stopped")

    async def register_websocket(self, user_id: str, websocket) -> WebSocketConnection:
        """Register a new WebSocket connection."""
        connection = WebSocketConnection(user_id, websocket)

        if user_id not in self.websocket_connections:
            self.websocket_connections[user_id] = []

        self.websocket_connections[user_id].append(connection)

        logger.info(f"Registered WebSocket connection for user: {user_id}")
        return connection

    async def unregister_websocket(self, user_id: str, connection: WebSocketConnection):
        """Unregister a WebSocket connection."""
        if user_id in self.websocket_connections:
            try:
                self.websocket_connections[user_id].remove(connection)
                if not self.websocket_connections[user_id]:
                    del self.websocket_connections[user_id]
            except ValueError:
                pass

        logger.info(f"Unregistered WebSocket connection for user: {user_id}")

    def subscribe_to_job(self, user_id: str, job_id: str):
        """Subscribe user to job progress updates."""
        if user_id in self.websocket_connections:
            for connection in self.websocket_connections[user_id]:
                connection.subscriptions.add(job_id)

        logger.info(f"User {user_id} subscribed to job: {job_id}")

    def unsubscribe_from_job(self, user_id: str, job_id: str):
        """Unsubscribe user from job progress updates."""
        if user_id in self.websocket_connections:
            for connection in self.websocket_connections[user_id]:
                connection.subscriptions.discard(job_id)

        logger.info(f"User {user_id} unsubscribed from job: {job_id}")

    def set_notification_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Set user notification preferences."""
        self.notification_preferences[user_id] = preferences

    async def send_progress_update(self, job_id: str, snapshot: JobProgressSnapshot):
        """Send progress update notification."""
        notification = RealTimeNotification(
            notification_id=f"progress_{job_id}_{int(time.time())}",
            user_id="",  # Will be determined per recipient
            notification_type=NotificationType.PROGRESS_UPDATE,
            title=f"Progress Update - Job {job_id}",
            message=f"Job progress: {snapshot.metrics.get_completion_percentage():.1f}% complete",
            data={
                "job_id": job_id,
                "snapshot": snapshot.to_dict(),
                "completion_percentage": snapshot.metrics.get_completion_percentage(),
                "current_phase": snapshot.current_phase.value
            },
            timestamp=datetime.now(timezone.utc),
            delivery_methods={DeliveryMethod.WEBSOCKET},
            priority=3
        )

        await self._queue_notification_for_job_subscribers(job_id, notification)

    async def send_job_completion(self, job_id: str, snapshot: JobProgressSnapshot, success: bool):
        """Send job completion notification."""
        notification_type = NotificationType.JOB_COMPLETED if success else NotificationType.JOB_FAILED
        title = f"Job {'Completed' if success else 'Failed'} - {job_id}"
        message = f"Crawling job {job_id} has {'completed successfully' if success else 'failed'}"

        notification = RealTimeNotification(
            notification_id=f"completion_{job_id}_{int(time.time())}",
            user_id="",  # Will be determined per recipient
            notification_type=notification_type,
            title=title,
            message=message,
            data={
                "job_id": job_id,
                "snapshot": snapshot.to_dict(),
                "success": success,
                "total_pages": snapshot.metrics.pages_crawled,
                "total_duration": (snapshot.estimated_completion - snapshot.started_at).total_seconds() if snapshot.estimated_completion and snapshot.started_at else 0
            },
            timestamp=datetime.now(timezone.utc),
            delivery_methods={DeliveryMethod.WEBSOCKET, DeliveryMethod.EMAIL} if not success else {DeliveryMethod.WEBSOCKET},
            priority=8 if not success else 6
        )

        await self._queue_notification_for_job_subscribers(job_id, notification)

    async def send_alert(self, alert: ProgressAlert):
        """Send alert notification."""
        notification = RealTimeNotification(
            notification_id=f"alert_{alert.alert_id}",
            user_id="",  # Will be determined per recipient
            notification_type=NotificationType.ALERT_CREATED,
            title=f"Alert - {alert.level.value.title()}",
            message=alert.message,
            data={
                "alert": alert.to_dict(),
                "job_id": alert.job_id
            },
            timestamp=datetime.now(timezone.utc),
            delivery_methods={DeliveryMethod.WEBSOCKET} if alert.level == AlertLevel.INFO else {DeliveryMethod.WEBSOCKET, DeliveryMethod.EMAIL},
            priority=self._get_alert_priority(alert.level)
        )

        await self._queue_notification_for_job_subscribers(alert.job_id, notification)

    async def send_milestone(self, job_id: str, milestone: str, data: Dict[str, Any]):
        """Send milestone notification."""
        notification = RealTimeNotification(
            notification_id=f"milestone_{job_id}_{int(time.time())}",
            user_id="",  # Will be determined per recipient
            notification_type=NotificationType.MILESTONE_REACHED,
            title=f"Milestone Reached - {milestone}",
            message=f"Job {job_id} reached milestone: {milestone}",
            data={
                "job_id": job_id,
                "milestone": milestone,
                "milestone_data": data
            },
            timestamp=datetime.now(timezone.utc),
            delivery_methods={DeliveryMethod.WEBSOCKET},
            priority=4
        )

        await self._queue_notification_for_job_subscribers(job_id, notification)

    async def send_system_status(self, status_data: Dict[str, Any]):
        """Send system status update."""
        notification = RealTimeNotification(
            notification_id=f"system_status_{int(time.time())}",
            user_id="*",  # Broadcast to all users
            notification_type=NotificationType.SYSTEM_STATUS,
            title="System Status Update",
            message="System status information updated",
            data=status_data,
            timestamp=datetime.now(timezone.utc),
            delivery_methods={DeliveryMethod.WEBSOCKET},
            priority=2
        )

        await self.notification_queue.put(notification)

    def _get_alert_priority(self, level: AlertLevel) -> int:
        """Get priority based on alert level."""
        priority_map = {
            AlertLevel.INFO: 3,
            AlertLevel.WARNING: 6,
            AlertLevel.ERROR: 8,
            AlertLevel.CRITICAL: 10
        }
        return priority_map.get(level, 5)

    async def _queue_notification_for_job_subscribers(self, job_id: str, notification: RealTimeNotification):
        """Queue notification for all subscribers of a job."""
        subscribers = set()

        # Find all users subscribed to this job
        for user_id, connections in self.websocket_connections.items():
            for connection in connections:
                if job_id in connection.subscriptions:
                    subscribers.add(user_id)

        # Queue notification for each subscriber
        for user_id in subscribers:
            user_notification = RealTimeNotification(
                notification_id=notification.notification_id,
                user_id=user_id,
                notification_type=notification.notification_type,
                title=notification.title,
                message=notification.message,
                data=notification.data,
                timestamp=notification.timestamp,
                delivery_methods=self._get_user_delivery_methods(user_id, notification.notification_type),
                priority=notification.priority
            )

            if not self._is_rate_limited(user_id):
                await self.notification_queue.put(user_notification)

    def _get_user_delivery_methods(self, user_id: str, notification_type: NotificationType) -> Set[DeliveryMethod]:
        """Get delivery methods for user based on preferences."""
        preferences = self.notification_preferences.get(user_id)
        if not preferences:
            return {DeliveryMethod.WEBSOCKET}

        method_map = {
            NotificationType.PROGRESS_UPDATE: preferences.progress_milestones,
            NotificationType.JOB_COMPLETED: preferences.job_completed,
            NotificationType.JOB_FAILED: preferences.job_failed,
            NotificationType.ALERT_CREATED: preferences.alerts,
            NotificationType.MILESTONE_REACHED: preferences.progress_milestones,
            NotificationType.SYSTEM_STATUS: preferences.system_status
        }

        return method_map.get(notification_type, {DeliveryMethod.WEBSOCKET})

    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited."""
        current_time = time.time()

        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Remove old timestamps
        self.rate_limits[user_id] = [
            ts for ts in self.rate_limits[user_id]
            if current_time - ts < self.rate_limit_window
        ]

        # Check if rate limit exceeded
        if len(self.rate_limits[user_id]) >= self.rate_limit_max:
            return True

        # Add current timestamp
        self.rate_limits[user_id].append(current_time)
        return False

    async def _delivery_worker(self, worker_name: str):
        """Worker for delivering notifications."""
        logger.info(f"Notification delivery worker {worker_name} started")

        while not self._shutdown_event.is_set():
            try:
                # Get notification with timeout
                notification = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )

                await self._deliver_notification(notification)
                self.notification_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in delivery worker {worker_name}: {e}")

        logger.info(f"Notification delivery worker {worker_name} stopped")

    async def _deliver_notification(self, notification: RealTimeNotification):
        """Deliver notification via appropriate methods."""
        delivery_results = {}

        for method in notification.delivery_methods:
            try:
                if method == DeliveryMethod.WEBSOCKET:
                    success = await self._deliver_websocket(notification)
                    delivery_results[method.value] = success

                elif method == DeliveryMethod.SERVER_SENT_EVENTS:
                    success = await self._deliver_sse(notification)
                    delivery_results[method.value] = success

                elif method == DeliveryMethod.WEBHOOK:
                    success = await self._deliver_webhook(notification)
                    delivery_results[method.value] = success

                elif method == DeliveryMethod.EMAIL:
                    success = await self._deliver_email(notification)
                    delivery_results[method.value] = success

                elif method == DeliveryMethod.PUSH_NOTIFICATION:
                    success = await self._deliver_push(notification)
                    delivery_results[method.value] = success

            except Exception as e:
                logger.error(f"Error delivering {method.value} notification: {e}")
                delivery_results[method.value] = False

        # Store in history
        self.notification_history.append(notification)
        if len(self.notification_history) > 10000:
            self.notification_history = self.notification_history[-5000:]

    async def _deliver_websocket(self, notification: RealTimeNotification) -> bool:
        """Deliver notification via WebSocket."""
        if notification.user_id == "*":
            # Broadcast to all connected users
            success_count = 0
            for user_connections in self.websocket_connections.values():
                for connection in user_connections:
                    if await connection.send_message(notification.to_dict()):
                        success_count += 1
            return success_count > 0

        else:
            # Send to specific user
            if notification.user_id in self.websocket_connections:
                success_count = 0
                for connection in self.websocket_connections[notification.user_id]:
                    if await connection.send_message(notification.to_dict()):
                        success_count += 1
                return success_count > 0

        return False

    async def _deliver_sse(self, notification: RealTimeNotification) -> bool:
        """Deliver notification via Server-Sent Events."""
        # Implementation would depend on SSE framework
        logger.info(f"SSE delivery for notification {notification.notification_id} (mock)")
        return True

    async def _deliver_webhook(self, notification: RealTimeNotification) -> bool:
        """Deliver notification via webhook."""
        webhook_url = self.webhook_endpoints.get(notification.user_id)
        if webhook_url:
            # Implementation would use aiohttp to POST to webhook
            logger.info(f"Webhook delivery to {webhook_url} for notification {notification.notification_id} (mock)")
            return True
        return False

    async def _deliver_email(self, notification: RealTimeNotification) -> bool:
        """Deliver notification via email."""
        # Implementation would use email service
        logger.info(f"Email delivery for notification {notification.notification_id} (mock)")
        return True

    async def _deliver_push(self, notification: RealTimeNotification) -> bool:
        """Deliver notification via push notification."""
        # Implementation would use push notification service
        logger.info(f"Push notification delivery for notification {notification.notification_id} (mock)")
        return True

    async def _cleanup_connections(self):
        """Cleanup stale connections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                # Cleanup stale WebSocket connections
                for user_id, connections in list(self.websocket_connections.items()):
                    active_connections = []
                    for connection in connections:
                        if connection.is_stale() or not await connection.ping():
                            logger.info(f"Removing stale WebSocket connection for user {user_id}")
                        else:
                            active_connections.append(connection)

                    if active_connections:
                        self.websocket_connections[user_id] = active_connections
                    else:
                        del self.websocket_connections[user_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total_websocket_connections = sum(len(conns) for conns in self.websocket_connections.values())
        total_users = len(self.websocket_connections)

        return {
            "websocket_connections": {
                "total_connections": total_websocket_connections,
                "total_users": total_users,
                "connections_per_user": {
                    user_id: len(connections)
                    for user_id, connections in self.websocket_connections.items()
                }
            },
            "notification_queue": {
                "queue_size": self.notification_queue.qsize(),
                "max_size": self.notification_queue.maxsize
            },
            "notification_history": {
                "total_notifications": len(self.notification_history)
            }
        }


# Global real-time progress service instance
realtime_service = RealTimeProgressService()