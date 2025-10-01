"""
Update Triggers and Scheduling System

This module provides a comprehensive scheduling system for managing crawl triggers,
automated update checks, and coordinating the overall update detection workflow.
"""

import asyncio
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Coroutine
from dataclasses import dataclass, asdict
from enum import Enum
import json
import aiofiles
from contextlib import asynccontextmanager

from app.core.logging import get_logger
from app.services.incremental_update_service import (
    incremental_service, ChangeType, UpdatePriority, ContentChange
)
from app.services.smart_recrawl_service import (
    smart_recrawl_service, CrawlReason, CrawlRequest, initialize_smart_recrawl_service
)
from app.services.content_versioning import versioning_service

logger = get_logger(__name__)


class TriggerType(Enum):
    """Types of update triggers."""
    TIME_BASED = "time_based"          # Scheduled at specific times
    INTERVAL_BASED = "interval_based"  # Recurring intervals
    EVENT_BASED = "event_based"        # Triggered by events
    DEPENDENCY_BASED = "dependency_based"  # Triggered by dependency changes
    THRESHOLD_BASED = "threshold_based"  # Triggered when thresholds are met
    MANUAL = "manual"                  # Manually triggered


class TriggerStatus(Enum):
    """Status of a trigger."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class UpdateTrigger:
    """Configuration for an update trigger."""
    trigger_id: str
    name: str
    trigger_type: TriggerType
    target_urls: Set[str]
    status: TriggerStatus
    schedule_config: Dict[str, Any]
    priority: UpdatePriority
    conditions: Dict[str, Any]
    created_at: datetime
    last_triggered: Optional[datetime] = None
    next_trigger: Optional[datetime] = None
    trigger_count: int = 0
    error_count: int = 0
    max_errors: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['trigger_type'] = self.trigger_type.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['target_urls'] = list(self.target_urls)
        data['created_at'] = self.created_at.isoformat()
        if self.last_triggered:
            data['last_triggered'] = self.last_triggered.isoformat()
        if self.next_trigger:
            data['next_trigger'] = self.next_trigger.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpdateTrigger':
        """Create instance from dictionary."""
        data = data.copy()
        data['trigger_type'] = TriggerType(data['trigger_type'])
        data['status'] = TriggerStatus(data['status'])
        data['priority'] = UpdatePriority(data['priority'])
        data['target_urls'] = set(data['target_urls'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_triggered'):
            data['last_triggered'] = datetime.fromisoformat(data['last_triggered'])
        if data.get('next_trigger'):
            data['next_trigger'] = datetime.fromisoformat(data['next_trigger'])
        return cls(**data)


@dataclass
class ScheduledTask:
    """Represents a scheduled task in the system."""
    task_id: str
    trigger_id: str
    scheduled_time: datetime
    target_urls: Set[str]
    priority: UpdatePriority
    task_data: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'trigger_id': self.trigger_id,
            'scheduled_time': self.scheduled_time.isoformat(),
            'target_urls': list(self.target_urls),
            'priority': self.priority.value,
            'task_data': self.task_data,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class EventBus:
    """Simple event bus for trigger communication."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)

    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event to all subscribers."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")


class UpdateScheduler:
    """Main scheduler service for managing update triggers and tasks."""

    def __init__(self, storage_path: str = "/tmp/scheduler"):
        self.storage_path = storage_path
        self.triggers: Dict[str, UpdateTrigger] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.event_bus = EventBus()
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

        # Scheduling configuration
        self.check_interval = timedelta(seconds=30)  # Check for due tasks every 30 seconds
        self.max_concurrent_tasks = 20

        # Initialize storage
        asyncio.create_task(self._initialize_storage())

        # Set up default event handlers
        self._setup_default_handlers()

    async def _initialize_storage(self):
        """Initialize storage directory and load existing data."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        await self.load_triggers()

    def _setup_default_handlers(self):
        """Set up default event handlers."""
        self.event_bus.subscribe("content_change", self._handle_content_change)
        self.event_bus.subscribe("crawl_completed", self._handle_crawl_completed)
        self.event_bus.subscribe("system_status", self._handle_system_status)

    async def _handle_content_change(self, event_data: Dict[str, Any]):
        """Handle content change events."""
        url = event_data.get('url')
        change_type = event_data.get('change_type')

        if url:
            # Trigger dependency-based updates
            await self._trigger_dependency_updates(url, change_type)

    async def _handle_crawl_completed(self, event_data: Dict[str, Any]):
        """Handle crawl completion events."""
        url = event_data.get('url')
        success = event_data.get('success', False)
        change_detected = event_data.get('change_detected', False)

        if url and change_detected:
            await self.event_bus.publish("content_change", {
                'url': url,
                'change_type': 'content_modified',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

    async def _handle_system_status(self, event_data: Dict[str, Any]):
        """Handle system status events."""
        status = event_data.get('status')
        if status == 'high_load':
            # Pause low-priority triggers during high load
            await self._adjust_triggers_for_load(True)
        elif status == 'normal_load':
            # Resume paused triggers
            await self._adjust_triggers_for_load(False)

    async def create_trigger(
        self,
        name: str,
        trigger_type: TriggerType,
        target_urls: Set[str],
        schedule_config: Dict[str, Any],
        priority: UpdatePriority = UpdatePriority.MEDIUM,
        conditions: Dict[str, Any] = None
    ) -> UpdateTrigger:
        """Create a new update trigger."""
        trigger_id = f"trigger_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(self.triggers)}"

        trigger = UpdateTrigger(
            trigger_id=trigger_id,
            name=name,
            trigger_type=trigger_type,
            target_urls=target_urls,
            status=TriggerStatus.ACTIVE,
            schedule_config=schedule_config,
            priority=priority,
            conditions=conditions or {},
            created_at=datetime.now(timezone.utc)
        )

        # Calculate next trigger time
        trigger.next_trigger = self._calculate_next_trigger_time(trigger)

        self.triggers[trigger_id] = trigger
        await self.save_triggers()

        logger.info(f"Created trigger {trigger_id}: {name}")
        return trigger

    def _calculate_next_trigger_time(self, trigger: UpdateTrigger) -> Optional[datetime]:
        """Calculate the next trigger time based on trigger configuration."""
        now = datetime.now(timezone.utc)
        config = trigger.schedule_config

        if trigger.trigger_type == TriggerType.TIME_BASED:
            # Scheduled at specific times
            if 'time' in config:
                target_time = datetime.fromisoformat(config['time'])
                if target_time > now:
                    return target_time
                elif 'repeat' in config and config['repeat']:
                    # Calculate next occurrence
                    if config.get('daily'):
                        return now.replace(
                            hour=target_time.hour,
                            minute=target_time.minute,
                            second=0,
                            microsecond=0
                        ) + timedelta(days=1)

        elif trigger.trigger_type == TriggerType.INTERVAL_BASED:
            # Recurring intervals
            if 'interval_seconds' in config:
                interval = timedelta(seconds=config['interval_seconds'])
                return now + interval
            elif 'interval' in config:
                # Parse interval string like "1h", "30m", "2d"
                interval = self._parse_interval(config['interval'])
                return now + interval

        elif trigger.trigger_type == TriggerType.THRESHOLD_BASED:
            # Check thresholds - return None, will be triggered by conditions
            return None

        elif trigger.trigger_type == TriggerType.EVENT_BASED:
            # Event-driven - return None, will be triggered by events
            return None

        elif trigger.trigger_type == TriggerType.DEPENDENCY_BASED:
            # Dependency-driven - return None, will be triggered by dependency changes
            return None

        return None

    def _parse_interval(self, interval_str: str) -> timedelta:
        """Parse interval string to timedelta."""
        import re

        pattern = r'(\d+)([smhd])'
        match = re.match(pattern, interval_str.lower())

        if not match:
            return timedelta(hours=1)  # Default

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 's':
            return timedelta(seconds=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)

        return timedelta(hours=1)

    async def start_scheduler(self):
        """Start the scheduler background task."""
        if self.running:
            return

        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Update scheduler started")

    async def stop_scheduler(self):
        """Stop the scheduler background task."""
        if not self.running:
            return

        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Update scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                await self._process_due_triggers()
                await self._process_scheduled_tasks()
                await self._cleanup_completed_tasks()
                await asyncio.sleep(self.check_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _process_due_triggers(self):
        """Process triggers that are due to fire."""
        now = datetime.now(timezone.utc)
        due_triggers = []

        for trigger in self.triggers.values():
            if (trigger.status == TriggerStatus.ACTIVE and
                trigger.next_trigger and
                trigger.next_trigger <= now):
                due_triggers.append(trigger)

        for trigger in due_triggers:
            await self._fire_trigger(trigger)

    async def _fire_trigger(self, trigger: UpdateTrigger):
        """Fire a trigger and create scheduled tasks."""
        try:
            # Check conditions if any
            if trigger.conditions and not await self._check_trigger_conditions(trigger):
                logger.debug(f"Trigger {trigger.trigger_id} conditions not met")
                return

            # Create scheduled tasks for target URLs
            for url in trigger.target_urls:
                task_id = f"task_{trigger.trigger_id}_{url}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

                task = ScheduledTask(
                    task_id=task_id,
                    trigger_id=trigger.trigger_id,
                    scheduled_time=datetime.now(timezone.utc),
                    target_urls={url},
                    priority=trigger.priority,
                    task_data={
                        'trigger_type': trigger.trigger_type.value,
                        'trigger_name': trigger.name
                    }
                )

                self.scheduled_tasks[task_id] = task

            # Update trigger state
            trigger.last_triggered = datetime.now(timezone.utc)
            trigger.trigger_count += 1
            trigger.next_trigger = self._calculate_next_trigger_time(trigger)

            logger.info(f"Fired trigger {trigger.trigger_id}: {trigger.name}")

        except Exception as e:
            trigger.error_count += 1
            logger.error(f"Error firing trigger {trigger.trigger_id}: {e}")

            if trigger.error_count >= trigger.max_errors:
                trigger.status = TriggerStatus.ERROR
                logger.warning(f"Trigger {trigger.trigger_id} disabled due to errors")

    async def _check_trigger_conditions(self, trigger: UpdateTrigger) -> bool:
        """Check if trigger conditions are met."""
        conditions = trigger.conditions

        # Check system load condition
        if 'max_system_load' in conditions:
            # This would check actual system metrics
            # For now, return True
            pass

        # Check change frequency condition
        if 'min_change_frequency' in conditions:
            # Check if URLs have minimum change frequency
            for url in trigger.target_urls:
                pattern = smart_recrawl_service.url_patterns.get(url) if smart_recrawl_service else None
                if pattern and pattern.change_frequency < conditions['min_change_frequency']:
                    return False

        # Check time window condition
        if 'time_window' in conditions:
            window = conditions['time_window']
            now = datetime.now(timezone.utc)
            start_time = datetime.strptime(window['start'], '%H:%M').time()
            end_time = datetime.strptime(window['end'], '%H:%M').time()
            current_time = now.time()

            if not (start_time <= current_time <= end_time):
                return False

        return True

    async def _process_scheduled_tasks(self):
        """Process scheduled tasks that are ready to execute."""
        now = datetime.now(timezone.utc)
        ready_tasks = []

        for task in self.scheduled_tasks.values():
            if task.scheduled_time <= now:
                ready_tasks.append(task)

        # Limit concurrent tasks
        if len(ready_tasks) > self.max_concurrent_tasks:
            # Sort by priority and take top tasks
            ready_tasks.sort(key=lambda t: t.priority.value)
            ready_tasks = ready_tasks[:self.max_concurrent_tasks]

        for task in ready_tasks:
            await self._execute_task(task)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            # Remove from scheduled tasks
            if task.task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task.task_id]

            # Schedule crawls via smart recrawl service
            if smart_recrawl_service:
                for url in task.target_urls:
                    await smart_recrawl_service.schedule_crawl(
                        url=url,
                        reason=CrawlReason.SCHEDULED_CHECK,
                        priority=task.priority
                    )

            logger.info(f"Executed task {task.task_id}")

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=5 * task.retry_count)
                self.scheduled_tasks[task.task_id] = task
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")

    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks."""
        # Remove tasks older than 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        completed_tasks = [
            task_id for task_id, task in self.scheduled_tasks.items()
            if task.scheduled_time < cutoff_time
        ]

        for task_id in completed_tasks:
            del self.scheduled_tasks[task_id]

        if completed_tasks:
            logger.debug(f"Cleaned up {len(completed_tasks)} old tasks")

    async def _trigger_dependency_updates(self, url: str, change_type: str):
        """Trigger updates for URLs that depend on the changed URL."""
        dependent_triggers = [
            trigger for trigger in self.triggers.values()
            if (trigger.trigger_type == TriggerType.DEPENDENCY_BASED and
                trigger.status == TriggerStatus.ACTIVE and
                url in trigger.conditions.get('dependencies', []))
        ]

        for trigger in dependent_triggers:
            await self._fire_trigger(trigger)

    async def _adjust_triggers_for_load(self, high_load: bool):
        """Adjust triggers based on system load."""
        for trigger in self.triggers.values():
            if trigger.priority in [UpdatePriority.LOW, UpdatePriority.DEFERRED]:
                if high_load and trigger.status == TriggerStatus.ACTIVE:
                    trigger.status = TriggerStatus.PAUSED
                elif not high_load and trigger.status == TriggerStatus.PAUSED:
                    trigger.status = TriggerStatus.ACTIVE

    async def pause_trigger(self, trigger_id: str) -> bool:
        """Pause a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].status = TriggerStatus.PAUSED
            await self.save_triggers()
            logger.info(f"Paused trigger {trigger_id}")
            return True
        return False

    async def resume_trigger(self, trigger_id: str) -> bool:
        """Resume a paused trigger."""
        if trigger_id in self.triggers:
            trigger = self.triggers[trigger_id]
            if trigger.status == TriggerStatus.PAUSED:
                trigger.status = TriggerStatus.ACTIVE
                trigger.next_trigger = self._calculate_next_trigger_time(trigger)
                await self.save_triggers()
                logger.info(f"Resumed trigger {trigger_id}")
                return True
        return False

    async def delete_trigger(self, trigger_id: str) -> bool:
        """Delete a trigger."""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            await self.save_triggers()
            logger.info(f"Deleted trigger {trigger_id}")
            return True
        return False

    async def manual_trigger(self, trigger_id: str) -> bool:
        """Manually fire a trigger."""
        if trigger_id in self.triggers:
            trigger = self.triggers[trigger_id]
            await self._fire_trigger(trigger)
            return True
        return False

    async def save_triggers(self):
        """Save triggers to storage."""
        triggers_file = f"{self.storage_path}/triggers.json"
        data = {trigger_id: trigger.to_dict() for trigger_id, trigger in self.triggers.items()}

        async with aiofiles.open(triggers_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def load_triggers(self):
        """Load triggers from storage."""
        triggers_file = f"{self.storage_path}/triggers.json"

        try:
            async with aiofiles.open(triggers_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            self.triggers = {
                trigger_id: UpdateTrigger.from_dict(trigger_data)
                for trigger_id, trigger_data in data.items()
            }

            logger.info(f"Loaded {len(self.triggers)} triggers")

        except FileNotFoundError:
            logger.info("No existing triggers found")
        except Exception as e:
            logger.error(f"Error loading triggers: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        trigger_counts = {}
        status_counts = {}

        for trigger in self.triggers.values():
            trigger_type = trigger.trigger_type.value
            status = trigger.status.value

            trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_triggers': len(self.triggers),
            'scheduled_tasks': len(self.scheduled_tasks),
            'trigger_type_distribution': trigger_counts,
            'trigger_status_distribution': status_counts,
            'scheduler_running': self.running,
            'check_interval_seconds': self.check_interval.total_seconds()
        }

    async def shutdown(self):
        """Gracefully shutdown the scheduler."""
        await self.stop_scheduler()
        await self.save_triggers()
        logger.info("Scheduler shutdown complete")


# Global scheduler instance
update_scheduler = UpdateScheduler()


# Context manager for scheduler lifecycle
@asynccontextmanager
async def managed_scheduler():
    """Context manager for scheduler lifecycle management."""
    try:
        await update_scheduler.start_scheduler()
        yield update_scheduler
    finally:
        await update_scheduler.shutdown()


# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(update_scheduler.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)