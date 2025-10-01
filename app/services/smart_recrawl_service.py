"""
Smart Re-crawling Logic Service

This module provides intelligent re-crawling logic that optimizes resource usage
by determining when and how to re-crawl content based on change patterns,
priorities, and system resources.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import math
import statistics

from app.core.logging import get_logger
from app.services.incremental_update_service import (
    IncrementalUpdateService, UpdatePriority, ChangeType, ContentChange, UpdateSchedule
)
from app.services.content_versioning import ContentVersioningService, ContentVersion

logger = get_logger(__name__)


class CrawlReason(Enum):
    """Reasons for scheduling a crawl."""
    SCHEDULED_CHECK = "scheduled_check"
    CHANGE_DETECTED = "change_detected"
    HIGH_PRIORITY = "high_priority"
    USER_REQUESTED = "user_requested"
    DEPENDENCY_UPDATED = "dependency_updated"
    ERROR_RECOVERY = "error_recovery"
    INITIAL_CRAWL = "initial_crawl"


class CrawlStrategy(Enum):
    """Different crawling strategies."""
    AGGRESSIVE = "aggressive"  # Check frequently, crawl immediately
    BALANCED = "balanced"      # Normal frequency and delays
    CONSERVATIVE = "conservative"  # Minimal crawling, longer delays
    ADAPTIVE = "adaptive"      # Adjust based on change patterns


@dataclass
class CrawlRequest:
    """Represents a request to crawl a URL."""
    url: str
    reason: CrawlReason
    priority: UpdatePriority
    scheduled_time: datetime
    request_time: datetime
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: Optional[timedelta] = None
    dependencies: Set[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'reason': self.reason.value,
            'priority': self.priority.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'request_time': self.request_time.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'estimated_duration': self.estimated_duration.total_seconds() if self.estimated_duration else None,
            'dependencies': list(self.dependencies)
        }


@dataclass
class UrlChangePattern:
    """Tracks change patterns for a URL to optimize re-crawling."""
    url: str
    change_frequency: float  # Changes per day
    last_change_time: Optional[datetime]
    change_history: List[datetime]
    average_change_interval: timedelta
    change_predictability: float  # 0.0 to 1.0, higher = more predictable
    content_volatility: float  # 0.0 to 1.0, higher = more volatile content
    crawl_success_rate: float  # 0.0 to 1.0
    last_significant_change: Optional[datetime]

    def update_with_change(self, change_time: datetime, significant: bool = True):
        """Update pattern with a new change."""
        self.last_change_time = change_time
        self.change_history.append(change_time)

        if significant:
            self.last_significant_change = change_time

        # Keep only recent history (last 30 changes or 90 days)
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=90)
        self.change_history = [
            t for t in self.change_history
            if t > cutoff_time
        ][-30:]

        # Recalculate metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate derived metrics from change history."""
        if len(self.change_history) < 2:
            self.change_frequency = 0.0
            self.average_change_interval = timedelta(days=7)
            self.change_predictability = 0.0
            return

        # Calculate change frequency (changes per day)
        time_span = (self.change_history[-1] - self.change_history[0]).days
        if time_span > 0:
            self.change_frequency = len(self.change_history) / time_span
        else:
            self.change_frequency = 1.0

        # Calculate average interval between changes
        intervals = []
        for i in range(1, len(self.change_history)):
            interval = self.change_history[i] - self.change_history[i-1]
            intervals.append(interval.total_seconds())

        if intervals:
            avg_seconds = statistics.mean(intervals)
            self.average_change_interval = timedelta(seconds=avg_seconds)

            # Calculate predictability based on interval consistency
            if len(intervals) > 1:
                std_dev = statistics.stdev(intervals)
                coefficient_of_variation = std_dev / avg_seconds if avg_seconds > 0 else 1.0
                self.change_predictability = max(0.0, 1.0 - coefficient_of_variation)
            else:
                self.change_predictability = 0.5
        else:
            self.average_change_interval = timedelta(days=7)
            self.change_predictability = 0.0


@dataclass
class SystemResourceMetrics:
    """Tracks system resource usage for crawling optimization."""
    active_crawls: int
    queue_size: int
    cpu_usage: float
    memory_usage: float
    bandwidth_usage: float
    error_rate: float
    average_response_time: float
    timestamp: datetime

    def get_load_factor(self) -> float:
        """Calculate overall system load factor (0.0 to 1.0+)."""
        factors = [
            self.cpu_usage,
            self.memory_usage,
            self.bandwidth_usage,
            min(self.error_rate * 2, 1.0),  # Error rate penalty
            min(self.average_response_time / 5.0, 1.0)  # Response time penalty
        ]
        return statistics.mean(factors)


class SmartRecrawlService:
    """Main service for intelligent re-crawling logic."""

    def __init__(
        self,
        incremental_service: IncrementalUpdateService,
        versioning_service: ContentVersioningService
    ):
        self.incremental_service = incremental_service
        self.versioning_service = versioning_service

        # Configuration
        self.max_concurrent_crawls = 10
        self.default_strategy = CrawlStrategy.BALANCED
        self.resource_threshold = 0.8  # Throttle when resources exceed this

        # State tracking
        self.crawl_queue: List[CrawlRequest] = []
        self.active_crawls: Set[str] = set()
        self.url_patterns: Dict[str, UrlChangePattern] = {}
        self.strategy_overrides: Dict[str, CrawlStrategy] = {}
        self.system_metrics = SystemResourceMetrics(
            active_crawls=0,
            queue_size=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            bandwidth_usage=0.0,
            error_rate=0.0,
            average_response_time=1.0,
            timestamp=datetime.now(timezone.utc)
        )

        # Strategy configurations
        self.strategy_configs = {
            CrawlStrategy.AGGRESSIVE: {
                'min_check_interval': timedelta(minutes=15),
                'max_check_interval': timedelta(hours=2),
                'immediate_crawl_threshold': 0.7,
                'resource_usage_limit': 0.9
            },
            CrawlStrategy.BALANCED: {
                'min_check_interval': timedelta(hours=1),
                'max_check_interval': timedelta(hours=12),
                'immediate_crawl_threshold': 0.8,
                'resource_usage_limit': 0.8
            },
            CrawlStrategy.CONSERVATIVE: {
                'min_check_interval': timedelta(hours=6),
                'max_check_interval': timedelta(days=1),
                'immediate_crawl_threshold': 0.9,
                'resource_usage_limit': 0.6
            },
            CrawlStrategy.ADAPTIVE: {
                'min_check_interval': timedelta(minutes=30),
                'max_check_interval': timedelta(hours=24),
                'immediate_crawl_threshold': 0.75,
                'resource_usage_limit': 0.75
            }
        }

    async def analyze_change_patterns(self, url: str) -> UrlChangePattern:
        """Analyze and update change patterns for a URL."""
        if url not in self.url_patterns:
            # Initialize new pattern
            self.url_patterns[url] = UrlChangePattern(
                url=url,
                change_frequency=0.0,
                last_change_time=None,
                change_history=[],
                average_change_interval=timedelta(days=7),
                change_predictability=0.0,
                content_volatility=0.0,
                crawl_success_rate=1.0,
                last_significant_change=None
            )

        pattern = self.url_patterns[url]

        # Get version history to analyze patterns
        history = await self.versioning_service.get_version_history(url)
        if history and len(history.versions) > 1:
            # Update change history from versions
            pattern.change_history = [v.created_at for v in history.versions]
            pattern._calculate_metrics()

            # Calculate content volatility based on version diffs
            await self._calculate_content_volatility(pattern, history)

        return pattern

    async def _calculate_content_volatility(
        self,
        pattern: UrlChangePattern,
        history: 'VersionHistory'
    ):
        """Calculate content volatility based on version differences."""
        if len(history.versions) < 2:
            pattern.content_volatility = 0.0
            return

        # Analyze last few versions to determine volatility
        recent_versions = sorted(history.versions, key=lambda v: v.created_at)[-5:]
        volatility_scores = []

        for i in range(1, len(recent_versions)):
            old_version = recent_versions[i-1]
            new_version = recent_versions[i]

            # Calculate content change ratio
            old_size = old_version.size_bytes
            new_size = new_version.size_bytes
            size_change = abs(old_size - new_size) / max(old_size, new_size, 1)

            # Simple volatility score based on size change
            volatility_scores.append(min(size_change, 1.0))

        if volatility_scores:
            pattern.content_volatility = statistics.mean(volatility_scores)
        else:
            pattern.content_volatility = 0.0

    def calculate_optimal_check_interval(
        self,
        url: str,
        strategy: CrawlStrategy = None
    ) -> timedelta:
        """Calculate optimal check interval for a URL."""
        if strategy is None:
            strategy = self.strategy_overrides.get(url, self.default_strategy)

        config = self.strategy_configs[strategy]
        min_interval = config['min_check_interval']
        max_interval = config['max_check_interval']

        pattern = self.url_patterns.get(url)
        if not pattern or pattern.change_frequency == 0:
            # Default interval for new URLs
            return min_interval * 2

        # Base interval on change frequency
        if pattern.change_frequency > 0:
            # Convert frequency (changes/day) to interval
            changes_per_second = pattern.change_frequency / (24 * 3600)
            suggested_interval = timedelta(seconds=1 / (changes_per_second * 2))
        else:
            suggested_interval = max_interval

        # Adjust based on predictability
        if pattern.change_predictability > 0.7:
            # High predictability - can use longer intervals
            suggested_interval *= 1.5
        elif pattern.change_predictability < 0.3:
            # Low predictability - use shorter intervals
            suggested_interval *= 0.7

        # Adjust based on content volatility
        if pattern.content_volatility > 0.7:
            # High volatility - check more frequently
            suggested_interval *= 0.8
        elif pattern.content_volatility < 0.3:
            # Low volatility - can wait longer
            suggested_interval *= 1.3

        # Apply strategy constraints
        suggested_interval = max(min_interval, min(suggested_interval, max_interval))

        # Adaptive strategy adjustments
        if strategy == CrawlStrategy.ADAPTIVE:
            load_factor = self.system_metrics.get_load_factor()
            if load_factor > 0.8:
                suggested_interval *= 1.5  # Reduce frequency under high load
            elif load_factor < 0.3:
                suggested_interval *= 0.8  # Increase frequency under low load

        return suggested_interval

    async def should_crawl_immediately(
        self,
        change: ContentChange,
        strategy: CrawlStrategy = None
    ) -> bool:
        """Determine if a change should trigger immediate crawling."""
        if strategy is None:
            strategy = self.strategy_overrides.get(change.url, self.default_strategy)

        config = self.strategy_configs[strategy]
        threshold = config['immediate_crawl_threshold']

        # Priority-based decisions
        if change.priority == UpdatePriority.CRITICAL:
            return True
        elif change.priority == UpdatePriority.LOW:
            return False

        # Change type considerations
        if change.change_type == ChangeType.CONTENT_MODIFIED:
            return change.confidence_score >= threshold
        elif change.change_type == ChangeType.STRUCTURE_CHANGED:
            return change.confidence_score >= threshold * 0.9
        elif change.change_type == ChangeType.FIRST_CRAWL:
            return True

        # Resource availability check
        load_factor = self.system_metrics.get_load_factor()
        resource_limit = config['resource_usage_limit']

        if load_factor > resource_limit:
            return False  # Defer if resources are constrained

        return change.confidence_score >= threshold

    async def schedule_crawl(
        self,
        url: str,
        reason: CrawlReason,
        priority: UpdatePriority = UpdatePriority.MEDIUM,
        delay: Optional[timedelta] = None,
        dependencies: Set[str] = None
    ) -> CrawlRequest:
        """Schedule a crawl request."""
        now = datetime.now(timezone.utc)

        if delay is None:
            if reason == CrawlReason.SCHEDULED_CHECK:
                delay = timedelta(0)  # Immediate for scheduled checks
            elif reason == CrawlReason.CHANGE_DETECTED:
                # Check if should crawl immediately
                pattern = self.url_patterns.get(url)
                if pattern and pattern.change_frequency > 1.0:  # More than 1 change/day
                    delay = timedelta(minutes=5)  # Small delay for frequent changes
                else:
                    delay = timedelta(0)
            else:
                delay = timedelta(0)

        request = CrawlRequest(
            url=url,
            reason=reason,
            priority=priority,
            scheduled_time=now + delay,
            request_time=now,
            dependencies=dependencies or set()
        )

        # Insert request in priority order
        self._insert_crawl_request(request)

        logger.info(f"Scheduled crawl for {url}: {reason.value} (priority: {priority.value})")
        return request

    def _insert_crawl_request(self, request: CrawlRequest):
        """Insert crawl request in the queue maintaining priority order."""
        # Priority order: CRITICAL, HIGH, MEDIUM, LOW, DEFERRED
        priority_order = {
            UpdatePriority.CRITICAL: 0,
            UpdatePriority.HIGH: 1,
            UpdatePriority.MEDIUM: 2,
            UpdatePriority.LOW: 3,
            UpdatePriority.DEFERRED: 4
        }

        # Find insertion point
        insert_index = 0
        for i, existing_request in enumerate(self.crawl_queue):
            existing_priority = priority_order[existing_request.priority]
            new_priority = priority_order[request.priority]

            if new_priority < existing_priority:
                insert_index = i
                break
            elif new_priority == existing_priority:
                # Same priority, order by scheduled time
                if request.scheduled_time < existing_request.scheduled_time:
                    insert_index = i
                    break
            insert_index = i + 1

        self.crawl_queue.insert(insert_index, request)

    async def get_next_crawl_requests(self, limit: int = None) -> List[CrawlRequest]:
        """Get next crawl requests to process."""
        if limit is None:
            limit = self.max_concurrent_crawls - len(self.active_crawls)

        now = datetime.now(timezone.utc)
        ready_requests = []

        # Check resource constraints
        load_factor = self.system_metrics.get_load_factor()
        if load_factor > self.resource_threshold:
            limit = max(1, limit // 2)  # Reduce concurrency under high load

        i = 0
        while i < len(self.crawl_queue) and len(ready_requests) < limit:
            request = self.crawl_queue[i]

            # Check if request is ready
            if request.scheduled_time <= now:
                # Check dependencies
                if self._dependencies_satisfied(request):
                    ready_requests.append(request)
                    self.crawl_queue.pop(i)
                    self.active_crawls.add(request.url)
                    continue

            i += 1

        return ready_requests

    def _dependencies_satisfied(self, request: CrawlRequest) -> bool:
        """Check if all dependencies for a crawl request are satisfied."""
        for dep_url in request.dependencies:
            if dep_url in self.active_crawls:
                return False  # Dependency still being processed
        return True

    async def mark_crawl_completed(
        self,
        url: str,
        success: bool,
        change_detected: bool = False,
        processing_time: Optional[timedelta] = None
    ):
        """Mark a crawl as completed and update patterns."""
        if url in self.active_crawls:
            self.active_crawls.remove(url)

        # Update URL pattern
        pattern = self.url_patterns.get(url)
        if pattern:
            if change_detected:
                pattern.update_with_change(datetime.now(timezone.utc), significant=True)

            # Update success rate
            if success:
                pattern.crawl_success_rate = min(1.0, pattern.crawl_success_rate + 0.1)
            else:
                pattern.crawl_success_rate = max(0.0, pattern.crawl_success_rate - 0.1)

        # Schedule next check
        await self._schedule_next_check(url, success)

        logger.info(f"Crawl completed for {url}: success={success}, change={change_detected}")

    async def _schedule_next_check(self, url: str, last_success: bool):
        """Schedule the next check for a URL."""
        interval = self.calculate_optimal_check_interval(url)

        # Adjust interval based on last result
        if not last_success:
            interval *= 2  # Back off on failure

        await self.schedule_crawl(
            url=url,
            reason=CrawlReason.SCHEDULED_CHECK,
            priority=UpdatePriority.MEDIUM,
            delay=interval
        )

    async def handle_change_detection(self, change: ContentChange):
        """Handle a detected change by scheduling appropriate crawls."""
        # Update change pattern
        pattern = self.url_patterns.get(change.url)
        if pattern:
            pattern.update_with_change(change.detected_at, significant=True)

        # Determine if immediate crawl is needed
        if await self.should_crawl_immediately(change):
            await self.schedule_crawl(
                url=change.url,
                reason=CrawlReason.CHANGE_DETECTED,
                priority=change.priority
            )
        else:
            # Schedule with delay
            delay = self.calculate_optimal_check_interval(change.url) // 2
            await self.schedule_crawl(
                url=change.url,
                reason=CrawlReason.CHANGE_DETECTED,
                priority=change.priority,
                delay=delay
            )

    def set_crawl_strategy(self, url: str, strategy: CrawlStrategy):
        """Set crawling strategy for a specific URL."""
        self.strategy_overrides[url] = strategy
        logger.info(f"Set crawl strategy for {url}: {strategy.value}")

    def update_system_metrics(self, metrics: SystemResourceMetrics):
        """Update system resource metrics."""
        self.system_metrics = metrics

    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the crawl queue."""
        priority_counts = {}
        reason_counts = {}

        for request in self.crawl_queue:
            priority = request.priority.value
            reason = request.reason.value

            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        now = datetime.now(timezone.utc)
        overdue_count = sum(1 for r in self.crawl_queue if r.scheduled_time <= now)

        return {
            'queue_size': len(self.crawl_queue),
            'active_crawls': len(self.active_crawls),
            'overdue_requests': overdue_count,
            'priority_distribution': priority_counts,
            'reason_distribution': reason_counts,
            'tracked_patterns': len(self.url_patterns),
            'system_load_factor': self.system_metrics.get_load_factor()
        }

    async def optimize_queue(self):
        """Optimize the crawl queue by reordering and merging requests."""
        # Remove duplicate requests for the same URL
        seen_urls = set()
        optimized_queue = []

        for request in self.crawl_queue:
            if request.url not in seen_urls:
                optimized_queue.append(request)
                seen_urls.add(request.url)

        self.crawl_queue = optimized_queue

        # Reorder based on current priorities and system state
        load_factor = self.system_metrics.get_load_factor()
        if load_factor > 0.8:
            # Under high load, prioritize high-priority requests more aggressively
            self.crawl_queue.sort(key=lambda r: (
                r.priority.value,
                r.scheduled_time.timestamp()
            ))

        logger.info(f"Optimized crawl queue: {len(self.crawl_queue)} requests")


# Global service instance (will be initialized with dependencies)
smart_recrawl_service: Optional[SmartRecrawlService] = None


def initialize_smart_recrawl_service(
    incremental_service: IncrementalUpdateService,
    versioning_service: ContentVersioningService
) -> SmartRecrawlService:
    """Initialize the global smart recrawl service."""
    global smart_recrawl_service
    smart_recrawl_service = SmartRecrawlService(incremental_service, versioning_service)
    return smart_recrawl_service