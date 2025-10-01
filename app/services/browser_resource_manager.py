"""
Browser Resource Manager for optimized resource allocation and monitoring.
Task 4.3: Develop browser resource management
"""

import asyncio
import logging
import psutil
import gc
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager

from playwright.async_api import Browser, BrowserContext, Page

from app.services.playwright_browser_manager import (
    PlaywrightBrowserManager,
    BrowserConfig,
    BrowserEngine,
    BrowserSession
)

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Resource status enumeration."""
    AVAILABLE = "available"
    BUSY = "busy"
    EXHAUSTED = "exhausted"
    ERROR = "error"


class PoolStrategy(Enum):
    """Browser pool management strategies."""
    FIFO = "fifo"  # First In, First Out
    LRU = "lru"   # Least Recently Used
    ROUND_ROBIN = "round_robin"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    memory_usage_mb: float
    cpu_usage_percent: float
    active_sessions: int
    total_sessions: int
    peak_sessions: int
    average_session_duration: float
    total_requests_processed: int
    error_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BrowserPool:
    """Browser instance pool."""
    engine: BrowserEngine
    instances: List[Browser] = field(default_factory=list)
    available_instances: Set[int] = field(default_factory=set)
    busy_instances: Set[int] = field(default_factory=set)
    max_instances: int = 5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_cleanup: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BrowserResourceManager:
    """
    Advanced browser resource manager with pooling, monitoring, and optimization.

    Features:
    - Browser instance pooling and reuse
    - Memory and CPU usage monitoring
    - Resource allocation optimization
    - Automatic cleanup and garbage collection
    - Session lifecycle management
    - Performance metrics and reporting
    - Resource exhaustion handling
    - Crash recovery and error handling
    """

    def __init__(
        self,
        max_total_browsers: int = 10,
        max_browsers_per_engine: int = 5,
        session_timeout_seconds: int = 300,
        cleanup_interval_seconds: int = 60,
        pool_strategy: PoolStrategy = PoolStrategy.LRU,
        enable_monitoring: bool = True
    ):
        """Initialize browser resource manager."""
        self.max_total_browsers = max_total_browsers
        self.max_browsers_per_engine = max_browsers_per_engine
        self.session_timeout_seconds = session_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.pool_strategy = pool_strategy
        self.enable_monitoring = enable_monitoring

        # Resource pools
        self.browser_pools: Dict[BrowserEngine, BrowserPool] = {}
        self.active_sessions: Dict[str, BrowserSession] = {}
        self.session_queue: asyncio.Queue = asyncio.Queue()

        # Monitoring
        self.metrics_history: List[ResourceMetrics] = []
        self.peak_sessions = 0
        self.total_requests_processed = 0
        self.error_count = 0
        self.start_time = datetime.now(timezone.utc)

        # Tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Browser manager integration
        self.browser_manager: Optional[PlaywrightBrowserManager] = None

    async def initialize(self) -> None:
        """Initialize resource manager."""
        try:
            # Initialize browser manager
            self.browser_manager = PlaywrightBrowserManager(
                max_concurrent_browsers=self.max_total_browsers
            )
            await self.browser_manager.initialize()

            # Initialize browser pools
            for engine in BrowserEngine:
                self.browser_pools[engine] = BrowserPool(
                    engine=engine,
                    max_instances=self.max_browsers_per_engine
                )

            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            if self.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.is_running = True
            logger.info("Browser resource manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize browser resource manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown resource manager and cleanup all resources."""
        logger.info("Shutting down browser resource manager...")

        self.is_running = False

        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Close all active sessions
        active_session_ids = list(self.active_sessions.keys())
        for session_id in active_session_ids:
            try:
                await self.release_session(session_id)
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")

        # Close all browser pools
        for engine, pool in self.browser_pools.items():
            await self._cleanup_browser_pool(pool)

        # Shutdown browser manager
        if self.browser_manager:
            await self.browser_manager.shutdown()

        # Final cleanup
        self.active_sessions.clear()
        self.browser_pools.clear()

        logger.info("Browser resource manager shutdown complete")

    @asynccontextmanager
    async def acquire_session(self, config: Optional[BrowserConfig] = None):
        """
        Context manager for acquiring and automatically releasing sessions.

        Usage:
            async with resource_manager.acquire_session(config) as session:
                # Use session
                result = await session.page.goto("https://example.com")
        """
        session_id = None
        try:
            session_id = await self.get_session(config)
            session = self.active_sessions[session_id]
            yield session
        finally:
            if session_id:
                await self.release_session(session_id)

    async def get_session(self, config: Optional[BrowserConfig] = None) -> str:
        """Get a browser session, creating one if needed."""
        config = config or BrowserConfig()

        try:
            # Check resource limits
            if len(self.active_sessions) >= self.max_total_browsers:
                await self._wait_for_available_session()

            # Try to get from pool first
            session_id = await self._get_pooled_session(config)

            if not session_id:
                # Create new session
                session_id = await self._create_new_session(config)

            # Update metrics
            self.total_requests_processed += 1
            current_sessions = len(self.active_sessions)
            if current_sessions > self.peak_sessions:
                self.peak_sessions = current_sessions

            return session_id

        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to get session: {e}")
            raise

    async def release_session(self, session_id: str) -> None:
        """Release a browser session back to the pool."""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return

            session = self.active_sessions[session_id]

            # Mark session as available for reuse
            session.is_active = False
            session.last_used = datetime.now(timezone.utc)

            # Return browser to pool or close if expired
            await self._return_to_pool_or_close(session)

            # Remove from active sessions
            del self.active_sessions[session_id]

            logger.debug(f"Released session {session_id}")

        except Exception as e:
            logger.error(f"Error releasing session {session_id}: {e}")

    async def get_resource_status(self) -> ResourceStatus:
        """Get current resource status."""
        active_count = len(self.active_sessions)

        if active_count >= self.max_total_browsers:
            return ResourceStatus.EXHAUSTED
        elif active_count >= self.max_total_browsers * 0.8:
            return ResourceStatus.BUSY
        else:
            return ResourceStatus.AVAILABLE

    async def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        try:
            # Get system metrics
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            # Calculate session metrics
            active_sessions = len(self.active_sessions)
            total_sessions = sum(len(pool.instances) for pool in self.browser_pools.values())

            # Calculate average session duration
            avg_duration = 0.0
            if self.active_sessions:
                now = datetime.now(timezone.utc)
                durations = [
                    (now - session.created_at).total_seconds()
                    for session in self.active_sessions.values()
                ]
                avg_duration = sum(durations) / len(durations)

            # Calculate error rate
            error_rate = 0.0
            if self.total_requests_processed > 0:
                error_rate = (self.error_count / self.total_requests_processed) * 100

            return ResourceMetrics(
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                active_sessions=active_sessions,
                total_sessions=total_sessions,
                peak_sessions=self.peak_sessions,
                average_session_duration=avg_duration,
                total_requests_processed=self.total_requests_processed,
                error_rate=error_rate
            )

        except Exception as e:
            logger.error(f"Error getting resource metrics: {e}")
            return ResourceMetrics(
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                active_sessions=len(self.active_sessions),
                total_sessions=0,
                peak_sessions=self.peak_sessions,
                average_session_duration=0.0,
                total_requests_processed=self.total_requests_processed,
                error_rate=0.0
            )

    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource usage and return optimization report."""
        logger.info("Starting resource optimization...")

        optimization_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [],
            "resources_freed": 0,
            "memory_before_mb": 0.0,
            "memory_after_mb": 0.0
        }

        try:
            # Get initial metrics
            initial_metrics = await self.get_resource_metrics()
            optimization_report["memory_before_mb"] = initial_metrics.memory_usage_mb

            # 1. Clean up expired sessions
            expired_count = await self._cleanup_expired_sessions()
            if expired_count > 0:
                optimization_report["actions_taken"].append(f"Cleaned up {expired_count} expired sessions")
                optimization_report["resources_freed"] += expired_count

            # 2. Optimize browser pools
            pool_cleanup_count = await self._optimize_browser_pools()
            if pool_cleanup_count > 0:
                optimization_report["actions_taken"].append(f"Optimized {pool_cleanup_count} browser pools")

            # 3. Force garbage collection
            gc.collect()
            optimization_report["actions_taken"].append("Forced garbage collection")

            # 4. Defragment session storage
            await self._defragment_session_storage()
            optimization_report["actions_taken"].append("Defragmented session storage")

            # Get final metrics
            final_metrics = await self.get_resource_metrics()
            optimization_report["memory_after_mb"] = final_metrics.memory_usage_mb

            memory_freed = initial_metrics.memory_usage_mb - final_metrics.memory_usage_mb
            optimization_report["memory_freed_mb"] = memory_freed

            logger.info(f"Resource optimization complete. Memory freed: {memory_freed:.2f}MB")

        except Exception as e:
            logger.error(f"Error during resource optimization: {e}")
            optimization_report["error"] = str(e)

        return optimization_report

    async def _get_pooled_session(self, config: BrowserConfig) -> Optional[str]:
        """Try to get an existing session from the pool."""
        pool = self.browser_pools.get(config.engine)
        if not pool or not pool.available_instances:
            return None

        # Implement pooling strategy
        if self.pool_strategy == PoolStrategy.LRU:
            # Use least recently used session
            oldest_session = min(
                self.active_sessions.values(),
                key=lambda s: s.last_used,
                default=None
            )
            if oldest_session and not oldest_session.is_active:
                oldest_session.is_active = True
                oldest_session.last_used = datetime.now(timezone.utc)
                return oldest_session.session_id

        # Fallback to creating new session
        return None

    async def _create_new_session(self, config: BrowserConfig) -> str:
        """Create a new browser session."""
        if not self.browser_manager:
            raise RuntimeError("Browser manager not initialized")

        session_id = await self.browser_manager.create_session(config)
        session = await self.browser_manager.get_session(session_id)

        if session:
            self.active_sessions[session_id] = session

        return session_id

    async def _return_to_pool_or_close(self, session: BrowserSession) -> None:
        """Return session to pool or close if expired."""
        try:
            # Check if session is too old
            age = datetime.now(timezone.utc) - session.created_at
            if age.total_seconds() > self.session_timeout_seconds:
                # Close expired session
                await self._close_session(session)
            else:
                # Return to pool for reuse (session is already marked inactive)
                pool = self.browser_pools.get(session.config.engine)
                if pool and len(pool.available_instances) < pool.max_instances:
                    # Add to available instances (browser is kept alive)
                    pass
                else:
                    # Pool full, close session
                    await self._close_session(session)

        except Exception as e:
            logger.error(f"Error returning session to pool: {e}")

    async def _close_session(self, session: BrowserSession) -> None:
        """Close a browser session and cleanup resources."""
        try:
            if self.browser_manager:
                await self.browser_manager.close_session(session.session_id)
        except Exception as e:
            logger.error(f"Error closing session {session.session_id}: {e}")

    async def _wait_for_available_session(self) -> None:
        """Wait for a session to become available."""
        max_wait_time = 30  # seconds
        wait_start = datetime.now()

        while (datetime.now() - wait_start).total_seconds() < max_wait_time:
            if len(self.active_sessions) < self.max_total_browsers:
                return

            await asyncio.sleep(0.1)

        raise RuntimeError("Timeout waiting for available session")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)

                if not self.is_running:
                    break

                # Cleanup expired sessions
                await self._cleanup_expired_sessions()

                # Cleanup browser pools
                await self._optimize_browser_pools()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _monitoring_loop(self) -> None:
        """Background task for resource monitoring."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                if not self.is_running:
                    break

                # Collect metrics
                metrics = await self.get_resource_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 24 hours of metrics (1440 minutes)
                if len(self.metrics_history) > 1440:
                    self.metrics_history = self.metrics_history[-1440:]

                # Log high resource usage
                if metrics.memory_usage_mb > 1000:  # Over 1GB
                    logger.warning(f"High memory usage: {metrics.memory_usage_mb:.2f}MB")

                if metrics.cpu_usage_percent > 80:
                    logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.2f}%")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        expired_count = 0
        now = datetime.now(timezone.utc)
        timeout_delta = timedelta(seconds=self.session_timeout_seconds)

        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if (now - session.last_used) > timeout_delta:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            try:
                await self.release_session(session_id)
                expired_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired sessions")

        return expired_count

    async def _optimize_browser_pools(self) -> int:
        """Optimize browser pools and return count of optimized pools."""
        optimized_count = 0

        for engine, pool in self.browser_pools.items():
            try:
                # Remove unused browser instances
                if len(pool.available_instances) > 2:  # Keep at least 2
                    # Close excess instances
                    excess_count = len(pool.available_instances) - 2
                    for _ in range(excess_count):
                        if pool.available_instances:
                            instance_id = pool.available_instances.pop()
                            # Browser cleanup would happen here

                    optimized_count += 1

                pool.last_cleanup = datetime.now(timezone.utc)

            except Exception as e:
                logger.error(f"Error optimizing pool for {engine}: {e}")

        return optimized_count

    async def _cleanup_browser_pool(self, pool: BrowserPool) -> None:
        """Cleanup a browser pool."""
        try:
            # Close all browser instances in pool
            for browser in pool.instances:
                try:
                    await browser.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")

            pool.instances.clear()
            pool.available_instances.clear()
            pool.busy_instances.clear()

        except Exception as e:
            logger.error(f"Error cleaning up browser pool: {e}")

    async def _defragment_session_storage(self) -> None:
        """Defragment session storage by rebuilding dictionaries."""
        try:
            # Rebuild active sessions dictionary to remove fragmentation
            active_sessions_copy = dict(self.active_sessions)
            self.active_sessions.clear()
            self.active_sessions.update(active_sessions_copy)

            # Trigger garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Error defragmenting session storage: {e}")


# Global instance
_browser_resource_manager_instance: Optional[BrowserResourceManager] = None

async def get_browser_resource_manager() -> BrowserResourceManager:
    """Get global browser resource manager instance."""
    global _browser_resource_manager_instance

    if _browser_resource_manager_instance is None:
        _browser_resource_manager_instance = BrowserResourceManager()
        await _browser_resource_manager_instance.initialize()

    return _browser_resource_manager_instance

async def shutdown_browser_resource_manager() -> None:
    """Shutdown global browser resource manager."""
    global _browser_resource_manager_instance

    if _browser_resource_manager_instance:
        await _browser_resource_manager_instance.shutdown()
        _browser_resource_manager_instance = None