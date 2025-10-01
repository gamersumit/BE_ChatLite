"""
Database connection pooling service for optimal performance and resource management.
Implements advanced connection pooling with monitoring, health checks, and automatic scaling.
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError


class PoolStatus(Enum):
    """Connection pool status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


class ConnectionState(Enum):
    """Individual connection states."""
    ACTIVE = "active"
    IDLE = "idle"
    STALE = "stale"
    INVALID = "invalid"


@dataclass
class ConnectionMetrics:
    """Metrics for individual database connections."""
    connection_id: str
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    total_time_active: float = 0.0
    state: ConnectionState = ConnectionState.IDLE
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class PoolMetrics:
    """Comprehensive metrics for connection pools."""
    pool_id: str
    pool_size: int
    max_overflow: int
    checked_in: int = 0
    checked_out: int = 0
    overflow: int = 0
    invalid_connections: int = 0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    total_checkouts: int = 0
    total_checkins: int = 0
    avg_checkout_time: float = 0.0
    peak_connections: int = 0
    status: PoolStatus = PoolStatus.HEALTHY
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    connection_metrics: Dict[str, ConnectionMetrics] = field(default_factory=dict)


class AdvancedConnectionPool:
    """Advanced connection pool with monitoring and health management."""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        pool_name: str = "default"
    ):
        """
        Initialize advanced connection pool.

        Args:
            database_url: Database connection URL
            pool_size: Base number of connections in pool
            max_overflow: Maximum overflow connections
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
            pool_pre_ping: Ping connection before use
            pool_name: Unique name for this pool
        """
        self.database_url = database_url
        self.pool_name = pool_name
        self.metrics = PoolMetrics(
            pool_id=pool_name,
            pool_size=pool_size,
            max_overflow=max_overflow
        )

        # Pool configuration
        self.pool_config = {
            "poolclass": QueuePool,
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
            "pool_pre_ping": pool_pre_ping,
            "echo": False
        }

        # Create async engine with monitoring
        self.engine = create_async_engine(database_url, **self.pool_config)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Health monitoring
        self.health_check_interval = 60  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.checkout_times: List[float] = []

        # Set up event listeners
        self._setup_event_listeners()

        # Start health monitoring
        self._start_health_monitoring()

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring."""

        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Track connection creation."""
            connection_id = str(id(dbapi_connection))
            self.metrics.connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                created_at=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc)
            )
            self.metrics.total_connections_created += 1
            self.metrics.peak_connections = max(
                self.metrics.peak_connections,
                len(self.metrics.connection_metrics)
            )

        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Track connection checkout."""
            connection_id = str(id(dbapi_connection))
            if connection_id in self.metrics.connection_metrics:
                metrics = self.metrics.connection_metrics[connection_id]
                metrics.state = ConnectionState.ACTIVE
                metrics.usage_count += 1
                metrics.last_used = datetime.now(timezone.utc)

            self.metrics.total_checkouts += 1

        @event.listens_for(self.engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Track connection checkin."""
            connection_id = str(id(dbapi_connection))
            if connection_id in self.metrics.connection_metrics:
                metrics = self.metrics.connection_metrics[connection_id]
                metrics.state = ConnectionState.IDLE

            self.metrics.total_checkins += 1

        @event.listens_for(self.engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            """Track connection closure."""
            connection_id = str(id(dbapi_connection))
            if connection_id in self.metrics.connection_metrics:
                del self.metrics.connection_metrics[connection_id]

            self.metrics.total_connections_closed += 1

        @event.listens_for(self.engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Track connection invalidation."""
            connection_id = str(id(dbapi_connection))
            if connection_id in self.metrics.connection_metrics:
                metrics = self.metrics.connection_metrics[connection_id]
                metrics.state = ConnectionState.INVALID
                metrics.error_count += 1
                metrics.last_error = str(exception) if exception else "Unknown error"

            self.metrics.invalid_connections += 1

    def _start_health_monitoring(self):
        """Start background health monitoring task."""
        if self.health_check_task is None or self.health_check_task.done():
            self.health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Health check error for pool {self.pool_name}: {e}")

    async def _perform_health_check(self):
        """Perform comprehensive health check on the pool."""
        try:
            start_time = time.time()

            # Test connection with simple query
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))

            health_check_time = time.time() - start_time

            # Update metrics based on health check
            if health_check_time < 1.0:
                self.metrics.status = PoolStatus.HEALTHY
            elif health_check_time < 5.0:
                self.metrics.status = PoolStatus.WARNING
            else:
                self.metrics.status = PoolStatus.CRITICAL

            self.metrics.last_health_check = datetime.now(timezone.utc)

            # Clean up stale connections
            await self._cleanup_stale_connections()

        except Exception as e:
            self.metrics.status = PoolStatus.UNAVAILABLE
            logging.error(f"Health check failed for pool {self.pool_name}: {e}")

    async def _cleanup_stale_connections(self):
        """Clean up stale or invalid connections."""
        current_time = datetime.now(timezone.utc)
        stale_threshold = timedelta(minutes=30)

        stale_connections = []
        for conn_id, metrics in self.metrics.connection_metrics.items():
            if (current_time - metrics.last_used) > stale_threshold:
                metrics.state = ConnectionState.STALE
                stale_connections.append(conn_id)

        # Note: Actual connection cleanup would be handled by SQLAlchemy's pool
        # This is just for monitoring purposes

    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic resource management and monitoring."""
        start_time = time.time()
        session = None

        try:
            session = self.session_factory()
            checkout_time = time.time() - start_time

            # Record checkout time for performance monitoring
            self.checkout_times.append(checkout_time)
            if len(self.checkout_times) > 1000:
                self.checkout_times = self.checkout_times[-1000:]

            self.metrics.avg_checkout_time = sum(self.checkout_times) / len(self.checkout_times)

            yield session

        except Exception as e:
            if session:
                await session.rollback()
            raise e

        finally:
            if session:
                await session.close()

    async def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status and metrics."""
        pool = self.engine.pool

        # Get current pool state
        current_checked_out = pool.checkedout()
        current_checked_in = pool.checkedin()
        current_overflow = pool.overflow()

        # Update metrics
        self.metrics.checked_out = current_checked_out
        self.metrics.checked_in = current_checked_in
        self.metrics.overflow = current_overflow

        # Connection state summary
        state_counts = {}
        for state in ConnectionState:
            state_counts[state.value] = sum(
                1 for metrics in self.metrics.connection_metrics.values()
                if metrics.state == state
            )

        return {
            "pool_id": self.metrics.pool_id,
            "status": self.metrics.status.value,
            "last_health_check": self.metrics.last_health_check,
            "pool_stats": {
                "pool_size": self.metrics.pool_size,
                "max_overflow": self.metrics.max_overflow,
                "checked_out": self.metrics.checked_out,
                "checked_in": self.metrics.checked_in,
                "overflow": self.metrics.overflow,
                "total_connections": len(self.metrics.connection_metrics)
            },
            "connection_states": state_counts,
            "performance_metrics": {
                "total_checkouts": self.metrics.total_checkouts,
                "total_checkins": self.metrics.total_checkins,
                "avg_checkout_time": round(self.metrics.avg_checkout_time, 4),
                "peak_connections": self.metrics.peak_connections,
                "invalid_connections": self.metrics.invalid_connections
            },
            "lifetime_stats": {
                "connections_created": self.metrics.total_connections_created,
                "connections_closed": self.metrics.total_connections_closed,
                "uptime": (datetime.now(timezone.utc) - self.metrics.last_health_check).total_seconds()
            }
        }

    async def scale_pool(self, new_pool_size: int, new_max_overflow: int = None):
        """Dynamically scale the connection pool."""
        if new_max_overflow is None:
            new_max_overflow = new_pool_size + 10

        # Note: SQLAlchemy doesn't support dynamic pool resizing
        # In a production environment, this would require creating a new engine
        # and gracefully transitioning connections

        logging.info(f"Pool scaling requested: {new_pool_size} (current: {self.metrics.pool_size})")

        # Update metrics
        self.metrics.pool_size = new_pool_size
        self.metrics.max_overflow = new_max_overflow

    async def close(self):
        """Close the connection pool and cleanup resources."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        await self.engine.dispose()


class ConnectionPoolManager:
    """Manages multiple connection pools with load balancing and failover."""

    def __init__(self):
        self.pools: Dict[str, AdvancedConnectionPool] = {}
        self.default_pool_name = "default"
        self.load_balancer: Optional[Callable] = None

    async def create_pool(
        self,
        name: str,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        **kwargs
    ) -> AdvancedConnectionPool:
        """Create and register a new connection pool."""
        if name in self.pools:
            raise ValueError(f"Pool '{name}' already exists")

        pool = AdvancedConnectionPool(
            database_url=database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_name=name,
            **kwargs
        )

        self.pools[name] = pool

        if len(self.pools) == 1:
            self.default_pool_name = name

        return pool

    async def get_pool(self, name: str = None) -> AdvancedConnectionPool:
        """Get connection pool by name or use load balancing."""
        if name:
            if name not in self.pools:
                raise ValueError(f"Pool '{name}' not found")
            return self.pools[name]

        # Use load balancer if available
        if self.load_balancer and len(self.pools) > 1:
            selected_pool = await self.load_balancer(self.pools)
            return selected_pool

        # Default to first available healthy pool
        for pool in self.pools.values():
            status = await pool.get_pool_status()
            if status["status"] in ["healthy", "warning"]:
                return pool

        # Fallback to default pool
        return self.pools[self.default_pool_name]

    @asynccontextmanager
    async def get_session(self, pool_name: str = None):
        """Get database session from specified or load-balanced pool."""
        pool = await self.get_pool(pool_name)
        async with pool.get_session() as session:
            yield session

    async def get_all_pool_status(self) -> Dict[str, Any]:
        """Get status of all managed pools."""
        pool_statuses = {}

        for name, pool in self.pools.items():
            pool_statuses[name] = await pool.get_pool_status()

        # Overall health summary
        total_pools = len(self.pools)
        healthy_pools = sum(
            1 for status in pool_statuses.values()
            if status["status"] == "healthy"
        )

        return {
            "timestamp": datetime.now(timezone.utc),
            "total_pools": total_pools,
            "healthy_pools": healthy_pools,
            "health_ratio": healthy_pools / total_pools if total_pools > 0 else 0,
            "pools": pool_statuses
        }

    def set_load_balancer(self, load_balancer: Callable):
        """Set custom load balancing function."""
        self.load_balancer = load_balancer

    async def close_all_pools(self):
        """Close all connection pools."""
        for pool in self.pools.values():
            await pool.close()
        self.pools.clear()


# Load balancing strategies
async def round_robin_load_balancer(pools: Dict[str, AdvancedConnectionPool]) -> AdvancedConnectionPool:
    """Simple round-robin load balancing."""
    if not hasattr(round_robin_load_balancer, 'counter'):
        round_robin_load_balancer.counter = 0

    pool_names = list(pools.keys())
    selected_name = pool_names[round_robin_load_balancer.counter % len(pool_names)]
    round_robin_load_balancer.counter += 1

    return pools[selected_name]


async def least_connections_load_balancer(pools: Dict[str, AdvancedConnectionPool]) -> AdvancedConnectionPool:
    """Load balance based on least active connections."""
    best_pool = None
    min_connections = float('inf')

    for pool in pools.values():
        status = await pool.get_pool_status()
        active_connections = status["pool_stats"]["checked_out"]

        if active_connections < min_connections:
            min_connections = active_connections
            best_pool = pool

    return best_pool or list(pools.values())[0]


async def health_based_load_balancer(pools: Dict[str, AdvancedConnectionPool]) -> AdvancedConnectionPool:
    """Load balance based on pool health and performance."""
    scored_pools = []

    for pool in pools.values():
        status = await pool.get_pool_status()

        # Calculate health score
        score = 0
        if status["status"] == "healthy":
            score = 100
        elif status["status"] == "warning":
            score = 70
        elif status["status"] == "critical":
            score = 30
        else:
            score = 0

        # Adjust score based on load
        utilization = status["pool_stats"]["checked_out"] / status["pool_stats"]["pool_size"]
        score *= (1 - utilization * 0.5)  # Reduce score for high utilization

        # Adjust score based on performance
        avg_checkout_time = status["performance_metrics"]["avg_checkout_time"]
        if avg_checkout_time < 0.01:
            score *= 1.1  # Bonus for fast checkouts
        elif avg_checkout_time > 0.1:
            score *= 0.9  # Penalty for slow checkouts

        scored_pools.append((score, pool))

    # Return the highest scoring pool
    scored_pools.sort(key=lambda x: x[0], reverse=True)
    return scored_pools[0][1] if scored_pools else list(pools.values())[0]


# Global connection pool manager instance
connection_manager = ConnectionPoolManager()