"""
Query caching service for database performance optimization.
Implements multi-level caching with TTL, invalidation, and performance monitoring.
"""
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
import gzip

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


class CacheLevel(Enum):
    """Cache levels for different performance requirements."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class CachePolicy(Enum):
    """Cache policies for different data types."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0
    compression: bool = False

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:  # Never expires
            return False
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() > self.ttl_seconds

    def access(self):
        """Record access to this cache entry."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate: float = 0.0

    def calculate_hit_rate(self):
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class QueryCacheService:
    """Advanced query caching service with multiple strategies."""

    def __init__(self, max_memory_size: int = 100 * 1024 * 1024, enable_compression: bool = True):
        """
        Initialize query cache service.

        Args:
            max_memory_size: Maximum memory cache size in bytes (default 100MB)
            enable_compression: Whether to compress large cache entries
        """
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_size = max_memory_size
        self.enable_compression = enable_compression
        self.stats = CacheStats()
        self.access_times: List[float] = []

    def _generate_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate consistent cache key for query and parameters."""
        # Normalize query by removing extra whitespace
        normalized_query = " ".join(query.split())

        # Include parameters in key if provided
        if params:
            # Sort parameters for consistent hashing
            sorted_params = json.dumps(params, sort_keys=True, default=str)
            cache_data = f"{normalized_query}:{sorted_params}"
        else:
            cache_data = normalized_query

        # Generate hash
        return hashlib.sha256(cache_data.encode()).hexdigest()[:32]

    def _serialize_value(self, value: Any, use_compression: bool = False) -> bytes:
        """Serialize and optionally compress cache value."""
        serialized = pickle.dumps(value)

        if use_compression and len(serialized) > 1024:  # Compress if > 1KB
            serialized = gzip.compress(serialized)

        return serialized

    def _deserialize_value(self, data: bytes, is_compressed: bool = False) -> Any:
        """Deserialize and optionally decompress cache value."""
        if is_compressed:
            data = gzip.decompress(data)

        return pickle.loads(data)

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(self._serialize_value(value))
        except Exception:
            # Fallback estimation
            return len(str(value).encode())

    def _evict_if_needed(self, new_entry_size: int):
        """Evict cache entries if memory limit would be exceeded."""
        current_size = sum(entry.size_bytes for entry in self.memory_cache.values())

        if current_size + new_entry_size <= self.max_memory_size:
            return

        # LRU eviction strategy
        entries_by_access = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )

        for key, entry in entries_by_access:
            del self.memory_cache[key]
            self.stats.evictions += 1
            current_size -= entry.size_bytes

            if current_size + new_entry_size <= self.max_memory_size:
                break

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()

        try:
            if key in self.memory_cache:
                entry = self.memory_cache[key]

                if entry.is_expired():
                    del self.memory_cache[key]
                    self.stats.misses += 1
                    return None

                entry.access()
                self.stats.hits += 1
                return entry.value
            else:
                self.stats.misses += 1
                return None

        finally:
            access_time = (time.time() - start_time) * 1000  # Convert to ms
            self.access_times.append(access_time)

            # Keep only recent access times for moving average
            if len(self.access_times) > 1000:
                self.access_times = self.access_times[-1000:]

            self.stats.avg_access_time_ms = sum(self.access_times) / len(self.access_times)

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            size_bytes = self._calculate_size(value)
            use_compression = self.enable_compression and size_bytes > 1024

            # Evict if needed
            self._evict_if_needed(size_bytes)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                compression=use_compression
            )

            self.memory_cache[key] = entry
            self.stats.sets += 1
            self.stats.total_size_bytes = sum(e.size_bytes for e in self.memory_cache.values())

            return True

        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
            self.stats.deletes += 1
            self.stats.total_size_bytes = sum(e.size_bytes for e in self.memory_cache.values())
            return True
        return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        cleared_count = len(self.memory_cache)
        self.memory_cache.clear()
        self.stats.deletes += cleared_count
        self.stats.total_size_bytes = 0
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        self.stats.calculate_hit_rate()

        return {
            "cache_stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": round(self.stats.hit_rate, 4),
                "sets": self.stats.sets,
                "deletes": self.stats.deletes,
                "evictions": self.stats.evictions
            },
            "memory_stats": {
                "total_size_bytes": self.stats.total_size_bytes,
                "total_size_mb": round(self.stats.total_size_bytes / (1024 * 1024), 2),
                "max_size_mb": round(self.max_memory_size / (1024 * 1024), 2),
                "utilization": round(self.stats.total_size_bytes / self.max_memory_size, 4),
                "entry_count": len(self.memory_cache)
            },
            "performance_stats": {
                "avg_access_time_ms": round(self.stats.avg_access_time_ms, 4),
                "compression_enabled": self.enable_compression
            }
        }

    async def cached_query(self, session: AsyncSession, query: str, params: Dict[str, Any] = None, ttl_seconds: int = 3600) -> Any:
        """Execute query with automatic caching."""
        cache_key = self._generate_cache_key(query, params)

        # Try to get from cache first
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute query if not in cache
        if params:
            result = await session.execute(text(query), params)
        else:
            result = await session.execute(text(query))

        # Convert result to cacheable format
        if hasattr(result, 'all'):
            query_result = [dict(row._mapping) for row in result.all()]
        elif hasattr(result, 'scalar'):
            query_result = result.scalar()
        else:
            query_result = result

        # Cache the result
        await self.set(cache_key, query_result, ttl_seconds)

        return query_result


class SmartQueryOptimizer:
    """Smart query optimizer with caching and performance monitoring."""

    def __init__(self, cache_service: QueryCacheService):
        self.cache = cache_service
        self.query_performance: Dict[str, List[float]] = {}
        self.optimization_rules: List[Callable] = []

    def add_optimization_rule(self, rule: Callable[[str, Dict[str, Any]], str]):
        """Add custom query optimization rule."""
        self.optimization_rules.append(rule)

    def _optimize_query(self, query: str, params: Dict[str, Any] = None) -> str:
        """Apply optimization rules to query."""
        optimized_query = query

        for rule in self.optimization_rules:
            try:
                optimized_query = rule(optimized_query, params or {})
            except Exception as e:
                print(f"Optimization rule failed: {e}")

        return optimized_query

    def _record_performance(self, query_signature: str, execution_time: float):
        """Record query performance metrics."""
        if query_signature not in self.query_performance:
            self.query_performance[query_signature] = []

        self.query_performance[query_signature].append(execution_time)

        # Keep only recent measurements
        if len(self.query_performance[query_signature]) > 100:
            self.query_performance[query_signature] = self.query_performance[query_signature][-100:]

    async def execute_optimized_query(self, session: AsyncSession, query: str, params: Dict[str, Any] = None, cache_ttl: int = 3600) -> Any:
        """Execute query with optimization and caching."""
        start_time = time.time()

        # Apply optimizations
        optimized_query = self._optimize_query(query, params)

        # Execute with caching
        result = await self.cache.cached_query(session, optimized_query, params, cache_ttl)

        # Record performance
        execution_time = time.time() - start_time
        query_signature = self.cache._generate_cache_key(optimized_query)[:8]
        self._record_performance(query_signature, execution_time)

        return result

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "query_count": len(self.query_performance),
            "queries": {}
        }

        for signature, times in self.query_performance.items():
            if times:
                report["queries"][signature] = {
                    "executions": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "recent_avg": sum(times[-10:]) / min(len(times), 10)
                }

        return report


class DatabaseCacheManager:
    """Comprehensive database cache manager."""

    def __init__(self):
        self.query_cache = QueryCacheService()
        self.optimizer = SmartQueryOptimizer(self.query_cache)
        self._setup_default_optimizations()

    def _setup_default_optimizations(self):
        """Set up default query optimizations."""

        def add_limit_if_missing(query: str, params: Dict[str, Any]) -> str:
            """Add LIMIT clause to SELECT queries without one."""
            query_upper = query.upper().strip()
            if (query_upper.startswith("SELECT") and
                "LIMIT" not in query_upper and
                "COUNT" not in query_upper):
                return f"{query} LIMIT 1000"
            return query

        def optimize_order_by(query: str, params: Dict[str, Any]) -> str:
            """Optimize ORDER BY clauses."""
            # Add index hints for common patterns
            if "ORDER BY created_at DESC" in query:
                query = query.replace(
                    "ORDER BY created_at DESC",
                    "ORDER BY created_at DESC NULLS LAST"
                )
            return query

        def add_query_hints(query: str, params: Dict[str, Any]) -> str:
            """Add database-specific query hints."""
            # For PostgreSQL, add hints for large result sets
            if "SELECT * FROM" in query and "JOIN" in query:
                query = f"/* Use index scan */ {query}"
            return query

        self.optimizer.add_optimization_rule(add_limit_if_missing)
        self.optimizer.add_optimization_rule(optimize_order_by)
        self.optimizer.add_optimization_rule(add_query_hints)

    async def get_cached_job_queue(self, session: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cached job queue with optimization."""
        query = """
            SELECT id, website_id, user_id, job_type, priority, status, created_at
            FROM crawling_jobs
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT :limit
        """

        return await self.optimizer.execute_optimized_query(
            session, query, {"limit": limit}, cache_ttl=60  # 1 minute cache
        )

    async def get_cached_dashboard_stats(self, session: AsyncSession, website_id: str) -> Dict[str, Any]:
        """Get cached dashboard statistics."""
        query = """
            SELECT
                status,
                COUNT(*) as count,
                AVG(priority) as avg_priority,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM crawling_jobs
            WHERE website_id = :website_id
            GROUP BY status
        """

        stats = await self.optimizer.execute_optimized_query(
            session, query, {"website_id": website_id}, cache_ttl=300  # 5 minute cache
        )

        # Transform to dashboard format
        dashboard_stats = {
            "total_jobs": sum(row["count"] for row in stats),
            "status_breakdown": {
                row["status"]: {
                    "count": row["count"],
                    "avg_priority": float(row["avg_priority"]) if row["avg_priority"] else 0
                }
                for row in stats
            }
        }

        return dashboard_stats

    async def get_cached_analytics_data(self, session: AsyncSession, period_type: str = "daily", days: int = 30) -> List[Dict[str, Any]]:
        """Get cached analytics data."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        query = """
            SELECT
                period_start,
                period_type,
                total_jobs,
                successful_jobs,
                failed_jobs,
                average_execution_time_seconds
            FROM crawling_analytics
            WHERE period_type = :period_type
                AND period_start >= :start_date
                AND period_start <= :end_date
            ORDER BY period_start DESC
        """

        return await self.optimizer.execute_optimized_query(
            session,
            query,
            {
                "period_type": period_type,
                "start_date": start_date,
                "end_date": end_date
            },
            cache_ttl=1800  # 30 minute cache
        )

    async def get_cached_error_summary(self, session: AsyncSession, hours: int = 24) -> Dict[str, Any]:
        """Get cached error summary."""
        since_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        query = """
            SELECT
                error_type,
                COUNT(*) as count,
                COUNT(DISTINCT job_id) as affected_jobs,
                AVG(retry_count) as avg_retries
            FROM crawling_errors
            WHERE occurred_at >= :since_time
            GROUP BY error_type
            ORDER BY count DESC
        """

        errors = await self.optimizer.execute_optimized_query(
            session, query, {"since_time": since_time}, cache_ttl=600  # 10 minute cache
        )

        return {
            "total_errors": sum(row["count"] for row in errors),
            "error_types": [
                {
                    "type": row["error_type"],
                    "count": row["count"],
                    "affected_jobs": row["affected_jobs"],
                    "avg_retries": float(row["avg_retries"]) if row["avg_retries"] else 0
                }
                for row in errors
            ]
        }

    async def invalidate_cache_for_website(self, website_id: str):
        """Invalidate all cache entries related to a specific website."""
        # In a real implementation, this would use cache tagging
        # For now, we'll clear related patterns
        keys_to_delete = []

        for key in self.query_cache.memory_cache.keys():
            # Check if key likely contains website-related data
            if website_id in key or "dashboard" in key:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            await self.query_cache.delete(key)

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache and optimization statistics."""
        cache_stats = await self.query_cache.get_stats()
        performance_report = self.optimizer.get_performance_report()

        return {
            "timestamp": datetime.now(timezone.utc),
            "cache": cache_stats,
            "performance": performance_report,
            "optimization_rules": len(self.optimizer.optimization_rules)
        }