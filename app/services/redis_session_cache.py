"""
Redis Session Caching Service for high-performance session management.
Provides fast access to active session data and context.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime, timezone, timedelta
import redis.asyncio as redis

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisSessionCache:
    """
    Redis-based caching service for session data and context optimization.
    """
    
    # Cache key prefixes
    SESSION_PREFIX = "session:"
    CONTEXT_PREFIX = "context:"
    ANALYTICS_PREFIX = "analytics:"
    IMPORTANCE_PREFIX = "importance:"
    
    # Default TTL values (in seconds)
    SESSION_TTL = 86400 * 7  # 7 days (matches session duration)
    CONTEXT_TTL = 3600       # 1 hour
    ANALYTICS_TTL = 1800     # 30 minutes
    
    def __init__(self):
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """
        Initialize Redis connection.
        """
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info("✅ REDIS: Initialized Redis connection")
        except Exception as e:
            logger.warning(f"⚠️ REDIS: Failed to initialize Redis: {e}")
            self.redis_client = None
    
    async def is_available(self) -> bool:
        """
        Check if Redis is available.
        """
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"⚠️ REDIS: Connection check failed: {e}")
            return False
    
    async def cache_session(
        self,
        session_token: str,
        session_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache session data in Redis.
        
        Args:
            session_token: Session token key
            session_data: Session data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            key = f"{self.SESSION_PREFIX}{session_token}"
            
            # Add cache metadata
            cached_data = {
                **session_data,
                '_cached_at': datetime.now(timezone.utc).isoformat(),
                '_cache_version': '1.0'
            }
            
            await self.redis_client.set(
                key,
                json.dumps(cached_data, default=str),
                ex=ttl or self.SESSION_TTL
            )
            
            logger.debug(f"📦 REDIS: Cached session {session_token[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to cache session: {e}")
            return False
    
    async def get_cached_session(
        self,
        session_token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached session data from Redis.
        
        Args:
            session_token: Session token key
            
        Returns:
            Cached session data or None if not found
        """
        if not await self.is_available():
            return None
        
        try:
            key = f"{self.SESSION_PREFIX}{session_token}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                session_data = json.loads(cached_data)
                
                # Remove cache metadata
                session_data.pop('_cached_at', None)
                session_data.pop('_cache_version', None)
                
                logger.debug(f"📦 REDIS: Retrieved cached session {session_token[:20]}...")
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to get cached session: {e}")
            return None
    
    async def cache_context(
        self,
        conversation_id: UUID,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache conversation context for quick access.
        
        Args:
            conversation_id: Conversation UUID
            context_data: Context data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            key = f"{self.CONTEXT_PREFIX}{conversation_id}"
            
            # Optimize context data for caching
            cached_context = {
                'messages': context_data.get('messages', [])[-10:],  # Keep only recent messages
                'summary': context_data.get('context_summary'),
                'total_tokens': context_data.get('total_tokens', 0),
                'messages_included': context_data.get('messages_included', 0),
                'last_optimized': context_data.get('last_optimized'),
                '_cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_client.set(
                key,
                json.dumps(cached_context, default=str),
                ex=ttl or self.CONTEXT_TTL
            )
            
            logger.debug(f"📦 REDIS: Cached context for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to cache context: {e}")
            return False
    
    async def get_cached_context(
        self,
        conversation_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached conversation context.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Cached context data or None if not found
        """
        if not await self.is_available():
            return None
        
        try:
            key = f"{self.CONTEXT_PREFIX}{conversation_id}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                context_data = json.loads(cached_data)
                context_data.pop('_cached_at', None)
                
                logger.debug(f"📦 REDIS: Retrieved cached context for {conversation_id}")
                return context_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to get cached context: {e}")
            return None
    
    async def cache_importance_scores(
        self,
        conversation_id: UUID,
        scores: Dict[str, float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache message importance scores.
        
        Args:
            conversation_id: Conversation UUID
            scores: Message ID to importance score mapping
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            key = f"{self.IMPORTANCE_PREFIX}{conversation_id}"
            
            cached_scores = {
                'scores': scores,
                '_cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_client.set(
                key,
                json.dumps(cached_scores, default=str),
                ex=ttl or self.CONTEXT_TTL
            )
            
            logger.debug(f"📦 REDIS: Cached importance scores for {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to cache importance scores: {e}")
            return False
    
    async def get_cached_importance_scores(
        self,
        conversation_id: UUID
    ) -> Optional[Dict[str, float]]:
        """
        Get cached importance scores.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Cached importance scores or None if not found
        """
        if not await self.is_available():
            return None
        
        try:
            key = f"{self.IMPORTANCE_PREFIX}{conversation_id}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data.get('scores', {})
            
            return None
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to get cached importance scores: {e}")
            return None
    
    async def invalidate_session(self, session_token: str) -> bool:
        """
        Remove session from cache.
        
        Args:
            session_token: Session token to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            key = f"{self.SESSION_PREFIX}{session_token}"
            await self.redis_client.delete(key)
            
            logger.debug(f"🗑️ REDIS: Invalidated session {session_token[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to invalidate session: {e}")
            return False
    
    async def invalidate_context(self, conversation_id: UUID) -> bool:
        """
        Remove context from cache.
        
        Args:
            conversation_id: Conversation UUID to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            keys = [
                f"{self.CONTEXT_PREFIX}{conversation_id}",
                f"{self.IMPORTANCE_PREFIX}{conversation_id}"
            ]
            
            await self.redis_client.delete(*keys)
            
            logger.debug(f"🗑️ REDIS: Invalidated context for {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to invalidate context: {e}")
            return False
    
    async def cache_analytics(
        self,
        key: str,
        analytics_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache analytics data for performance.
        
        Args:
            key: Analytics cache key
            analytics_data: Analytics data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_available():
            return False
        
        try:
            cache_key = f"{self.ANALYTICS_PREFIX}{key}"
            
            cached_data = {
                **analytics_data,
                '_cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_client.set(
                cache_key,
                json.dumps(cached_data, default=str),
                ex=ttl or self.ANALYTICS_TTL
            )
            
            logger.debug(f"📦 REDIS: Cached analytics for {key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to cache analytics: {e}")
            return False
    
    async def get_cached_analytics(
        self,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached analytics data.
        
        Args:
            key: Analytics cache key
            
        Returns:
            Cached analytics data or None if not found
        """
        if not await self.is_available():
            return None
        
        try:
            cache_key = f"{self.ANALYTICS_PREFIX}{key}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                data.pop('_cached_at', None)
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to get cached analytics: {e}")
            return None
    
    async def warm_session_cache(
        self,
        session_tokens: List[str]
    ) -> int:
        """
        Pre-warm cache with frequently accessed sessions.
        
        Args:
            session_tokens: List of session tokens to warm
            
        Returns:
            Number of sessions successfully warmed
        """
        if not await self.is_available():
            return 0
        
        warmed_count = 0
        
        # This would typically fetch session data from database and cache it
        # Implementation depends on specific session storage mechanism
        
        logger.info(f"🔥 REDIS: Warmed {warmed_count} sessions in cache")
        return warmed_count
    
    async def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        if not await self.is_available():
            return 0
        
        cleaned_count = 0
        
        try:
            # Get all cache keys
            all_keys = []
            for prefix in [self.SESSION_PREFIX, self.CONTEXT_PREFIX, self.ANALYTICS_PREFIX]:
                pattern = f"{prefix}*"
                async for key in self.redis_client.scan_iter(match=pattern):
                    all_keys.append(key)
            
            # Check TTL and remove expired ones
            for key in all_keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    continue
                elif ttl == -2:  # Key doesn't exist
                    cleaned_count += 1
            
            logger.info(f"🧹 REDIS: Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to cleanup expired cache: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        if not await self.is_available():
            return {'available': False, 'error': 'Redis not available'}
        
        try:
            # Get Redis info
            info = await self.redis_client.info()
            
            # Count keys by prefix
            key_counts = {}
            for prefix in [self.SESSION_PREFIX, self.CONTEXT_PREFIX, self.ANALYTICS_PREFIX, self.IMPORTANCE_PREFIX]:
                pattern = f"{prefix}*"
                count = 0
                async for key in self.redis_client.scan_iter(match=pattern):
                    count += 1
                key_counts[prefix.rstrip(':')] = count
            
            return {
                'available': True,
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'key_counts': key_counts,
                'total_keys': sum(key_counts.values())
            }
            
        except Exception as e:
            logger.error(f"❌ REDIS: Failed to get cache stats: {e}")
            return {'available': False, 'error': str(e)}
    
    async def close(self):
        """
        Close Redis connection.
        """
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔌 REDIS: Connection closed")