"""
Optimized Widget Configuration Service
Provides high-performance configuration management with caching, versioning, and atomic operations.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
import redis.asyncio as redis
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from supabase import Client

from ..core.database import get_supabase_admin_client


logger = logging.getLogger(__name__)


@dataclass
class ConfigUpdateResult:
    """Result of a configuration update operation."""
    success: bool
    new_version: int
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    cached: bool = False


@dataclass
class PerformanceMetrics:
    """Performance tracking for configuration operations."""
    operation: str
    duration_ms: float
    cache_hit: bool
    db_queries: int
    bytes_transferred: int


class OptimizedWidgetConfigService:
    """
    High-performance widget configuration service with caching and optimization.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.cache_prefix = "widget_config:"
        self.version_cache_prefix = "widget_version:"
        self.performance_metrics: List[PerformanceMetrics] = []
        
    async def get_widget_configuration(
        self, 
        widget_id: str, 
        version: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Get widget configuration with optimal caching and performance.
        
        Returns:
            Tuple of (config_data, is_cached)
        """
        start_time = asyncio.get_event_loop().time()
        cache_hit = False
        
        try:
            # Try cache first if enabled
            if use_cache and self.redis_client:
                cached_config = await self._get_from_cache(widget_id, version)
                if cached_config:
                    cache_hit = True
                    await self._record_performance_metric(
                        "get_config_cached",
                        (asyncio.get_event_loop().time() - start_time) * 1000,
                        cache_hit=True,
                        db_queries=0,
                        bytes_transferred=len(json.dumps(cached_config))
                    )
                    return cached_config, True
            
            # Fetch from database using optimized function
            supabase = get_supabase_admin_client()
            result = supabase.rpc('get_widget_config_optimized', {'p_widget_id': widget_id}).execute()
            
            if not result.data:
                logger.warning(f"Widget configuration not found: {widget_id}")
                return None, False
            
            widget_data = result.data[0]
            
            # Check if widget is active and verified
            if not widget_data.get('is_active') or widget_data.get('verification_status') != 'verified':
                logger.warning(f"Widget not active or verified: {widget_id}")
                return None, False
            
            # Build structured configuration
            config = self._build_config_response(widget_data, widget_id)
            
            # Cache the result
            if self.redis_client:
                await self._set_cache(widget_id, config, widget_data.get('config_version'))
            
            await self._record_performance_metric(
                "get_config_db",
                (asyncio.get_event_loop().time() - start_time) * 1000,
                cache_hit=False,
                db_queries=1,
                bytes_transferred=len(json.dumps(config))
            )
            
            return config, False
            
        except Exception as e:
            logger.error(f"Error getting widget configuration for {widget_id}: {str(e)}")
            await self._record_performance_metric(
                "get_config_error",
                (asyncio.get_event_loop().time() - start_time) * 1000,
                cache_hit=cache_hit,
                db_queries=1,
                bytes_transferred=0
            )
            raise
    
    async def update_widget_configuration(
        self,
        website_id: str,
        user_id: str,
        config_updates: Dict[str, Any],
        expected_version: Optional[int] = None,
        create_backup: bool = True
    ) -> ConfigUpdateResult:
        """
        Atomically update widget configuration with version control.
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            supabase = get_supabase_admin_client()
            
            # Get current configuration for backup if needed
            if create_backup:
                current_config = await self._get_current_config_for_backup(website_id, supabase)
                if current_config and expected_version is None:
                    expected_version = current_config.get('config_version', 1)
            
            # Validate configuration updates
            validation_errors = self._validate_config_updates(config_updates)
            if validation_errors:
                return ConfigUpdateResult(
                    success=False,
                    new_version=0,
                    error_message=f"Validation errors: {', '.join(validation_errors)}"
                )
            
            # Perform atomic update using database function
            update_params = {
                'p_website_id': website_id,
                'p_expected_version': expected_version or 1,
                **{f'p_{key}': value for key, value in self._map_config_updates(config_updates).items()}
            }
            
            result = supabase.rpc('update_widget_config_atomic', update_params).execute()
            
            if not result.data:
                return ConfigUpdateResult(
                    success=False,
                    new_version=0,
                    error_message="Failed to execute atomic update"
                )
            
            update_result = result.data[0]
            
            if not update_result['success']:
                return ConfigUpdateResult(
                    success=False,
                    new_version=update_result['new_version'],
                    error_message=update_result['error_message']
                )
            
            # Create version backup if requested
            if create_backup and current_config:
                await self._create_version_backup(
                    website_id, 
                    current_config, 
                    user_id, 
                    f"Backup before update to v{update_result['new_version']}",
                    supabase
                )
            
            # Invalidate cache
            if self.redis_client:
                await self._invalidate_widget_cache(website_id)
            
            # Calculate new configuration checksum
            new_checksum = self._calculate_config_checksum(config_updates)
            
            await self._record_performance_metric(
                "update_config_atomic",
                (asyncio.get_event_loop().time() - start_time) * 1000,
                cache_hit=False,
                db_queries=2 if create_backup else 1,
                bytes_transferred=len(json.dumps(config_updates))
            )
            
            logger.info(f"Successfully updated widget config for website {website_id} to version {update_result['new_version']}")
            
            return ConfigUpdateResult(
                success=True,
                new_version=update_result['new_version'],
                checksum=new_checksum
            )
            
        except IntegrityError as e:
            logger.error(f"Integrity error updating config for {website_id}: {str(e)}")
            return ConfigUpdateResult(
                success=False,
                new_version=0,
                error_message="Configuration conflict - please refresh and try again"
            )
        except Exception as e:
            logger.error(f"Error updating widget configuration for {website_id}: {str(e)}")
            return ConfigUpdateResult(
                success=False,
                new_version=0,
                error_message=f"Internal error: {str(e)}"
            )
    
    async def get_configuration_versions(
        self, 
        website_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get configuration version history with pagination."""
        try:
            supabase = get_supabase_admin_client()
            
            result = supabase.table('widget_configuration_versions')\
                .select('*')\
                .eq('website_id', website_id)\
                .order('created_at', desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error getting configuration versions for {website_id}: {str(e)}")
            return []
    
    async def rollback_to_version(
        self,
        website_id: str,
        version_id: str,
        user_id: str
    ) -> ConfigUpdateResult:
        """
        Rollback configuration to a specific version with atomic operations.
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            supabase = get_supabase_admin_client()
            
            # Get the target version
            version_result = supabase.table('widget_configuration_versions')\
                .select('*')\
                .eq('id', version_id)\
                .eq('website_id', website_id)\
                .execute()
            
            if not version_result.data:
                return ConfigUpdateResult(
                    success=False,
                    new_version=0,
                    error_message="Version not found"
                )
            
            version_data = version_result.data[0]
            target_config = version_data['configuration_data']
            
            # Get current configuration for backup
            current_config = await self._get_current_config_for_backup(website_id, supabase)
            
            # Perform rollback update
            rollback_updates = self._flatten_config_for_db(target_config)
            update_result = await self.update_widget_configuration(
                website_id=website_id,
                user_id=user_id,
                config_updates=rollback_updates,
                expected_version=current_config.get('config_version') if current_config else None,
                create_backup=True
            )
            
            if update_result.success:
                # Log the rollback action
                await self._log_audit_action(
                    website_id,
                    'rolled_back',
                    current_config,
                    target_config,
                    user_id,
                    f"Rollback to version {version_data['version_number']}",
                    supabase
                )
                
                logger.info(f"Successfully rolled back website {website_id} to version {version_data['version_number']}")
            
            await self._record_performance_metric(
                "rollback_config",
                (asyncio.get_event_loop().time() - start_time) * 1000,
                cache_hit=False,
                db_queries=3,
                bytes_transferred=len(json.dumps(target_config))
            )
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error rolling back configuration for {website_id}: {str(e)}")
            return ConfigUpdateResult(
                success=False,
                new_version=0,
                error_message=f"Rollback failed: {str(e)}"
            )
    
    async def batch_update_configurations(
        self,
        updates: List[Dict[str, Any]]
    ) -> List[ConfigUpdateResult]:
        """
        Perform batch configuration updates with optimized database operations.
        """
        results = []
        
        # Process updates in batches to avoid overwhelming the database
        batch_size = 10
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[
                    self.update_widget_configuration(**update)
                    for update in batch
                ],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ConfigUpdateResult(
                        success=False,
                        new_version=0,
                        error_message=str(result)
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring and optimization."""
        if not self.performance_metrics:
            return {}
        
        metrics_by_operation = {}
        for metric in self.performance_metrics:
            if metric.operation not in metrics_by_operation:
                metrics_by_operation[metric.operation] = []
            metrics_by_operation[metric.operation].append(metric)
        
        summary = {}
        for operation, metrics in metrics_by_operation.items():
            durations = [m.duration_ms for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            
            summary[operation] = {
                'count': len(metrics),
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'cache_hit_rate': cache_hits / len(metrics),
                'total_db_queries': sum(m.db_queries for m in metrics),
                'total_bytes': sum(m.bytes_transferred for m in metrics)
            }
        
        return summary
    
    async def refresh_configuration_cache(self, widget_id: Optional[str] = None):
        """Refresh materialized view cache for better performance."""
        try:
            supabase = get_supabase_admin_client()
            
            if widget_id:
                # Refresh specific widget cache
                if self.redis_client:
                    await self._invalidate_widget_cache(widget_id)
            else:
                # Refresh materialized view
                supabase.rpc('refresh_widget_config_cache').execute()
                logger.info("Refreshed widget configuration cache")
            
        except Exception as e:
            logger.error(f"Error refreshing configuration cache: {str(e)}")
    
    # Private helper methods
    
    async def _get_from_cache(
        self, 
        widget_id: str, 
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get configuration from Redis cache."""
        try:
            cache_key = f"{self.cache_prefix}{widget_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                config = json.loads(cached_data)
                # Check version match if provided
                if version and config.get('version') != version:
                    return None
                return config
                
        except Exception as e:
            logger.warning(f"Cache read error for {widget_id}: {str(e)}")
        
        return None
    
    async def _set_cache(
        self, 
        widget_id: str, 
        config: Dict[str, Any], 
        version: Optional[int] = None
    ):
        """Set configuration in Redis cache."""
        try:
            cache_key = f"{self.cache_prefix}{widget_id}"
            await self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(config, default=str)
            )
            
            # Also cache version info
            if version:
                version_key = f"{self.version_cache_prefix}{widget_id}"
                await self.redis_client.setex(version_key, self.cache_ttl, str(version))
                
        except Exception as e:
            logger.warning(f"Cache write error for {widget_id}: {str(e)}")
    
    async def _invalidate_widget_cache(self, website_id: str):
        """Invalidate cache for a specific widget."""
        try:
            # Get widget_id from website_id
            supabase = get_supabase_admin_client()
            result = supabase.table('websites').select('widget_id').eq('id', website_id).execute()
            
            if result.data:
                widget_id = result.data[0]['widget_id']
                if widget_id:
                    cache_key = f"{self.cache_prefix}{widget_id}"
                    version_key = f"{self.version_cache_prefix}{widget_id}"
                    
                    await asyncio.gather(
                        self.redis_client.delete(cache_key),
                        self.redis_client.delete(version_key),
                        return_exceptions=True
                    )
                    
        except Exception as e:
            logger.warning(f"Cache invalidation error for {website_id}: {str(e)}")
    
    def _build_config_response(
        self, 
        widget_data: Dict[str, Any], 
        widget_id: str
    ) -> Dict[str, Any]:
        """Build structured configuration response from database data."""
        return {
            'widget_id': widget_id,
            'config': {
                'appearance': {
                    'primaryColor': widget_data.get('widget_color', '#0066CC'),
                    'theme': widget_data.get('widget_theme', 'light'),
                    'borderRadius': widget_data.get('border_radius', 12),
                    'position': widget_data.get('widget_position', 'bottom-right'),
                    'size': widget_data.get('widget_size', 'medium')
                },
                'messages': {
                    'welcomeMessage': widget_data.get('welcome_message', 'Hi! How can I help you today?'),
                    'placeholder': widget_data.get('placeholder_text', 'Type your message...'),
                    'offlineMessage': widget_data.get('offline_message', "We're currently offline. We'll get back to you soon!"),
                    'title': widget_data.get('name', 'Chat Support')
                },
                'behavior': {
                    'showAvatar': widget_data.get('show_avatar', True),
                    'soundEnabled': widget_data.get('enable_sound', True),
                    'typingIndicator': True,
                    'quickReplies': ['Help', 'Contact Support', 'Pricing', 'Features']
                }
            },
            'version': widget_data.get('updated_at', datetime.now(timezone.utc).isoformat()),
            'cached': False,
            'config_version': widget_data.get('config_version', 1),
            'checksum': widget_data.get('config_checksum')
        }
    
    def _validate_config_updates(self, config_updates: Dict[str, Any]) -> List[str]:
        """Validate configuration updates."""
        errors = []
        
        # Validate color format
        if 'primaryColor' in config_updates:
            color = config_updates['primaryColor']
            if not re.match(r'^#[0-9A-Fa-f]{6}$', color):
                errors.append('Invalid color format')
        
        # Validate enum values
        valid_themes = ['light', 'dark', 'auto']
        if 'theme' in config_updates and config_updates['theme'] not in valid_themes:
            errors.append(f'Invalid theme. Must be one of: {valid_themes}')
        
        valid_positions = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
        if 'position' in config_updates and config_updates['position'] not in valid_positions:
            errors.append(f'Invalid position. Must be one of: {valid_positions}')
        
        valid_sizes = ['small', 'medium', 'large']
        if 'size' in config_updates and config_updates['size'] not in valid_sizes:
            errors.append(f'Invalid size. Must be one of: {valid_sizes}')
        
        # Validate message lengths
        message_limits = {
            'welcomeMessage': 500,
            'placeholder': 200,
            'offlineMessage': 500,
            'title': 100
        }
        
        for field, limit in message_limits.items():
            if field in config_updates and len(str(config_updates[field])) > limit:
                errors.append(f'{field} must be {limit} characters or less')
        
        return errors
    
    def _map_config_updates(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend config updates to database field names."""
        field_mapping = {
            'primaryColor': 'widget_color',
            'theme': 'widget_theme',
            'position': 'widget_position',
            'size': 'widget_size',
            'borderRadius': 'border_radius',
            'welcomeMessage': 'welcome_message',
            'placeholder': 'placeholder_text',
            'offlineMessage': 'offline_message',
            'title': 'name',
            'showAvatar': 'show_avatar',
            'soundEnabled': 'enable_sound'
        }
        
        return {
            field_mapping.get(key, key): value
            for key, value in config_updates.items()
            if key in field_mapping
        }
    
    def _flatten_config_for_db(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration for database update."""
        flattened = {}
        
        if 'appearance' in config:
            appearance = config['appearance']
            flattened.update({
                'primaryColor': appearance.get('primaryColor'),
                'theme': appearance.get('theme'),
                'borderRadius': appearance.get('borderRadius'),
                'position': appearance.get('position'),
                'size': appearance.get('size')
            })
        
        if 'messages' in config:
            messages = config['messages']
            flattened.update({
                'welcomeMessage': messages.get('welcomeMessage'),
                'placeholder': messages.get('placeholder'),
                'offlineMessage': messages.get('offlineMessage'),
                'title': messages.get('title')
            })
        
        if 'behavior' in config:
            behavior = config['behavior']
            flattened.update({
                'showAvatar': behavior.get('showAvatar'),
                'soundEnabled': behavior.get('soundEnabled')
            })
        
        # Remove None values
        return {k: v for k, v in flattened.items() if v is not None}
    
    def _calculate_config_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate configuration checksum for integrity verification."""
        config_string = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_string.encode()).hexdigest()
    
    async def _get_current_config_for_backup(
        self, 
        website_id: str, 
        supabase: Client
    ) -> Optional[Dict[str, Any]]:
        """Get current configuration for backup purposes."""
        try:
            result = supabase.table('websites').select('*').eq('id', website_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting current config for backup: {str(e)}")
            return None
    
    async def _create_version_backup(
        self,
        website_id: str,
        config_data: Dict[str, Any],
        user_id: str,
        description: str,
        supabase: Client
    ):
        """Create a version backup in the database."""
        try:
            # Get next version number
            version_result = supabase.table('widget_configuration_versions')\
                .select('version_number')\
                .eq('website_id', website_id)\
                .order('version_number', desc=True)\
                .limit(1)\
                .execute()
            
            next_version = 1
            if version_result.data:
                next_version = version_result.data[0]['version_number'] + 1
            
            # Create version record
            structured_config = self._build_config_from_db_row(config_data)
            
            supabase.table('widget_configuration_versions').insert({
                'website_id': website_id,
                'version_number': next_version,
                'description': description,
                'configuration_data': structured_config,
                'created_by': user_id,
                'is_active': False
            }).execute()
            
        except Exception as e:
            logger.error(f"Error creating version backup: {str(e)}")
    
    def _build_config_from_db_row(self, db_row: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured config from database row."""
        return {
            'appearance': {
                'primaryColor': db_row.get('widget_color'),
                'theme': db_row.get('widget_theme'),
                'borderRadius': db_row.get('border_radius'),
                'position': db_row.get('widget_position'),
                'size': db_row.get('widget_size')
            },
            'messages': {
                'welcomeMessage': db_row.get('welcome_message'),
                'placeholder': db_row.get('placeholder_text'),
                'offlineMessage': db_row.get('offline_message'),
                'title': db_row.get('name')
            },
            'behavior': {
                'showAvatar': db_row.get('show_avatar'),
                'soundEnabled': db_row.get('enable_sound'),
                'typingIndicator': True,
                'quickReplies': ['Help', 'Contact Support', 'Pricing', 'Features']
            }
        }
    
    async def _log_audit_action(
        self,
        website_id: str,
        action: str,
        old_config: Optional[Dict[str, Any]],
        new_config: Optional[Dict[str, Any]],
        user_id: str,
        description: str,
        supabase: Client
    ):
        """Log audit action for configuration changes."""
        try:
            supabase.table('widget_configuration_audit').insert({
                'website_id': website_id,
                'action': action,
                'old_config': old_config,
                'new_config': new_config,
                'user_id': user_id,
                'description': description
            }).execute()
        except Exception as e:
            logger.error(f"Error logging audit action: {str(e)}")
    
    async def _record_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        cache_hit: bool,
        db_queries: int,
        bytes_transferred: int
    ):
        """Record performance metric for monitoring."""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            db_queries=db_queries,
            bytes_transferred=bytes_transferred
        )
        
        self.performance_metrics.append(metric)
        
        # Keep only last 1000 metrics to avoid memory issues
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        # Log slow operations
        if duration_ms > 1000:  # > 1 second
            logger.warning(f"Slow operation detected: {operation} took {duration_ms:.2f}ms")


# Global service instance
_widget_config_service: Optional[OptimizedWidgetConfigService] = None


def get_optimized_widget_config_service() -> OptimizedWidgetConfigService:
    """Get or create the optimized widget configuration service instance."""
    global _widget_config_service
    
    if _widget_config_service is None:
        # Initialize Redis client if available
        redis_client = None
        try:
            redis_client = redis.Redis.from_url(
                "redis://localhost:6379",  # This should be configurable
                decode_responses=False
            )
        except Exception as e:
            logger.warning(f"Redis not available, running without cache: {str(e)}")
        
        _widget_config_service = OptimizedWidgetConfigService(redis_client)
    
    return _widget_config_service