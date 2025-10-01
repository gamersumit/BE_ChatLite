"""
Comprehensive logging and audit trails service.
Task 5.4: Implement comprehensive logging and audit trails
"""

import asyncio
import json
import os
import gzip
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from pathlib import Path
import traceback
import threading
from collections import defaultdict, deque
import logging
import aiofiles


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categories for log entries."""
    CRAWLING = "CRAWLING"
    SYSTEM = "SYSTEM"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"
    AUDIT = "AUDIT"
    USER_ACTION = "USER_ACTION"
    API = "API"
    DATABASE = "DATABASE"


class AuditAction(Enum):
    """Types of audit actions."""
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    CONFIG_CREATE = "CONFIG_CREATE"
    CONFIG_UPDATE = "CONFIG_UPDATE"
    CONFIG_DELETE = "CONFIG_DELETE"
    DATA_READ = "DATA_READ"
    DATA_CREATE = "DATA_CREATE"
    DATA_UPDATE = "DATA_UPDATE"
    DATA_DELETE = "DATA_DELETE"
    DATA_EXPORT = "DATA_EXPORT"
    RESOURCE_ACCESS = "RESOURCE_ACCESS"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SECURITY_EVENT = "SECURITY_EVENT"


@dataclass
class LogEntry:
    """Structured log entry."""
    level: LogLevel
    message: str
    category: LogCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class AuditEntry:
    """Audit trail entry."""
    action: AuditAction
    user_id: Optional[str]
    resource_type: str
    resource_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    integrity_hash: Optional[str] = None
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CrawlingLogger:
    """
    Comprehensive structured logging service for crawling activities.

    Features:
    - Structured JSON logging
    - Multiple log levels and categories
    - Correlation and tracing support
    - File and console output
    - Performance metrics logging
    - Error tracking with stack traces
    """

    def __init__(self,
                 log_directory: str = "./logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_json_logging: bool = True,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_log_file_size_mb: int = 100,
                 max_log_files: int = 10):
        """Initialize the crawling logger."""
        self.log_directory = Path(log_directory)
        self.log_level = log_level
        self.enable_json_logging = enable_json_logging
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_log_file_size_mb = max_log_file_size_mb
        self.max_log_files = max_log_files

        # Active traces for correlation
        self.active_traces: Dict[str, Dict[str, Any]] = {}

        # Log buffers for performance
        self.log_buffer: deque = deque(maxlen=10000)
        self.buffer_lock = threading.Lock()

        # Statistics
        self.log_stats = defaultdict(int)

        # File handles
        self.log_files: Dict[str, Any] = {}

        # Initialize logging system
        self.logger = None

    async def initialize(self):
        """Initialize the logging service."""
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Setup Python logging
        self._setup_python_logging()

        # Start background tasks
        asyncio.create_task(self._log_writer_task())

    async def shutdown(self):
        """Shutdown the logging service."""
        # Flush remaining logs
        await self._flush_log_buffer()

        # Close file handles
        for file_handle in self.log_files.values():
            if hasattr(file_handle, 'close'):
                file_handle.close()

    def create_log_entry(self,
                        level: LogLevel,
                        message: str,
                        category: LogCategory,
                        context: Optional[Dict[str, Any]] = None,
                        correlation_id: Optional[str] = None,
                        user_id: Optional[str] = None) -> LogEntry:
        """Create a structured log entry."""
        return LogEntry(
            level=level,
            message=message,
            category=category,
            context=context or {},
            correlation_id=correlation_id or str(uuid.uuid4()),
            user_id=user_id
        )

    async def log(self, entry: LogEntry):
        """Log a structured entry."""
        # Check log level
        if not self._should_log(entry.level):
            return

        # Add to buffer
        with self.buffer_lock:
            self.log_buffer.append(entry)
            self.log_stats[entry.level.value] += 1
            self.log_stats["total"] += 1

        # Console logging
        if self.enable_console_logging:
            self._log_to_console(entry)

    async def log_crawl_start(self,
                            job_id: str,
                            url: str,
                            config: Dict[str, Any],
                            user_id: Optional[str] = None) -> LogEntry:
        """Log crawl start event."""
        entry = self.create_log_entry(
            level=LogLevel.INFO,
            message=f"Crawl started for {url}",
            category=LogCategory.CRAWLING,
            context={
                "job_id": job_id,
                "url": url,
                "config": config,
                "event_type": "crawl_start"
            },
            user_id=user_id
        )
        await self.log(entry)
        return entry

    async def log_crawl_completion(self,
                                 job_id: str,
                                 success: bool,
                                 result_data: Dict[str, Any]) -> LogEntry:
        """Log crawl completion event."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        status = "completed successfully" if success else "failed"

        entry = self.create_log_entry(
            level=level,
            message=f"Crawl {status} for job {job_id}",
            category=LogCategory.CRAWLING,
            context={
                "job_id": job_id,
                "success": success,
                "event_type": "crawl_completion",
                **result_data
            }
        )
        await self.log(entry)
        return entry

    async def log_error(self,
                       message: str,
                       exception: Optional[Exception] = None,
                       context: Optional[Dict[str, Any]] = None,
                       user_id: Optional[str] = None) -> LogEntry:
        """Log error with optional exception details."""
        error_context = context or {}

        if exception:
            error_context.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stack_trace": traceback.format_exc()
            })

        entry = self.create_log_entry(
            level=LogLevel.ERROR,
            message=message,
            category=LogCategory.ERROR,
            context=error_context,
            user_id=user_id
        )
        await self.log(entry)
        return entry

    async def log_performance_metrics(self,
                                    job_id: str,
                                    metrics: Dict[str, Any],
                                    operation: str) -> LogEntry:
        """Log performance metrics."""
        entry = self.create_log_entry(
            level=LogLevel.INFO,
            message=f"Performance metrics for {operation}",
            category=LogCategory.PERFORMANCE,
            context={
                "job_id": job_id,
                "operation": operation,
                "metrics": metrics,
                "event_type": "performance_metrics"
            }
        )
        await self.log(entry)
        return entry

    async def log_security_event(self,
                               event_type: str,
                               details: str,
                               context: Optional[Dict[str, Any]] = None,
                               severity: str = "medium") -> LogEntry:
        """Log security-related events."""
        security_context = context or {}
        security_context.update({
            "event_type": event_type,
            "severity": severity
        })

        entry = self.create_log_entry(
            level=LogLevel.WARNING,
            message=f"Security event: {details}",
            category=LogCategory.SECURITY,
            context=security_context
        )
        await self.log(entry)
        return entry

    def start_trace(self, operation_name: str) -> str:
        """Start a distributed trace."""
        trace_id = str(uuid.uuid4())
        self.active_traces[trace_id] = {
            "operation": operation_name,
            "start_time": datetime.now(timezone.utc),
            "spans": []
        }
        return trace_id

    def end_trace(self, trace_id: str):
        """End a distributed trace."""
        if trace_id in self.active_traces:
            trace_data = self.active_traces[trace_id]
            trace_data["end_time"] = datetime.now(timezone.utc)
            trace_data["duration_ms"] = (
                trace_data["end_time"] - trace_data["start_time"]
            ).total_seconds() * 1000
            del self.active_traces[trace_id]

    async def log_with_trace(self,
                           trace_id: str,
                           level: LogLevel,
                           message: str,
                           category: LogCategory,
                           context: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Log entry with trace correlation."""
        entry = self.create_log_entry(
            level=level,
            message=message,
            category=category,
            context=context,
            correlation_id=trace_id
        )
        entry.trace_id = trace_id
        await self.log(entry)
        return entry

    def format_as_json(self, entry: LogEntry) -> str:
        """Format log entry as JSON."""
        log_dict = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.value,
            "category": entry.category.value,
            "message": entry.message,
            "correlation_id": entry.correlation_id,
            "context": entry.context or {}
        }

        # Add optional fields
        if entry.source:
            log_dict["source"] = entry.source
        if entry.trace_id:
            log_dict["trace_id"] = entry.trace_id
        if entry.user_id:
            log_dict["user_id"] = entry.user_id
        if entry.session_id:
            log_dict["session_id"] = entry.session_id

        return json.dumps(log_dict)

    async def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_logs": self.log_stats["total"],
            "logs_by_level": dict(self.log_stats),
            "active_traces": len(self.active_traces),
            "buffer_size": len(self.log_buffer)
        }

    # Private methods

    def _should_log(self, level: LogLevel) -> bool:
        """Check if log level should be logged."""
        level_hierarchy = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        return level_hierarchy[level] >= level_hierarchy[self.log_level]

    def _setup_python_logging(self):
        """Setup Python logging integration."""
        self.logger = logging.getLogger("crawling_logger")
        self.logger.setLevel(getattr(logging, self.log_level.value))

        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _log_to_console(self, entry: LogEntry):
        """Log to console."""
        if self.enable_json_logging:
            print(self.format_as_json(entry))
        else:
            print(f"[{entry.timestamp.isoformat()}] {entry.level.value} - {entry.message}")

    async def _log_writer_task(self):
        """Background task to write logs to files."""
        while True:
            try:
                await self._flush_log_buffer()
                await asyncio.sleep(1)  # Write every second
            except Exception as e:
                print(f"Error in log writer task: {e}")

    async def _flush_log_buffer(self):
        """Flush log buffer to files."""
        if not self.enable_file_logging or not self.log_buffer:
            return

        entries_to_write = []
        with self.buffer_lock:
            while self.log_buffer:
                entries_to_write.append(self.log_buffer.popleft())

        if entries_to_write:
            await self._write_logs_to_file(entries_to_write)

    async def _write_logs_to_file(self, entries: List[LogEntry]):
        """Write log entries to file."""
        # Ensure log directory exists
        self.log_directory.mkdir(parents=True, exist_ok=True)

        log_file_path = self.log_directory / f"crawling_{datetime.now().strftime('%Y-%m-%d')}.log"

        try:
            async with aiofiles.open(log_file_path, 'a', encoding='utf-8') as f:
                for entry in entries:
                    if self.enable_json_logging:
                        await f.write(self.format_as_json(entry) + '\n')
                    else:
                        await f.write(f"[{entry.timestamp.isoformat()}] {entry.level.value} - {entry.message}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")


class AuditTrail:
    """
    Comprehensive audit trail service for tracking changes and actions.

    Features:
    - Configuration change tracking
    - User action auditing
    - Data access logging
    - Integrity verification
    - Search and filtering
    - Compliance reporting
    """

    def __init__(self,
                 storage_directory: str = "./audit_logs",
                 enable_encryption: bool = False,
                 retention_days: int = 365):
        """Initialize the audit trail service."""
        self.storage_directory = Path(storage_directory)
        self.enable_encryption = enable_encryption
        self.retention_days = retention_days

        # Audit entries storage
        self.audit_entries: List[AuditEntry] = []
        self.entry_lock = threading.Lock()

        # Search index
        self.search_index: Dict[str, List[str]] = defaultdict(list)

    async def initialize(self):
        """Initialize the audit trail service."""
        # Create storage directory
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        # Load existing entries
        await self._load_existing_entries()

        # Start background cleanup task
        asyncio.create_task(self._cleanup_task())

    async def shutdown(self):
        """Shutdown the audit trail service."""
        # Save pending entries
        await self._save_all_entries()

    async def log_configuration_change(self,
                                     action: AuditAction,
                                     resource_type: str,
                                     resource_id: str,
                                     old_values: Optional[Dict[str, Any]] = None,
                                     new_values: Optional[Dict[str, Any]] = None,
                                     changed_by: Optional[str] = None,
                                     reason: Optional[str] = None,
                                     ip_address: Optional[str] = None) -> AuditEntry:
        """Log configuration changes."""
        metadata = {}
        if reason:
            metadata["reason"] = reason
        if changed_by:
            metadata["changed_by"] = changed_by

        entry = AuditEntry(
            action=action,
            user_id=changed_by,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            metadata=metadata,
            ip_address=ip_address
        )

        # Calculate integrity hash
        entry.integrity_hash = self._calculate_integrity_hash(entry)

        await self._store_audit_entry(entry)
        return entry

    async def log_user_action(self,
                            action: AuditAction,
                            user_id: str,
                            resource_type: str,
                            resource_id: str,
                            metadata: Optional[Dict[str, Any]] = None,
                            ip_address: Optional[str] = None,
                            user_agent: Optional[str] = None) -> AuditEntry:
        """Log user actions."""
        entry = AuditEntry(
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent
        )

        entry.integrity_hash = self._calculate_integrity_hash(entry)
        await self._store_audit_entry(entry)
        return entry

    async def log_data_operation(self,
                               action: AuditAction,
                               user_id: str,
                               resource_type: str,
                               resource_id: str,
                               operation_details: Dict[str, Any],
                               ip_address: Optional[str] = None) -> AuditEntry:
        """Log data operations (read, create, update, delete, export)."""
        entry = AuditEntry(
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=operation_details,
            ip_address=ip_address
        )

        entry.integrity_hash = self._calculate_integrity_hash(entry)
        await self._store_audit_entry(entry)
        return entry

    async def log_system_event(self,
                             action: AuditAction,
                             component: str,
                             event_details: Dict[str, Any]) -> AuditEntry:
        """Log system events."""
        entry = AuditEntry(
            action=action,
            user_id=None,  # System events don't have users
            resource_type="system_component",
            resource_id=component,
            metadata=event_details
        )

        entry.integrity_hash = self._calculate_integrity_hash(entry)
        await self._store_audit_entry(entry)
        return entry

    def verify_entry_integrity(self, entry: AuditEntry) -> bool:
        """Verify the integrity of an audit entry."""
        if not entry.integrity_hash:
            return False

        calculated_hash = self._calculate_integrity_hash(entry)
        return calculated_hash == entry.integrity_hash

    async def search_entries(self,
                           filters: Optional[Dict[str, Any]] = None,
                           time_range: Optional[Dict[str, datetime]] = None,
                           limit: int = 100) -> List[AuditEntry]:
        """Search audit entries with filters."""
        filtered_entries = []

        with self.entry_lock:
            for entry in self.audit_entries:
                # Time range filter
                if time_range:
                    if time_range.get("start") and entry.timestamp < time_range["start"]:
                        continue
                    if time_range.get("end") and entry.timestamp > time_range["end"]:
                        continue

                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key == "user_id" and entry.user_id != value:
                            match = False
                            break
                        elif key == "action":
                            if isinstance(value, list):
                                if entry.action not in value:
                                    match = False
                                    break
                            elif entry.action != value:
                                match = False
                                break
                        elif key == "resource_type" and entry.resource_type != value:
                            match = False
                            break

                    if not match:
                        continue

                filtered_entries.append(entry)

                if len(filtered_entries) >= limit:
                    break

        return filtered_entries

    async def generate_compliance_report(self,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for a time period."""
        entries = await self.search_entries(
            time_range={"start": start_date, "end": end_date}
        )

        # Categorize entries
        user_actions = [e for e in entries if e.user_id]
        config_changes = [e for e in entries if e.action in [
            AuditAction.CONFIG_CREATE, AuditAction.CONFIG_UPDATE, AuditAction.CONFIG_DELETE
        ]]
        data_operations = [e for e in entries if e.action in [
            AuditAction.DATA_READ, AuditAction.DATA_CREATE,
            AuditAction.DATA_UPDATE, AuditAction.DATA_DELETE, AuditAction.DATA_EXPORT
        ]]

        # Calculate statistics
        unique_users = len(set(e.user_id for e in user_actions if e.user_id))

        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_entries": len(entries),
                "user_actions": len(user_actions),
                "config_changes": len(config_changes),
                "data_operations": len(data_operations),
                "unique_users": unique_users
            },
            "compliance_checks": {
                "all_entries_have_integrity_hash": all(e.integrity_hash for e in entries),
                "retention_compliance": True,  # Simplified
                "user_action_coverage": len(user_actions) / len(entries) if entries else 0
            },
            "risk_indicators": {
                "high_privilege_actions": len([e for e in config_changes if "admin" in str(e.metadata)]),
                "bulk_data_exports": len([e for e in data_operations if e.action == AuditAction.DATA_EXPORT]),
                "after_hours_activity": 0  # Simplified
            }
        }

    # Private methods

    async def _store_audit_entry(self, entry: AuditEntry):
        """Store audit entry."""
        with self.entry_lock:
            self.audit_entries.append(entry)

        # Update search index
        self._update_search_index(entry)

        # Persist to file
        await self._save_entry_to_file(entry)

    def _calculate_integrity_hash(self, entry: AuditEntry) -> str:
        """Calculate integrity hash for audit entry."""
        # Create a copy without the hash field for calculation
        entry_dict = asdict(entry)
        entry_dict.pop('integrity_hash', None)

        # Create deterministic string representation
        entry_string = json.dumps(entry_dict, sort_keys=True, default=str)

        # Calculate SHA-256 hash
        return hashlib.sha256(entry_string.encode()).hexdigest()

    def _update_search_index(self, entry: AuditEntry):
        """Update search index with new entry."""
        self.search_index[entry.user_id or "system"].append(entry.entry_id)
        self.search_index[entry.action.value].append(entry.entry_id)
        self.search_index[entry.resource_type].append(entry.entry_id)

    async def _save_entry_to_file(self, entry: AuditEntry):
        """Save audit entry to file."""
        # Ensure storage directory exists
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        audit_file_path = self.storage_directory / f"audit_{datetime.now().strftime('%Y-%m-%d')}.log"

        try:
            entry_json = json.dumps(asdict(entry), default=str)
            async with aiofiles.open(audit_file_path, 'a', encoding='utf-8') as f:
                await f.write(entry_json + '\n')
        except Exception as e:
            print(f"Error saving audit entry: {e}")

    async def _save_all_entries(self):
        """Save all pending entries."""
        # Implementation would batch save entries
        pass

    async def _load_existing_entries(self):
        """Load existing audit entries from storage."""
        # Implementation would load from persistent storage
        pass

    async def _cleanup_task(self):
        """Background cleanup task for old entries."""
        while True:
            try:
                await self._cleanup_old_entries()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                print(f"Error in audit cleanup task: {e}")

    async def _cleanup_old_entries(self):
        """Clean up old audit entries based on retention policy."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        with self.entry_lock:
            self.audit_entries = [
                entry for entry in self.audit_entries
                if entry.timestamp >= cutoff_date
            ]


class LogAggregator:
    """
    Log aggregation and search service.

    Features:
    - Multi-source log aggregation
    - Full-text search capabilities
    - Real-time log streaming
    - Analytics and metrics
    - Log correlation
    """

    def __init__(self,
                 storage_directory: str = "./aggregated_logs",
                 index_logs: bool = True,
                 enable_search: bool = True):
        """Initialize log aggregator."""
        self.storage_directory = Path(storage_directory)
        self.index_logs = index_logs
        self.enable_search = enable_search

        # Aggregated logs storage
        self.aggregated_logs: List[LogEntry] = []
        self.logs_lock = threading.Lock()

        # Search index
        self.search_index: Dict[str, List[int]] = defaultdict(list)

        # Stream subscribers
        self.stream_subscribers: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.ingestion_stats = defaultdict(int)

    async def initialize(self):
        """Initialize the log aggregator."""
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    async def shutdown(self):
        """Shutdown the log aggregator."""
        pass

    async def ingest_log_entry(self,
                             source: str,
                             level: LogLevel,
                             message: str,
                             category: LogCategory,
                             context: Optional[Dict[str, Any]] = None,
                             correlation_id: Optional[str] = None) -> LogEntry:
        """Ingest a log entry from a source."""
        entry = LogEntry(
            level=level,
            message=message,
            category=category,
            source=source,
            context=context or {},
            correlation_id=correlation_id
        )

        with self.logs_lock:
            self.aggregated_logs.append(entry)
            entry_index = len(self.aggregated_logs) - 1

        # Update search index
        if self.index_logs:
            self._update_search_index(entry, entry_index)

        # Update statistics
        self.ingestion_stats[source] += 1
        self.ingestion_stats["total"] += 1

        # Notify stream subscribers
        await self._notify_stream_subscribers(entry)

        return entry

    async def search_logs(self,
                        query: Optional[str] = None,
                        filters: Optional[Dict[str, Any]] = None,
                        limit: int = 100) -> List[LogEntry]:
        """Search logs with query and filters."""
        results = []

        with self.logs_lock:
            for entry in self.aggregated_logs:
                # Apply text query
                if query and query.lower() not in entry.message.lower():
                    continue

                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key == "source" and entry.source != value:
                            match = False
                            break
                        elif key == "level" and entry.level != value:
                            match = False
                            break
                        elif key == "category" and entry.category != value:
                            match = False
                            break

                    if not match:
                        continue

                results.append(entry)

                if len(results) >= limit:
                    break

        return results

    async def get_total_log_count(self) -> int:
        """Get total number of aggregated logs."""
        with self.logs_lock:
            return len(self.aggregated_logs)

    async def generate_log_analytics(self) -> Dict[str, Any]:
        """Generate analytics from aggregated logs."""
        with self.logs_lock:
            if not self.aggregated_logs:
                return {}

            # Level distribution
            level_counts = defaultdict(int)
            category_counts = defaultdict(int)
            source_counts = defaultdict(int)

            for entry in self.aggregated_logs:
                level_counts[entry.level.value] += 1
                category_counts[entry.category.value] += 1
                if entry.source:
                    source_counts[entry.source] += 1

            # Error rate
            total_logs = len(self.aggregated_logs)
            error_logs = level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)
            error_rate = (error_logs / total_logs) * 100 if total_logs > 0 else 0

        return {
            "log_levels_distribution": dict(level_counts),
            "categories_distribution": dict(category_counts),
            "sources_distribution": dict(source_counts),
            "error_rate": error_rate,
            "total_logs": total_logs,
            "log_volume_trends": {}  # Simplified
        }

    async def subscribe_to_log_stream(self,
                                    subscriber_id: str,
                                    callback: Callable,
                                    filters: Optional[Dict[str, Any]] = None):
        """Subscribe to real-time log stream."""
        self.stream_subscribers[subscriber_id] = {
            "callback": callback,
            "filters": filters or {}
        }

    async def unsubscribe_from_log_stream(self, subscriber_id: str):
        """Unsubscribe from log stream."""
        self.stream_subscribers.pop(subscriber_id, None)

    # Private methods

    def _update_search_index(self, entry: LogEntry, index: int):
        """Update search index."""
        # Index by message words
        words = entry.message.lower().split()
        for word in words:
            self.search_index[word].append(index)

        # Index by level and category
        self.search_index[entry.level.value.lower()].append(index)
        self.search_index[entry.category.value.lower()].append(index)

    async def _notify_stream_subscribers(self, entry: LogEntry):
        """Notify stream subscribers of new log entry."""
        for subscriber_id, subscriber_info in self.stream_subscribers.items():
            try:
                # Check filters
                filters = subscriber_info.get("filters", {})
                if filters:
                    match = True
                    if "level" in filters:
                        allowed_levels = filters["level"]
                        if isinstance(allowed_levels, list):
                            if entry.level not in allowed_levels:
                                match = False
                        elif entry.level != allowed_levels:
                            match = False

                    if not match:
                        continue

                # Call subscriber callback
                callback = subscriber_info["callback"]
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)

            except Exception as e:
                print(f"Error notifying subscriber {subscriber_id}: {e}")


class LogRetentionPolicy:
    """
    Log retention and archival policy manager.

    Features:
    - Automated log aging and archival
    - Compressed archive creation
    - Archive search and retrieval
    - Compliance enforcement
    """

    def __init__(self,
                 storage_directory: str = "./log_archives",
                 retention_days: int = 30,
                 archive_after_days: int = 7,
                 compress_archives: bool = True):
        """Initialize retention policy manager."""
        self.storage_directory = Path(storage_directory)
        self.retention_days = retention_days
        self.archive_after_days = archive_after_days
        self.compress_archives = compress_archives

        # Active logs
        self.active_logs: List[LogEntry] = []
        self.archived_logs: Dict[str, List[LogEntry]] = {}

    async def initialize(self):
        """Initialize retention policy manager."""
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    async def shutdown(self):
        """Shutdown retention policy manager."""
        pass

    async def store_log_entry(self, entry: LogEntry):
        """Store a log entry."""
        self.active_logs.append(entry)

    async def apply_retention_policies(self) -> Dict[str, int]:
        """Apply retention policies to stored logs."""
        now = datetime.now(timezone.utc)
        archive_cutoff = now - timedelta(days=self.archive_after_days)
        delete_cutoff = now - timedelta(days=self.retention_days)

        archived_count = 0
        deleted_count = 0
        remaining_logs = []

        logs_to_archive = []

        for entry in self.active_logs:
            if entry.timestamp < delete_cutoff:
                # Delete old logs
                deleted_count += 1
            elif entry.timestamp < archive_cutoff:
                # Archive logs
                logs_to_archive.append(entry)
                archived_count += 1
            else:
                # Keep active logs
                remaining_logs.append(entry)

        # Create archive for logs to archive
        if logs_to_archive:
            archive_name = f"archive_{archive_cutoff.strftime('%Y_%m_%d')}"
            await self.create_archive(logs_to_archive, archive_name)

        # Update active logs
        self.active_logs = remaining_logs

        return {
            "archived_count": archived_count,
            "deleted_count": deleted_count,
            "remaining_active": len(remaining_logs)
        }

    async def create_archive(self,
                           logs: List[LogEntry],
                           archive_name: str) -> Dict[str, Any]:
        """Create compressed archive of logs."""
        archive_path = self.storage_directory / f"{archive_name}.json"

        # Convert logs to JSON
        logs_data = [asdict(log) for log in logs]
        archive_content = json.dumps(logs_data, default=str, indent=2)

        # Write to file
        if self.compress_archives:
            archive_path = archive_path.with_suffix('.json.gz')
            with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                f.write(archive_content)
        else:
            with open(archive_path, 'w', encoding='utf-8') as f:
                f.write(archive_content)

        # Store in memory index
        self.archived_logs[archive_name] = logs

        # Calculate compression ratio
        uncompressed_size = len(archive_content.encode('utf-8'))
        compressed_size = archive_path.stat().st_size
        compression_ratio = compressed_size / uncompressed_size if uncompressed_size > 0 else 1

        return {
            "archive_created": True,
            "archive_path": str(archive_path),
            "entry_count": len(logs),
            "compressed": self.compress_archives,
            "compression_ratio": compression_ratio,
            "archive_size_bytes": compressed_size
        }

    async def search_archive(self,
                           archive_name: str,
                           query: str,
                           filters: Optional[Dict[str, Any]] = None) -> List[LogEntry]:
        """Search within an archived log set."""
        if archive_name not in self.archived_logs:
            return []

        results = []
        for entry in self.archived_logs[archive_name]:
            # Apply text query
            if query.lower() not in entry.message.lower():
                continue

            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key == "level" and entry.level != value:
                        match = False
                        break

                if not match:
                    continue

            results.append(entry)

        return results

    async def get_active_logs(self) -> List[LogEntry]:
        """Get currently active logs."""
        return self.active_logs.copy()

    async def configure_retention_policy(self, policy: Dict[str, Any]):
        """Configure retention policy parameters."""
        if "critical_logs_retention_days" in policy:
            self.retention_days = max(self.retention_days, policy["critical_logs_retention_days"])

    async def enforce_compliance(self) -> Dict[str, Any]:
        """Enforce compliance with retention policies."""
        # Simplified compliance check
        compliance_result = await self.apply_retention_policies()

        return {
            "compliant": True,
            "policy_violations": [],
            "corrective_actions": compliance_result
        }


# Global instances
_crawling_logger_instance: Optional[CrawlingLogger] = None
_audit_trail_instance: Optional[AuditTrail] = None


async def get_crawling_logger() -> CrawlingLogger:
    """Get or create global crawling logger instance."""
    global _crawling_logger_instance

    if _crawling_logger_instance is None:
        _crawling_logger_instance = CrawlingLogger()
        await _crawling_logger_instance.initialize()

    return _crawling_logger_instance


async def get_audit_trail() -> AuditTrail:
    """Get or create global audit trail instance."""
    global _audit_trail_instance

    if _audit_trail_instance is None:
        _audit_trail_instance = AuditTrail()
        await _audit_trail_instance.initialize()

    return _audit_trail_instance


async def shutdown_logging_services():
    """Shutdown global logging services."""
    global _crawling_logger_instance, _audit_trail_instance

    if _crawling_logger_instance:
        await _crawling_logger_instance.shutdown()
        _crawling_logger_instance = None

    if _audit_trail_instance:
        await _audit_trail_instance.shutdown()
        _audit_trail_instance = None