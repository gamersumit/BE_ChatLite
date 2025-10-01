"""
JavaScript Error Handler for comprehensive error detection, categorization, and debugging.
Task 4.4: Create JavaScript error handling and debugging
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, Counter
import traceback

from app.services.playwright_browser_manager import (
    PlaywrightBrowserManager,
    BrowserConfig,
    BrowserEngine,
    CrawlResult,
    WaitStrategy
)

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """JavaScript error categories."""
    RUNTIME_ERROR = "runtime_error"
    SYNTAX_ERROR = "syntax_error"
    REFERENCE_ERROR = "reference_error"
    TYPE_ERROR = "type_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    SECURITY_ERROR = "security_error"
    WARNING = "warning"
    INFO = "info"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    DISABLE_JAVASCRIPT = "disable_javascript"
    CHANGE_USER_AGENT = "change_user_agent"
    INCREASE_TIMEOUT = "increase_timeout"
    USE_DIFFERENT_ENGINE = "use_different_engine"
    FALLBACK_CRAWL = "fallback_crawl"


@dataclass
class ErrorRecord:
    """JavaScript error record."""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    url: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stack_trace: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    source_file: Optional[str] = None
    user_agent: Optional[str] = None
    page_title: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorPattern:
    """Error pattern analysis."""
    pattern_type: str
    frequency: int
    affected_urls: List[str]
    first_seen: datetime
    last_seen: datetime
    recovery_success_rate: float = 0.0


@dataclass
class DebugSession:
    """Debug session information."""
    session_id: str
    url: str
    start_time: datetime
    config: BrowserConfig
    errors: List[ErrorRecord] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[bytes] = field(default_factory=list)
    console_logs: List[Dict[str, str]] = field(default_factory=list)
    network_logs: List[Dict[str, Any]] = field(default_factory=list)


class JavaScriptErrorHandler:
    """
    Comprehensive JavaScript error handler with debugging capabilities.

    Features:
    - JavaScript console error capture and analysis
    - Error categorization and severity assessment
    - Automatic error recovery and retry mechanisms
    - Performance profiling and debugging tools
    - Error pattern analysis and reporting
    - Screenshot capture on errors
    - Detailed error reporting and troubleshooting
    """

    def __init__(
        self,
        debug_mode: bool = False,
        capture_screenshots: bool = True,
        enable_profiling: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize JavaScript error handler."""
        self.debug_mode = debug_mode
        self.capture_screenshots = capture_screenshots
        self.enable_profiling = enable_profiling
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_stats: Dict[str, int] = defaultdict(int)

        # Debug sessions
        self.debug_sessions: Dict[str, DebugSession] = {}

        # Error classification patterns
        self._init_error_patterns()

    def _init_error_patterns(self):
        """Initialize error classification patterns."""
        self.error_classification = {
            ErrorCategory.TYPE_ERROR: [
                r"TypeError",
                r"Cannot read propert(y|ies) of",
                r"is not a function",
                r"is not defined"
            ],
            ErrorCategory.REFERENCE_ERROR: [
                r"ReferenceError",
                r"is not defined",
                r"undefined variable"
            ],
            ErrorCategory.SYNTAX_ERROR: [
                r"SyntaxError",
                r"Unexpected token",
                r"Invalid or unexpected token"
            ],
            ErrorCategory.NETWORK_ERROR: [
                r"Failed to load resource",
                r"NetworkError",
                r"ERR_NETWORK",
                r"ERR_INTERNET_DISCONNECTED"
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                r"timeout",
                r"Navigation timeout",
                r"Request timeout"
            ],
            ErrorCategory.RESOURCE_ERROR: [
                r"Memory limit exceeded",
                r"Maximum call stack",
                r"out of memory"
            ],
            ErrorCategory.SECURITY_ERROR: [
                r"SecurityError",
                r"CORS",
                r"Cross-origin",
                r"Blocked by CORS"
            ],
            ErrorCategory.WARNING: [
                r"deprecated",
                r"warning",
                r"will be removed"
            ]
        }

        self.severity_rules = {
            ErrorCategory.RUNTIME_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.SYNTAX_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.REFERENCE_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.TYPE_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.TIMEOUT_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.RESOURCE_ERROR: ErrorSeverity.CRITICAL,
            ErrorCategory.SECURITY_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.WARNING: ErrorSeverity.LOW,
            ErrorCategory.INFO: ErrorSeverity.INFO,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }

    def categorize_error(self, error_message: str) -> Dict[str, Any]:
        """Categorize error message and determine severity."""
        error_message_lower = error_message.lower()

        # Find matching category
        category = ErrorCategory.UNKNOWN
        for error_cat, patterns in self.error_classification.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    category = error_cat
                    break
            if category != ErrorCategory.UNKNOWN:
                break

        # Determine severity
        severity = self.severity_rules.get(category, ErrorSeverity.MEDIUM)

        return {
            "category": category.value,
            "severity": severity.value,
            "message": error_message,
            "matched_patterns": self._find_matched_patterns(error_message)
        }

    def _find_matched_patterns(self, error_message: str) -> List[str]:
        """Find all patterns that match the error message."""
        matched = []
        for patterns in self.error_classification.values():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    matched.append(pattern)
        return matched

    def record_error(self, error_message: str, url: str, **kwargs) -> ErrorRecord:
        """Record an error for analysis."""
        categorization = self.categorize_error(error_message)

        error_record = ErrorRecord(
            message=error_message,
            category=ErrorCategory(categorization["category"]),
            severity=ErrorSeverity(categorization["severity"]),
            url=url,
            stack_trace=kwargs.get("stack_trace"),
            line_number=kwargs.get("line_number"),
            column_number=kwargs.get("column_number"),
            source_file=kwargs.get("source_file"),
            user_agent=kwargs.get("user_agent"),
            page_title=kwargs.get("page_title"),
            additional_context=kwargs.get("additional_context", {})
        )

        self.error_records.append(error_record)
        self.error_stats[categorization["category"]] += 1

        # Update error patterns
        self._update_error_patterns(error_record)

        return error_record

    def _update_error_patterns(self, error_record: ErrorRecord):
        """Update error pattern analysis."""
        pattern_key = f"{error_record.category.value}:{error_record.message[:50]}"

        if pattern_key in self.error_patterns:
            pattern = self.error_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_seen = error_record.timestamp
            if error_record.url not in pattern.affected_urls:
                pattern.affected_urls.append(error_record.url)
        else:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern_type=error_record.category.value,
                frequency=1,
                affected_urls=[error_record.url],
                first_seen=error_record.timestamp,
                last_seen=error_record.timestamp
            )

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        if not self.error_records:
            return {"message": "No errors recorded"}

        # Most common errors
        error_counter = Counter([record.message for record in self.error_records])
        most_common = error_counter.most_common(5)

        # Error frequency by category
        category_stats = defaultdict(int)
        for record in self.error_records:
            category_stats[record.category.value] += 1

        # Error trends over time
        recent_errors = [
            record for record in self.error_records
            if record.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]

        # Affected URLs
        affected_urls = list(set([record.url for record in self.error_records]))

        return {
            "total_errors": len(self.error_records),
            "unique_errors": len(error_counter),
            "most_common_errors": most_common,
            "errors_by_category": dict(category_stats),
            "recent_errors_24h": len(recent_errors),
            "affected_urls": len(affected_urls),
            "error_patterns": len(self.error_patterns),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": len(self.error_records),
            "errors_by_severity": self._get_errors_by_severity(),
            "errors_by_category": dict(self.error_stats),
            "most_common_errors": Counter([r.message for r in self.error_records]).most_common(10),
            "error_rate_by_hour": self._calculate_error_rate_by_hour(),
            "top_error_urls": self._get_top_error_urls()
        }

    def _get_errors_by_severity(self) -> Dict[str, int]:
        """Get error count by severity."""
        severity_count = defaultdict(int)
        for record in self.error_records:
            severity_count[record.severity.value] += 1
        return dict(severity_count)

    def _calculate_error_rate_by_hour(self) -> Dict[str, int]:
        """Calculate error rate by hour for the last 24 hours."""
        now = datetime.now(timezone.utc)
        hourly_errors = defaultdict(int)

        for record in self.error_records:
            if record.timestamp > now - timedelta(hours=24):
                hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
                hourly_errors[hour_key] += 1

        return dict(hourly_errors)

    def _get_top_error_urls(self) -> List[Tuple[str, int]]:
        """Get URLs with the most errors."""
        url_errors = defaultdict(int)
        for record in self.error_records:
            url_errors[record.url] += 1

        return sorted(url_errors.items(), key=lambda x: x[1], reverse=True)[:10]

    async def crawl_with_error_handling(
        self,
        browser_manager: PlaywrightBrowserManager,
        url: str,
        config: Optional[BrowserConfig] = None
    ) -> CrawlResult:
        """Crawl with comprehensive error handling."""
        config = config or BrowserConfig()

        # Enable debug mode if requested
        if self.debug_mode:
            config = self.enable_debug_mode(config)

        # Start debug session if enabled
        debug_session = None
        if self.debug_mode:
            debug_session = self._start_debug_session(url, config)

        try:
            result = await browser_manager.crawl_url(url, config=config)

            # Process result for errors
            await self._process_crawl_result(result, debug_session)

            return result

        except Exception as e:
            # Record the exception
            error_record = self.record_error(
                error_message=str(e),
                url=url,
                stack_trace=traceback.format_exc()
            )

            # Return failed result
            return CrawlResult(
                success=False,
                url=url,
                error_message=str(e)
            )

        finally:
            if debug_session:
                self._end_debug_session(debug_session.session_id)

    async def crawl_with_retry(
        self,
        browser_manager: PlaywrightBrowserManager,
        url: str,
        config: Optional[BrowserConfig] = None,
        max_retries: Optional[int] = None
    ) -> CrawlResult:
        """Crawl with automatic retry on errors."""
        max_retries = max_retries or self.max_retries
        config = config or BrowserConfig()

        last_result = None

        for attempt in range(max_retries + 1):
            try:
                result = await self.crawl_with_error_handling(
                    browser_manager, url, config
                )

                if result.success:
                    return result

                last_result = result

                # Don't retry on final attempt
                if attempt < max_retries:
                    logger.info(f"Retry {attempt + 1}/{max_retries} for {url}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return CrawlResult(
                        success=False,
                        url=url,
                        error_message=str(e)
                    )

        return last_result or CrawlResult(
            success=False,
            url=url,
            error_message="All retry attempts failed"
        )

    async def crawl_with_recovery(
        self,
        browser_manager: PlaywrightBrowserManager,
        url: str,
        config: Optional[BrowserConfig] = None
    ) -> CrawlResult:
        """Crawl with automatic error recovery strategies."""
        config = config or BrowserConfig()

        # Try normal crawl first
        result = await self.crawl_with_error_handling(browser_manager, url, config)

        if result.success:
            return result

        # Determine recovery strategies based on error
        strategies = self._determine_recovery_strategies(result)

        for strategy in strategies:
            logger.info(f"Trying recovery strategy: {strategy.value}")

            recovery_config = self.create_fallback_config(config, {strategy.value: True})

            recovery_result = await self.crawl_with_retry(
                browser_manager, url, recovery_config, max_retries=1
            )

            if recovery_result.success:
                logger.info(f"Recovery successful with strategy: {strategy.value}")
                return recovery_result

        # All recovery attempts failed
        return result

    def create_fallback_config(
        self,
        original_config: BrowserConfig,
        strategy: Dict[str, Any]
    ) -> BrowserConfig:
        """Create fallback configuration based on recovery strategy."""
        # Create a copy of the original config
        fallback_config = BrowserConfig(
            engine=original_config.engine,
            viewport=original_config.viewport,
            user_agent=original_config.user_agent,
            timeout=original_config.timeout,
            wait_strategy=original_config.wait_strategy,
            enable_javascript=original_config.enable_javascript,
            block_images=original_config.block_images,
            block_fonts=original_config.block_fonts,
            block_ads=original_config.block_ads
        )

        # Apply strategy modifications
        if strategy.get("disable_javascript"):
            fallback_config.enable_javascript = False

        if strategy.get("change_user_agent"):
            fallback_config.user_agent = "Mozilla/5.0 (compatible; ErrorRecoveryBot/1.0)"

        if strategy.get("increase_timeout"):
            fallback_config.timeout = original_config.timeout * 2

        if strategy.get("use_different_engine"):
            if original_config.engine == BrowserEngine.CHROMIUM:
                fallback_config.engine = BrowserEngine.FIREFOX
            else:
                fallback_config.engine = BrowserEngine.CHROMIUM

        return fallback_config

    def _determine_recovery_strategies(self, failed_result: CrawlResult) -> List[RecoveryStrategy]:
        """Determine appropriate recovery strategies based on error."""
        strategies = []

        if not failed_result.error_message:
            return [RecoveryStrategy.RETRY]

        error_msg = failed_result.error_message.lower()

        if "javascript" in error_msg or "script" in error_msg:
            strategies.extend([
                RecoveryStrategy.DISABLE_JAVASCRIPT,
                RecoveryStrategy.CHANGE_USER_AGENT
            ])

        if "timeout" in error_msg:
            strategies.extend([
                RecoveryStrategy.INCREASE_TIMEOUT,
                RecoveryStrategy.USE_DIFFERENT_ENGINE
            ])

        if "network" in error_msg or "connection" in error_msg:
            strategies.extend([
                RecoveryStrategy.CHANGE_USER_AGENT,
                RecoveryStrategy.RETRY
            ])

        # Default fallback
        if not strategies:
            strategies = [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CHANGE_USER_AGENT,
                RecoveryStrategy.INCREASE_TIMEOUT
            ]

        return strategies

    def enable_debug_mode(self, config: BrowserConfig) -> BrowserConfig:
        """Enable debug mode configuration."""
        debug_config = BrowserConfig(
            engine=config.engine,
            viewport=config.viewport,
            user_agent=config.user_agent,
            timeout=config.timeout * 2,  # Increase timeout for debugging
            wait_strategy=WaitStrategy.NETWORKIDLE,  # Use most thorough wait strategy
            enable_javascript=config.enable_javascript,
            block_images=False,  # Don't block resources in debug mode
            block_fonts=False,
            block_ads=False
        )

        return debug_config

    def create_detailed_error_report(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed error report for debugging."""
        return {
            "error_message": error_info.get("message", ""),
            "stack_trace": error_info.get("stack", ""),
            "location": {
                "line": error_info.get("line"),
                "column": error_info.get("column"),
                "url": error_info.get("url", "")
            },
            "source_file": error_info.get("filename", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_agent": error_info.get("user_agent", ""),
            "page_title": error_info.get("page_title", ""),
            "categorization": self.categorize_error(error_info.get("message", "")),
            "debug_context": {
                "debug_mode": self.debug_mode,
                "capture_screenshots": self.capture_screenshots,
                "enable_profiling": self.enable_profiling
            }
        }

    def enhance_error_with_debug_info(self, result: CrawlResult) -> Dict[str, Any]:
        """Enhance error result with debug information."""
        debug_info = {
            "original_error": result.error_message,
            "url": result.url,
            "success": result.success,
            "load_time": result.load_time,
            "has_screenshot": False,
            "screenshot_size": 0,
            "console_log_count": len(result.console_logs) if result.console_logs else 0,
            "network_request_count": len(result.network_requests) if result.network_requests else 0
        }

        # Add screenshot info if available
        if hasattr(result, 'screenshot_data') and result.screenshot_data:
            debug_info["has_screenshot"] = True
            debug_info["screenshot_size"] = len(result.screenshot_data)

        # Add performance metrics if available
        if hasattr(result, 'page_metrics') and result.page_metrics:
            debug_info["performance_metrics"] = result.page_metrics

        return debug_info

    def create_performance_profile(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance profile for debugging."""
        profile = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "navigation_timing": performance_data.get("navigation_timing", {}),
            "resource_timing": performance_data.get("resource_timing", []),
            "memory_usage": performance_data.get("memory_usage", {}),
            "total_load_time": 0,
            "performance_score": 0
        }

        # Calculate total load time
        nav_timing = performance_data.get("navigation_timing", {})
        if nav_timing.get("load_complete"):
            profile["total_load_time"] = nav_timing["load_complete"]

        # Calculate basic performance score (0-100)
        load_time = profile["total_load_time"]
        if load_time > 0:
            if load_time < 1000:  # < 1 second
                profile["performance_score"] = 100
            elif load_time < 3000:  # < 3 seconds
                profile["performance_score"] = 80
            elif load_time < 5000:  # < 5 seconds
                profile["performance_score"] = 60
            else:
                profile["performance_score"] = 40

        return profile

    def _start_debug_session(self, url: str, config: BrowserConfig) -> DebugSession:
        """Start a debug session."""
        session_id = f"debug_{datetime.now().timestamp()}"

        debug_session = DebugSession(
            session_id=session_id,
            url=url,
            start_time=datetime.now(timezone.utc),
            config=config
        )

        self.debug_sessions[session_id] = debug_session
        return debug_session

    def _end_debug_session(self, session_id: str):
        """End a debug session."""
        if session_id in self.debug_sessions:
            session = self.debug_sessions[session_id]
            # Could save session data, generate reports, etc.
            del self.debug_sessions[session_id]

    async def _process_crawl_result(self, result: CrawlResult, debug_session: Optional[DebugSession]):
        """Process crawl result for error analysis."""
        # Record console errors
        if result.console_logs:
            for log in result.console_logs:
                if log.get("type") == "error":
                    self.record_error(
                        error_message=log.get("message", ""),
                        url=result.url,
                        additional_context={"log_type": "console"}
                    )

        # Record if crawl failed
        if not result.success and result.error_message:
            self.record_error(
                error_message=result.error_message,
                url=result.url,
                additional_context={"crawl_failure": True}
            )

        # Add to debug session if active
        if debug_session:
            if result.console_logs:
                debug_session.console_logs.extend(result.console_logs)

            if result.network_requests:
                debug_session.network_logs.extend(result.network_requests)

            if hasattr(result, 'page_metrics'):
                debug_session.performance_metrics = result.page_metrics


# Global instance
_javascript_error_handler_instance: Optional[JavaScriptErrorHandler] = None

def get_javascript_error_handler() -> JavaScriptErrorHandler:
    """Get global JavaScript error handler instance."""
    global _javascript_error_handler_instance

    if _javascript_error_handler_instance is None:
        _javascript_error_handler_instance = JavaScriptErrorHandler(
            debug_mode=False,
            capture_screenshots=True,
            enable_profiling=True
        )

    return _javascript_error_handler_instance

async def crawl_with_error_handling(
    browser_manager: PlaywrightBrowserManager,
    url: str,
    config: Optional[BrowserConfig] = None
) -> CrawlResult:
    """Convenience function for crawling with error handling."""
    error_handler = get_javascript_error_handler()
    return await error_handler.crawl_with_error_handling(browser_manager, url, config)