"""
Playwright Browser Automation Framework for JavaScript-heavy page crawling.

This service provides comprehensive browser automation capabilities using Playwright
with support for multiple browser engines, browser pool management, and session lifecycle.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    ViewportSize,
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError
)

logger = logging.getLogger(__name__)


class BrowserEngine(Enum):
    """Supported browser engines."""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class BrowserMode(Enum):
    """Browser execution modes."""
    HEADLESS = "headless"
    HEADED = "headed"


class WaitStrategy(Enum):
    """Page load wait strategies."""
    NETWORKIDLE = "networkidle"
    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    COMMIT = "commit"


@dataclass
class BrowserConfig:
    """Browser configuration settings."""
    engine: BrowserEngine = BrowserEngine.CHROMIUM
    mode: BrowserMode = BrowserMode.HEADLESS
    viewport: ViewportSize = field(default_factory=lambda: {"width": 1920, "height": 1080})
    user_agent: Optional[str] = None
    timeout: int = 30000  # 30 seconds
    wait_strategy: WaitStrategy = WaitStrategy.NETWORKIDLE
    enable_javascript: bool = True
    block_images: bool = False
    block_fonts: bool = False
    block_ads: bool = False
    extra_args: List[str] = field(default_factory=list)


@dataclass
class BrowserSession:
    """Browser session information."""
    session_id: str
    browser: Browser
    context: BrowserContext
    page: Page
    config: BrowserConfig
    created_at: datetime
    last_used: datetime
    is_active: bool = True


@dataclass
class CrawlResult:
    """Crawl operation result."""
    url: str
    success: bool
    status_code: Optional[int] = None
    content: Optional[str] = None
    title: Optional[str] = None
    screenshot_path: Optional[str] = None
    pdf_path: Optional[str] = None
    error_message: Optional[str] = None
    load_time: float = 0.0
    page_metrics: Dict[str, Any] = field(default_factory=dict)
    console_logs: List[Dict[str, str]] = field(default_factory=list)
    network_requests: List[Dict[str, Any]] = field(default_factory=list)


class PlaywrightBrowserManager:
    """
    Comprehensive Playwright browser automation manager.

    Features:
    - Multiple browser engine support (Chromium, Firefox, WebKit)
    - Browser pool management with configurable concurrency
    - Session lifecycle management
    - Resource optimization and performance monitoring
    - Screenshot and PDF generation
    - Network request interception
    - Console log capture
    """

    def __init__(
        self,
        max_concurrent_browsers: int = 5,
        session_timeout: int = 300,  # 5 minutes
        default_config: Optional[BrowserConfig] = None
    ):
        """Initialize the browser manager."""
        self.max_concurrent_browsers = max_concurrent_browsers
        self.session_timeout = session_timeout
        self.default_config = default_config or BrowserConfig()

        self._playwright: Optional[Playwright] = None
        self._browser_sessions: Dict[str, BrowserSession] = {}
        self._session_queue = asyncio.Queue()
        self._is_initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize Playwright and start the browser manager."""
        if self._is_initialized:
            return

        try:
            self._playwright = await async_playwright().start()
            self._is_initialized = True

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

            logger.info("Playwright browser manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the browser manager and cleanup resources."""
        if not self._is_initialized:
            return

        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()

            # Close all browser sessions
            for session_id in list(self._browser_sessions.keys()):
                await self._close_session(session_id)

            # Stop Playwright
            if self._playwright:
                await self._playwright.stop()

            self._is_initialized = False
            logger.info("Playwright browser manager shutdown successfully")

        except Exception as e:
            logger.error(f"Error during browser manager shutdown: {e}")

    async def create_session(
        self,
        config: Optional[BrowserConfig] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new browser session.

        Args:
            config: Browser configuration (uses default if not provided)
            session_id: Custom session ID (auto-generated if not provided)

        Returns:
            Session ID for the created session
        """
        if not self._is_initialized:
            await self.initialize()

        if len(self._browser_sessions) >= self.max_concurrent_browsers:
            raise RuntimeError(f"Maximum concurrent browsers ({self.max_concurrent_browsers}) reached")

        config = config or self.default_config
        session_id = session_id or f"session_{datetime.now().timestamp()}"

        try:
            # Get browser instance
            browser = await self._get_browser_instance(config.engine, config)

            # Create browser context
            context = await self._create_browser_context(browser, config)

            # Create page
            page = await context.new_page()

            # Configure page
            await self._configure_page(page, config)

            # Create session object
            session = BrowserSession(
                session_id=session_id,
                browser=browser,
                context=context,
                page=page,
                config=config,
                created_at=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc)
            )

            self._browser_sessions[session_id] = session

            logger.info(f"Created browser session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """Get an existing browser session."""
        session = self._browser_sessions.get(session_id)
        if session:
            session.last_used = datetime.now(timezone.utc)
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close a browser session and cleanup resources."""
        return await self._close_session(session_id)

    async def crawl_url(
        self,
        url: str,
        session_id: Optional[str] = None,
        config: Optional[BrowserConfig] = None,
        capture_screenshot: bool = False,
        generate_pdf: bool = False,
        custom_js: Optional[str] = None
    ) -> CrawlResult:
        """
        Crawl a URL and extract content.

        Args:
            url: URL to crawl
            session_id: Existing session ID (creates new if not provided)
            config: Browser configuration
            capture_screenshot: Whether to capture a screenshot
            generate_pdf: Whether to generate a PDF
            custom_js: Custom JavaScript to execute

        Returns:
            CrawlResult with extracted content and metadata
        """
        start_time = datetime.now()

        try:
            # Get or create session
            if session_id:
                session = await self.get_session(session_id)
                if not session:
                    raise ValueError(f"Session not found: {session_id}")
            else:
                session_id = await self.create_session(config)
                session = await self.get_session(session_id)

            page = session.page

            # Setup request/response monitoring
            console_logs = []
            network_requests = []

            def handle_console(msg):
                console_logs.append({
                    "type": msg.type,
                    "text": msg.text,
                    "timestamp": datetime.now().isoformat()
                })

            def handle_request(request):
                network_requests.append({
                    "url": request.url,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "timestamp": datetime.now().isoformat()
                })

            page.on("console", handle_console)
            page.on("request", handle_request)

            # Navigate to URL
            response = await page.goto(
                url,
                wait_until=session.config.wait_strategy.value,
                timeout=session.config.timeout
            )

            # Wait for additional time if needed
            if session.config.wait_strategy == WaitStrategy.NETWORKIDLE:
                await page.wait_for_load_state("networkidle")

            # Execute custom JavaScript if provided
            if custom_js:
                await page.evaluate(custom_js)

            # Extract content
            content = await page.content()
            title = await page.title()

            # Get performance metrics
            page_metrics = await page.evaluate("""() => {
                const navigation = performance.getEntriesByType('navigation')[0];
                return {
                    loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
                    domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
                    firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                    firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
                };
            }""")

            # Capture screenshot if requested
            screenshot_path = None
            if capture_screenshot:
                screenshot_path = await self._capture_screenshot(page, url)

            # Generate PDF if requested
            pdf_path = None
            if generate_pdf:
                pdf_path = await self._generate_pdf(page, url)

            # Calculate total load time
            load_time = (datetime.now() - start_time).total_seconds()

            return CrawlResult(
                url=url,
                success=True,
                status_code=response.status if response else None,
                content=content,
                title=title,
                screenshot_path=screenshot_path,
                pdf_path=pdf_path,
                load_time=load_time,
                page_metrics=page_metrics,
                console_logs=console_logs,
                network_requests=network_requests[:100]  # Limit to prevent memory issues
            )

        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout crawling {url}: {e}")
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Timeout: {str(e)}",
                load_time=(datetime.now() - start_time).total_seconds()
            )

        except PlaywrightError as e:
            logger.error(f"Playwright error crawling {url}: {e}")
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Playwright error: {str(e)}",
                load_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {e}")
            return CrawlResult(
                url=url,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                load_time=(datetime.now() - start_time).total_seconds()
            )

    async def get_browser_health(self) -> Dict[str, Any]:
        """Get browser manager health information."""
        active_sessions = len([s for s in self._browser_sessions.values() if s.is_active])

        return {
            "initialized": self._is_initialized,
            "total_sessions": len(self._browser_sessions),
            "active_sessions": active_sessions,
            "max_concurrent_browsers": self.max_concurrent_browsers,
            "session_timeout": self.session_timeout,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "engine": session.config.engine.value,
                    "mode": session.config.mode.value,
                    "created_at": session.created_at.isoformat(),
                    "last_used": session.last_used.isoformat(),
                    "is_active": session.is_active
                }
                for session in self._browser_sessions.values()
            ]
        }

    async def _get_browser_instance(self, engine: BrowserEngine, config: BrowserConfig) -> Browser:
        """Get a browser instance for the specified engine."""
        browser_args = {
            "headless": config.mode == BrowserMode.HEADLESS,
            "args": config.extra_args
        }

        if engine == BrowserEngine.CHROMIUM:
            # Add Chromium-specific optimizations
            browser_args["args"].extend([
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ])
            return await self._playwright.chromium.launch(**browser_args)

        elif engine == BrowserEngine.FIREFOX:
            return await self._playwright.firefox.launch(**browser_args)

        elif engine == BrowserEngine.WEBKIT:
            return await self._playwright.webkit.launch(**browser_args)

        else:
            raise ValueError(f"Unsupported browser engine: {engine}")

    async def _create_browser_context(self, browser: Browser, config: BrowserConfig) -> BrowserContext:
        """Create a browser context with optimized settings."""
        context_options = {
            "viewport": config.viewport,
            "user_agent": config.user_agent,
            "java_script_enabled": config.enable_javascript,
            "bypass_csp": True,  # Bypass Content Security Policy
            "ignore_https_errors": True
        }

        context = await browser.new_context(**context_options)

        # Set up request interception for resource blocking
        if config.block_images or config.block_fonts or config.block_ads:
            await context.route("**/*", self._handle_route_interception(config))

        return context

    def _handle_route_interception(self, config: BrowserConfig):
        """Create a route handler for request interception."""
        async def handle_route(route):
            request = route.request
            resource_type = request.resource_type

            # Block resources based on config
            should_block = False

            if config.block_images and resource_type in ["image", "imageset"]:
                should_block = True
            elif config.block_fonts and resource_type == "font":
                should_block = True
            elif config.block_ads and self._is_ad_request(request.url):
                should_block = True

            if should_block:
                await route.abort()
            else:
                await route.continue_()

        return handle_route

    def _is_ad_request(self, url: str) -> bool:
        """Check if a request URL is likely an advertisement."""
        ad_patterns = [
            "doubleclick.net", "googleadservices.com", "googlesyndication.com",
            "amazon-adsystem.com", "facebook.com/tr", "google-analytics.com",
            "googletagmanager.com", "/ads/", "/ad/", "adsystem", "advertising"
        ]
        return any(pattern in url.lower() for pattern in ad_patterns)

    async def _configure_page(self, page: Page, config: BrowserConfig) -> None:
        """Configure page settings."""
        # Set timeout
        page.set_default_timeout(config.timeout)

        # Add custom headers if needed
        await page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
        })

    async def _capture_screenshot(self, page: Page, url: str) -> str:
        """Capture a screenshot of the current page."""
        try:
            screenshots_dir = Path("screenshots")
            screenshots_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{hash(url) % 10000}.png"
            filepath = screenshots_dir / filename

            await page.screenshot(path=str(filepath), full_page=True)
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

    async def _generate_pdf(self, page: Page, url: str) -> str:
        """Generate a PDF of the current page."""
        try:
            pdfs_dir = Path("pdfs")
            pdfs_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pdf_{timestamp}_{hash(url) % 10000}.pdf"
            filepath = pdfs_dir / filename

            await page.pdf(path=str(filepath), format="A4")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return None

    async def _close_session(self, session_id: str) -> bool:
        """Close a specific browser session."""
        session = self._browser_sessions.get(session_id)
        if not session:
            return False

        try:
            await session.page.close()
            await session.context.close()
            await session.browser.close()

            del self._browser_sessions[session_id]
            logger.info(f"Closed browser session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    async def _cleanup_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_sessions = []

                for session_id, session in self._browser_sessions.items():
                    time_since_last_use = (current_time - session.last_used).total_seconds()
                    if time_since_last_use > self.session_timeout:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    await self._close_session(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")

                # Sleep for 30 seconds before next cleanup
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(30)


# Global browser manager instance
_browser_manager: Optional[PlaywrightBrowserManager] = None


async def get_browser_manager() -> PlaywrightBrowserManager:
    """Get the global browser manager instance."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = PlaywrightBrowserManager()
        await _browser_manager.initialize()
    return _browser_manager


async def create_browser_session(config: Optional[BrowserConfig] = None) -> str:
    """Create a new browser session using the global manager."""
    manager = await get_browser_manager()
    return await manager.create_session(config)


async def crawl_with_playwright(
    url: str,
    session_id: Optional[str] = None,
    config: Optional[BrowserConfig] = None,
    **kwargs
) -> CrawlResult:
    """Crawl a URL using Playwright with the global manager."""
    manager = await get_browser_manager()
    return await manager.crawl_url(url, session_id, config, **kwargs)


async def shutdown_browser_manager() -> None:
    """Shutdown the global browser manager."""
    global _browser_manager
    if _browser_manager:
        await _browser_manager.shutdown()
        _browser_manager = None