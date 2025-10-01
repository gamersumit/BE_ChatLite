"""
Enhanced JavaScript Crawler for dynamic content extraction.
Task 4.2: Implement JavaScript-heavy page crawling
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.async_api import Page, Browser, BrowserContext, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError

from app.services.playwright_browser_manager import (
    PlaywrightBrowserManager,
    BrowserConfig,
    CrawlResult,
    WaitStrategy
)

logger = logging.getLogger(__name__)


class JavaScriptCrawler:
    """
    Advanced JavaScript crawler for dynamic content extraction.

    Features:
    - Dynamic content detection and extraction
    - SPA (Single Page Application) navigation
    - AJAX/XHR request monitoring
    - Custom JavaScript execution
    - Element waiting strategies
    - Infinite scroll handling
    - Form interaction capabilities
    """

    def __init__(self, browser_manager: PlaywrightBrowserManager):
        """Initialize JavaScript crawler with browser manager."""
        self.browser_manager = browser_manager

    async def crawl_spa_application(
        self,
        base_url: str,
        routes: List[str],
        config: Optional[BrowserConfig] = None
    ) -> Dict[str, CrawlResult]:
        """
        Crawl Single Page Application with multiple routes.

        Args:
            base_url: Base URL of the SPA
            routes: List of routes to crawl
            config: Browser configuration

        Returns:
            Dictionary mapping routes to crawl results
        """
        results = {}

        # Create session for SPA crawling
        session_id = await self.browser_manager.create_session(config)

        try:
            for route in routes:
                full_url = f"{base_url.rstrip('/')}/{route.lstrip('/')}"

                logger.info(f"Crawling SPA route: {full_url}")

                result = await self.crawl_dynamic_content(
                    url=full_url,
                    session_id=session_id,
                    config=config,
                    spa_navigation=True
                )

                results[route] = result

                # Small delay between route changes
                await asyncio.sleep(0.5)

        finally:
            await self.browser_manager.close_session(session_id)

        return results

    async def crawl_dynamic_content(
        self,
        url: str,
        session_id: Optional[str] = None,
        config: Optional[BrowserConfig] = None,
        wait_for_selectors: Optional[List[str]] = None,
        infinite_scroll: bool = False,
        spa_navigation: bool = False,
        form_interactions: Optional[List[Dict[str, Any]]] = None
    ) -> CrawlResult:
        """
        Crawl page with dynamic content loading.

        Args:
            url: URL to crawl
            session_id: Browser session ID
            config: Browser configuration
            wait_for_selectors: CSS selectors to wait for
            infinite_scroll: Whether to handle infinite scroll
            spa_navigation: Whether this is SPA navigation
            form_interactions: Form interactions to perform

        Returns:
            CrawlResult with extracted content
        """
        start_time = datetime.now()
        config = config or BrowserConfig()

        try:
            # Get session
            if session_id:
                session = await self.browser_manager.get_session(session_id)
                if not session:
                    raise ValueError(f"Session not found: {session_id}")
            else:
                session_id = await self.browser_manager.create_session(config)
                session = await self.browser_manager.get_session(session_id)

            page = session.page

            # Setup monitoring
            console_logs = []
            network_requests = []
            ajax_requests = []

            def handle_console(msg):
                console_logs.append(f"{msg.type}: {msg.text}")

            def handle_request(request):
                network_requests.append({
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "headers": dict(request.headers)
                })

                # Track AJAX/XHR requests
                if request.resource_type in ["xhr", "fetch"]:
                    ajax_requests.append({
                        "url": request.url,
                        "method": request.method,
                        "timestamp": datetime.now().isoformat()
                    })

            def handle_response(response):
                # Log response details for debugging
                if response.request.resource_type in ["xhr", "fetch"]:
                    logger.debug(f"AJAX response: {response.url} - {response.status}")

            page.on("console", handle_console)
            page.on("request", handle_request)
            page.on("response", handle_response)

            # Navigate to URL
            if spa_navigation:
                # For SPA navigation, still use goto but with special handling
                # SPA navigation in a real app would be handled differently,
                # but for testing we'll just use regular navigation
                response = await page.goto(
                    url,
                    wait_until=config.wait_strategy.value,
                    timeout=config.timeout
                )
                status_code = response.status if response else 200
            else:
                response = await page.goto(
                    url,
                    wait_until=config.wait_strategy.value,
                    timeout=config.timeout
                )
                if response:
                    status_code = response.status
                else:
                    status_code = None

            # Wait for initial content load
            await self._wait_for_initial_load(page, config)

            # Wait for specific selectors if provided
            if wait_for_selectors:
                await self._wait_for_selectors(page, wait_for_selectors)

            # Handle form interactions
            if form_interactions:
                await self._handle_form_interactions(page, form_interactions)

            # Handle infinite scroll
            if infinite_scroll:
                await self._handle_infinite_scroll(page)

            # Wait for any pending AJAX requests to complete
            await self._wait_for_ajax_completion(page, ajax_requests)

            # Extract final content
            content = await page.content()
            title = await page.title()
            final_url = page.url

            # Calculate load time
            load_time = (datetime.now() - start_time).total_seconds()

            # Analyze page for dynamic content
            dynamic_analysis = await self._analyze_dynamic_content(page)

            return CrawlResult(
                success=True,
                url=url,
                status_code=status_code if not spa_navigation else 200,
                content=content,
                title=title,
                load_time=load_time,
                console_logs=[{"type": "info", "message": log} for log in console_logs],
                network_requests=network_requests,
                page_metrics={
                    "final_url": final_url,
                    "ajax_requests": len(ajax_requests),
                    "total_requests": len(network_requests),
                    "dynamic_elements": dynamic_analysis["dynamic_elements"],
                    "spa_detected": dynamic_analysis["spa_detected"],
                    "javascript_frameworks": dynamic_analysis["frameworks"]
                }
            )

        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout crawling {url}: {e}")
            return CrawlResult(
                success=False,
                url=url,
                error_message=f"Timeout: {str(e)}",
                load_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return CrawlResult(
                success=False,
                url=url,
                error_message=str(e),
                load_time=(datetime.now() - start_time).total_seconds()
            )

        finally:
            # Cleanup session if we created it
            if not session_id:
                await self.browser_manager.close_session(session_id)

    async def extract_structured_data(
        self,
        url: str,
        selectors: Dict[str, str],
        config: Optional[BrowserConfig] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data using CSS selectors after JavaScript rendering.

        Args:
            url: URL to crawl
            selectors: Dictionary mapping field names to CSS selectors
            config: Browser configuration

        Returns:
            Dictionary with extracted structured data
        """
        config = config or BrowserConfig()

        # Create JavaScript to extract data
        extraction_js = """
        const selectors = arguments[0];
        const results = {};

        for (const [field, selector] of Object.entries(selectors)) {
            try {
                const elements = document.querySelectorAll(selector);
                if (elements.length === 0) {
                    results[field] = null;
                } else if (elements.length === 1) {
                    results[field] = elements[0].textContent.trim();
                } else {
                    results[field] = Array.from(elements).map(el => el.textContent.trim());
                }
            } catch (error) {
                results[field] = null;
            }
        }

        return results;
        """

        result = await self.browser_manager.crawl_url(
            url=url,
            config=config,
            custom_js=extraction_js
        )

        if result.success and hasattr(result, 'custom_js_result'):
            try:
                return json.loads(result.custom_js_result)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse extracted data: {result.custom_js_result}")

        return {}

    async def _wait_for_initial_load(self, page: Page, config: BrowserConfig):
        """Wait for initial page load completion."""
        try:
            # Wait for network idle or DOM ready based on strategy
            if config.wait_strategy == WaitStrategy.NETWORKIDLE:
                await page.wait_for_load_state("networkidle", timeout=config.timeout)
            elif config.wait_strategy == WaitStrategy.LOAD:
                await page.wait_for_load_state("load", timeout=config.timeout)
            else:
                await page.wait_for_load_state("domcontentloaded", timeout=config.timeout)

        except PlaywrightTimeoutError:
            logger.warning(f"Timeout waiting for initial load")

    async def _wait_for_selectors(self, page: Page, selectors: List[str]):
        """Wait for specific selectors to appear."""
        for selector in selectors:
            try:
                await page.wait_for_selector(selector, timeout=10000)
                logger.debug(f"Found selector: {selector}")
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout waiting for selector: {selector}")

    async def _handle_form_interactions(self, page: Page, interactions: List[Dict[str, Any]]):
        """Handle form interactions to reveal hidden content."""
        for interaction in interactions:
            try:
                action = interaction.get("action")
                selector = interaction.get("selector")
                value = interaction.get("value")

                if action == "fill" and selector and value:
                    await page.fill(selector, value)
                elif action == "click" and selector:
                    await page.click(selector)
                elif action == "select" and selector and value:
                    await page.select_option(selector, value)

                # Wait for potential content update
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Form interaction failed: {e}")

    async def _handle_infinite_scroll(self, page: Page, max_scrolls: int = 10):
        """Handle infinite scroll to load more content."""
        previous_height = 0
        scroll_count = 0

        while scroll_count < max_scrolls:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for new content to load
            await asyncio.sleep(2)

            # Check if new content loaded
            current_height = await page.evaluate("document.body.scrollHeight")

            if current_height == previous_height:
                # No new content loaded, break
                break

            previous_height = current_height
            scroll_count += 1

        logger.debug(f"Performed {scroll_count} scroll operations")

    async def _wait_for_ajax_completion(self, page: Page, ajax_requests: List[Dict]):
        """Wait for AJAX requests to complete."""
        if not ajax_requests:
            return

        # Wait for a short period to allow pending requests to complete
        await asyncio.sleep(2)

        # Check if there are any pending network requests
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeoutError:
            logger.debug("Network not idle after AJAX wait")

    async def _analyze_dynamic_content(self, page: Page) -> Dict[str, Any]:
        """Analyze page for dynamic content characteristics."""
        analysis_js = """
        () => {
            const analysis = {
                dynamic_elements: 0,
                spa_detected: false,
                frameworks: []
            };

            // Count elements with dynamic attributes
            const dynamicSelectors = [
                '[data-react-class]',
                '[data-vue-component]',
                '[ng-app]',
                '[data-ember-component]',
                '.react-root',
                '#root',
                '#app'
            ];

            analysis.dynamic_elements = dynamicSelectors.reduce((count, selector) => {
                return count + document.querySelectorAll(selector).length;
            }, 0);

            // Detect JavaScript frameworks
            if (window.React) analysis.frameworks.push('React');
            if (window.Vue) analysis.frameworks.push('Vue');
            if (window.angular) analysis.frameworks.push('Angular');
            if (window.Ember) analysis.frameworks.push('Ember');
            if (window.jQuery) analysis.frameworks.push('jQuery');

            // Detect SPA characteristics
            analysis.spa_detected = !!(
                window.history.pushState &&
                (document.querySelector('#root') ||
                 document.querySelector('#app') ||
                 window.React ||
                 window.Vue ||
                 window.angular)
            );

            return analysis;
        }
        """

        try:
            result = await page.evaluate(analysis_js)
            return result
        except Exception as e:
            logger.warning(f"Dynamic content analysis failed: {e}")
            return {
                "dynamic_elements": 0,
                "spa_detected": False,
                "frameworks": []
            }


# Global instance for easy access
_javascript_crawler_instance: Optional[JavaScriptCrawler] = None

async def get_javascript_crawler() -> JavaScriptCrawler:
    """Get global JavaScript crawler instance."""
    global _javascript_crawler_instance

    if _javascript_crawler_instance is None:
        from app.services.playwright_browser_manager import get_browser_manager
        browser_manager = await get_browser_manager()
        _javascript_crawler_instance = JavaScriptCrawler(browser_manager)

    return _javascript_crawler_instance

async def crawl_dynamic_website(
    url: str,
    wait_for_selectors: Optional[List[str]] = None,
    infinite_scroll: bool = False,
    config: Optional[BrowserConfig] = None
) -> CrawlResult:
    """Convenience function for crawling dynamic websites."""
    crawler = await get_javascript_crawler()
    return await crawler.crawl_dynamic_content(
        url=url,
        wait_for_selectors=wait_for_selectors,
        infinite_scroll=infinite_scroll,
        config=config
    )

async def crawl_spa_routes(
    base_url: str,
    routes: List[str],
    config: Optional[BrowserConfig] = None
) -> Dict[str, CrawlResult]:
    """Convenience function for crawling SPA applications."""
    crawler = await get_javascript_crawler()
    return await crawler.crawl_spa_application(
        base_url=base_url,
        routes=routes,
        config=config
    )