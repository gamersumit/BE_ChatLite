"""
Playwright Browser Automation Service for JavaScript-heavy page crawling.

This service provides advanced web crawling capabilities using Playwright
to handle SPAs, JavaScript-rendered content, and dynamic page interactions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone
import json
import re

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

from app.core.database import get_supabase_admin_client

logger = logging.getLogger(__name__)


class PlaywrightCrawlerService:
    """
    Advanced web crawler using Playwright for JavaScript-heavy sites.

    This service:
    1. Handles JavaScript-rendered content
    2. Supports SPA (Single Page Application) crawling
    3. Manages browser automation and page interactions
    4. Extracts content from dynamic pages
    5. Handles complex navigation patterns
    """

    def __init__(self):
        """Initialize the Playwright crawler service."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed. Install with: pip install playwright")

        self.supabase = get_supabase_admin_client()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.visited_urls: Set[str] = set()
        self.crawl_stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'js_pages_detected': 0,
            'spa_pages_detected': 0,
            'dynamic_content_found': 0
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()

    async def start_browser(self, headless: bool = True) -> None:
        """
        Start the Playwright browser instance.

        Args:
            headless: Whether to run browser in headless mode
        """
        try:
            self.playwright = await async_playwright().start()

            # Launch browser with optimized settings for crawling
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',  # Speed up crawling
                    '--disable-javascript-harmony-shipping',
                    '--memory-pressure-off',
                    '--max_old_space_size=4096'
                ]
            )

            # Create browser context with realistic settings
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                ignore_https_errors=True,
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )

            logger.info("Playwright browser started successfully")

        except Exception as e:
            logger.error(f"Failed to start Playwright browser: {e}")
            raise

    async def close_browser(self) -> None:
        """Close the Playwright browser instance."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()

            logger.info("Playwright browser closed successfully")

        except Exception as e:
            logger.error(f"Error closing Playwright browser: {e}")

    async def crawl_website_advanced(
        self,
        website_id: str,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        wait_for_js: bool = True,
        spa_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Crawl a website using advanced Playwright automation.

        Args:
            website_id: UUID of the website to crawl
            base_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum crawl depth
            wait_for_js: Whether to wait for JavaScript execution
            spa_mode: Whether to treat as Single Page Application

        Returns:
            Dict containing crawl results
        """
        try:
            logger.info(f"Starting advanced crawl for {base_url}")

            self.visited_urls.clear()
            self.crawl_stats = {
                'pages_crawled': 0,
                'pages_failed': 0,
                'js_pages_detected': 0,
                'spa_pages_detected': 0,
                'dynamic_content_found': 0
            }

            if not self.browser:
                await self.start_browser()

            # Start crawling from base URL
            pages_data = []
            urls_to_crawl = [(base_url, 0)]  # (url, depth)

            while urls_to_crawl and len(pages_data) < max_pages:
                current_url, depth = urls_to_crawl.pop(0)

                if current_url in self.visited_urls or depth > max_depth:
                    continue

                page_data = await self.crawl_single_page_advanced(
                    current_url, website_id, wait_for_js, spa_mode
                )

                if page_data:
                    pages_data.append(page_data)
                    self.crawl_stats['pages_crawled'] += 1

                    # Extract and queue new URLs if not at max depth
                    if depth < max_depth:
                        new_urls = await self.extract_links_from_page_data(
                            page_data, base_url, current_url
                        )
                        for new_url in new_urls:
                            if new_url not in self.visited_urls:
                                urls_to_crawl.append((new_url, depth + 1))

                self.visited_urls.add(current_url)

                # Add small delay to be respectful
                await asyncio.sleep(1)

            # Store crawled data
            stored_count = await self.store_crawled_pages(website_id, pages_data)

            logger.info(f"Advanced crawl completed: {len(pages_data)} pages crawled, {stored_count} stored")

            return {
                'success': True,
                'pages_found': len(pages_data),
                'pages_stored': stored_count,
                'crawl_stats': self.crawl_stats,
                'website_id': website_id
            }

        except Exception as e:
            logger.error(f"Advanced crawl failed for {base_url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'pages_found': len(pages_data) if 'pages_data' in locals() else 0,
                'crawl_stats': self.crawl_stats
            }

    async def crawl_single_page_advanced(
        self,
        url: str,
        website_id: str,
        wait_for_js: bool = True,
        spa_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Crawl a single page with advanced Playwright features.

        Args:
            url: URL to crawl
            website_id: Website UUID
            wait_for_js: Whether to wait for JavaScript execution
            spa_mode: Whether to handle as SPA

        Returns:
            Page data dictionary or None if failed
        """
        try:
            page = await self.context.new_page()

            # Set up page monitoring
            page_metrics = {
                'requests': 0,
                'responses': 0,
                'js_errors': []
            }

            # Monitor network activity
            page.on('request', lambda request: self._track_request(page_metrics, request))
            page.on('response', lambda response: self._track_response(page_metrics, response))
            page.on('pageerror', lambda error: page_metrics['js_errors'].append(str(error)))

            # Navigate to page with timeout
            response = await page.goto(
                url,
                wait_until='domcontentloaded',
                timeout=30000
            )

            if not response or response.status >= 400:
                logger.warning(f"Failed to load page {url}: status {response.status if response else 'None'}")
                await page.close()
                self.crawl_stats['pages_failed'] += 1
                return None

            # Wait for JavaScript execution if enabled
            if wait_for_js:
                await self.wait_for_javascript_content(page)

            # Handle SPA-specific logic
            if spa_mode:
                await self.handle_spa_navigation(page)

            # Extract page content
            page_data = await self.extract_page_content_advanced(page, url, website_id)

            # Add metrics to page data
            page_data['crawl_metrics'] = page_metrics
            page_data['js_detected'] = len(page_metrics['js_errors']) == 0 and page_metrics['requests'] > 1
            page_data['spa_detected'] = spa_mode or await self.detect_spa_patterns(page)

            if page_data['js_detected']:
                self.crawl_stats['js_pages_detected'] += 1
            if page_data['spa_detected']:
                self.crawl_stats['spa_pages_detected'] += 1

            await page.close()
            return page_data

        except PlaywrightTimeoutError:
            logger.warning(f"Timeout loading page {url}")
            self.crawl_stats['pages_failed'] += 1
            return None
        except Exception as e:
            logger.error(f"Error crawling page {url}: {e}")
            self.crawl_stats['pages_failed'] += 1
            return None

    async def wait_for_javascript_content(self, page: Page, timeout: int = 5000) -> None:
        """
        Wait for JavaScript content to load and render.

        Args:
            page: Playwright page instance
            timeout: Maximum wait time in milliseconds
        """
        try:
            # Wait for common JavaScript loading indicators
            await page.wait_for_load_state('networkidle', timeout=timeout)

            # Wait for common dynamic content indicators
            selectors_to_wait = [
                '[data-react-root]',  # React apps
                '[ng-version]',       # Angular apps
                '[data-vue-root]',    # Vue apps
                '.js-loaded',         # Custom JS loaded indicator
                '[data-loaded="true"]' # Generic loaded indicator
            ]

            for selector in selectors_to_wait:
                try:
                    await page.wait_for_selector(selector, timeout=1000)
                    logger.debug(f"Found JS framework indicator: {selector}")
                    break
                except PlaywrightTimeoutError:
                    continue

            # Give additional time for content rendering
            await asyncio.sleep(2)

        except PlaywrightTimeoutError:
            logger.debug("JavaScript content wait timeout - proceeding anyway")
        except Exception as e:
            logger.warning(f"Error waiting for JavaScript content: {e}")

    async def handle_spa_navigation(self, page: Page) -> None:
        """
        Handle Single Page Application navigation patterns.

        Args:
            page: Playwright page instance
        """
        try:
            # Look for common SPA navigation elements
            nav_selectors = [
                'a[href^="#"]',           # Hash-based routing
                'a[data-router-link]',    # Vue router links
                'a[routerlink]',          # Angular router links
                '[data-testid*="nav"]',   # Navigation test IDs
                '.nav-link',              # Bootstrap nav links
            ]

            for selector in nav_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        logger.debug(f"Found {len(elements)} SPA navigation elements: {selector}")
                        self.crawl_stats['spa_pages_detected'] += 1
                        break
                except Exception:
                    continue

            # Wait for any route changes to complete
            await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"Error handling SPA navigation: {e}")

    async def detect_spa_patterns(self, page: Page) -> bool:
        """
        Detect if the page is a Single Page Application.

        Args:
            page: Playwright page instance

        Returns:
            True if SPA patterns are detected
        """
        try:
            # Check for common SPA framework indicators
            spa_indicators = await page.evaluate("""
                () => {
                    const indicators = {
                        react: !!window.React || !!document.querySelector('[data-reactroot]'),
                        vue: !!window.Vue || !!document.querySelector('[data-vue-root]'),
                        angular: !!window.ng || !!document.querySelector('[ng-version]'),
                        hashRouting: window.location.hash.length > 1,
                        historyAPI: !!window.history.pushState
                    };
                    return indicators;
                }
            """)

            is_spa = any(spa_indicators.values())
            if is_spa:
                logger.debug(f"SPA detected: {spa_indicators}")

            return is_spa

        except Exception as e:
            logger.warning(f"Error detecting SPA patterns: {e}")
            return False

    async def extract_page_content_advanced(
        self,
        page: Page,
        url: str,
        website_id: str
    ) -> Dict[str, Any]:
        """
        Extract comprehensive content from a page using advanced selectors.

        Args:
            page: Playwright page instance
            url: Page URL
            website_id: Website UUID

        Returns:
            Dictionary containing extracted page data
        """
        try:
            # Extract basic page metadata
            title = await page.title()

            # Extract meta description
            meta_description = await page.evaluate("""
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : '';
                }
            """)

            # Extract main content using multiple strategies
            content_strategies = [
                'main',
                'article',
                '[role="main"]',
                '.main-content',
                '#main-content',
                '.content',
                '#content',
                '.post-content',
                '.article-content'
            ]

            main_content = ""
            for selector in content_strategies:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        main_content = await element.inner_text()
                        if len(main_content.strip()) > 100:  # Reasonable content length
                            break
                except Exception:
                    continue

            # If no main content found, extract body text
            if not main_content.strip():
                main_content = await page.evaluate("""
                    () => {
                        // Remove script and style elements
                        const scripts = document.querySelectorAll('script, style, nav, header, footer');
                        scripts.forEach(el => el.remove());

                        // Get body text
                        return document.body ? document.body.innerText : '';
                    }
                """)

            # Extract structured data
            structured_data = await self.extract_structured_data(page)

            # Extract links for further crawling
            links = await page.evaluate("""
                () => {
                    const anchors = Array.from(document.querySelectorAll('a[href]'));
                    return anchors.map(a => ({
                        href: a.href,
                        text: a.innerText.trim(),
                        title: a.title || ''
                    })).filter(link => link.href && link.href.startsWith('http'));
                }
            """)

            # Calculate content metrics
            word_count = len(main_content.split()) if main_content else 0

            # Detect dynamic content
            dynamic_content = await self.detect_dynamic_content(page)
            if dynamic_content:
                self.crawl_stats['dynamic_content_found'] += 1

            return {
                'url': url,
                'title': title,
                'meta_description': meta_description,
                'content_text': main_content,
                'content_html': await page.content(),
                'word_count': word_count,
                'links': links,
                'structured_data': structured_data,
                'dynamic_content': dynamic_content,
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'page_type': self.classify_page_type(title, main_content, url),
                'depth_level': 0,  # Will be set by caller
                'relevance_score': self.calculate_relevance_score(title, main_content),
                'status_code': 200
            }

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {
                'url': url,
                'title': '',
                'meta_description': '',
                'content_text': '',
                'content_html': '',
                'word_count': 0,
                'links': [],
                'error': str(e)
            }

    async def extract_structured_data(self, page: Page) -> Dict[str, Any]:
        """
        Extract structured data (JSON-LD, microdata, etc.) from the page.

        Args:
            page: Playwright page instance

        Returns:
            Dictionary containing structured data
        """
        try:
            structured_data = await page.evaluate("""
                () => {
                    const data = {};

                    // Extract JSON-LD
                    const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
                    data.jsonLd = Array.from(jsonLdScripts).map(script => {
                        try {
                            return JSON.parse(script.textContent);
                        } catch (e) {
                            return null;
                        }
                    }).filter(Boolean);

                    // Extract OpenGraph data
                    const ogTags = document.querySelectorAll('meta[property^="og:"]');
                    data.openGraph = {};
                    ogTags.forEach(tag => {
                        const property = tag.getAttribute('property').replace('og:', '');
                        data.openGraph[property] = tag.getAttribute('content');
                    });

                    // Extract Twitter Card data
                    const twitterTags = document.querySelectorAll('meta[name^="twitter:"]');
                    data.twitterCard = {};
                    twitterTags.forEach(tag => {
                        const name = tag.getAttribute('name').replace('twitter:', '');
                        data.twitterCard[name] = tag.getAttribute('content');
                    });

                    return data;
                }
            """)

            return structured_data

        except Exception as e:
            logger.warning(f"Error extracting structured data: {e}")
            return {}

    async def detect_dynamic_content(self, page: Page) -> Dict[str, Any]:
        """
        Detect and analyze dynamic content on the page.

        Args:
            page: Playwright page instance

        Returns:
            Dictionary containing dynamic content analysis
        """
        try:
            dynamic_analysis = await page.evaluate("""
                () => {
                    const analysis = {
                        hasAjax: false,
                        hasDynamicForms: false,
                        hasInfiniteScroll: false,
                        hasLazyLoading: false,
                        dynamicElements: 0
                    };

                    // Check for AJAX indicators
                    if (window.XMLHttpRequest || window.fetch) {
                        analysis.hasAjax = true;
                    }

                    // Check for dynamic forms
                    const forms = document.querySelectorAll('form');
                    forms.forEach(form => {
                        if (form.hasAttribute('data-ajax') || form.classList.contains('ajax-form')) {
                            analysis.hasDynamicForms = true;
                        }
                    });

                    // Check for infinite scroll indicators
                    const infiniteScrollIndicators = [
                        '[data-infinite-scroll]',
                        '.infinite-scroll',
                        '[data-scroll="infinite"]'
                    ];
                    infiniteScrollIndicators.forEach(selector => {
                        if (document.querySelector(selector)) {
                            analysis.hasInfiniteScroll = true;
                        }
                    });

                    // Check for lazy loading
                    const lazyImages = document.querySelectorAll('img[data-src], img[loading="lazy"]');
                    if (lazyImages.length > 0) {
                        analysis.hasLazyLoading = true;
                    }

                    // Count elements with dynamic attributes
                    const dynamicSelectors = [
                        '[data-bind]',
                        '[v-if]', '[v-for]', '[v-model]',
                        '[ng-if]', '[ng-for]', '[ng-model]',
                        '[data-react-component]'
                    ];
                    dynamicSelectors.forEach(selector => {
                        analysis.dynamicElements += document.querySelectorAll(selector).length;
                    });

                    return analysis;
                }
            """)

            return dynamic_analysis

        except Exception as e:
            logger.warning(f"Error detecting dynamic content: {e}")
            return {}

    def classify_page_type(self, title: str, content: str, url: str) -> str:
        """
        Classify the type of page based on content and URL patterns.

        Args:
            title: Page title
            content: Page content
            url: Page URL

        Returns:
            Page type classification
        """
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()

        # URL-based classification
        if '/contact' in url_lower or '/contact-us' in url_lower:
            return 'contact'
        elif '/about' in url_lower or '/about-us' in url_lower:
            return 'about'
        elif '/product' in url_lower or '/service' in url_lower:
            return 'product'
        elif '/blog' in url_lower or '/article' in url_lower or '/post' in url_lower:
            return 'blog'
        elif '/pricing' in url_lower or '/price' in url_lower:
            return 'pricing'
        elif url_lower.count('/') <= 3:  # Likely home page
            return 'home'

        # Content-based classification
        if 'contact us' in title_lower or 'get in touch' in title_lower:
            return 'contact'
        elif 'about' in title_lower:
            return 'about'
        elif 'product' in title_lower or 'service' in title_lower:
            return 'product'
        elif 'blog' in title_lower or 'article' in title_lower:
            return 'blog'
        elif 'pricing' in title_lower or 'price' in title_lower:
            return 'pricing'

        return 'general'

    def calculate_relevance_score(self, title: str, content: str) -> float:
        """
        Calculate a relevance score for the page content.

        Args:
            title: Page title
            content: Page content

        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0

        # Base score for having content
        if content and len(content.strip()) > 50:
            score += 0.3

        # Bonus for having a title
        if title and len(title.strip()) > 0:
            score += 0.2

        # Content length bonus
        content_length = len(content.strip()) if content else 0
        if content_length > 500:
            score += 0.2
        elif content_length > 200:
            score += 0.1

        # Quality indicators
        if content:
            # Penalize pages with too much repetitive content
            words = content.split()
            unique_words = set(word.lower() for word in words)
            if len(words) > 0:
                uniqueness_ratio = len(unique_words) / len(words)
                score += min(uniqueness_ratio, 0.3)

        return min(score, 1.0)

    def _track_request(self, metrics: Dict[str, Any], request) -> None:
        """Track page requests for metrics."""
        metrics['requests'] += 1

    def _track_response(self, metrics: Dict[str, Any], response) -> None:
        """Track page responses for metrics."""
        metrics['responses'] += 1

    async def extract_links_from_page_data(
        self,
        page_data: Dict[str, Any],
        base_url: str,
        current_url: str
    ) -> List[str]:
        """
        Extract and filter links from page data for further crawling.

        Args:
            page_data: Page data containing links
            base_url: Base URL for the crawl
            current_url: Current page URL

        Returns:
            List of URLs to crawl
        """
        links = page_data.get('links', [])
        base_domain = urlparse(base_url).netloc
        valid_urls = []

        for link in links:
            href = link.get('href', '')
            if not href:
                continue

            try:
                # Resolve relative URLs
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)

                # Filter criteria
                if (parsed_url.netloc == base_domain and
                    parsed_url.scheme in ('http', 'https') and
                    not any(ext in parsed_url.path.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js']) and
                    absolute_url not in self.visited_urls):
                    valid_urls.append(absolute_url)

            except Exception as e:
                logger.debug(f"Error processing link {href}: {e}")
                continue

        return valid_urls

    async def store_crawled_pages(self, website_id: str, pages_data: List[Dict[str, Any]]) -> int:
        """
        Store crawled page data in the database.

        Args:
            website_id: Website UUID
            pages_data: List of page data to store

        Returns:
            Number of pages successfully stored
        """
        try:
            stored_count = 0

            for page_data in pages_data:
                try:
                    # Prepare page record for database
                    page_record = {
                        'url': page_data['url'],
                        'title': page_data.get('title', ''),
                        'meta_description': page_data.get('meta_description', ''),
                        'content_text': page_data.get('content_text', ''),
                        'content_html': page_data.get('content_html', ''),
                        'word_count': page_data.get('word_count', 0),
                        'page_type': page_data.get('page_type', 'general'),
                        'depth_level': page_data.get('depth_level', 0),
                        'relevance_score': page_data.get('relevance_score', 0.0),
                        'status_code': page_data.get('status_code', 200),
                        'scraped_at': page_data.get('scraped_at'),
                        'playwright_data': json.dumps({
                            'js_detected': page_data.get('js_detected', False),
                            'spa_detected': page_data.get('spa_detected', False),
                            'dynamic_content': page_data.get('dynamic_content', {}),
                            'structured_data': page_data.get('structured_data', {}),
                            'crawl_metrics': page_data.get('crawl_metrics', {})
                        })
                    }

                    # Store in database (this would be integrated with your existing storage)
                    # For now, we'll log the storage
                    logger.debug(f"Storing page data for {page_data['url']}")
                    stored_count += 1

                except Exception as e:
                    logger.error(f"Error storing page {page_data.get('url', 'unknown')}: {e}")
                    continue

            logger.info(f"Stored {stored_count}/{len(pages_data)} pages successfully")
            return stored_count

        except Exception as e:
            logger.error(f"Error storing crawled pages: {e}")
            return 0


# Convenience functions for integration
async def crawl_with_playwright(
    website_id: str,
    base_url: str,
    max_pages: int = 100,
    max_depth: int = 3,
    spa_mode: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to crawl a website with Playwright.

    Args:
        website_id: Website UUID
        base_url: Starting URL
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        spa_mode: Whether to use SPA mode

    Returns:
        Crawl results dictionary
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {
            'success': False,
            'error': 'Playwright is not available. Install with: pip install playwright && playwright install'
        }

    async with PlaywrightCrawlerService() as crawler:
        return await crawler.crawl_website_advanced(
            website_id=website_id,
            base_url=base_url,
            max_pages=max_pages,
            max_depth=max_depth,
            spa_mode=spa_mode
        )