"""
Enhanced Web Crawler with SPA Support using Playwright
Handles both traditional HTML websites and modern SPAs (React, Vue, Angular)
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Set, Any
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone
from uuid import UUID

from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import aiohttp

from ..models.scraper_schemas import ScrapedWebsiteCreate, ScrapedPageCreate
from ..core.supabase_client import get_supabase_admin

logger = logging.getLogger(__name__)


class SPACrawler:
    """Enhanced crawler that handles both static HTML and JavaScript-rendered SPAs"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_pages = config.get('max_pages', 100)
        self.max_depth = config.get('max_depth', 3)
        self.wait_for_network = config.get('wait_for_network', True)
        self.screenshot_enabled = config.get('screenshot_enabled', False)
        self.user_agent = config.get('user_agent', 'ChatLite-Crawler/1.0')

        self.supabase = get_supabase_admin()
        self.browser: Optional[Browser] = None
        self.crawled_urls: Set[str] = set()
        self.pages_processed = 0

    async def __aenter__(self):
        """Initialize Playwright browser on context enter"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--disable-dev-shm-usage', '--no-sandbox']
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser on context exit"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def is_spa_website(self, url: str) -> bool:
        """Detect if a website is likely a SPA by checking initial HTML"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status != 200:
                        return False

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Check for common SPA indicators
                    spa_indicators = [
                        # React indicators
                        soup.find('div', id='root'),
                        soup.find('div', id='app'),
                        soup.find('script', src=lambda x: x and ('react' in x.lower() or 'vue' in x.lower() or 'angular' in x.lower())),
                        # Check if body has minimal content
                        len(soup.body.get_text(strip=True)) < 100 if soup.body else False,
                        # Check for common SPA frameworks
                        '__NEXT_DATA__' in html,  # Next.js
                        'ng-app' in html,  # Angular
                        'v-app' in html,  # Vue
                    ]

                    # If multiple indicators present, likely a SPA
                    indicators_found = sum(1 for indicator in spa_indicators if indicator)
                    return indicators_found >= 2

        except Exception as e:
            logger.warning(f"Could not detect if {url} is SPA: {e}")
            return False

    async def crawl_with_playwright(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """Crawl a page using Playwright for JavaScript rendering"""
        page = await self.browser.new_page(user_agent=self.user_agent)

        try:
            # Navigate to page with network idle wait
            await page.goto(url, wait_until='networkidle' if self.wait_for_network else 'load', timeout=30000)

            # Wait for common SPA elements to load
            try:
                # Try to wait for common content containers
                await page.wait_for_selector('main, article, [role="main"], .content, #content', timeout=5000)
            except PlaywrightTimeout:
                # Continue anyway if selectors not found
                pass

            # Additional wait for dynamic content
            await page.wait_for_timeout(2000)

            # Get fully rendered HTML
            html_content = await page.content()

            # Extract content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title
            title = await page.title() or self._extract_title(soup)

            # Extract meta description
            meta_desc = await page.evaluate('''
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : null;
                }
            ''') or self._extract_meta_description(soup)

            # Clean and extract text content
            content_text = self._extract_text_content(soup)

            # Extract all links for crawling - Enhanced for SPAs
            links = await page.evaluate('''
                () => {
                    const links = [];
                    const baseUrl = window.location.origin;

                    // Method 1: Traditional href links
                    document.querySelectorAll('a[href]').forEach(a => {
                        links.push(a.href);
                    });

                    // Method 2: React Router links (to= attribute)
                    document.querySelectorAll('a[to], [to]').forEach(a => {
                        const to = a.getAttribute('to');
                        if (to) links.push(baseUrl + (to.startsWith('/') ? to : '/' + to));
                    });

                    // Method 3: Navigation with onclick handlers
                    document.querySelectorAll('[onclick*="push"], [onclick*="navigate"]').forEach(el => {
                        const onclick = el.getAttribute('onclick');
                        const match = onclick.match(/['"]([^'"]+)['"]/);
                        if (match) links.push(baseUrl + (match[1].startsWith('/') ? match[1] : '/' + match[1]));
                    });

                    // Method 4: Common SPA route discovery
                    const commonRoutes = ['/', '/home', '/about', '/contact', '/services', '/products', '/faq', '/help', '/support', '/blog', '/news'];
                    const pageText = document.body.textContent.toLowerCase();

                    commonRoutes.forEach(route => {
                        const routeName = route === '/' ? 'home' : route.slice(1);
                        // Check if route name appears in page content (suggests it exists)
                        if (pageText.includes(routeName) || document.querySelector(`[href*="${route}"], [to*="${route}"]`)) {
                            links.push(baseUrl + route);
                        }
                    });

                    // Method 5: Look for navigation menu items with data attributes
                    document.querySelectorAll('[data-href], [data-to], [data-route]').forEach(el => {
                        const href = el.getAttribute('data-href') || el.getAttribute('data-to') || el.getAttribute('data-route');
                        if (href) links.push(baseUrl + (href.startsWith('/') ? href : '/' + href));
                    });

                    // Method 6: Extract from React Router's history or route config if available
                    if (window.__REACT_ROUTER__ || window.history) {
                        // Try to detect route patterns from navigation elements
                        document.querySelectorAll('nav *, .nav *, .navigation *, .menu *').forEach(el => {
                            const text = el.textContent?.trim().toLowerCase();
                            if (text && text.length < 20 && /^[a-z\\s]+$/.test(text)) {
                                const possibleRoute = '/' + text.replace(/\\s+/g, '-');
                                if (['home', 'about', 'contact', 'services', 'products', 'faq', 'help', 'support', 'blog'].includes(text.replace(/\\s+/g, ''))) {
                                    links.push(baseUrl + (text === 'home' ? '/' : possibleRoute));
                                }
                            }
                        });
                    }

                    // Method 7: Look for external domain links and redirects
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.href;
                        const text = a.textContent.trim().toLowerCase();

                        // Check if it's an external link (different domain) - should be crawled if it's the main site
                        if (href.startsWith('http') && !href.includes(window.location.hostname)) {
                            // Only add if it looks like a main website (not social media, etc.)
                            if (!href.includes('facebook') && !href.includes('twitter') && !href.includes('instagram') &&
                                !href.includes('linkedin') && !href.includes('youtube') && !href.includes('github')) {
                                links.push(href);
                            }
                        }

                        // Check for redirect indicators
                        if (text.includes('visit') || text.includes('main site') || text.includes('official') ||
                            text.includes('more info') || text.includes('learn more') || text.includes('website')) {
                            links.push(href);
                        }
                    });

                    // Method 8: Look for links in buttons or clickable elements
                    document.querySelectorAll('button, [role="button"], .btn').forEach(el => {
                        const onclick = el.getAttribute('onclick');
                        const dataHref = el.getAttribute('data-href') || el.getAttribute('data-url');
                        const text = el.textContent.trim().toLowerCase();

                        if (dataHref) {
                            links.push(dataHref.startsWith('http') ? dataHref : baseUrl + dataHref);
                        }

                        if (onclick && onclick.includes('location.href')) {
                            const match = onclick.match(/location\.href\s*=\s*['"]([^'"]+)['"]/);
                            if (match) {
                                links.push(match[1].startsWith('http') ? match[1] : baseUrl + match[1]);
                            }
                        }

                        // Look for external site indicators in button text
                        if (text.includes('visit') || text.includes('main site') || text.includes('official')) {
                            // Try to find associated href
                            const parentLink = el.closest('a');
                            if (parentLink && parentLink.href) {
                                links.push(parentLink.href);
                            }
                        }
                    });

                    // Method 9: Enhanced logo and image detection with JavaScript navigation
                    document.querySelectorAll('img, [role="img"], .logo').forEach(el => {
                        const src = el.src || el.getAttribute('src');
                        const alt = el.alt || el.getAttribute('alt') || '';

                        // Check if it's likely a logo or brand image
                        const isLogo = src && (src.includes('logo') || alt.toLowerCase().includes('logo') ||
                                              alt.toLowerCase().includes('brand') || alt.toLowerCase().includes('bloom') ||
                                              alt.toLowerCase().includes('floral') || el.className.includes('logo'));

                        if (isLogo) {
                            // Check if image is wrapped in a link
                            const parentLink = el.closest('a');
                            if (parentLink && parentLink.href) {
                                links.push(parentLink.href);
                            }

                            // Enhanced JavaScript navigation detection (onClick handlers)
                            else if (el.onclick || el.parentElement?.onclick || el.getAttribute('onclick')) {
                                // Try to extract navigation destination from common patterns
                                let destination = null;

                                // Check onclick attribute
                                const onclickAttr = el.getAttribute('onclick') || el.parentElement?.getAttribute('onclick');
                                if (onclickAttr) {
                                    // Look for URL patterns in onclick
                                    const urlMatch = onclickAttr.match(/(?:window\.location|location\.href|navigate|push)\s*=?\s*['"]([^'"]+)['"]/);
                                    if (urlMatch) {
                                        destination = urlMatch[1];
                                    }
                                }

                                // Check for data attributes indicating destination
                                if (!destination) {
                                    destination = el.getAttribute('data-href') ||
                                                el.getAttribute('data-url') ||
                                                el.getAttribute('data-link') ||
                                                el.parentElement?.getAttribute('data-href');
                                }

                                // Only add if destination is same-domain (no external domains)
                                if (destination && !destination.startsWith('http')) {
                                    links.push(baseUrl + (destination.startsWith('/') ? destination : '/' + destination));
                                } else if (destination && destination.includes(window.location.hostname)) {
                                    links.push(destination);
                                }
                            }

                            // Check for clickable images with data attributes
                            const dataHref = el.getAttribute('data-href') || el.getAttribute('data-url');
                            if (dataHref && !dataHref.startsWith('http')) {
                                links.push(dataHref.startsWith('http') ? dataHref : baseUrl + dataHref);
                            }
                        }
                    });

                    // Method 10: Look for logo/brand elements that might be clickable
                    document.querySelectorAll('[class*="logo"], [class*="brand"], [id*="logo"], [id*="brand"]').forEach(el => {
                        // Check if the logo element itself is clickable
                        const onclick = el.getAttribute('onclick');
                        if (onclick && onclick.includes('location.href')) {
                            const match = onclick.match(/location\.href\s*=\s*['"]([^'"]+)['"]/);
                            if (match) {
                                links.push(match[1].startsWith('http') ? match[1] : baseUrl + match[1]);
                            }
                        }

                        // Check if logo is wrapped in a link
                        const parentLink = el.closest('a');
                        if (parentLink && parentLink.href) {
                            links.push(parentLink.href);
                        }

                        // Check for data attributes on logo elements
                        const dataHref = el.getAttribute('data-href') || el.getAttribute('data-url');
                        if (dataHref) {
                            links.push(dataHref.startsWith('http') ? dataHref : baseUrl + dataHref);
                        }
                    });

                    return [...new Set(links)]; // Remove duplicates
                }
            ''')

            # Filter and normalize links
            domain = urlparse(url).netloc
            filtered_links = []
            for link in links:
                try:
                    parsed = urlparse(link)

                    # Keep same-domain links only (exclude external domains)
                    if parsed.netloc == domain or not parsed.netloc:
                        full_url = urljoin(url, link)
                        if full_url not in self.crawled_urls:
                            filtered_links.append(full_url)
                except:
                    continue

            # Take screenshot if enabled
            screenshot = None
            if self.screenshot_enabled:
                screenshot = await page.screenshot(full_page=True)

            return {
                'url': url,
                'title': title,
                'meta_description': meta_desc,
                'content_text': content_text,
                'content_html': html_content,
                'links': filtered_links[:50],  # Limit links to prevent explosion
                'word_count': len(content_text.split()),
                'content_hash': hashlib.md5(content_text.encode()).hexdigest(),
                'screenshot': screenshot,
                'depth': depth,
                'is_spa': True
            }

        except Exception as e:
            logger.error(f"Error crawling {url} with Playwright: {e}")
            return None
        finally:
            await page.close()

    async def crawl_with_aiohttp(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """Traditional crawling for static HTML sites"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': self.user_agent}) as response:
                    if response.status != 200:
                        return None

                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Extract content
                    title = self._extract_title(soup)
                    meta_desc = self._extract_meta_description(soup)
                    content_text = self._extract_text_content(soup)

                    # Extract links
                    links = []
                    domain = urlparse(url).netloc
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(url, href)
                        parsed = urlparse(full_url)
                        if parsed.netloc == domain or not parsed.netloc:
                            if full_url not in self.crawled_urls:
                                links.append(full_url)

                    return {
                        'url': url,
                        'title': title,
                        'meta_description': meta_desc,
                        'content_text': content_text,
                        'content_html': html_content,
                        'links': links[:50],
                        'word_count': len(content_text.split()),
                        'content_hash': hashlib.md5(content_text.encode()).hexdigest(),
                        'depth': depth,
                        'is_spa': False
                    }

        except Exception as e:
            logger.error(f"Error crawling {url} with aiohttp: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)

        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return "Untitled Page"

    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description"""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            return meta_tag.get('content', '')
        return None

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean text content from HTML"""
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()

        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main', '#main']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content found, use body
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Get text
        text = main_content.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = ' '.join(text.split())

        return text

    async def crawl_website(self, website_id: str, start_url: str) -> Dict[str, Any]:
        """Main crawl orchestrator"""
        domain = urlparse(start_url).netloc

        # Check if it's a SPA
        is_spa = await self.is_spa_website(start_url)
        logger.info(f"Website {start_url} detected as {'SPA' if is_spa else 'static HTML'}")

        # Create scraped_website record
        scraped_website_data = ScrapedWebsiteCreate(
            website_id=UUID(website_id),
            domain=domain,
            base_url=start_url,
            crawl_status="running",
            crawl_depth=self.max_depth,
            max_pages=self.max_pages
        )

        result = self.supabase.table('scraped_websites').insert(
            scraped_website_data.model_dump()
        ).execute()

        scraped_website_id = result.data[0]['id']

        # URLs to crawl queue
        urls_to_crawl = [(start_url, 0)]  # (url, depth)
        crawled_data = []

        try:
            while urls_to_crawl and self.pages_processed < self.max_pages:
                url, depth = urls_to_crawl.pop(0)

                if url in self.crawled_urls:
                    logger.info(f"⏭️  Skipping already crawled: {url}")
                    continue

                if depth > self.max_depth:
                    logger.info(f"⏭️  Skipping depth {depth} > max_depth {self.max_depth}: {url}")
                    continue

                logger.info(f"🔍 Crawling page {self.pages_processed + 1}/{self.max_pages}: {url} (depth: {depth}, queue: {len(urls_to_crawl)} remaining)")

                # Choose crawl method based on website type
                if is_spa:
                    page_data = await self.crawl_with_playwright(url, depth)
                else:
                    page_data = await self.crawl_with_aiohttp(url, depth)

                if page_data:
                    self.crawled_urls.add(url)
                    self.pages_processed += 1

                    # Save page to database
                    scraped_page = ScrapedPageCreate(
                        scraped_website_id=UUID(scraped_website_id),
                        url=url,
                        title=page_data['title'],
                        meta_description=page_data['meta_description'],
                        content_text=page_data['content_text'],
                        content_html=page_data['content_html'],
                        content_hash=page_data['content_hash'],
                        page_type=self._determine_page_type(url, page_data['title']),
                        depth_level=depth,
                        word_count=page_data['word_count'],
                        scraped_at=datetime.now(timezone.utc),
                        status_code=200
                    )

                    self.supabase.table('scraped_pages').insert(
                        scraped_page.model_dump()
                    ).execute()

                    # Content is stored in Supabase - no local storage needed
                    crawled_data.append(page_data)

                    # Add new URLs to queue
                    new_links_added = 0
                    for link in page_data.get('links', []):
                        if link not in self.crawled_urls and link not in [u[0] for u in urls_to_crawl]:
                            urls_to_crawl.append((link, depth + 1))
                            new_links_added += 1

                    logger.info(f"➕ Added {new_links_added} new links to queue (total queue: {len(urls_to_crawl)})")

                    # Update progress
                    self.supabase.table('scraped_websites').update({
                        'pages_processed': self.pages_processed,
                        'total_pages_found': len(self.crawled_urls) + len(urls_to_crawl)
                    }).eq('id', scraped_website_id).execute()

            logger.info(f"🏁 Crawl finished: {self.pages_processed} pages processed, {len(urls_to_crawl)} URLs remaining in queue")

            # Mark as completed
            self.supabase.table('scraped_websites').update({
                'crawl_status': 'completed',
                'pages_processed': self.pages_processed,
                'last_crawled_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', scraped_website_id).execute()

            # Don't return crawled_data to avoid passing large payloads through Redis
            # All content is already stored in Supabase
            return {
                'scraped_website_id': scraped_website_id,
                'pages_processed': self.pages_processed,
                'pages_found': len(crawled_data),
                'is_spa': is_spa,
                'crawl_time': (datetime.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
            }

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            # Mark as failed
            self.supabase.table('scraped_websites').update({
                'crawl_status': 'failed',
                'pages_processed': self.pages_processed
            }).eq('id', scraped_website_id).execute()
            raise


    def _determine_page_type(self, url: str, title: str) -> str:
        """Determine page type based on URL and title"""
        url_lower = url.lower()

        if any(keyword in url_lower for keyword in ['about', 'about-us']):
            return 'about'
        elif any(keyword in url_lower for keyword in ['contact', 'contact-us']):
            return 'contact'
        elif any(keyword in url_lower for keyword in ['product', 'products']):
            return 'product'
        elif any(keyword in url_lower for keyword in ['service', 'services']):
            return 'service'
        elif any(keyword in url_lower for keyword in ['blog', 'news', 'article']):
            return 'blog'
        elif url_lower.rstrip('/').endswith(('/', '/index.html', '/home')):
            return 'home'
        else:
            return 'page'


async def crawl_spa_website(website_id: str, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to crawl a website with SPA support"""
    async with SPACrawler(config) as crawler:
        return await crawler.crawl_website(website_id, url)