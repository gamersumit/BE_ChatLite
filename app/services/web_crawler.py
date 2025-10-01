"""
Core Web Scraping Engine using aiohttp and BeautifulSoup4.
Handles ethical web crawling with robots.txt respect and rate limiting.
Integrates with Supabase for data persistence.
"""
import asyncio
import aiohttp
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from uuid import UUID
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from ..core.supabase_client import SupabaseClient
from ..models.scraper_schemas import (
    ScrapedWebsiteCreate, ScrapedWebsiteUpdate,
    ScrapedPageCreate, ScrapedContentChunkCreate,
    CrawlStatusUpdate
)

logger = logging.getLogger(__name__)


class CrawlerConfig:
    """Configuration class for web crawler settings."""
    
    def __init__(
        self,
        max_pages: int = 100,
        crawl_depth: int = 3,
        respect_robots: bool = True,
        crawl_delay: float = 1.0,
        max_concurrent: int = 5,
        timeout: int = 30,
        retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "LiteChat-Scraper/1.0"
    ):
        self.max_pages = max_pages
        self.crawl_depth = crawl_depth
        self.respect_robots = respect_robots
        self.crawl_delay = crawl_delay
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CrawlerConfig':
        """Create config from dictionary (e.g., from Supabase scraping_config)."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class RobotsTxtParser:
    """Parser for robots.txt files with crawl delay support."""
    
    def __init__(self, robots_content: str, user_agent: str = "*"):
        self.user_agent = user_agent
        self.robots_content = robots_content
        self._parse_robots_txt()
    
    def _parse_robots_txt(self):
        """Parse robots.txt content."""
        import tempfile
        import os
        
        # Create a temporary file with the robots.txt content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(self.robots_content)
            temp_file = f.name
        
        try:
            self.rp = RobotFileParser()
            self.rp.set_url(f"file://{temp_file}")
            self.rp.read()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Extract crawl delay
        self.crawl_delay = self._extract_crawl_delay()
    
    def _extract_crawl_delay(self) -> float:
        """Extract crawl delay from robots.txt."""
        lines = self.robots_content.lower().split('\n')
        current_user_agent = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('user-agent:'):
                current_user_agent = line.split(':', 1)[1].strip()
            elif line.startswith('crawl-delay:') and (
                current_user_agent == '*' or 
                current_user_agent == self.user_agent.lower()
            ):
                try:
                    return float(line.split(':', 1)[1].strip())
                except ValueError:
                    continue
        
        return 1.0  # Default crawl delay
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        return self.rp.can_fetch(self.user_agent, url)


class ContentExtractor:
    """Extracts and processes content from HTML pages."""
    
    @staticmethod
    def extract_content(html: str, url: str) -> Dict[str, Any]:
        """Extract structured content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Basic metadata extraction
        title = ContentExtractor._extract_title(soup)
        meta_description = ContentExtractor._extract_meta_description(soup)
        
        # Remove unwanted elements (scripts, styles, nav, footer, etc.)
        ContentExtractor._clean_soup(soup)
        
        # Extract main content - look for main content containers
        main_content = ContentExtractor._extract_main_content(soup)
        
        # Extract clean text
        content_text = main_content.get_text(separator=' ', strip=True) if main_content else ""
        content_text = ContentExtractor._normalize_text(content_text)
        
        # Get cleaned HTML
        content_html = str(main_content) if main_content else ""
        
        # Calculate word count
        word_count = len(content_text.split()) if content_text else 0
        
        # Generate content hash
        content_hash = ContentExtractor._generate_content_hash(content_text)
        
        return {
            'title': title,
            'meta_description': meta_description,
            'content_text': content_text,
            'content_html': content_html,
            'content_hash': content_hash,
            'word_count': word_count,
            'scraped_at': datetime.now(timezone.utc)
        }
    
    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None
    
    @staticmethod
    def _extract_meta_description(soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content') if meta_desc else None
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def _generate_content_hash(content: str) -> str:
        """Generate SHA-256 hash for content deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _clean_soup(soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup."""
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Remove navigation and footer elements
        for element in soup.find_all(['nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Remove elements with common navigation/ad class names
        unwanted_classes = ['nav', 'navigation', 'menu', 'sidebar', 'footer', 'header', 'ad', 'ads', 'advertisement']
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=lambda x: x and class_name in x.lower() if x else False):
                element.decompose()
    
    @staticmethod
    def _extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content from cleaned soup."""
        # Look for main content containers in order of preference
        main_selectors = [
            'main',
            'article', 
            '[role="main"]',
            '.main-content',
            '.content',
            '#main',
            '#content',
            'body'
        ]
        
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return element
        
        # Fallback to body if nothing else found
        return soup.find('body') or soup


class SitemapParser:
    """Parser for XML sitemaps."""
    
    @staticmethod
    async def parse_sitemap(session: aiohttp.ClientSession, sitemap_url: str) -> List[str]:
        """Parse sitemap XML and extract URLs."""
        try:
            async with session.get(sitemap_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return SitemapParser._extract_urls_from_xml(content)
        except Exception as e:
            logger.warning(f"Failed to parse sitemap {sitemap_url}: {e}")
        
        return []
    
    @staticmethod
    def _extract_urls_from_xml(xml_content: str) -> List[str]:
        """Extract URLs from sitemap XML."""
        urls = []
        try:
            root = ET.fromstring(xml_content)
            # Handle different sitemap namespaces
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text.strip())
        except ET.ParseError as e:
            logger.warning(f"Failed to parse sitemap XML: {e}")
        
        return urls


class WebCrawler:
    """Asynchronous web crawler with ethical scraping practices."""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.supabase = SupabaseClient().client
        self.session: Optional[aiohttp.ClientSession] = None
        self.robots_cache: Dict[str, RobotsTxtParser] = {}
        self.last_request_time: Dict[str, float] = {}
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={'User-Agent': self.config.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def crawl_website(self, website_id: UUID, base_url: str, domain: str) -> Dict[str, Any]:
        """Crawl a complete website and store results in Supabase."""
        # Create scraped_website record
        scraped_website_data = ScrapedWebsiteCreate(
            website_id=website_id,
            domain=domain,
            base_url=base_url,
            crawl_status="crawling",
            total_pages_found=0,
            pages_processed=0,
            crawl_depth=self.config.crawl_depth,
            max_pages=self.config.max_pages
        )
        
        # Insert scraped_website record
        result = self.supabase.table('scraped_websites').insert(
            scraped_website_data.model_dump()
        ).execute()
        
        if not result.data:
            raise Exception("Failed to create scraped website record")
        
        scraped_website_id = result.data[0]['id']
        
        try:
            # Get robots.txt
            await self._fetch_robots_txt(domain)
            
            # Discover URLs from sitemap
            sitemap_urls = await self._discover_sitemap_urls(base_url)
            
            # Update scraped_website with sitemap URLs
            if sitemap_urls:
                self.supabase.table('scraped_websites').update({
                    'sitemap_urls': sitemap_urls
                }).eq('id', scraped_website_id).execute()
            
            # Start crawling from base URL
            urls_to_crawl = {base_url: 0}  # URL: depth
            crawled_urls: Set[str] = set()
            pages_processed = 0
            
            while urls_to_crawl and pages_processed < self.config.max_pages:
                # Get next batch of URLs to crawl
                current_batch = []
                for url, depth in list(urls_to_crawl.items()):
                    if len(current_batch) >= self.config.max_concurrent:
                        break
                    if url not in crawled_urls and depth <= self.config.crawl_depth:
                        current_batch.append((url, depth))
                        del urls_to_crawl[url]
                
                if not current_batch:
                    break
                
                # Crawl batch concurrently
                batch_tasks = [
                    self._crawl_page(scraped_website_id, url, depth, domain)
                    for url, depth in current_batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results and extract new URLs
                for (url, depth), result in zip(current_batch, batch_results):
                    crawled_urls.add(url)
                    pages_processed += 1
                    
                    if isinstance(result, dict) and 'new_urls' in result:
                        # Add new URLs to crawl queue
                        for new_url in result['new_urls']:
                            if new_url not in crawled_urls and new_url not in urls_to_crawl:
                                urls_to_crawl[new_url] = depth + 1
                
                # Update progress
                await self._update_crawl_progress(
                    scraped_website_id, 
                    len(urls_to_crawl) + len(crawled_urls),
                    pages_processed
                )
            
            # Mark crawl as completed
            await self._update_crawl_status(scraped_website_id, "completed", pages_processed)
            
            return {
                'scraped_website_id': scraped_website_id,
                'pages_processed': pages_processed,
                'total_pages_found': len(urls_to_crawl) + len(crawled_urls),
                'status': 'completed'
            }
        
        except Exception as e:
            logger.error(f"Crawl failed for {base_url}: {e}")
            await self._update_crawl_status(scraped_website_id, "failed", pages_processed)
            raise
    
    async def _crawl_page(self, scraped_website_id: str, url: str, depth: int, domain: str) -> Dict[str, Any]:
        """Crawl a single page and extract content."""
        async with self.semaphore:
            try:
                # Check robots.txt compliance
                if self.config.respect_robots:
                    robots_parser = self.robots_cache.get(domain)
                    if robots_parser and not robots_parser.can_fetch(url):
                        logger.info(f"Skipping {url} due to robots.txt restrictions")
                        return {'new_urls': []}
                
                # Enforce rate limiting
                await self._enforce_rate_limit(domain)
                
                # Fetch page with retries
                html_content = await self._fetch_with_retries(url)
                if not html_content:
                    return {'new_urls': []}
                
                # Extract content
                content_data = ContentExtractor.extract_content(html_content, url)
                
                # Determine page type (basic heuristic)
                page_type = self._determine_page_type(url, content_data.get('title', ''))
                
                # Create scraped page record
                scraped_page_data = ScrapedPageCreate(
                    scraped_website_id=UUID(scraped_website_id),
                    url=url,
                    title=content_data.get('title'),
                    meta_description=content_data.get('meta_description'),
                    content_text=content_data.get('content_text'),
                    content_html=content_data.get('content_html'),
                    content_hash=content_data.get('content_hash'),
                    page_type=page_type,
                    depth_level=depth,
                    word_count=content_data.get('word_count', 0),
                    scraped_at=content_data.get('scraped_at'),
                    status_code=200
                )
                
                # Store page in Supabase
                self.supabase.table('scraped_pages').insert(
                    scraped_page_data.model_dump()
                ).execute()
                
                # Extract new URLs to crawl
                new_urls = self._extract_links(html_content, url, domain)
                
                return {'new_urls': new_urls}
                
            except Exception as e:
                logger.error(f"Failed to crawl page {url}: {e}")
                return {'new_urls': []}
    
    async def _fetch_robots_txt(self, domain: str) -> None:
        """Fetch and parse robots.txt for domain."""
        if domain in self.robots_cache:
            return
        
        robots_url = f"https://{domain}/robots.txt"
        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    self.robots_cache[domain] = RobotsTxtParser(
                        robots_content, 
                        self.config.user_agent
                    )
                    
                    # Store robots.txt content in database
                    self.supabase.table('scraped_websites').update({
                        'robots_txt_content': robots_content
                    }).eq('domain', domain).execute()
                    
                    logger.info(f"Loaded robots.txt for {domain}")
                else:
                    # Create permissive robots.txt if none found
                    self.robots_cache[domain] = RobotsTxtParser("User-agent: *\nDisallow:", self.config.user_agent)
        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
            # Create permissive robots.txt on error
            self.robots_cache[domain] = RobotsTxtParser("User-agent: *\nDisallow:", self.config.user_agent)
    
    async def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover sitemap URLs."""
        domain = urlparse(base_url).netloc
        sitemap_candidates = [
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemaps.xml"
        ]
        
        discovered_urls = []
        for sitemap_url in sitemap_candidates:
            urls = await SitemapParser.parse_sitemap(self.session, sitemap_url)
            discovered_urls.extend(urls)
        
        return list(set(discovered_urls))  # Remove duplicates
    
    async def _enforce_rate_limit(self, domain: str) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        last_time = self.last_request_time.get(domain, 0)
        
        # Get crawl delay from robots.txt or use config default
        crawl_delay = self.config.crawl_delay
        if domain in self.robots_cache:
            crawl_delay = max(crawl_delay, self.robots_cache[domain].crawl_delay)
        
        elapsed = current_time - last_time
        if elapsed < crawl_delay:
            await asyncio.sleep(crawl_delay - elapsed)
        
        self.last_request_time[domain] = time.time()
    
    async def _fetch_with_retries(self, url: str) -> Optional[str]:
        """Fetch URL with retry logic and exponential backoff."""
        for attempt in range(self.config.retries + 1):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Rate limited
                        retry_after = response.headers.get('Retry-After', str(self.config.retry_delay))
                        await asyncio.sleep(float(retry_after))
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Error fetching {url} (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.retries:
                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        
        return None
    
    def _extract_links(self, html_content: str, base_url: str, domain: str) -> List[str]:
        """Extract links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if not href or href.startswith('#'):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            
            # Only include links from the same domain
            if parsed_url.netloc == domain:
                # Clean URL (remove fragments)
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if parsed_url.query:
                    clean_url += f"?{parsed_url.query}"
                
                links.append(clean_url)
        
        return list(set(links))  # Remove duplicates
    
    def _determine_page_type(self, url: str, title: str) -> str:
        """Determine page type based on URL and title."""
        url_lower = url.lower()
        title_lower = title.lower() if title else ""
        
        if any(keyword in url_lower for keyword in ['about', 'about-us', 'company']):
            return 'about'
        elif any(keyword in url_lower for keyword in ['contact', 'contact-us']):
            return 'contact'
        elif any(keyword in url_lower for keyword in ['product', 'products']):
            return 'product'
        elif any(keyword in url_lower for keyword in ['service', 'services']):
            return 'service'
        elif url_lower.rstrip('/').split('/')[-1] in ['', 'index.html', 'home']:
            return 'home'
        elif any(keyword in url_lower for keyword in ['blog', 'news', 'article']):
            return 'blog'
        else:
            return 'page'
    
    async def _update_crawl_progress(self, scraped_website_id: str, total_found: int, processed: int) -> None:
        """Update crawl progress in database."""
        self.supabase.table('scraped_websites').update({
            'total_pages_found': total_found,
            'pages_processed': processed,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('id', scraped_website_id).execute()
    
    async def _update_crawl_status(self, scraped_website_id: str, status: str, processed: int) -> None:
        """Update crawl status in database."""
        self.supabase.table('scraped_websites').update({
            'crawl_status': status,
            'pages_processed': processed,
            'last_crawled_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('id', scraped_website_id).execute()