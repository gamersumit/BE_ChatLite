"""
Vector Crawler Service - Cloud-ready crawler using Supabase vector storage
Replaces local file storage with Supabase vector database
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set
from uuid import UUID
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag

from .supabase_vector_service import get_vector_service
from ..core.supabase_client import SupabaseClient, get_supabase_admin

logger = logging.getLogger(__name__)


class VectorCrawlerService:
    """
    Cloud-ready crawler service using Supabase vector storage
    No local file dependencies - everything stored in vector database
    """

    def __init__(self, use_admin_client=False):
        """Initialize crawler with vector storage"""
        if use_admin_client:
            self.supabase = get_supabase_admin()
        else:
            self.supabase = SupabaseClient().client

        # Initialize vector storage service
        self.vector_service = get_vector_service()

        # Crawler configuration
        self.max_pages_default = 50
        self.max_concurrent_requests = 5
        self.request_timeout = 30
        self.min_content_length = 100
        self.max_content_length = 1000000  # 1MB max

    async def start_crawl(
        self,
        website_id: UUID,
        base_url: str,
        domain: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start crawling a website with vector storage

        Args:
            website_id: UUID of the website to crawl
            base_url: Base URL to start crawling from
            domain: Domain of the website
            config_override: Optional configuration overrides

        Returns:
            Dictionary with crawl results including status and metrics
        """
        logger.info(f"🚀 Starting vector crawl for website_id: {website_id}, base_url: {base_url}")

        try:
            # Validate website configuration
            website_data = await self._validate_website(website_id, base_url, domain)

            # Create crawl job in vector database
            crawl_config = {
                'max_pages': config_override.get('max_pages', self.max_pages_default) if config_override else self.max_pages_default,
                'domain': domain,
                'base_url': base_url,
                'crawler_version': '2.0-vector'
            }

            job_id = await self.vector_service.create_crawl_job(
                website_id=str(website_id),
                config=crawl_config
            )

            if not job_id:
                raise RuntimeError("Failed to create crawl job")

            logger.info(f"✅ Created crawl job: {job_id}")

            # Update job status to processing
            await self.vector_service.update_crawl_job(
                job_id=job_id,
                status='processing'
            )

            # Perform the crawl
            crawl_results = await self._perform_vector_crawl(
                website_id=str(website_id),
                job_id=job_id,
                base_url=base_url,
                domain=domain,
                max_pages=crawl_config['max_pages']
            )

            # Update final job status
            await self.vector_service.update_crawl_job(
                job_id=job_id,
                status='completed',
                pages_found=crawl_results['pages_found'],
                pages_processed=crawl_results['pages_processed']
            )

            logger.info(f"🎉 Vector crawl completed: {crawl_results['pages_processed']} pages processed")

            return {
                'job_id': job_id,
                'status': 'completed',
                'pages_processed': crawl_results['pages_processed'],
                'pages_found': crawl_results['pages_found'],
                'pages_with_errors': crawl_results['pages_with_errors'],
                'domain': domain,
                'base_url': base_url,
                'storage_type': 'vector'
            }

        except Exception as e:
            logger.error(f"💥 Vector crawl failed for website {website_id}: {str(e)}", exc_info=True)

            # Update job status to failed if we have job_id
            try:
                if 'job_id' in locals():
                    await self.vector_service.update_crawl_job(
                        job_id=job_id,
                        status='failed',
                        error_message=str(e)
                    )
            except:
                pass

            raise

    async def _validate_website(self, website_id: UUID, base_url: str, domain: str) -> Dict[str, Any]:
        """Validate website configuration"""
        logger.info(f"📋 Validating website configuration for {website_id}")

        website_result = self.supabase.table('websites').select(
            'id, name, domain, url, scraping_enabled, business_description'
        ).eq('id', str(website_id)).execute()

        if not website_result.data:
            raise ValueError(f"Website {website_id} not found")

        website_data = website_result.data[0]
        logger.info(f"✅ Found website: {website_data['name']} ({website_data['domain']})")

        if not website_data.get('scraping_enabled', False):
            raise ValueError(f"Scraping not enabled for website {website_id}")

        return website_data

    async def _perform_vector_crawl(
        self,
        website_id: str,
        job_id: str,
        base_url: str,
        domain: str,
        max_pages: int
    ) -> Dict[str, Any]:
        """
        Perform the actual crawling with vector storage

        Args:
            website_id: Website identifier
            job_id: Crawl job identifier
            base_url: Base URL to crawl
            domain: Domain name
            max_pages: Maximum pages to crawl

        Returns:
            Crawl results dictionary
        """
        logger.info(f"🌐 Starting vector crawl for {base_url} (max {max_pages} pages)")

        # Initialize crawl state
        urls_to_crawl = [base_url]
        crawled_urls: Set[str] = set()
        failed_urls: Set[str] = set()
        pages_processed = 0
        pages_with_errors = 0

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        headers = {
            'User-Agent': 'ChatLite-Crawler/2.0 (+https://chatlite.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            while urls_to_crawl and pages_processed < max_pages:
                # Process URLs in batches for better performance
                batch_size = min(self.max_concurrent_requests, len(urls_to_crawl))
                current_batch = []

                for _ in range(batch_size):
                    if urls_to_crawl and pages_processed < max_pages:
                        url = urls_to_crawl.pop(0)
                        if url not in crawled_urls and url not in failed_urls:
                            current_batch.append(url)

                if not current_batch:
                    break

                # Process batch concurrently
                tasks = [
                    self._crawl_single_page(
                        session, website_id, url, domain, crawled_urls, failed_urls
                    )
                    for url in current_batch
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(batch_results):
                    url = current_batch[i]

                    if isinstance(result, Exception):
                        logger.error(f"❌ Failed to crawl {url}: {result}")
                        failed_urls.add(url)
                        pages_with_errors += 1
                    elif result:
                        crawled_urls.add(url)
                        pages_processed += 1

                        # Add new URLs found on this page
                        new_urls = result.get('new_urls', [])
                        for new_url in new_urls:
                            if (new_url not in crawled_urls and
                                new_url not in failed_urls and
                                new_url not in urls_to_crawl and
                                self._is_valid_url(new_url, domain)):
                                urls_to_crawl.append(new_url)

                        logger.info(f"✅ Processed {url} (found {len(new_urls)} new URLs)")

                # Update job progress
                await self.vector_service.update_crawl_job(
                    job_id=job_id,
                    pages_found=len(crawled_urls) + len(urls_to_crawl),
                    pages_processed=pages_processed
                )

        return {
            'pages_found': len(crawled_urls) + len(failed_urls),
            'pages_processed': pages_processed,
            'pages_with_errors': pages_with_errors
        }

    async def _crawl_single_page(
        self,
        session: aiohttp.ClientSession,
        website_id: str,
        url: str,
        domain: str,
        crawled_urls: Set[str],
        failed_urls: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Crawl a single page and store in vector database

        Args:
            session: HTTP session
            website_id: Website identifier
            url: URL to crawl
            domain: Domain name
            crawled_urls: Set of already crawled URLs
            failed_urls: Set of failed URLs

        Returns:
            Crawl result or None if failed
        """
        try:
            logger.info(f"🔍 Crawling page: {url}")

            # Fetch page content
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ HTTP {response.status} for {url}")
                    return None

                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"⚠️ Non-HTML content for {url}: {content_type}")
                    return None

                html_content = await response.text()

            # Parse and extract content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '').strip() if meta_desc else ''

            # Clean and extract text content
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            # Extract clean text
            text_content = soup.get_text()
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)

            # Validate content length
            if len(clean_text) < self.min_content_length:
                logger.warning(f"⚠️ Content too short ({len(clean_text)} chars), skipping: {url}")
                return None

            if len(clean_text) > self.max_content_length:
                logger.warning(f"⚠️ Content too long ({len(clean_text)} chars), truncating: {url}")
                clean_text = clean_text[:self.max_content_length]

            logger.info(f"📝 Extracted {len(clean_text)} chars from {url}")

            # Store content in vector database
            metadata = {
                'meta_description': meta_description,
                'word_count': len(clean_text.split()),
                'status_code': response.status,
                'content_type': content_type,
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'crawler_version': '2.0-vector'
            }

            store_result = await self.vector_service.store_content(
                website_id=website_id,
                url=url,
                title=title_text,
                content=clean_text,
                metadata=metadata
            )

            if store_result['success']:
                logger.info(f"💾 Stored in vector DB: {store_result['chunks_created']} chunks for {url}")
            else:
                logger.error(f"❌ Failed to store in vector DB: {store_result.get('error')}")
                return None

            # Extract new URLs for crawling
            new_urls = self._extract_urls(soup, url, domain)

            return {
                'url': url,
                'title': title_text,
                'content_length': len(clean_text),
                'chunks_created': store_result['chunks_created'],
                'new_urls': new_urls
            }

        except Exception as e:
            logger.error(f"❌ Error crawling {url}: {e}")
            return None

    def _extract_urls(self, soup: BeautifulSoup, base_url: str, domain: str) -> List[str]:
        """Extract valid URLs from a page"""
        urls = []

        for link in soup.find_all('a', href=True):
            href = link['href']

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Remove fragment identifier
            full_url, _ = urldefrag(full_url)

            if self._is_valid_url(full_url, domain):
                urls.append(full_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def _is_valid_url(self, url: str, domain: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)

            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False

            # Must be same domain
            if not parsed.netloc.endswith(domain):
                return False

            # Skip common file extensions
            skip_extensions = {
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.rar', '.tar', '.gz', '.7z',
                '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv',
                '.css', '.js', '.xml', '.json'
            }

            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in skip_extensions):
                return False

            # Skip common patterns
            skip_patterns = [
                '/wp-admin/', '/admin/', '/login/', '/logout/',
                '/search/', '/tag/', '/category/', '/archive/',
                '/feed/', '/rss/', '/sitemap'
            ]

            if any(pattern in path_lower for pattern in skip_patterns):
                return False

            return True

        except Exception:
            return False

    async def get_crawl_status(self, website_id: str) -> Dict[str, Any]:
        """Get crawl status for a website"""
        try:
            # Get latest crawl job
            result = self.supabase.table('crawl_jobs').select('*').eq(
                'website_id', website_id
            ).order('created_at', desc=True).limit(1).execute()

            if not result.data:
                return {'status': 'no_crawls', 'message': 'No crawl jobs found'}

            job = result.data[0]

            # Get content statistics
            stats = await self.vector_service.get_content_stats(website_id)

            return {
                'status': job['status'],
                'job_id': job['id'],
                'pages_found': job.get('pages_found', 0),
                'pages_processed': job.get('pages_processed', 0),
                'pages_with_errors': job.get('pages_with_errors', 0),
                'started_at': job.get('started_at'),
                'completed_at': job.get('completed_at'),
                'error_message': job.get('error_message'),
                'content_stats': stats
            }

        except Exception as e:
            logger.error(f"Error getting crawl status: {e}")
            return {'status': 'error', 'message': str(e)}

    async def delete_website_content(self, website_id: str) -> bool:
        """Delete all content for a website"""
        try:
            success = await self.vector_service.delete_content(website_id=website_id)

            if success:
                logger.info(f"✅ Deleted all content for website {website_id}")
            else:
                logger.error(f"❌ Failed to delete content for website {website_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting website content: {e}")
            return False


# Singleton instance
_vector_crawler_service = None

def get_vector_crawler_service() -> VectorCrawlerService:
    """Get or create vector crawler service instance"""
    global _vector_crawler_service
    if _vector_crawler_service is None:
        _vector_crawler_service = VectorCrawlerService()
    return _vector_crawler_service