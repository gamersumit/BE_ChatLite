"""
Crawler Service - High-level interface for website scraping operations.
Integrates WebCrawler with Supabase services and provides easy-to-use API.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID

from .web_crawler import WebCrawler, CrawlerConfig
from .scraper_service import ScrapedWebsiteService
from ..core.supabase_client import SupabaseClient, get_supabase_admin
import hashlib

logger = logging.getLogger(__name__)


class CrawlerService:
    """High-level service for managing website crawling operations."""
    
    def __init__(self, use_admin_client=False):
        if use_admin_client:
            self.supabase = get_supabase_admin()
        else:
            self.supabase = SupabaseClient().client
        self.scraped_website_service = ScrapedWebsiteService()
    
    async def start_crawl(
        self, 
        website_id: UUID, 
        base_url: str, 
        domain: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start crawling a website with specified configuration.
        
        Args:
            website_id: UUID of the website to crawl
            base_url: Base URL to start crawling from
            domain: Domain of the website
            config_override: Optional configuration overrides
            
        Returns:
            Dictionary with crawl results including status and metrics
        """
        print(f"🚀 CRAWLER: Starting crawl for website_id: {website_id}, base_url: {base_url}, domain: {domain}")
        logger.info(f"🚀 Starting crawl for website_id: {website_id}, base_url: {base_url}, domain: {domain}")
        
        try:
            # Get website scraping configuration
            print(f"📋 CRAWLER: Fetching website configuration for {website_id}")
            logger.info(f"📋 Fetching website configuration for {website_id}")
            website_result = self.supabase.table('websites').select(
                'id, name, domain, url, scraping_enabled'
            ).eq('id', str(website_id)).execute()
            
            if not website_result.data:
                print(f"❌ CRAWLER: Website {website_id} not found in database")
                logger.error(f"❌ Website {website_id} not found in database")
                raise ValueError(f"Website {website_id} not found")
            
            website_data = website_result.data[0]
            print(f"✅ CRAWLER: Found website: {website_data['name']} ({website_data['domain']})")
            logger.info(f"✅ Found website: {website_data['name']} ({website_data['domain']})")
            
            if not website_data.get('scraping_enabled', False):
                logger.error(f"❌ Scraping not enabled for website {website_id}")
                raise ValueError(f"Scraping not enabled for website {website_id}")
            
            # Use the URL from the website data if base_url not provided
            if not base_url:
                base_url = website_data.get('url') or f"https://{domain}"
                logger.info(f"🔗 Using base URL from website data: {base_url}")
            
            if not base_url:
                logger.error(f"❌ No base URL available for website {website_id}")
                raise ValueError(f"No base URL available for website {website_id}")
            
            # Create or find scraped_website record
            logger.info(f"🔍 Looking for existing scraped_website record")
            scraped_website_result = self.supabase.table('scraped_websites').select('*').eq('website_id', str(website_id)).execute()
            
            if scraped_website_result.data:
                scraped_website = scraped_website_result.data[0]
                logger.info(f"📝 Found existing scraped_website: {scraped_website['id']}")
                # Update status to crawling
                self.supabase.table('scraped_websites').update({
                    'crawl_status': 'crawling',
                    'last_crawled_at': datetime.now(timezone.utc).isoformat()
                }).eq('id', scraped_website['id']).execute()
            else:
                logger.info(f"➕ Creating new scraped_website record")
                scraped_website_data = {
                    'website_id': str(website_id),
                    'domain': domain,
                    'base_url': base_url,
                    'crawl_status': 'crawling',
                    'last_crawled_at': datetime.now(timezone.utc).isoformat()
                }
                result = self.supabase.table('scraped_websites').insert(scraped_website_data).execute()
                scraped_website = result.data[0]
                logger.info(f"✅ Created scraped_website: {scraped_website['id']}")
            
            # Start simple crawling process
            print(f"🕷️ CRAWLER: Starting to crawl {base_url}")
            logger.info(f"🕷️ Starting to crawl {base_url}")
            # Get max_pages from config_override or use default
            max_pages = config_override.get('max_pages', 100) if config_override else 100
            pages_crawled = await self._simple_crawl(scraped_website['id'], base_url, domain, max_pages=max_pages)
            
            # Update final status
            self.supabase.table('scraped_websites').update({
                'crawl_status': 'completed',
                'pages_processed': pages_crawled,
                'total_pages_found': pages_crawled
            }).eq('id', scraped_website['id']).execute()
            
            print(f"🎉 CRAWLER: Crawl completed for {domain}: {pages_crawled} pages processed")
            logger.info(f"🎉 Crawl completed for {domain}: {pages_crawled} pages processed")
            
            return {
                'scraped_website_id': scraped_website['id'],
                'status': 'completed',
                'pages_processed': pages_crawled,
                'total_pages_found': pages_crawled,
                'domain': domain,
                'base_url': base_url
            }
                
        except Exception as e:
            logger.error(f"💥 Crawl failed for website {website_id}: {str(e)}", exc_info=True)
            # Update status to failed if we have scraped_website record
            try:
                self.supabase.table('scraped_websites').update({
                    'crawl_status': 'failed'
                }).eq('website_id', str(website_id)).execute()
            except:
                pass
            raise
    
    async def _simple_crawl(self, scraped_website_id: str, base_url: str, domain: str, max_pages: int = 100) -> int:
        """
        Simple crawling implementation that scrapes a website and stores content.

        Args:
            scraped_website_id: ID of the scraped_website record
            base_url: Base URL to crawl
            domain: Domain name
            max_pages: Maximum number of pages to crawl (default: 100)

        Returns:
            Number of pages successfully crawled
        """
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        from ..services.vector_search_service import VectorSearchService

        print(f"🌐 CRAWLER: Starting simple crawl for {base_url}")
        logger.info(f"🌐 Starting simple crawl for {base_url}")
        pages_crawled = 0
        vector_service = VectorSearchService()

        try:
            # Create aiohttp session with reasonable timeout
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:

                # Start with the base URL
                urls_to_crawl = [base_url]
                crawled_urls = set()
                
                print(f"📝 CRAWLER: Will crawl max {max_pages} pages starting from {base_url}")
                logger.info(f"📝 Will crawl max {max_pages} pages starting from {base_url}")
                
                while len(urls_to_crawl) > 0 and pages_crawled < max_pages:
                    url = urls_to_crawl.pop(0)

                    if url in crawled_urls:
                        print(f"⏭️  CRAWLER: Skipping already crawled: {url}")
                        continue

                    try:
                        print(f"🔍 CRAWLER: Crawling page {pages_crawled + 1}/{max_pages}: {url} (Queue: {len(urls_to_crawl)} remaining)")
                        logger.info(f"🔍 Crawling page {pages_crawled + 1}/{max_pages}: {url}")
                        
                        # Make HTTP request
                        async with session.get(url, headers={'User-Agent': 'LiteChat-Bot/1.0'}) as response:
                            if response.status != 200:
                                logger.warning(f"⚠️ HTTP {response.status} for {url}")
                                continue
                            
                            content_type = response.headers.get('content-type', '').lower()
                            if 'text/html' not in content_type:
                                logger.warning(f"⚠️ Non-HTML content for {url}: {content_type}")
                                continue
                                
                            html_content = await response.text()
                            logger.info(f"✅ Downloaded {len(html_content)} chars from {url}")
                        
                        # Parse HTML
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Extract content
                        title = soup.find('title')
                        title_text = title.get_text().strip() if title else url
                        
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        meta_description = meta_desc.get('content', '').strip() if meta_desc else ''
                        
                        # Extract main text content
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        
                        text_content = soup.get_text()
                        # Clean up whitespace
                        lines = (line.strip() for line in text_content.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        clean_text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        if len(clean_text) < 100:
                            logger.warning(f"⚠️ Page too short, skipping: {url}")
                            continue
                        
                        logger.info(f"📝 Extracted {len(clean_text)} chars of text from {url}")

                        # Content will be stored in Supabase only - no local storage
                        # Vector embeddings will be generated on-demand by the vector service

                        # First, check if this page exists in global_scraped_pages (improved schema)
                        global_page = await self._get_or_create_global_page(url, {
                            'title': title_text,
                            'meta_description': meta_description,
                            'content_text': clean_text,
                            'content_html': html_content[:50000],
                            'status_code': response.status
                        })
                        
                        if global_page:
                            # Use improved schema - create association
                            print(f"🔗 CRAWLER: Using global page system for {url}")
                            await self._create_website_page_association(scraped_website_id, global_page['id'], 0)
                            
                            # Create embeddings for global page if not exists
                            existing_chunks = self.supabase.table('scraped_content_chunks').select('id').eq('global_page_id', global_page['id']).execute()
                            
                            if not existing_chunks.data:
                                try:
                                    chunk_ids = await vector_service.chunk_and_embed_content_improved(global_page['id'], clean_text)
                                    print(f"🧠 CRAWLER: Created {len(chunk_ids)} new chunks for global page")
                                except Exception as e:
                                    print(f"❌ CRAWLER: Embedding failed for global page: {e}")
                            else:
                                print(f"♻️ CRAWLER: Reusing existing {len(existing_chunks.data)} chunks")
                        else:
                            # Fallback to old schema
                            print(f"📄 CRAWLER: Using legacy page system for {url}")
                            page_data = {
                                'scraped_website_id': scraped_website_id,
                                'url': url,
                                'title': title_text[:500],
                                'meta_description': meta_description[:1000] if meta_description else None,
                                'content_text': clean_text,
                                'content_html': html_content[:50000],
                                'word_count': len(clean_text.split()),
                                'status_code': response.status,
                                'scraped_at': datetime.now(timezone.utc).isoformat()
                            }
                        
                            # Try to insert the page (legacy mode), or update if it already exists
                            try:
                                page_result = self.supabase.table('scraped_pages').insert(page_data).execute()
                                page_id = page_result.data[0]['id']
                                print(f"💾 CRAWLER: Saved new legacy page: {page_id}")
                                logger.info(f"💾 Saved new legacy page: {page_id}")
                                
                                # Create embeddings for legacy page
                                try:
                                    chunk_ids = await vector_service.chunk_and_embed_content(page_id, clean_text)
                                    print(f"🧠 CRAWLER: Created {len(chunk_ids)} chunks for legacy page")
                                    logger.info(f"🧠 Created {len(chunk_ids)} embedding chunks for {url}")
                                except Exception as e:
                                    print(f"❌ CRAWLER: Failed to create embeddings: {e}")
                                    logger.error(f"❌ Failed to create embeddings for {url}: {e}")
                                    
                            except Exception as e:
                                if 'duplicate key' in str(e):
                                    # Page already exists, update it instead
                                    print(f"🔄 CRAWLER: Legacy page already exists, updating: {url}")
                                    logger.info(f"🔄 Legacy page already exists, updating: {url}")
                                    update_result = self.supabase.table('scraped_pages').update({
                                        'title': page_data['title'],
                                        'meta_description': page_data['meta_description'],
                                        'content_text': page_data['content_text'],
                                        'content_html': page_data['content_html'],
                                        'word_count': page_data['word_count'],
                                        'scraped_at': page_data['scraped_at']
                                    }).eq('scraped_website_id', scraped_website_id).eq('url', url).execute()
                                    if update_result.data:
                                        page_id = update_result.data[0]['id']
                                        print(f"🔄 CRAWLER: Updated existing legacy page: {page_id}")
                                        logger.info(f"🔄 Updated existing legacy page: {page_id}")
                                    else:
                                        print(f"❌ CRAWLER: Failed to update legacy page: {url}")
                                        logger.error(f"❌ Failed to update existing page: {url}")
                                        continue
                                else:
                                    print(f"❌ CRAWLER: Database error for {url}: {e}")
                                    logger.error(f"❌ Database error for {url}: {e}")
                                    continue
                        
                        pages_crawled += 1
                        crawled_urls.add(url)
                        
                        # Find more links to crawl (simple implementation)
                        if pages_crawled < max_pages:
                            links = soup.find_all('a', href=True)
                            print(f"🔗 CRAWLER: Found {len(links)} links on {url}")
                            logger.info(f"🔗 Found {len(links)} links on {url}")

                            # Process all links, not just first 20
                            for link in links:
                                href = link.get('href', '').strip()
                                if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                                    continue
                                    
                                full_url = urljoin(url, href)
                                parsed = urlparse(full_url)
                                base_parsed = urlparse(base_url)
                                
                                # Strip fragment from URL before checking
                                full_url_no_fragment = full_url.split('#')[0]

                                # Only crawl same domain, avoid certain file types
                                if (parsed.netloc == base_parsed.netloc and
                                    not any(full_url_no_fragment.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.zip', '.doc']) and
                                    full_url_no_fragment not in crawled_urls and
                                    full_url_no_fragment not in urls_to_crawl):
                                    
                                    urls_to_crawl.append(full_url_no_fragment)
                                    print(f"➕ CRAWLER: Added to queue ({len(urls_to_crawl)} total): {full_url_no_fragment}")
                                    logger.info(f"➕ Added to crawl queue: {full_url_no_fragment}")
                        else:
                            print(f"📊 CRAWLER: Reached max pages limit ({max_pages}), stopping link discovery for: {url}")
                            logger.info(f"📊 Reached max pages limit ({max_pages}), stopping link discovery for: {url}")
                            
                    except Exception as e:
                        logger.error(f"💥 Error crawling {url}: {str(e)}")
                        continue
                
                print(f"🏁 CRAWLER: Crawling completed. Total pages: {pages_crawled}, Remaining queue: {len(urls_to_crawl)}")
                logger.info(f"🏁 Crawling completed. Total pages: {pages_crawled}")
                return pages_crawled
                
        except Exception as e:
            logger.error(f"💥 Fatal error in crawling: {str(e)}", exc_info=True)
            return pages_crawled
    
    async def get_crawl_status(self, website_id: UUID) -> Dict[str, Any]:
        """
        Get current crawl status for a website.
        
        Args:
            website_id: UUID of the website
            
        Returns:
            Dictionary with crawl status and progress information
        """
        result = self.supabase.table('scraped_websites').select(
            'id, crawl_status, total_pages_found, pages_processed, '
            'last_crawled_at, created_at, updated_at'
        ).eq('website_id', str(website_id)).order('created_at', desc=True).limit(1).execute()
        
        if not result.data:
            return {
                'status': 'not_started',
                'pages_processed': 0,
                'total_pages_found': 0,
                'last_crawled_at': None
            }
        
        crawl_data = result.data[0]
        return {
            'status': crawl_data.get('crawl_status', 'unknown'),
            'pages_processed': crawl_data.get('pages_processed', 0),
            'total_pages_found': crawl_data.get('total_pages_found', 0),
            'last_crawled_at': crawl_data.get('last_crawled_at'),
            'progress_percentage': (
                (crawl_data.get('pages_processed', 0) / max(crawl_data.get('total_pages_found', 1), 1)) * 100
                if crawl_data.get('total_pages_found', 0) > 0 else 0
            )
        }
    
    async def get_scraped_pages(
        self, 
        website_id: UUID, 
        page_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get scraped pages for a website.
        
        Args:
            website_id: UUID of the website
            page_type: Optional filter by page type
            limit: Maximum number of pages to return
            offset: Number of pages to skip
            
        Returns:
            Dictionary with pages and pagination info
        """
        # Build query
        query = self.supabase.table('scraped_pages').select(
            'id, url, title, meta_description, page_type, word_count, '
            'relevance_score, scraped_at, depth_level'
        ).eq('scraped_website_id', str(website_id))
        
        if page_type:
            query = query.eq('page_type', page_type)
        
        # Get total count
        count_result = query.execute(count='exact')
        total_count = count_result.count if hasattr(count_result, 'count') else 0
        
        # Get paginated results
        result = query.order('relevance_score', desc=True).range(offset, offset + limit - 1).execute()
        
        return {
            'pages': result.data or [],
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total_count
        }
    
    async def search_content(
        self, 
        website_id: UUID, 
        query: str,
        page_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search scraped content using PostgreSQL full-text search.
        
        Args:
            website_id: UUID of the website
            query: Search query string
            page_type: Optional filter by page type
            limit: Maximum number of results
            
        Returns:
            List of matching pages with relevance scores
        """
        # Note: This would use PostgreSQL's full-text search capabilities
        # For now, implement basic text matching as Supabase equivalent
        search_query = self.supabase.table('scraped_pages').select(
            'id, url, title, meta_description, content_text, page_type, '
            'word_count, scraped_at'
        ).eq('scraped_website_id', str(website_id))
        
        # Add text search filters
        search_terms = query.lower().split()
        for term in search_terms:
            search_query = search_query.or_(
                f'title.ilike.%{term}%,'
                f'meta_description.ilike.%{term}%,'
                f'content_text.ilike.%{term}%'
            )
        
        if page_type:
            search_query = search_query.eq('page_type', page_type)
        
        result = search_query.limit(limit).execute()
        
        return result.data or []
    
    async def update_scraping_config(
        self, 
        website_id: UUID, 
        config: Dict[str, Any]
    ) -> bool:
        """
        Update scraping configuration for a website.
        
        Args:
            website_id: UUID of the website
            config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate configuration
            CrawlerConfig.from_dict(config)  # This will raise if invalid
            
            # Update in database
            result = self.supabase.table('websites').update({
                'scraping_config': config,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', str(website_id)).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Failed to update scraping config for {website_id}: {e}")
            return False
    
    async def enable_scraping(self, website_id: UUID, enabled: bool = True) -> bool:
        """
        Enable or disable scraping for a website.
        
        Args:
            website_id: UUID of the website
            enabled: Whether to enable or disable scraping
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.supabase.table('websites').update({
                'scraping_enabled': enabled,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', str(website_id)).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Failed to {'enable' if enabled else 'disable'} scraping for {website_id}: {e}")
            return False
    
    async def delete_scraped_data(self, website_id: UUID) -> bool:
        """
        Delete all scraped data for a website.
        
        Args:
            website_id: UUID of the website
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete scraped_websites record (cascades to related data)
            result = self.supabase.table('scraped_websites').delete().eq(
                'website_id', str(website_id)
            ).execute()
            
            logger.info(f"Deleted scraped data for website {website_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete scraped data for {website_id}: {e}")
            return False
    
    async def get_crawl_statistics(self, website_id: UUID) -> Dict[str, Any]:
        """
        Get comprehensive crawl statistics for a website.
        
        Args:
            website_id: UUID of the website
            
        Returns:
            Dictionary with detailed crawl statistics
        """
        # Get scraped website info
        website_result = self.supabase.table('scraped_websites').select(
            '*'
        ).eq('website_id', str(website_id)).order('created_at', desc=True).limit(1).execute()
        
        if not website_result.data:
            return {'status': 'no_data'}
        
        scraped_website = website_result.data[0]
        scraped_website_id = scraped_website['id']
        
        # Get page statistics
        pages_result = self.supabase.table('scraped_pages').select(
            'page_type, word_count, depth_level'
        ).eq('scraped_website_id', scraped_website_id).execute()
        
        pages_data = pages_result.data or []
        
        # Calculate statistics
        page_types = {}
        depth_distribution = {}
        total_words = 0
        
        for page in pages_data:
            # Page type distribution
            page_type = page.get('page_type', 'unknown')
            page_types[page_type] = page_types.get(page_type, 0) + 1
            
            # Depth distribution
            depth = page.get('depth_level', 0)
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
            
            # Word count
            total_words += page.get('word_count', 0)
        
        return {
            'status': scraped_website.get('crawl_status', 'unknown'),
            'total_pages': len(pages_data),
            'pages_processed': scraped_website.get('pages_processed', 0),
            'total_pages_found': scraped_website.get('total_pages_found', 0),
            'total_words': total_words,
            'average_words_per_page': total_words / max(len(pages_data), 1),
            'page_type_distribution': page_types,
            'depth_distribution': depth_distribution,
            'last_crawled_at': scraped_website.get('last_crawled_at'),
            'crawl_config': {
                'max_pages': scraped_website.get('max_pages', 100),
                'crawl_depth': scraped_website.get('crawl_depth', 3)
            }
        }
    
    async def _get_or_create_global_page(self, url: str, page_content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get existing global page or create new one (improved schema)."""
        try:
            # Check if global_scraped_pages table exists
            existing_page = self.supabase.table('global_scraped_pages').select('*').eq('url', url).execute()
            
            if existing_page.data:
                print(f"♻️ CRAWLER: Found existing global page for {url}")
                return existing_page.data[0]
            else:
                # Create new global page
                content_hash = hashlib.md5(page_content['content_text'].encode()).hexdigest()
                
                global_page_data = {
                    'url': url,
                    'title': page_content['title'][:500],
                    'meta_description': page_content['meta_description'][:1000] if page_content['meta_description'] else None,
                    'content_text': page_content['content_text'],
                    'content_html': page_content['content_html'],
                    'content_hash': content_hash,
                    'word_count': len(page_content['content_text'].split()),
                    'status_code': page_content['status_code']
                }
                
                result = self.supabase.table('global_scraped_pages').insert(global_page_data).execute()
                if result.data:
                    print(f"🆕 CRAWLER: Created new global page for {url}")
                    return result.data[0]
                    
        except Exception as e:
            # If global_scraped_pages doesn't exist or other error, fall back to legacy
            print(f"⚠️ CRAWLER: Global pages not available, using legacy: {e}")
            return None
        
        return None
    
    async def _create_website_page_association(self, scraped_website_id: str, global_page_id: str, depth_level: int):
        """Create association between website and global page."""
        try:
            association_data = {
                'scraped_website_id': scraped_website_id,
                'global_page_id': global_page_id,
                'depth_level': depth_level,
                'relevance_score': 1.0
            }
            
            self.supabase.table('website_page_associations').insert(association_data).execute()
            print(f"🔗 CRAWLER: Created website-page association")
            
        except Exception as e:
            # Might already exist (unique constraint) or table doesn't exist
            if 'duplicate key' in str(e):
                print(f"🔗 CRAWLER: Association already exists")
            else:
                print(f"⚠️ CRAWLER: Failed to create association: {e}")
    
    async def get_website_shared_content(self, website_id: UUID) -> List[Dict[str, Any]]:
        """Get all content associated with a website (improved schema)."""
        try:
            # Try to get content from improved schema
            result = self.supabase.table('website_page_associations').select(
                '''
                global_page_id,
                depth_level,
                relevance_score,
                global_scraped_pages!inner(
                    id, url, title, meta_description, content_text, word_count, last_scraped
                )
                '''
            ).join(
                'scraped_websites!inner'
            ).eq('scraped_websites.website_id', str(website_id)).execute()
            
            if result.data:
                print(f"📊 CRAWLER: Found {len(result.data)} shared pages for website {website_id}")
                return result.data
                
        except Exception as e:
            print(f"⚠️ CRAWLER: Shared content query failed: {e}")
        
        return []