"""
Improved Crawler Service with content sharing capabilities.
Addresses the issue where the same page content needs to be available to multiple websites.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID
import hashlib

from .web_crawler import WebCrawler, CrawlerConfig
from .scraper_service import ScrapedWebsiteService
from ..core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class ImprovedCrawlerService:
    """Improved crawler service with content sharing across websites."""
    
    def __init__(self):
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
        Start crawling with improved content sharing logic.
        """
        print(f"🚀 IMPROVED CRAWLER: Starting crawl for website_id: {website_id}")
        logger.info(f"🚀 Starting improved crawl for website_id: {website_id}")
        
        try:
            # Get website configuration
            print(f"📋 IMPROVED CRAWLER: Fetching website configuration")
            website_result = self.supabase.table('websites').select(
                'id, name, domain, url, scraping_enabled, business_description'
            ).eq('id', str(website_id)).execute()
            
            if not website_result.data:
                raise ValueError(f"Website {website_id} not found")
            
            website_data = website_result.data[0]
            print(f"✅ IMPROVED CRAWLER: Found website: {website_data['name']}")
            
            if not website_data.get('scraping_enabled', False):
                raise ValueError(f"Scraping not enabled for website {website_id}")
            
            # Use the URL from the website data if base_url not provided
            if not base_url:
                base_url = website_data.get('url') or f"https://{domain}"
            
            # Create or update scraped_website record (crawl session)
            scraped_website = await self._get_or_create_crawl_session(
                website_id, domain, base_url
            )
            
            # Start crawling process
            print(f"🕷️ IMPROVED CRAWLER: Starting to crawl {base_url}")
            pages_crawled = await self._crawl_with_sharing(
                scraped_website['id'], website_id, base_url, domain
            )
            
            # Update final status
            self.supabase.table('scraped_websites').update({
                'crawl_status': 'completed',
                'pages_processed': pages_crawled,
                'total_pages_found': pages_crawled
            }).eq('id', scraped_website['id']).execute()
            
            print(f"🎉 IMPROVED CRAWLER: Completed crawl: {pages_crawled} pages")
            
            return {
                'scraped_website_id': scraped_website['id'],
                'status': 'completed',
                'pages_processed': pages_crawled,
                'total_pages_found': pages_crawled,
                'domain': domain,
                'base_url': base_url
            }
                
        except Exception as e:
            logger.error(f"💥 Improved crawl failed: {str(e)}", exc_info=True)
            raise
    
    async def _get_or_create_crawl_session(self, website_id: UUID, domain: str, base_url: str) -> Dict[str, Any]:
        """Get or create a crawl session (scraped_website record)."""
        
        # Look for existing session
        existing_session = self.supabase.table('scraped_websites').select('*').eq(
            'website_id', str(website_id)
        ).execute()
        
        if existing_session.data:
            session = existing_session.data[0]
            print(f"📝 IMPROVED CRAWLER: Using existing session: {session['id']}")
            
            # Update status to crawling
            self.supabase.table('scraped_websites').update({
                'crawl_status': 'crawling',
                'last_crawled_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', session['id']).execute()
            
            return session
        else:
            print(f"➕ IMPROVED CRAWLER: Creating new crawl session")
            session_data = {
                'website_id': str(website_id),
                'domain': domain,
                'base_url': base_url,
                'crawl_status': 'crawling',
                'last_crawled_at': datetime.now(timezone.utc).isoformat()
            }
            result = self.supabase.table('scraped_websites').insert(session_data).execute()
            return result.data[0]
    
    async def _crawl_with_sharing(
        self, 
        scraped_website_id: str, 
        website_id: UUID,
        base_url: str, 
        domain: str
    ) -> int:
        """
        Crawl with content sharing - reuse existing pages where possible.
        """
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        from ..services.vector_search_service import VectorSearchService
        
        print(f"🌐 IMPROVED CRAWLER: Starting shared crawl for {base_url}")
        pages_processed = 0
        vector_service = VectorSearchService()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                
                urls_to_crawl = [base_url]
                crawled_urls = set()
                max_pages = 5
                
                print(f"📝 IMPROVED CRAWLER: Will crawl max {max_pages} pages")
                
                while len(urls_to_crawl) > 0 and pages_processed < max_pages:
                    url = urls_to_crawl.pop(0)
                    
                    if url in crawled_urls:
                        continue
                        
                    try:
                        print(f"🔍 IMPROVED CRAWLER: Processing {url}")
                        
                        # Check if this URL is already in global pages
                        existing_page = await self._get_existing_global_page(url)
                        
                        if existing_page:
                            print(f"♻️ IMPROVED CRAWLER: Reusing existing page: {existing_page['id']}")
                            global_page_id = existing_page['id']
                            
                            # Just create the association
                            await self._create_website_page_association(
                                scraped_website_id, global_page_id, 0
                            )
                            
                        else:
                            print(f"🆕 IMPROVED CRAWLER: Crawling new page: {url}")
                            
                            # Crawl the page
                            page_content = await self._fetch_page_content(session, url)
                            if not page_content:
                                continue
                            
                            # Create global page record
                            global_page_id = await self._create_global_page(url, page_content)
                            
                            # Create website association
                            await self._create_website_page_association(
                                scraped_website_id, global_page_id, 0
                            )
                            
                            # Create embeddings
                            try:
                                chunk_ids = await vector_service.chunk_and_embed_content_improved(
                                    global_page_id, page_content['content_text']
                                )
                                print(f"🧠 IMPROVED CRAWLER: Created {len(chunk_ids)} chunks")
                            except Exception as e:
                                print(f"❌ IMPROVED CRAWLER: Embedding failed: {e}")
                            
                            # Find more links
                            if pages_processed < max_pages:
                                new_links = self._extract_links(page_content['soup'], url, base_url)
                                for link in new_links:
                                    if link not in crawled_urls and link not in urls_to_crawl:
                                        urls_to_crawl.append(link)
                                        print(f"➕ IMPROVED CRAWLER: Added link: {link}")
                        
                        pages_processed += 1
                        crawled_urls.add(url)
                        
                    except Exception as e:
                        print(f"💥 IMPROVED CRAWLER: Error processing {url}: {e}")
                        continue
                
                print(f"🏁 IMPROVED CRAWLER: Completed. Pages: {pages_processed}")
                return pages_processed
                
        except Exception as e:
            print(f"💥 IMPROVED CRAWLER: Fatal error: {e}")
            return pages_processed
    
    async def _get_existing_global_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Check if a URL is already in global_scraped_pages."""
        result = self.supabase.table('global_scraped_pages').select('*').eq('url', url).execute()
        return result.data[0] if result.data else None
    
    async def _fetch_page_content(self, session, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse page content."""
        try:
            async with session.get(url, headers={'User-Agent': 'LiteChat-Bot/1.0'}) as response:
                if response.status != 200:
                    return None
                
                html_content = await response.text()
                
                # Parse HTML
                from bs4 import BeautifulSoup
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
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                
                if len(clean_text) < 100:
                    return None
                
                return {
                    'title': title_text,
                    'meta_description': meta_description,
                    'content_text': clean_text,
                    'content_html': html_content[:50000],
                    'status_code': response.status,
                    'soup': soup
                }
        except Exception as e:
            print(f"💥 IMPROVED CRAWLER: Fetch failed for {url}: {e}")
            return None
    
    async def _create_global_page(self, url: str, page_content: Dict[str, Any]) -> str:
        """Create a global page record."""
        content_hash = hashlib.md5(page_content['content_text'].encode()).hexdigest()
        
        page_data = {
            'url': url,
            'title': page_content['title'][:500],
            'meta_description': page_content['meta_description'][:1000] if page_content['meta_description'] else None,
            'content_text': page_content['content_text'],
            'content_html': page_content['content_html'],
            'content_hash': content_hash,
            'word_count': len(page_content['content_text'].split()),
            'status_code': page_content['status_code']
        }
        
        result = self.supabase.table('global_scraped_pages').insert(page_data).execute()
        return result.data[0]['id']
    
    async def _create_website_page_association(
        self, 
        scraped_website_id: str, 
        global_page_id: str, 
        depth_level: int
    ):
        """Create association between website and global page."""
        association_data = {
            'scraped_website_id': scraped_website_id,
            'global_page_id': global_page_id,
            'depth_level': depth_level,
            'relevance_score': 1.0
        }
        
        try:
            self.supabase.table('website_page_associations').insert(association_data).execute()
            print(f"🔗 IMPROVED CRAWLER: Created website-page association")
        except Exception as e:
            # Might already exist (unique constraint)
            print(f"🔗 IMPROVED CRAWLER: Association exists or failed: {e}")
    
    def _extract_links(self, soup, current_url: str, base_url: str) -> List[str]:
        """Extract valid internal links."""
        links = []
        found_links = soup.find_all('a', href=True)
        
        print(f"🔗 IMPROVED CRAWLER: Found {len(found_links)} links")
        
        for link in found_links[:20]:
            href = link.get('href', '').strip()
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
            
            full_url = urljoin(current_url, href)
            from urllib.parse import urlparse
            parsed = urlparse(full_url)
            base_parsed = urlparse(base_url)
            
            # Only crawl same domain, avoid certain file types
            if (parsed.netloc == base_parsed.netloc and 
                not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.zip', '.doc']) and
                not '#' in full_url):
                
                links.append(full_url)
        
        return links
    
    async def get_website_content(self, website_id: UUID) -> List[Dict[str, Any]]:
        """Get all content associated with a specific website."""
        # Get all pages associated with this website
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
            'scraped_websites!inner', 
            'scraped_websites.id', 
            'website_page_associations.scraped_website_id'
        ).eq('scraped_websites.website_id', str(website_id)).execute()
        
        return result.data or []