"""
Website Ownership Verification Service
Handles HTML tag and DNS record verification methods.
"""

import re
import dns.resolver
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.website import Website
from ..models.user_website import UserWebsite
from .web_crawler import WebCrawler, CrawlerConfig
from .crawler_service import CrawlerService


class WebsiteVerificationService:
    """Service for verifying website ownership."""
    
    def __init__(self):
        self.crawler_service = CrawlerService()
    
    async def get_verification_methods(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get available verification methods and instructions for a website.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user requesting verification methods
            db: Database session
            
        Returns:
            Dict containing verification methods and instructions
        """
        try:
            # Verify user has access to this website
            result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(
                    and_(
                        Website.id == website_id,
                        UserWebsite.user_id == user_id
                    )
                )
            )
            
            row = result.first()
            if not row:
                return {
                    "success": False,
                    "error": "Website not found or access denied"
                }
            
            website, _ = row
            
            # Generate verification instructions
            verification_token = website.verification_token
            domain = website.domain
            
            return {
                "success": True,
                "verification_token": verification_token,
                "methods": {
                    "html_tag": {
                        "name": "HTML Meta Tag",
                        "description": "Add a meta tag to your website's homepage",
                        "instructions": [
                            "Add the following meta tag to the <head> section of your homepage:",
                            f'<meta name="chatlite-verification" content="{verification_token}">',
                            "The tag must be present on the root page of your domain",
                            "Click 'Verify' once the tag is in place"
                        ],
                        "tag": f'<meta name="chatlite-verification" content="{verification_token}">',
                        "verification_url": f"{website.url}"
                    },
                    "dns_record": {
                        "name": "DNS TXT Record",
                        "description": "Add a TXT record to your domain's DNS settings",
                        "instructions": [
                            "Add a TXT record to your DNS settings with these details:",
                            f"Record Type: TXT",
                            f"Host/Name: _chatlite-verification",
                            f"Value: {verification_token}",
                            "DNS propagation may take up to 48 hours",
                            "Click 'Verify' once the record is active"
                        ],
                        "record_name": "_chatlite-verification",
                        "record_value": verification_token,
                        "domain": domain
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get verification methods: {str(e)}"
            }
    
    async def verify_html_tag(
        self,
        website: Website
    ) -> Dict[str, Any]:
        """
        Verify ownership using HTML meta tag method.
        
        Args:
            website: Website object to verify
            
        Returns:
            Dict containing verification result
        """
        try:
            # Fetch the homepage content using aiohttp directly
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(website.url) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to fetch website content. HTTP status: {response.status}"
                        }
                    
                    content = await response.text()
            
            if not content:
                return {
                    "success": False,
                    "error": "Failed to fetch website content. Please ensure your website is accessible."
                }
            
            # Look for the verification meta tag
            verification_token = website.verification_token
            tag_pattern = rf'<meta\s+name=["\']chatlite-verification["\']\s+content=["\']({re.escape(verification_token)})["\']\s*/?>'
            
            if re.search(tag_pattern, content, re.IGNORECASE):
                return {
                    "success": True,
                    "message": "HTML verification tag found successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Verification tag not found. Please ensure the meta tag is present in your homepage's <head> section."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"HTML verification failed: {str(e)}"
            }
    
    async def verify_dns_record(
        self,
        website: Website
    ) -> Dict[str, Any]:
        """
        Verify ownership using DNS TXT record method.
        
        Args:
            website: Website object to verify
            
        Returns:
            Dict containing verification result
        """
        try:
            domain = website.domain
            verification_token = website.verification_token
            record_name = f"_chatlite-verification.{domain}"
            
            # Query DNS for the TXT record
            try:
                answers = dns.resolver.resolve(record_name, 'TXT')
                
                for rdata in answers:
                    txt_value = rdata.to_text().strip('"')
                    if txt_value == verification_token:
                        return {
                            "success": True,
                            "message": "DNS TXT record verified successfully"
                        }
                
                return {
                    "success": False,
                    "error": f"DNS TXT record not found or token mismatch. Expected: {verification_token}"
                }
                
            except dns.resolver.NXDOMAIN:
                return {
                    "success": False,
                    "error": f"DNS record '_chatlite-verification.{domain}' not found. Please add the TXT record to your DNS settings."
                }
            except dns.resolver.NoAnswer:
                return {
                    "success": False,
                    "error": f"No TXT records found for '_chatlite-verification.{domain}'. Please add the required TXT record."
                }
            except Exception as dns_error:
                return {
                    "success": False,
                    "error": f"DNS lookup failed: {str(dns_error)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"DNS verification failed: {str(e)}"
            }
    
    async def verify_website_ownership(
        self,
        website_id: str,
        user_id: str,
        verification_method: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Verify website ownership using specified method.
        
        Args:
            website_id: ID of the website to verify
            user_id: ID of the user requesting verification
            verification_method: Method to use ('html_tag' or 'dns_record')
            db: Database session
            
        Returns:
            Dict containing verification result
        """
        try:
            # Verify user has access to this website
            result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(
                    and_(
                        Website.id == website_id,
                        UserWebsite.user_id == user_id
                    )
                )
            )
            
            row = result.first()
            if not row:
                return {
                    "success": False,
                    "error": "Website not found or access denied"
                }
            
            website, _ = row
            
            # Check if already verified
            if website.verification_status == 'verified':
                return {
                    "success": True,
                    "message": "Website is already verified",
                    "verified_at": website.verified_at.isoformat() if website.verified_at else None,
                    "verification_method": website.verification_method
                }
            
            # Perform verification based on method
            if verification_method == 'html_tag':
                verification_result = await self.verify_html_tag(website)
            elif verification_method == 'dns_record':
                verification_result = await self.verify_dns_record(website)
            else:
                return {
                    "success": False,
                    "error": "Invalid verification method. Use 'html_tag' or 'dns_record'"
                }
            
            # Update website status if verification successful
            if verification_result['success']:
                website.verification_status = 'verified'
                website.verification_method = verification_method
                website.verified_at = datetime.utcnow()
                website.updated_at = datetime.utcnow()
                
                # Update widget status if it's not configured
                if website.widget_status == 'not_configured':
                    website.widget_status = 'ready_for_configuration'
                
                await db.commit()
                
                # Initiate content crawling after successful verification
                try:
                    await self.initiate_content_crawling(website)
                except Exception as crawl_error:
                    # Log error but don't fail the verification
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to initiate crawling for website {website.id}: {crawl_error}")
                    # Update scraping status to indicate failure
                    website.scraping_status = 'failed'
                    await db.commit()
                
                return {
                    "success": True,
                    "verified": True,
                    "verification_status": website.verification_status,
                    "verification_method": website.verification_method,
                    "verified_at": website.verified_at.isoformat(),
                    "message": verification_result['message'],
                    "scraping_initiated": website.scraping_status == 'in_progress'
                }
            else:
                return {
                    "success": False,
                    "error": verification_result['error']
                }
                
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Verification failed: {str(e)}"
            }
    
    async def get_website_verification_status(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get current verification status of a website.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user requesting status
            db: Database session
            
        Returns:
            Dict containing verification status information
        """
        try:
            # Verify user has access to this website
            result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(
                    and_(
                        Website.id == website_id,
                        UserWebsite.user_id == user_id
                    )
                )
            )
            
            row = result.first()
            if not row:
                return {
                    "success": False,
                    "error": "Website not found or access denied"
                }
            
            website, _ = row
            
            return {
                "success": True,
                "website_id": website.id,
                "verification_status": website.verification_status,
                "verification_method": website.verification_method,
                "verified_at": website.verified_at.isoformat() if website.verified_at else None,
                "scraping_status": website.scraping_status,
                "widget_status": website.widget_status,
                "last_crawled": website.last_crawled.isoformat() if website.last_crawled else None,
                "is_verified": website.verification_status == 'verified',
                "next_steps": self._get_next_steps(website)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get verification status: {str(e)}"
            }
    
    def _get_next_steps(self, website: Website) -> list:
        """
        Get suggested next steps based on website status.
        
        Args:
            website: Website object
            
        Returns:
            List of suggested next steps
        """
        steps = []
        
        if website.verification_status == 'pending':
            steps.append("Verify website ownership using HTML tag or DNS record method")
        
        if website.verification_status == 'verified' and website.scraping_status == 'not_started':
            steps.append("Content scraping will begin automatically after verification")
        
        if website.verification_status == 'verified' and website.widget_status == 'ready_for_configuration':
            steps.append("Customize your chatbot widget appearance and behavior")
        
        if website.verification_status == 'verified' and website.widget_status == 'configured':
            steps.append("Generate and install the widget script on your website")
        
        if not steps:
            steps.append("Your website setup is complete!")
        
        return steps
    
    async def initiate_content_crawling(self, website: Website) -> Dict[str, Any]:
        """
        Initiate content crawling for a verified website.
        
        Args:
            website: Verified website object
            
        Returns:
            Dict containing crawling initiation result
        """
        try:
            # Update scraping status to in_progress
            website.scraping_status = 'in_progress'
            
            # Start the crawling process asynchronously
            from uuid import UUID
            website_uuid = UUID(website.id)
            
            # Use default crawling configuration
            crawl_config = {
                'max_pages': 50,  # Reasonable default for initial crawl
                'crawl_depth': 3,
                'respect_robots_txt': True,
                'delay_between_requests': 1.0
            }
            
            # Initiate crawling (this will run asynchronously)
            crawl_result = await self.crawler_service.start_crawl(
                website_id=website_uuid,
                base_url=website.url,
                domain=website.domain,
                config_override=crawl_config
            )
            
            # Update last_crawled timestamp
            website.last_crawled = datetime.utcnow()
            website.scraping_status = 'completed' if crawl_result.get('success', False) else 'failed'
            
            return {
                "success": True,
                "message": f"Crawling initiated for {website.domain}",
                "pages_found": crawl_result.get('pages_processed', 0)
            }
            
        except Exception as e:
            website.scraping_status = 'failed'
            return {
                "success": False,
                "error": f"Failed to initiate crawling: {str(e)}"
            }