"""
Website Registration Service
Handles website registration, URL validation, and duplicate detection.
"""

import uuid
import secrets
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError

from ..models.website import Website
from ..models.user import User
from ..models.user_website import UserWebsite
from .web_crawler import WebCrawler, CrawlerConfig
from .crawler_service import CrawlerService


class WebsiteRegistrationService:
    """Service for handling website registration and management."""
    
    def __init__(self):
        self.crawler_service = CrawlerService()
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate URL format and extract domain information.
        
        Args:
            url: The URL to validate
            
        Returns:
            Dict containing validation result and extracted info
        """
        try:
            # Basic URL format validation
            parsed = urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                return {
                    "valid": False,
                    "error": "Invalid URL format. URL must include scheme (http/https) and domain."
                }
            
            if parsed.scheme not in ['http', 'https']:
                return {
                    "valid": False,
                    "error": "URL must use HTTP or HTTPS protocol."
                }
            
            # Extract domain information
            domain = parsed.netloc.lower()
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            # Basic domain validation
            domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$'
            if not re.match(domain_pattern, domain):
                return {
                    "valid": False,
                    "error": "Invalid domain format."
                }
            
            return {
                "valid": True,
                "domain": domain,
                "parsed_url": parsed,
                "normalized_url": f"{parsed.scheme}://{domain}{parsed.path or '/'}"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"URL validation error: {str(e)}"
            }
    
    async def check_duplicate_website(self, url: str, domain: str, db: AsyncSession) -> Optional[Website]:
        """
        Check if website with given URL or domain already exists.
        
        Args:
            url: The full URL to check
            domain: The domain to check
            db: Database session
            
        Returns:
            Existing website if found, None otherwise
        """
        # Check for exact URL match first
        result = await db.execute(
            select(Website).where(Website.url == url)
        )
        existing_website = result.scalar_one_or_none()
        
        if existing_website:
            return existing_website
        
        # Check for domain match
        result = await db.execute(
            select(Website).where(Website.domain == domain)
        )
        return result.scalar_one_or_none()
    
    def generate_widget_id(self) -> str:
        """Generate unique widget ID for website."""
        return f"widget_{uuid.uuid4().hex[:16]}"
    
    def generate_verification_token(self) -> str:
        """Generate secure verification token for ownership verification."""
        return f"chatlite_{secrets.token_urlsafe(32)}"
    
    async def register_website(
        self,
        user_id: str,
        website_data: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Register a new website for a user.
        
        Args:
            user_id: ID of the user registering the website
            website_data: Dictionary containing website information
            db: Database session
            
        Returns:
            Dict containing registration result and website data
        """
        try:
            # Validate required fields
            required_fields = ['url', 'name']
            for field in required_fields:
                if field not in website_data or not website_data[field]:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }
            
            # Validate and normalize URL
            url_validation = self.validate_url(website_data['url'])
            if not url_validation['valid']:
                return {
                    "success": False,
                    "error": url_validation['error']
                }
            
            domain = url_validation['domain']
            normalized_url = url_validation['normalized_url']
            
            # Check for duplicates
            existing_website = await self.check_duplicate_website(normalized_url, domain, db)
            if existing_website:
                return {
                    "success": False,
                    "error": f"Website with URL '{normalized_url}' is already registered"
                }
            
            # Verify user exists
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            if not user:
                return {
                    "success": False,
                    "error": "User not found"
                }
            
            # Create new website
            website = Website(
                id=str(uuid.uuid4()),
                name=website_data['name'],
                url=normalized_url,
                domain=domain,
                widget_id=self.generate_widget_id(),
                verification_token=self.generate_verification_token(),
                
                # Optional fields
                business_name=website_data.get('business_name'),
                contact_email=website_data.get('contact_email'),
                business_description=website_data.get('business_description'),
                widget_color=website_data.get('widget_color', '#0066CC'),
                widget_position=website_data.get('widget_position', 'bottom-right'),
                welcome_message=website_data.get('welcome_message'),
                
                # Status fields
                verification_status='pending',
                scraping_status='not_started',
                widget_status='not_configured',
                is_active=True,
                scraping_enabled=True
            )
            
            db.add(website)
            await db.flush()  # Get the website ID
            
            # Create user-website relationship
            user_website = UserWebsite(
                id=str(uuid.uuid4()),
                user_id=user_id,
                website_id=website.id,
                role='owner'
            )
            
            db.add(user_website)
            await db.commit()

            # Trigger screenshot capture task (async, don't wait for it)
            try:
                from app.core.celery_config import celery_app
                celery_app.send_task(
                    'crawler.tasks.capture_screenshot',
                    args=[website.id, normalized_url]
                )
                logger.info(f"Screenshot capture task triggered for website {website.id}")
            except Exception as e:
                # Don't fail registration if screenshot task fails to trigger
                logger.error(f"Failed to trigger screenshot task: {e}")

            return {
                "success": True,
                "website": {
                    "id": website.id,
                    "name": website.name,
                    "url": website.url,
                    "domain": website.domain,
                    "widget_id": website.widget_id,
                    "verification_status": website.verification_status,
                    "scraping_status": website.scraping_status,
                    "widget_status": website.widget_status,
                    "business_name": website.business_name,
                    "contact_email": website.contact_email,
                    "business_description": website.business_description,
                    "widget_color": website.widget_color,
                    "widget_position": website.widget_position,
                    "welcome_message": website.welcome_message,
                    "is_active": website.is_active,
                    "created_at": website.created_at.isoformat() if website.created_at else None
                }
            }
            
        except IntegrityError as e:
            await db.rollback()
            return {
                "success": False,
                "error": "Website registration failed due to database constraint"
            }
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Website registration failed: {str(e)}"
            }
    
    async def get_user_websites(
        self,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get all websites associated with a user.
        
        Args:
            user_id: ID of the user
            db: Database session
            
        Returns:
            Dict containing user's websites
        """
        try:
            result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(UserWebsite.user_id == user_id)
                .order_by(Website.created_at.desc())
            )
            
            websites = []
            for website, user_website in result.all():
                websites.append({
                    "id": website.id,
                    "name": website.name,
                    "url": website.url,
                    "domain": website.domain,
                    "widget_id": website.widget_id,
                    "verification_status": website.verification_status,
                    "scraping_status": website.scraping_status,
                    "widget_status": website.widget_status,
                    "role": user_website.role,
                    "is_active": website.is_active,
                    "created_at": website.created_at.isoformat() if website.created_at else None,
                    "verified_at": website.verified_at.isoformat() if website.verified_at else None,
                    "last_crawled": website.last_crawled.isoformat() if website.last_crawled else None
                })
            
            return {
                "success": True,
                "websites": websites,
                "total_count": len(websites)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve websites: {str(e)}"
            }
    
    async def get_website_details(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get detailed information for a specific website.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user requesting the details
            db: Database session
            
        Returns:
            Dict containing website details
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
            
            website, user_website = row
            
            return {
                "success": True,
                "website": {
                    "id": website.id,
                    "name": website.name,
                    "url": website.url,
                    "domain": website.domain,
                    "widget_id": website.widget_id,
                    "verification_status": website.verification_status,
                    "verification_method": website.verification_method,
                    "verified_at": website.verified_at.isoformat() if website.verified_at else None,
                    "scraping_status": website.scraping_status,
                    "widget_status": website.widget_status,
                    "last_crawled": website.last_crawled.isoformat() if website.last_crawled else None,
                    "business_name": website.business_name,
                    "contact_email": website.contact_email,
                    "business_description": website.business_description,
                    "widget_color": website.widget_color,
                    "widget_position": website.widget_position,
                    "welcome_message": website.welcome_message,
                    "total_conversations": website.total_conversations,
                    "total_messages": website.total_messages,
                    "monthly_message_limit": website.monthly_message_limit,
                    "role": user_website.role,
                    "is_active": website.is_active,
                    "created_at": website.created_at.isoformat() if website.created_at else None,
                    "updated_at": website.updated_at.isoformat() if website.updated_at else None
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve website details: {str(e)}"
            }
    
    async def update_website(
        self,
        website_id: str,
        user_id: str,
        update_data: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Update website settings.
        
        Args:
            website_id: ID of the website to update
            user_id: ID of the user making the update
            update_data: Dictionary containing fields to update
            db: Database session
            
        Returns:
            Dict containing update result
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
            
            website, user_website = row
            
            # Define allowed update fields
            allowed_fields = {
                'name', 'business_name', 'contact_email', 'business_description',
                'widget_color', 'widget_position', 'welcome_message'
            }
            
            # Update allowed fields
            updated_fields = []
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(website, field):
                    setattr(website, field, value)
                    updated_fields.append(field)
            
            if updated_fields:
                website.updated_at = datetime.utcnow()
                await db.commit()
                
                return {
                    "success": True,
                    "message": f"Updated fields: {', '.join(updated_fields)}",
                    "website": {
                        "id": website.id,
                        "name": website.name,
                        "business_name": website.business_name,
                        "contact_email": website.contact_email,
                        "business_description": website.business_description,
                        "widget_color": website.widget_color,
                        "widget_position": website.widget_position,
                        "welcome_message": website.welcome_message,
                        "updated_at": website.updated_at.isoformat()
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No valid fields to update"
                }
                
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Failed to update website: {str(e)}"
            }
    
    async def delete_website(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Delete a website.
        
        Args:
            website_id: ID of the website to delete
            user_id: ID of the user requesting deletion
            db: Database session
            
        Returns:
            Dict containing deletion result
        """
        try:
            # Verify user is owner of this website
            result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(
                    and_(
                        Website.id == website_id,
                        UserWebsite.user_id == user_id,
                        UserWebsite.role == 'owner'
                    )
                )
            )
            
            row = result.first()
            if not row:
                return {
                    "success": False,
                    "error": "Website not found or you don't have permission to delete it"
                }
            
            website, user_website = row
            
            # Delete user-website relationship first (due to foreign key constraints)
            await db.execute(
                select(UserWebsite).where(UserWebsite.website_id == website_id)
            )
            await db.delete(user_website)
            
            # Delete the website (cascade will handle related records)
            await db.delete(website)
            await db.commit()
            
            return {
                "success": True,
                "message": f"Website '{website.name}' has been deleted successfully"
            }
            
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Failed to delete website: {str(e)}"
            }