"""
Widget Configuration Service
Handles widget customization, configuration validation, and versioning.
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ..models.website import Website
from ..models.user_website import UserWebsite
from ..models.widget_configuration import WidgetConfigurationVersion


class WidgetConfigurationService:
    """Service for managing widget configuration and customization."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "widget_color": "#0066CC",
        "widget_position": "bottom-right",
        "widget_size": "medium",
        "widget_theme": "light",
        "show_avatar": True,
        "enable_sound": True,
        "auto_open_delay": None,
        "show_online_status": True,
        "welcome_message": "Hello! How can I help you today?",
        "placeholder_text": "Type your message...",
        "offline_message": "We're currently offline, but we'll get back to you soon!",
        "thanks_message": "Thanks for chatting with us!",
        "show_branding": True,
        "custom_logo_url": None,
        "company_name": None,
        "support_email": None,
        "custom_css": None,
        "font_family": None,
        "border_radius": 8
    }
    
    # Valid options for enum-like fields
    VALID_POSITIONS = ["bottom-right", "bottom-left", "top-right", "top-left"]
    VALID_SIZES = ["small", "medium", "large"]
    VALID_THEMES = ["light", "dark", "auto"]
    
    async def get_website_configuration(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get current configuration for a website.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user requesting configuration
            db: Database session
            
        Returns:
            Dict containing current configuration
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
            
            # Build configuration from website fields
            config = self._build_configuration_from_website(website)
            
            return {
                "success": True,
                "configuration": config,
                "website_id": website_id,
                "last_updated": website.updated_at.isoformat() if website.updated_at else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get configuration: {str(e)}"
            }
    
    async def update_website_configuration(
        self,
        website_id: str,
        user_id: str,
        config_updates: Dict[str, Any],
        db: AsyncSession,
        create_version: bool = True
    ) -> Dict[str, Any]:
        """
        Update website configuration.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user updating configuration
            config_updates: Dictionary of configuration updates
            db: Database session
            create_version: Whether to create a configuration version
            
        Returns:
            Dict containing update result and new configuration
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
            
            # Create version before updating (if requested)
            if create_version:
                current_config = self._build_configuration_from_website(website)
                await self._create_configuration_version(website, current_config, user_id, db)
            
            # Validate configuration updates
            validation_result = self.validate_configuration(config_updates)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Configuration validation failed: {validation_result['errors']}"
                }
            
            # Apply updates to website model
            self._apply_configuration_to_website(website, config_updates)
            website.updated_at = datetime.utcnow()
            
            # Update widget status
            if website.widget_status == 'ready_for_configuration':
                website.widget_status = 'configured'
            
            await db.commit()
            
            # Return updated configuration
            updated_config = self._build_configuration_from_website(website)
            
            return {
                "success": True,
                "configuration": updated_config,
                "message": "Configuration updated successfully",
                "version_created": create_version
            }
            
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Failed to update configuration: {str(e)}"
            }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate widget configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dict with validation results
        """
        errors = []
        
        # Validate widget_color (hex color)
        if "widget_color" in config:
            color = config["widget_color"]
            if color and not re.match(r"^#[0-9A-Fa-f]{6}$", color):
                errors.append("widget_color must be a valid hex color (e.g., #FF0000)")
        
        # Validate widget_position
        if "widget_position" in config:
            position = config["widget_position"]
            if position not in self.VALID_POSITIONS:
                errors.append(f"widget_position must be one of: {', '.join(self.VALID_POSITIONS)}")
        
        # Validate widget_size
        if "widget_size" in config:
            size = config["widget_size"]
            if size not in self.VALID_SIZES:
                errors.append(f"widget_size must be one of: {', '.join(self.VALID_SIZES)}")
        
        # Validate widget_theme
        if "widget_theme" in config:
            theme = config["widget_theme"]
            if theme not in self.VALID_THEMES:
                errors.append(f"widget_theme must be one of: {', '.join(self.VALID_THEMES)}")
        
        # Validate message lengths
        message_fields = {
            "welcome_message": 1000,
            "placeholder_text": 200,
            "offline_message": 1000,
            "thanks_message": 1000
        }
        
        for field, max_length in message_fields.items():
            if field in config and config[field]:
                if len(config[field]) > max_length:
                    errors.append(f"{field} must be {max_length} characters or less")
        
        # Validate auto_open_delay
        if "auto_open_delay" in config:
            delay = config["auto_open_delay"]
            if delay is not None and (not isinstance(delay, int) or delay < 0 or delay > 300):
                errors.append("auto_open_delay must be an integer between 0 and 300 seconds")
        
        # Validate URLs
        url_fields = ["custom_logo_url"]
        for field in url_fields:
            if field in config and config[field]:
                url = config[field]
                if not self._is_valid_url(url):
                    errors.append(f"{field} must be a valid URL")
        
        # Validate CSS (basic check)
        if "custom_css" in config and config["custom_css"]:
            css = config["custom_css"]
            if len(css) > 10000:  # 10KB limit
                errors.append("custom_css must be 10KB or less")
        
        # Validate border_radius
        if "border_radius" in config:
            radius = config["border_radius"]
            if radius is not None and (not isinstance(radius, int) or radius < 0 or radius > 50):
                errors.append("border_radius must be an integer between 0 and 50 pixels")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default widget configuration."""
        return self.DEFAULT_CONFIG.copy()
    
    async def get_configuration_versions(
        self,
        website_id: str,
        user_id: str,
        db: AsyncSession,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get configuration version history for a website.
        
        Args:
            website_id: ID of the website
            user_id: ID of the user requesting versions
            db: Database session
            limit: Maximum number of versions to return
            
        Returns:
            Dict containing version history
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
            
            if not result.first():
                return {
                    "success": False,
                    "error": "Website not found or access denied"
                }
            
            # Get versions
            versions_result = await db.execute(
                select(WidgetConfigurationVersion)
                .where(WidgetConfigurationVersion.website_id == website_id)
                .order_by(desc(WidgetConfigurationVersion.created_at))
                .limit(limit)
            )
            
            versions = versions_result.scalars().all()
            
            return {
                "success": True,
                "versions": [version.to_dict() for version in versions],
                "total_count": len(versions)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get version history: {str(e)}"
            }
    
    async def rollback_to_version(
        self,
        website_id: str,
        version_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Rollback website configuration to a specific version.
        
        Args:
            website_id: ID of the website
            version_id: ID of the version to rollback to
            user_id: ID of the user performing rollback
            db: Database session
            
        Returns:
            Dict containing rollback result
        """
        try:
            # Verify user has access to this website
            website_result = await db.execute(
                select(Website, UserWebsite)
                .join(UserWebsite, Website.id == UserWebsite.website_id)
                .where(
                    and_(
                        Website.id == website_id,
                        UserWebsite.user_id == user_id
                    )
                )
            )
            
            website_row = website_result.first()
            if not website_row:
                return {
                    "success": False,
                    "error": "Website not found or access denied"
                }
            
            website, _ = website_row
            
            # Get the version to rollback to
            version_result = await db.execute(
                select(WidgetConfigurationVersion)
                .where(
                    and_(
                        WidgetConfigurationVersion.id == version_id,
                        WidgetConfigurationVersion.website_id == website_id
                    )
                )
            )
            
            version = version_result.scalar_one_or_none()
            if not version:
                return {
                    "success": False,
                    "error": "Configuration version not found"
                }
            
            # Create a version of current state before rollback
            current_config = self._build_configuration_from_website(website)
            await self._create_configuration_version(
                website, 
                current_config, 
                user_id, 
                db, 
                description=f"Before rollback to version {version.version_number}"
            )
            
            # Apply the version configuration
            self._apply_configuration_to_website(website, version.configuration_data)
            website.updated_at = datetime.utcnow()
            
            await db.commit()
            
            return {
                "success": True,
                "message": f"Successfully rolled back to version {version.version_number}",
                "configuration": self._build_configuration_from_website(website)
            }
            
        except Exception as e:
            await db.rollback()
            return {
                "success": False,
                "error": f"Failed to rollback configuration: {str(e)}"
            }
    
    def _build_configuration_from_website(self, website: Website) -> Dict[str, Any]:
        """Build configuration dictionary from website model."""
        return {
            "widget_color": website.widget_color,
            "widget_position": website.widget_position,
            "widget_size": website.widget_size,
            "widget_theme": website.widget_theme,
            "show_avatar": website.show_avatar,
            "enable_sound": website.enable_sound,
            "auto_open_delay": website.auto_open_delay,
            "show_online_status": website.show_online_status,
            "welcome_message": website.welcome_message,
            "placeholder_text": website.placeholder_text,
            "offline_message": website.offline_message,
            "thanks_message": website.thanks_message,
            "show_branding": website.show_branding,
            "custom_logo_url": website.custom_logo_url,
            "company_name": website.company_name,
            "support_email": website.support_email,
            "custom_css": website.custom_css,
            "font_family": website.font_family,
            "border_radius": website.border_radius
        }
    
    def _apply_configuration_to_website(self, website: Website, config: Dict[str, Any]) -> None:
        """Apply configuration dictionary to website model."""
        for key, value in config.items():
            if hasattr(website, key):
                setattr(website, key, value)
    
    async def _create_configuration_version(
        self,
        website: Website,
        config_data: Dict[str, Any],
        user_id: str,
        db: AsyncSession,
        description: Optional[str] = None
    ) -> WidgetConfigurationVersion:
        """Create a new configuration version."""
        # Get next version number
        latest_version_result = await db.execute(
            select(WidgetConfigurationVersion.version_number)
            .where(WidgetConfigurationVersion.website_id == website.id)
            .order_by(desc(WidgetConfigurationVersion.version_number))
            .limit(1)
        )
        
        latest_version = latest_version_result.scalar_one_or_none()
        next_version_number = (latest_version + 1) if latest_version else 1
        
        # Create version record
        version = WidgetConfigurationVersion(
            website_id=website.id,
            version_number=next_version_number,
            description=description,
            configuration_data=config_data,
            created_by=user_id
        )
        
        db.add(version)
        return version
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        import re
        pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return pattern.match(url) is not None