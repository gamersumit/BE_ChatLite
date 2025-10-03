"""
Cloudinary service for uploading website screenshots.
"""
import logging
import base64
from datetime import datetime
from typing import Optional
import cloudinary
import cloudinary.uploader
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class CloudinaryService:
    """Service for handling Cloudinary uploads"""

    def __init__(self):
        settings = get_settings()
        cloudinary.config(
            cloud_name=settings.cloudinary_cloud_name,
            api_key=settings.cloudinary_api_key,
            api_secret=settings.cloudinary_api_secret,
            secure=True
        )

    def upload_screenshot(self, screenshot_base64: str, website_domain: str, website_id: str) -> Optional[str]:
        """
        Upload screenshot to Cloudinary.

        Args:
            screenshot_base64: Base64 encoded screenshot image
            website_domain: Domain of the website (e.g., 'example.com')
            website_id: UUID of the website

        Returns:
            URL of uploaded screenshot or None if failed
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

            # Determine file extension from base64 data
            # Base64 format: data:image/png;base64,iVBORw0KG...
            if screenshot_base64.startswith('data:image/'):
                # Extract file extension
                file_extension = screenshot_base64.split(';')[0].split('/')[-1]
            else:
                # Default to png
                file_extension = 'png'

            # Construct public_id with folder structure: ChatLite/website_home/{domain}_{timestamp}
            public_id = f"ChatLite/website_home/{website_domain}_{timestamp}"

            # Upload to Cloudinary (don't use folder parameter since public_id already includes folder)
            result = cloudinary.uploader.upload(
                screenshot_base64,
                public_id=public_id,
                resource_type="image",
                format=file_extension,
                overwrite=True,
                invalidate=True,
                transformation=[
                    {'quality': 'auto:good'},
                    {'fetch_format': 'auto'}
                ]
            )

            screenshot_url = result.get('secure_url')
            logger.info(f"Screenshot uploaded successfully for {website_domain}: {screenshot_url}")

            return screenshot_url

        except Exception as e:
            logger.error(f"Failed to upload screenshot to Cloudinary for {website_domain}: {e}")
            return None

    def delete_screenshot(self, screenshot_url: str) -> bool:
        """
        Delete screenshot from Cloudinary.

        Args:
            screenshot_url: URL of the screenshot to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Extract public_id from URL
            # URL format: https://res.cloudinary.com/{cloud_name}/image/upload/{public_id}.{ext}
            if '/upload/' in screenshot_url:
                public_id = screenshot_url.split('/upload/')[-1]
                # Remove file extension
                public_id = public_id.rsplit('.', 1)[0]

                cloudinary.uploader.destroy(public_id, resource_type="image")
                logger.info(f"Screenshot deleted successfully: {public_id}")
                return True
            else:
                logger.warning(f"Invalid Cloudinary URL format: {screenshot_url}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete screenshot from Cloudinary: {e}")
            return False
