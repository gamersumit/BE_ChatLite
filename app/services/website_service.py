from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models import Website


class WebsiteService:
    
    async def get_website_by_widget_id(
        self,
        widget_id: str,
        db: AsyncSession
    ) -> Optional[Website]:
        """
        Get website by widget ID.
        """
        result = await db.execute(
            select(Website).where(Website.widget_id == widget_id)
        )
        return result.scalar_one_or_none()
    
    async def log_widget_interaction(
        self,
        website_id,
        interaction_type: str,
        data: Dict[str, Any],
        db: AsyncSession
    ):
        """
        Log widget interactions for analytics.
        Simplified implementation for MVP.
        """
        # In production, you'd save this to an analytics/events table
        # For MVP, we'll just update website metrics
        
        result = await db.execute(
            select(Website).where(Website.id == website_id)
        )
        website = result.scalar_one_or_none()
        
        if website:
            # Update basic counters based on interaction type
            if interaction_type == "widget_init":
                # Could increment page_views counter
                pass
            elif interaction_type == "conversation_start":
                website.total_conversations += 1
            elif interaction_type == "message_sent":
                website.total_messages += 1
            
            # Note: Don't commit here - let the calling function handle the transaction
            # await db.commit()