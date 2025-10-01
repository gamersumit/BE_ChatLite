"""
Dashboard Service for aggregating website statistics and analytics.
Handles data aggregation, metrics calculation, and analytics insights.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text, and_, or_, desc, asc
from sqlalchemy.orm import selectinload
from collections import defaultdict
import json

from ..models.website import Website
from ..models.conversation import Conversation
from ..models.message import Message
from ..models.user_website import UserWebsite
from ..models.scraper import ScrapedWebsite, ScrapedPage, ScrapedContentChunk


class DashboardService:
    """Service for dashboard data aggregation and analytics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_user_websites_summary(
        self, 
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        status_filter: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """Get summary of websites owned by a user with statistics."""
        
        # Build base query
        query = (
            select(Website)
            .join(UserWebsite, Website.id == UserWebsite.website_id)
            .where(UserWebsite.user_id == user_id)
            .options(selectinload(Website.conversations))
        )
        
        # Apply status filter
        if status_filter:
            query = query.where(Website.verification_status == status_filter)
        
        # Apply sorting
        sort_column = getattr(Website, sort_by, Website.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Get total count
        count_query = (
            select(func.count(Website.id))
            .join(UserWebsite, Website.id == UserWebsite.website_id)
            .where(UserWebsite.user_id == user_id)
        )
        if status_filter:
            count_query = count_query.where(Website.verification_status == status_filter)
        
        total_count_result = await self.session.execute(count_query)
        total_count = total_count_result.scalar() or 0
        
        # Apply pagination
        paginated_query = query.limit(limit).offset(offset)
        result = await self.session.execute(paginated_query)
        websites = result.scalars().all()
        
        # Calculate statistics for each website
        website_summaries = []
        summary_stats = {
            "total_websites": total_count,
            "active_websites": 0,
            "verified_websites": 0,
            "total_conversations": 0,
            "total_messages": 0,
            "avg_response_time": 0.0
        }
        
        for website in websites:
            # Get conversation stats
            conv_stats = await self._get_website_conversation_stats(str(website.id))
            
            website_summary = {
                "id": str(website.id),
                "name": website.name,
                "url": website.url,
                "domain": website.domain,
                "widget_id": website.widget_id,
                "is_active": website.is_active,
                "verification_status": website.verification_status,
                "widget_status": website.widget_status,
                "scraping_status": website.scraping_status,
                "created_at": website.created_at.isoformat() if website.created_at else None,
                "last_crawled": website.last_crawled.isoformat() if website.last_crawled else None,
                "stats": conv_stats
            }
            
            website_summaries.append(website_summary)
            
            # Update summary stats
            if website.is_active:
                summary_stats["active_websites"] += 1
            if website.verification_status == "verified":
                summary_stats["verified_websites"] += 1
                
            summary_stats["total_conversations"] += conv_stats["total_conversations"]
            summary_stats["total_messages"] += conv_stats["total_messages"]
        
        # Calculate average response time across all websites
        if summary_stats["total_conversations"] > 0:
            response_times = []
            for website in websites:
                avg_time = await self._get_average_response_time(str(website.id))
                if avg_time:
                    response_times.append(avg_time)
            
            if response_times:
                summary_stats["avg_response_time"] = sum(response_times) / len(response_times)
        
        return {
            "websites": website_summaries,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "summary": summary_stats
        }
    
    async def get_website_analytics(
        self,
        website_id: str,
        user_id: str,
        time_range: str = "7d",
        include_trends: bool = False,
        include_visitors: bool = False,
        include_questions: bool = False
    ) -> Dict[str, Any]:
        """Get detailed analytics for a specific website."""
        
        # Verify user has access to website
        access_query = (
            select(Website)
            .join(UserWebsite, Website.id == UserWebsite.website_id)
            .where(
                and_(
                    Website.id == website_id,
                    UserWebsite.user_id == user_id
                )
            )
        )
        result = await self.session.execute(access_query)
        website = result.scalar_one_or_none()
        
        if not website:
            return None
        
        # Parse time range
        start_date, end_date = self._parse_time_range(time_range)
        
        # Get core metrics
        metrics = await self._get_website_metrics(website_id, start_date, end_date)
        
        analytics = {
            "website_id": website_id,
            "website_name": website.name,
            "time_range": time_range,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": metrics
        }
        
        # Add trends if requested
        if include_trends:
            analytics["trends"] = await self._get_website_trends(website_id, start_date, end_date)
        
        # Add visitor analytics if requested
        if include_visitors:
            analytics["visitor_analytics"] = await self._get_visitor_analytics(website_id, start_date, end_date)
        
        # Add popular questions analysis if requested
        if include_questions:
            analytics["popular_questions"] = await self._get_popular_questions(website_id, start_date, end_date)
        
        return analytics
    
    async def _get_website_conversation_stats(self, website_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a website."""
        
        # Get conversation counts
        conv_count_query = (
            select(func.count(Conversation.id))
            .where(Conversation.website_id == website_id)
        )
        conv_count_result = await self.session.execute(conv_count_query)
        total_conversations = conv_count_result.scalar() or 0
        
        # Get message counts
        msg_count_query = (
            select(func.sum(Conversation.total_messages))
            .where(Conversation.website_id == website_id)
        )
        msg_count_result = await self.session.execute(msg_count_query)
        total_messages = msg_count_result.scalar() or 0
        
        # Get unique visitors
        visitor_count_query = (
            select(func.count(func.distinct(Conversation.visitor_id)))
            .where(Conversation.website_id == website_id)
        )
        visitor_count_result = await self.session.execute(visitor_count_query)
        unique_visitors = visitor_count_result.scalar() or 0
        
        # Get active conversations (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        active_conv_query = (
            select(func.count(Conversation.id))
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.last_activity_at >= yesterday,
                    Conversation.is_active == True
                )
            )
        )
        active_conv_result = await self.session.execute(active_conv_query)
        active_conversations = active_conv_result.scalar() or 0
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "unique_visitors": unique_visitors,
            "active_conversations": active_conversations
        }
    
    async def _get_website_metrics(self, website_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get detailed metrics for a website within a time range."""
        
        # Conversations in time range
        conv_query = (
            select(
                func.count(Conversation.id).label("total_conversations"),
                func.count(func.distinct(Conversation.visitor_id)).label("unique_visitors"),
                func.count(func.distinct(Conversation.session_id)).label("unique_sessions"),
                func.avg(Conversation.total_messages).label("avg_conversation_length"),
                func.sum(Conversation.total_messages).label("total_messages"),
                func.sum(Conversation.user_messages).label("user_messages"),
                func.sum(Conversation.ai_messages).label("ai_messages")
            )
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date
                )
            )
        )
        
        conv_result = await self.session.execute(conv_query)
        conv_data = conv_result.first()
        
        # Satisfaction metrics
        satisfaction_query = (
            select(
                func.avg(Conversation.satisfaction_rating).label("avg_satisfaction"),
                func.count().filter(Conversation.satisfaction_rating.isnot(None)).label("total_ratings")
            )
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date
                )
            )
        )
        
        satisfaction_result = await self.session.execute(satisfaction_query)
        satisfaction_data = satisfaction_result.first()
        
        # Lead capture metrics
        lead_query = (
            select(
                func.count().filter(Conversation.lead_captured == True).label("leads_captured"),
                func.count(Conversation.id).label("total_conversations_for_rate")
            )
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date
                )
            )
        )
        
        lead_result = await self.session.execute(lead_query)
        lead_data = lead_result.first()
        
        # Calculate response rate (conversations with AI responses)
        response_rate = 0.0
        if conv_data.total_conversations > 0 and conv_data.ai_messages > 0:
            conversations_with_ai = await self.session.execute(
                select(func.count(Conversation.id))
                .where(
                    and_(
                        Conversation.website_id == website_id,
                        Conversation.started_at >= start_date,
                        Conversation.started_at <= end_date,
                        Conversation.ai_messages > 0
                    )
                )
            )
            ai_conv_count = conversations_with_ai.scalar() or 0
            response_rate = (ai_conv_count / conv_data.total_conversations) * 100
        
        # Calculate lead conversion rate
        lead_conversion_rate = 0.0
        if conv_data.total_conversations > 0:
            lead_conversion_rate = (lead_data.leads_captured / conv_data.total_conversations) * 100
        
        return {
            "total_conversations": conv_data.total_conversations or 0,
            "total_messages": conv_data.total_messages or 0,
            "user_messages": conv_data.user_messages or 0,
            "ai_messages": conv_data.ai_messages or 0,
            "unique_visitors": conv_data.unique_visitors or 0,
            "unique_sessions": conv_data.unique_sessions or 0,
            "avg_conversation_length": float(conv_data.avg_conversation_length or 0),
            "response_rate": response_rate,
            "satisfaction_score": float(satisfaction_data.avg_satisfaction or 0),
            "total_satisfaction_ratings": satisfaction_data.total_ratings or 0,
            "leads_captured": lead_data.leads_captured or 0,
            "lead_conversion_rate": lead_conversion_rate
        }
    
    async def _get_website_trends(self, website_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get trend data for charts and graphs."""
        
        # Daily conversations trend
        daily_conv_query = text("""
            SELECT 
                DATE(started_at) as date,
                COUNT(*) as conversations,
                COUNT(DISTINCT visitor_id) as unique_visitors,
                SUM(total_messages) as messages
            FROM conversations 
            WHERE website_id = :website_id 
                AND started_at >= :start_date 
                AND started_at <= :end_date
            GROUP BY DATE(started_at)
            ORDER BY date
        """)
        
        daily_conv_result = await self.session.execute(
            daily_conv_query,
            {"website_id": website_id, "start_date": start_date, "end_date": end_date}
        )
        
        daily_conversations = []
        daily_messages = []
        daily_visitors = []
        
        for row in daily_conv_result:
            date_str = row.date.isoformat() if row.date else None
            daily_conversations.append({"date": date_str, "count": row.conversations})
            daily_messages.append({"date": date_str, "count": row.messages or 0})
            daily_visitors.append({"date": date_str, "count": row.unique_visitors})
        
        # Hourly activity pattern
        hourly_activity_query = text("""
            SELECT 
                EXTRACT(hour FROM started_at) as hour,
                COUNT(*) as conversations
            FROM conversations 
            WHERE website_id = :website_id 
                AND started_at >= :start_date 
                AND started_at <= :end_date
            GROUP BY EXTRACT(hour FROM started_at)
            ORDER BY hour
        """)
        
        hourly_result = await self.session.execute(
            hourly_activity_query,
            {"website_id": website_id, "start_date": start_date, "end_date": end_date}
        )
        
        hourly_activity = []
        for row in hourly_result:
            hourly_activity.append({"hour": int(row.hour), "conversations": row.conversations})
        
        return {
            "daily_conversations": daily_conversations,
            "daily_messages": daily_messages,
            "daily_visitors": daily_visitors,
            "hourly_activity": hourly_activity
        }
    
    async def _get_visitor_analytics(self, website_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get visitor behavior analytics."""
        
        # Visitor metrics
        visitor_query = text("""
            SELECT 
                COUNT(DISTINCT visitor_id) as unique_visitors,
                COUNT(DISTINCT CASE WHEN conversation_count > 1 THEN visitor_id END) as returning_visitors,
                AVG(session_duration) as avg_session_duration,
                COUNT(CASE WHEN total_messages = 0 THEN 1 END)::float / COUNT(*)::float * 100 as bounce_rate
            FROM (
                SELECT 
                    visitor_id,
                    COUNT(*) as conversation_count,
                    AVG(EXTRACT(epoch FROM (COALESCE(ended_at, NOW()) - started_at))/60) as session_duration,
                    SUM(total_messages) as total_messages
                FROM conversations 
                WHERE website_id = :website_id 
                    AND started_at >= :start_date 
                    AND started_at <= :end_date
                GROUP BY visitor_id
            ) visitor_stats
        """)
        
        visitor_result = await self.session.execute(
            visitor_query,
            {"website_id": website_id, "start_date": start_date, "end_date": end_date}
        )
        
        visitor_data = visitor_result.first()
        
        # Top pages
        top_pages_query = (
            select(
                Conversation.page_url,
                func.count(Conversation.id).label("visits"),
                func.count(func.distinct(Conversation.visitor_id)).label("unique_visitors")
            )
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date,
                    Conversation.page_url.isnot(None)
                )
            )
            .group_by(Conversation.page_url)
            .order_by(desc(func.count(Conversation.id)))
            .limit(10)
        )
        
        top_pages_result = await self.session.execute(top_pages_query)
        
        top_pages = []
        for row in top_pages_result:
            top_pages.append({
                "page_url": row.page_url,
                "visits": row.visits,
                "unique_visitors": row.unique_visitors
            })
        
        # Traffic sources (referrers)
        traffic_sources_query = (
            select(
                Conversation.referrer,
                func.count(Conversation.id).label("visits")
            )
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date,
                    Conversation.referrer.isnot(None)
                )
            )
            .group_by(Conversation.referrer)
            .order_by(desc(func.count(Conversation.id)))
            .limit(10)
        )
        
        traffic_result = await self.session.execute(traffic_sources_query)
        
        traffic_sources = []
        for row in traffic_result:
            # Extract domain from referrer
            referrer_domain = "Direct"
            if row.referrer:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(row.referrer)
                    referrer_domain = parsed.netloc or "Direct"
                except:
                    referrer_domain = "Unknown"
            
            traffic_sources.append({
                "source": referrer_domain,
                "visits": row.visits
            })
        
        return {
            "unique_visitors": visitor_data.unique_visitors or 0,
            "returning_visitors": visitor_data.returning_visitors or 0,
            "new_visitors": (visitor_data.unique_visitors or 0) - (visitor_data.returning_visitors or 0),
            "avg_session_duration": float(visitor_data.avg_session_duration or 0),
            "bounce_rate": float(visitor_data.bounce_rate or 0),
            "top_pages": top_pages,
            "traffic_sources": traffic_sources
        }
    
    async def _get_popular_questions(self, website_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze popular questions and topics from conversations."""
        
        # Get user messages from conversations
        messages_query = (
            select(Message.content)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date,
                    Message.message_type == "user",
                    Message.content.isnot(None)
                )
            )
        )
        
        messages_result = await self.session.execute(messages_query)
        user_messages = [row.content for row in messages_result if row.content]
        
        # Basic keyword analysis
        question_keywords = defaultdict(int)
        question_patterns = []
        
        for message in user_messages:
            message_lower = message.lower()
            
            # Look for question patterns
            if any(word in message_lower for word in ['how', 'what', 'when', 'where', 'why', 'can', 'do']):
                question_patterns.append(message)
            
            # Extract keywords (simple word frequency)
            words = message_lower.split()
            for word in words:
                if len(word) > 3 and word.isalpha():  # Filter out short words and non-alphabetic
                    question_keywords[word] += 1
        
        # Get top keywords
        top_keywords = sorted(question_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Categorize questions (simple keyword-based categorization)
        categories = {
            "pricing": ["price", "cost", "expensive", "cheap", "money", "payment"],
            "support": ["help", "support", "problem", "issue", "error"],
            "features": ["feature", "function", "capability", "does", "can"],
            "technical": ["technical", "setup", "install", "configure", "api"],
            "general": ["about", "company", "business", "service"]
        }
        
        question_categories = defaultdict(int)
        for message in user_messages:
            message_lower = message.lower()
            categorized = False
            for category, keywords in categories.items():
                if any(keyword in message_lower for keyword in keywords):
                    question_categories[category] += 1
                    categorized = True
                    break
            if not categorized:
                question_categories["other"] += 1
        
        # Find unanswered questions (conversations with user messages but no AI responses)
        unanswered_query = (
            select(func.count(Conversation.id))
            .where(
                and_(
                    Conversation.website_id == website_id,
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date,
                    Conversation.user_messages > 0,
                    Conversation.ai_messages == 0
                )
            )
        )
        
        unanswered_result = await self.session.execute(unanswered_query)
        unanswered_count = unanswered_result.scalar() or 0
        
        return {
            "top_questions": question_patterns[:10],  # Top 10 question-like messages
            "trending_topics": [{"topic": word, "frequency": count} for word, count in top_keywords[:10]],
            "question_categories": dict(question_categories),
            "unanswered_questions": unanswered_count,
            "total_analyzed_messages": len(user_messages)
        }
    
    async def _get_average_response_time(self, website_id: str) -> Optional[float]:
        """Calculate average response time for a website."""
        
        # This would require analysis of message timestamps
        # For now, return a placeholder
        response_time_query = text("""
            SELECT AVG(
                EXTRACT(epoch FROM (ai_msg.created_at - user_msg.created_at))
            ) as avg_response_time
            FROM messages user_msg
            JOIN conversations c ON user_msg.conversation_id = c.id
            JOIN messages ai_msg ON (
                ai_msg.conversation_id = c.id 
                AND ai_msg.created_at > user_msg.created_at
                AND ai_msg.message_type = 'assistant'
            )
            WHERE c.website_id = :website_id
                AND user_msg.message_type = 'user'
                AND ai_msg.id = (
                    SELECT MIN(id) FROM messages 
                    WHERE conversation_id = c.id 
                        AND created_at > user_msg.created_at 
                        AND message_type = 'assistant'
                )
        """)
        
        try:
            result = await self.session.execute(response_time_query, {"website_id": website_id})
            avg_time = result.scalar()
            return float(avg_time) if avg_time else None
        except:
            return None
    
    def _parse_time_range(self, time_range: str) -> Tuple[datetime, datetime]:
        """Parse time range string into start and end dates."""
        
        end_date = datetime.utcnow()
        
        if time_range == "1d":
            start_date = end_date - timedelta(days=1)
        elif time_range == "7d":
            start_date = end_date - timedelta(days=7)
        elif time_range == "30d":
            start_date = end_date - timedelta(days=30)
        elif time_range == "90d":
            start_date = end_date - timedelta(days=90)
        elif time_range == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            # Default to 7 days
            start_date = end_date - timedelta(days=7)
        
        return start_date, end_date
    
    async def export_analytics_data(
        self,
        website_id: str,
        user_id: str,
        format_type: str = "json",
        time_range: str = "30d"
    ) -> Dict[str, Any]:
        """Export analytics data in specified format."""
        
        # Get analytics data
        analytics = await self.get_website_analytics(
            website_id=website_id,
            user_id=user_id,
            time_range=time_range,
            include_trends=True,
            include_visitors=True,
            include_questions=True
        )
        
        if not analytics:
            return None
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "format": format_type,
            "website_id": website_id,
            "time_range": time_range,
            "analytics_data": analytics,
            "summary": {
                "total_conversations": analytics["metrics"]["total_conversations"],
                "total_messages": analytics["metrics"]["total_messages"],
                "unique_visitors": analytics["metrics"]["unique_visitors"],
                "satisfaction_score": analytics["metrics"]["satisfaction_score"]
            }
        }
        
        return export_data