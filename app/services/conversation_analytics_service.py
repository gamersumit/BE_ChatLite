"""
Conversation Analytics Service for comprehensive session and conversation analysis.
Provides insights into user engagement, conversation quality, and system performance.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone, timedelta
from collections import Counter
import statistics

from ..core.config import get_settings
from ..core.database import get_supabase_admin
from .conversation_threading_service import ConversationThreadingService
from .message_importance_service import MessageImportanceService
from .redis_session_cache import RedisSessionCache

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationAnalyticsService:
    """
    Comprehensive analytics service for conversations and sessions.
    """
    
    def __init__(self):
        self.supabase = get_supabase_admin()
        self.threading_service = ConversationThreadingService()
        self.importance_service = MessageImportanceService()
        self.redis_cache = RedisSessionCache()
    
    async def generate_conversation_analytics(
        self,
        website_id: UUID,
        days: int = 7,
        include_detailed_metrics: bool = True,
        cache_results: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive conversation analytics for a website.
        
        Args:
            website_id: Website UUID
            days: Number of days to analyze
            include_detailed_metrics: Whether to include detailed breakdowns
            cache_results: Whether to cache results in Redis
            
        Returns:
            Comprehensive analytics data
        """
        try:
            # Check cache first
            cache_key = f"conv_analytics_{website_id}_{days}_{include_detailed_metrics}"
            if cache_results:
                cached_result = await self.redis_cache.get_cached_analytics(cache_key)
                if cached_result:
                    logger.info(f"📦 ANALYTICS: Using cached results for {website_id}")
                    return cached_result
            
            logger.info(f"📊 ANALYTICS: Generating analytics for website {website_id} ({days} days)")
            
            # Get base data
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get conversations and sessions
            conversations_data = await self._get_conversations_data(website_id, cutoff_date)
            messages_data = await self._get_messages_data(website_id, cutoff_date)
            
            # Calculate core metrics
            core_metrics = await self._calculate_core_metrics(
                conversations_data, messages_data, days
            )
            
            # Calculate engagement metrics
            engagement_metrics = await self._calculate_engagement_metrics(
                conversations_data, messages_data
            )
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                conversations_data, messages_data
            )
            
            # Get threading analytics
            threading_analytics = await self.threading_service.get_threading_analytics(
                website_id, days
            )
            
            # Build result
            result = {
                'website_id': str(website_id),
                'analysis_period': {
                    'days': days,
                    'start_date': cutoff_date,
                    'end_date': datetime.now(timezone.utc).isoformat(),
                    'generated_at': datetime.now(timezone.utc).isoformat()
                },
                'core_metrics': core_metrics,
                'engagement_metrics': engagement_metrics,
                'quality_metrics': quality_metrics,
                'threading_analytics': threading_analytics,
                'summary': await self._generate_analytics_summary(
                    core_metrics, engagement_metrics, quality_metrics
                )
            }
            
            # Add detailed metrics if requested
            if include_detailed_metrics:
                result['detailed_metrics'] = await self._calculate_detailed_metrics(
                    conversations_data, messages_data
                )
                
                result['trends'] = await self._calculate_trends(
                    website_id, days
                )
            
            # Cache results
            if cache_results:
                await self.redis_cache.cache_analytics(cache_key, result, ttl=1800)  # 30 minutes
            
            logger.info(f"✅ ANALYTICS: Generated analytics with {core_metrics['total_conversations']} conversations")
            return result
            
        except Exception as e:
            logger.error(f"❌ ANALYTICS: Error generating conversation analytics: {e}")
            return self._empty_analytics_result(website_id, days)
    
    async def _get_conversations_data(
        self,
        website_id: UUID,
        cutoff_date: str
    ) -> List[Dict[str, Any]]:
        """Get conversation data for analysis."""
        try:
            result = self.supabase.table('conversations').select(
                '''
                id, session_id, visitor_id, user_id, is_active, status,
                total_messages, user_messages, ai_messages,
                started_at, last_activity_at, ended_at,
                satisfaction_rating, feedback,
                website_context_used, context_pages_referenced,
                context_tokens_used, session_expires_at
                '''
            ).eq('website_id', str(website_id)).gte('started_at', cutoff_date).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting conversations data: {e}")
            return []
    
    async def _get_messages_data(
        self,
        website_id: UUID,
        cutoff_date: str
    ) -> List[Dict[str, Any]]:
        """Get message data for analysis."""
        try:
            # Get messages for conversations in the time period
            conversations_result = self.supabase.table('conversations').select('id').eq('website_id', str(website_id)).gte('started_at', cutoff_date).execute()
            
            if not conversations_result.data:
                return []
            
            conversation_ids = [c['id'] for c in conversations_result.data]
            
            # Get messages for these conversations
            messages_result = self.supabase.table('messages').select(
                '''
                id, conversation_id, content, message_type, sequence_number,
                word_count, character_count, tokens_used, context_importance,
                created_at
                '''
            ).in_('conversation_id', conversation_ids).execute()
            
            return messages_result.data if messages_result.data else []
            
        except Exception as e:
            logger.error(f"Error getting messages data: {e}")
            return []
    
    async def _calculate_core_metrics(
        self,
        conversations: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        days: int
    ) -> Dict[str, Any]:
        """Calculate core conversation metrics."""
        
        total_conversations = len(conversations)
        active_conversations = len([c for c in conversations if c.get('is_active', True)])
        completed_conversations = len([c for c in conversations if c.get('status') == 'completed'])
        
        # Visitor metrics
        unique_visitors = len(set(c.get('visitor_id') for c in conversations if c.get('visitor_id')))
        returning_visitors = len(conversations) - unique_visitors if unique_visitors > 0 else 0
        
        # Message metrics
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.get('message_type') == 'user'])
        ai_messages = len([m for m in messages if m.get('message_type') == 'assistant'])
        
        # Average metrics
        avg_messages_per_conversation = total_messages / total_conversations if total_conversations > 0 else 0
        avg_user_messages = user_messages / total_conversations if total_conversations > 0 else 0
        avg_ai_messages = ai_messages / total_conversations if total_conversations > 0 else 0
        
        # Duration metrics
        durations = []
        for conv in conversations:
            if conv.get('ended_at') and conv.get('started_at'):
                try:
                    start = datetime.fromisoformat(conv['started_at'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(conv['ended_at'].replace('Z', '+00:00'))
                    duration_minutes = (end - start).total_seconds() / 60
                    durations.append(duration_minutes)
                except:
                    continue
        
        avg_duration_minutes = statistics.mean(durations) if durations else 0
        median_duration_minutes = statistics.median(durations) if durations else 0
        
        return {
            'total_conversations': total_conversations,
            'active_conversations': active_conversations,
            'completed_conversations': completed_conversations,
            'completion_rate': (completed_conversations / total_conversations * 100) if total_conversations > 0 else 0,
            'unique_visitors': unique_visitors,
            'returning_visitors': returning_visitors,
            'return_visitor_rate': (returning_visitors / total_conversations * 100) if total_conversations > 0 else 0,
            'total_messages': total_messages,
            'user_messages': user_messages,
            'ai_messages': ai_messages,
            'avg_messages_per_conversation': round(avg_messages_per_conversation, 1),
            'avg_user_messages_per_conversation': round(avg_user_messages, 1),
            'avg_ai_messages_per_conversation': round(avg_ai_messages, 1),
            'avg_duration_minutes': round(avg_duration_minutes, 1),
            'median_duration_minutes': round(median_duration_minutes, 1),
            'conversations_per_day': round(total_conversations / days, 1) if days > 0 else 0
        }
    
    async def _calculate_engagement_metrics(
        self,
        conversations: List[Dict[str, Any]],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate user engagement metrics."""
        
        if not conversations:
            return self._empty_engagement_metrics()
        
        # Message length analysis
        message_lengths = [m.get('character_count', 0) for m in messages if m.get('character_count')]
        word_counts = [m.get('word_count', 0) for m in messages if m.get('word_count')]
        
        avg_message_length = statistics.mean(message_lengths) if message_lengths else 0
        avg_word_count = statistics.mean(word_counts) if word_counts else 0
        
        # Conversation depth analysis
        message_counts_per_conv = Counter()
        for msg in messages:
            conv_id = msg.get('conversation_id')
            if conv_id:
                message_counts_per_conv[conv_id] += 1
        
        conv_depths = list(message_counts_per_conv.values())
        avg_conversation_depth = statistics.mean(conv_depths) if conv_depths else 0
        
        # Deep conversation analysis (> 5 messages)
        deep_conversations = sum(1 for depth in conv_depths if depth > 5)
        deep_conversation_rate = (deep_conversations / len(conversations) * 100) if conversations else 0
        
        # Context usage analysis
        context_used_count = sum(1 for c in conversations if c.get('website_context_used', False))
        context_usage_rate = (context_used_count / len(conversations) * 100) if conversations else 0
        
        # Session engagement
        active_sessions = sum(1 for c in conversations if c.get('is_active', True))
        session_engagement_rate = (active_sessions / len(conversations) * 100) if conversations else 0
        
        # Response time analysis (simplified - would need more detailed timestamps)
        response_engagement = {
            'quick_responses': sum(1 for c in conversations if c.get('total_messages', 0) > 3),
            'sustained_conversations': sum(1 for c in conversations if c.get('total_messages', 0) > 8),
        }
        
        return {
            'avg_message_length_chars': round(avg_message_length, 1),
            'avg_message_word_count': round(avg_word_count, 1),
            'avg_conversation_depth': round(avg_conversation_depth, 1),
            'deep_conversations': deep_conversations,
            'deep_conversation_rate': round(deep_conversation_rate, 1),
            'context_usage_rate': round(context_usage_rate, 1),
            'session_engagement_rate': round(session_engagement_rate, 1),
            'engagement_distribution': {
                'shallow_conversations': sum(1 for depth in conv_depths if depth <= 2),
                'medium_conversations': sum(1 for depth in conv_depths if 3 <= depth <= 5),
                'deep_conversations': sum(1 for depth in conv_depths if depth > 5)
            },
            'message_type_distribution': {
                'user_message_percentage': (len([m for m in messages if m.get('message_type') == 'user']) / len(messages) * 100) if messages else 0,
                'ai_message_percentage': (len([m for m in messages if m.get('message_type') == 'assistant']) / len(messages) * 100) if messages else 0
            }
        }
    
    async def _calculate_quality_metrics(
        self,
        conversations: List[Dict[str, Any]],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate conversation quality metrics."""
        
        if not conversations:
            return self._empty_quality_metrics()
        
        # Satisfaction analysis
        rated_conversations = [c for c in conversations if c.get('satisfaction_rating')]
        avg_satisfaction = statistics.mean([c['satisfaction_rating'] for c in rated_conversations]) if rated_conversations else 0
        
        satisfaction_distribution = Counter([c['satisfaction_rating'] for c in rated_conversations])
        
        # Feedback analysis
        feedback_count = sum(1 for c in conversations if c.get('feedback'))
        feedback_rate = (feedback_count / len(conversations) * 100) if conversations else 0
        
        # Resolution analysis
        resolved_conversations = sum(1 for c in conversations if c.get('status') == 'completed')
        resolution_rate = (resolved_conversations / len(conversations) * 100) if conversations else 0
        
        # Message importance analysis
        if messages:
            importance_scores = []
            for msg in messages:
                if msg.get('context_importance'):
                    importance_scores.append(float(msg['context_importance']))
            
            avg_message_importance = statistics.mean(importance_scores) if importance_scores else 0
            high_importance_messages = sum(1 for score in importance_scores if score > 1.2)
        else:
            avg_message_importance = 0
            high_importance_messages = 0
        
        # Context effectiveness
        context_conversations = [c for c in conversations if c.get('website_context_used')]
        context_effectiveness = len(context_conversations) / len(conversations) if conversations else 0
        
        # Token efficiency (if available)
        token_data = [c.get('context_tokens_used', 0) for c in conversations if c.get('context_tokens_used')]
        avg_tokens_per_conversation = statistics.mean(token_data) if token_data else 0
        
        return {
            'avg_satisfaction_rating': round(avg_satisfaction, 2),
            'satisfaction_distribution': dict(satisfaction_distribution),
            'satisfaction_rate': (len(rated_conversations) / len(conversations) * 100) if conversations else 0,
            'feedback_rate': round(feedback_rate, 1),
            'resolution_rate': round(resolution_rate, 1),
            'avg_message_importance': round(avg_message_importance, 2),
            'high_importance_messages': high_importance_messages,
            'context_effectiveness': round(context_effectiveness * 100, 1),
            'avg_tokens_per_conversation': round(avg_tokens_per_conversation, 1),
            'quality_indicators': {
                'highly_rated_conversations': sum(1 for c in rated_conversations if c['satisfaction_rating'] >= 4),
                'poorly_rated_conversations': sum(1 for c in rated_conversations if c['satisfaction_rating'] <= 2),
                'completed_successfully': resolved_conversations,
                'abandoned_conversations': sum(1 for c in conversations if c.get('status') == 'abandoned')
            }
        }
    
    async def _calculate_detailed_metrics(
        self,
        conversations: List[Dict[str, Any]],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate detailed analytics metrics."""
        
        # Time-based analysis
        hourly_distribution = Counter()
        daily_distribution = Counter()
        
        for conv in conversations:
            try:
                start_time = datetime.fromisoformat(conv['started_at'].replace('Z', '+00:00'))
                hour = start_time.hour
                day = start_time.strftime('%A')
                hourly_distribution[hour] += 1
                daily_distribution[day] += 1
            except:
                continue
        
        # Visitor journey analysis
        visitor_sessions = Counter()
        for conv in conversations:
            visitor_id = conv.get('visitor_id')
            if visitor_id:
                visitor_sessions[visitor_id] += 1
        
        # Conversation flow analysis
        conversation_flows = {
            'single_message_conversations': sum(1 for c in conversations if c.get('total_messages', 0) == 1),
            'short_conversations': sum(1 for c in conversations if 2 <= c.get('total_messages', 0) <= 4),
            'medium_conversations': sum(1 for c in conversations if 5 <= c.get('total_messages', 0) <= 10),
            'long_conversations': sum(1 for c in conversations if c.get('total_messages', 0) > 10)
        }
        
        return {
            'time_distribution': {
                'hourly': dict(hourly_distribution),
                'daily': dict(daily_distribution)
            },
            'visitor_behavior': {
                'single_session_visitors': sum(1 for count in visitor_sessions.values() if count == 1),
                'returning_visitors': sum(1 for count in visitor_sessions.values() if count > 1),
                'max_sessions_per_visitor': max(visitor_sessions.values()) if visitor_sessions else 0,
                'avg_sessions_per_visitor': statistics.mean(visitor_sessions.values()) if visitor_sessions else 0
            },
            'conversation_flows': conversation_flows,
            'performance_metrics': {
                'total_processing_time': sum(c.get('context_tokens_used', 0) for c in conversations),
                'avg_response_generation_time': 0,  # Would need detailed timing data
                'context_optimization_effectiveness': len([c for c in conversations if c.get('context_tokens_used', 0) > 0])
            }
        }
    
    async def _calculate_trends(
        self,
        website_id: UUID,
        days: int
    ) -> Dict[str, Any]:
        """Calculate trend analysis."""
        try:
            # Compare with previous period
            previous_cutoff = (datetime.now(timezone.utc) - timedelta(days=days*2)).isoformat()
            current_cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get data for both periods
            previous_convs = self.supabase.table('conversations').select('id, started_at, total_messages').eq('website_id', str(website_id)).gte('started_at', previous_cutoff).lt('started_at', current_cutoff).execute()
            current_convs = self.supabase.table('conversations').select('id, started_at, total_messages').eq('website_id', str(website_id)).gte('started_at', current_cutoff).execute()
            
            prev_data = previous_convs.data if previous_convs.data else []
            curr_data = current_convs.data if current_convs.data else []
            
            # Calculate trends
            prev_count = len(prev_data)
            curr_count = len(curr_data)
            
            conv_growth = ((curr_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
            
            prev_avg_msgs = statistics.mean([c.get('total_messages', 0) for c in prev_data]) if prev_data else 0
            curr_avg_msgs = statistics.mean([c.get('total_messages', 0) for c in curr_data]) if curr_data else 0
            
            msg_growth = ((curr_avg_msgs - prev_avg_msgs) / prev_avg_msgs * 100) if prev_avg_msgs > 0 else 0
            
            return {
                'conversation_growth_percentage': round(conv_growth, 1),
                'message_engagement_growth': round(msg_growth, 1),
                'trend_direction': 'increasing' if conv_growth > 0 else 'decreasing' if conv_growth < 0 else 'stable',
                'period_comparison': {
                    'previous_period_conversations': prev_count,
                    'current_period_conversations': curr_count,
                    'previous_avg_messages': round(prev_avg_msgs, 1),
                    'current_avg_messages': round(curr_avg_msgs, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {'error': 'Could not calculate trends'}
    
    async def _generate_analytics_summary(
        self,
        core_metrics: Dict[str, Any],
        engagement_metrics: Dict[str, Any],
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of analytics."""
        
        # Determine overall health score
        health_factors = [
            min(100, core_metrics.get('completion_rate', 0)),
            min(100, engagement_metrics.get('deep_conversation_rate', 0) * 2),  # Weight deep conversations
            min(100, quality_metrics.get('avg_satisfaction_rating', 0) * 20),    # Convert 1-5 scale to percentage
            min(100, quality_metrics.get('resolution_rate', 0)),
            min(100, engagement_metrics.get('context_usage_rate', 0))
        ]
        
        overall_health = statistics.mean(health_factors)
        
        # Key insights
        insights = []
        
        if core_metrics.get('return_visitor_rate', 0) > 30:
            insights.append("High return visitor rate indicates strong user engagement")
        
        if engagement_metrics.get('deep_conversation_rate', 0) > 25:
            insights.append("Users are having meaningful, in-depth conversations")
        
        if quality_metrics.get('avg_satisfaction_rating', 0) > 4:
            insights.append("Excellent user satisfaction scores")
        
        if quality_metrics.get('context_effectiveness', 0) > 60:
            insights.append("Website content is being effectively used in conversations")
        
        # Recommendations
        recommendations = []
        
        if core_metrics.get('completion_rate', 0) < 70:
            recommendations.append("Focus on improving conversation completion rates")
        
        if engagement_metrics.get('deep_conversation_rate', 0) < 20:
            recommendations.append("Consider strategies to encourage longer conversations")
        
        if quality_metrics.get('satisfaction_rate', 0) < 50:
            recommendations.append("Implement satisfaction rating collection to gather more feedback")
        
        return {
            'overall_health_score': round(overall_health, 1),
            'health_grade': self._get_health_grade(overall_health),
            'key_insights': insights,
            'recommendations': recommendations,
            'top_metrics': {
                'total_conversations': core_metrics.get('total_conversations', 0),
                'avg_satisfaction': quality_metrics.get('avg_satisfaction_rating', 0),
                'completion_rate': core_metrics.get('completion_rate', 0),
                'deep_conversation_rate': engagement_metrics.get('deep_conversation_rate', 0)
            }
        }
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _empty_analytics_result(self, website_id: UUID, days: int) -> Dict[str, Any]:
        """Return empty analytics result."""
        return {
            'website_id': str(website_id),
            'analysis_period': {'days': days, 'error': True},
            'core_metrics': {},
            'engagement_metrics': {},
            'quality_metrics': {},
            'threading_analytics': {},
            'summary': {'overall_health_score': 0, 'health_grade': 'F', 'error': True}
        }
    
    def _empty_engagement_metrics(self) -> Dict[str, Any]:
        """Return empty engagement metrics."""
        return {
            'avg_message_length_chars': 0,
            'avg_message_word_count': 0,
            'avg_conversation_depth': 0,
            'deep_conversations': 0,
            'deep_conversation_rate': 0,
            'context_usage_rate': 0,
            'session_engagement_rate': 0
        }
    
    def _empty_quality_metrics(self) -> Dict[str, Any]:
        """Return empty quality metrics."""
        return {
            'avg_satisfaction_rating': 0,
            'satisfaction_distribution': {},
            'satisfaction_rate': 0,
            'feedback_rate': 0,
            'resolution_rate': 0,
            'avg_message_importance': 0,
            'high_importance_messages': 0,
            'context_effectiveness': 0
        }