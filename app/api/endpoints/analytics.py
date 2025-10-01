"""
Analytics API endpoints for session and conversation analytics.
Provides comprehensive analytics data for website owners and administrators.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import logging

from fastapi import APIRouter, HTTPException, Query, Depends

from ...core.database import get_supabase_client
from ...core.auth_middleware import get_current_user
from ...services.conversation_analytics_service import ConversationAnalyticsService
from ...services.conversation_threading_service import ConversationThreadingService
from ...services.session_service import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/websites/{website_id}/overview")
async def get_website_analytics_overview(
    website_id: UUID,
    current_user: dict = Depends(get_current_user),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze")
) -> Dict[str, Any]:
    """
    Get comprehensive analytics overview for a website.
    
    Args:
        website_id: UUID of the website
        days: Number of days to analyze (1-90)
        
    Returns:
        Comprehensive analytics overview including metrics, trends, and insights
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Generate comprehensive analytics
        analytics_data = await analytics_service.generate_conversation_analytics(
            website_id=website_id,
            days=days
        )
        
        if not analytics_data or analytics_data.get('error'):
            raise HTTPException(
                status_code=404,
                detail=f"No analytics data found for website {website_id}"
            )
        
        # Calculate additional overview metrics
        core_metrics = analytics_data.get('core_metrics', {})
        engagement_metrics = analytics_data.get('engagement_metrics', {})
        quality_metrics = analytics_data.get('quality_metrics', {})
        
        # Build overview response
        overview = {
            'website_id': str(website_id),
            'analysis_period': {
                'days': days,
                'start_date': analytics_data.get('period', {}).get('start_date'),
                'end_date': analytics_data.get('period', {}).get('end_date'),
                'analyzed_conversations': core_metrics.get('total_conversations', 0)
            },
            'key_metrics': {
                'total_conversations': core_metrics.get('total_conversations', 0),
                'total_messages': core_metrics.get('total_messages', 0),
                'unique_visitors': core_metrics.get('unique_visitors', 0),
                'avg_conversation_length': core_metrics.get('avg_conversation_length', 0),
                'avg_messages_per_conversation': core_metrics.get('avg_messages_per_conversation', 0),
                'total_conversation_duration_hours': round(core_metrics.get('total_conversation_duration_minutes', 0) / 60, 1)
            },
            'engagement_summary': {
                'engagement_score': engagement_metrics.get('overall_engagement_score', 0),
                'response_rate': engagement_metrics.get('user_response_rate', 0),
                'conversation_completion_rate': engagement_metrics.get('conversation_completion_rate', 0),
                'avg_response_time_seconds': engagement_metrics.get('avg_response_time_seconds', 0)
            },
            'quality_indicators': {
                'overall_quality_score': quality_metrics.get('overall_conversation_quality', 0),
                'coherence_score': quality_metrics.get('avg_conversation_coherence', 0),
                'satisfaction_score': quality_metrics.get('avg_satisfaction_score', 0),
                'issue_resolution_rate': quality_metrics.get('issue_resolution_indicators', {}).get('resolution_rate', 0)
            },
            'trends': analytics_data.get('trends', {}),
            'insights': analytics_data.get('insights', []),
            'generated_at': datetime.now().isoformat(),
            'processing_time_ms': analytics_data.get('processing_time_ms', 0)
        }
        
        logger.info(f"✅ ANALYTICS: Generated overview for website {website_id} ({days} days)")
        return overview
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating overview for website {website_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics overview: {str(e)}"
        )


@router.get("/websites/{website_id}/sessions")
async def get_session_analytics(
    website_id: UUID,
    days: int = Query(7, ge=1, le=90),
    include_details: bool = Query(False, description="Include detailed session information")
) -> Dict[str, Any]:
    """
    Get detailed session analytics for a website.
    
    Args:
        website_id: UUID of the website
        days: Number of days to analyze
        include_details: Whether to include detailed session information
        
    Returns:
        Detailed session analytics and statistics
    """
    try:
        analytics_service = ConversationAnalyticsService()
        session_service = SessionService()
        
        # Get session statistics
        session_stats = await session_service.get_session_statistics(
            website_id=website_id,
            days=days
        )
        
        # Get conversation analytics for session context
        conv_analytics = await analytics_service.generate_conversation_analytics(
            website_id=website_id,
            days=days
        )
        
        response_data = {
            'website_id': str(website_id),
            'analysis_period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'session_metrics': {
                'total_sessions': session_stats.get('total_sessions', 0),
                'active_sessions': session_stats.get('active_sessions', 0),
                'expired_sessions': session_stats.get('expired_sessions', 0),
                'avg_session_duration_minutes': session_stats.get('avg_session_duration_minutes', 0),
                'sessions_per_day': session_stats.get('sessions_per_day', 0),
                'session_continuation_rate': session_stats.get('continuation_rate', 0)
            },
            'conversation_session_mapping': {
                'conversations_with_sessions': conv_analytics.get('core_metrics', {}).get('conversations_with_sessions', 0),
                'multi_session_conversations': session_stats.get('multi_session_conversations', 0),
                'avg_conversations_per_session': session_stats.get('avg_conversations_per_session', 1)
            },
            'context_optimization': {
                'sessions_with_context_optimization': session_stats.get('optimized_sessions', 0),
                'avg_context_tokens_saved': session_stats.get('avg_tokens_saved', 0),
                'optimization_success_rate': session_stats.get('optimization_success_rate', 0)
            }
        }
        
        if include_details:
            response_data['detailed_sessions'] = session_stats.get('session_details', [])
        
        logger.info(f"✅ ANALYTICS: Generated session analytics for website {website_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating session analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate session analytics: {str(e)}"
        )


@router.get("/websites/{website_id}/threads")
async def get_thread_analytics(
    website_id: UUID,
    days: int = Query(7, ge=1, le=90),
    include_thread_details: bool = Query(False, description="Include detailed thread information")
) -> Dict[str, Any]:
    """
    Get conversation thread analytics for multi-session tracking.
    
    Args:
        website_id: UUID of the website
        days: Number of days to analyze
        include_thread_details: Whether to include detailed thread information
        
    Returns:
        Thread analytics showing conversation continuity across sessions
    """
    try:
        threading_service = ConversationThreadingService()
        
        # Get thread analytics
        thread_analytics = await threading_service.analyze_conversation_threads(
            website_id=website_id,
            days=days
        )
        
        response_data = {
            'website_id': str(website_id),
            'analysis_period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'thread_metrics': {
                'total_threads': thread_analytics.get('total_threads', 0),
                'active_threads': thread_analytics.get('active_threads', 0),
                'avg_thread_length': thread_analytics.get('avg_thread_length', 0),
                'avg_conversations_per_thread': thread_analytics.get('avg_conversations_per_thread', 1),
                'thread_continuation_rate': thread_analytics.get('continuation_rate', 0)
            },
            'topic_analysis': thread_analytics.get('topic_distribution', {}),
            'visitor_journey': {
                'multi_session_visitors': thread_analytics.get('multi_session_visitors', 0),
                'avg_sessions_per_visitor': thread_analytics.get('avg_sessions_per_visitor', 1),
                'visitor_return_rate': thread_analytics.get('return_rate', 0)
            },
            'thread_health': {
                'coherent_threads': thread_analytics.get('coherent_threads', 0),
                'broken_threads': thread_analytics.get('broken_threads', 0),
                'avg_coherence_score': thread_analytics.get('avg_coherence_score', 0)
            }
        }
        
        if include_thread_details:
            response_data['thread_details'] = thread_analytics.get('thread_summaries', [])
        
        logger.info(f"✅ ANALYTICS: Generated thread analytics for website {website_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating thread analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate thread analytics: {str(e)}"
        )


@router.get("/websites/{website_id}/performance")
async def get_performance_analytics(
    website_id: UUID,
    days: int = Query(7, ge=1, le=90)
) -> Dict[str, Any]:
    """
    Get performance analytics including response times, optimization metrics, and system health.
    
    Args:
        website_id: UUID of the website
        days: Number of days to analyze
        
    Returns:
        Performance analytics and system health metrics
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Get performance metrics
        performance_data = await analytics_service.analyze_performance_metrics(
            website_id=website_id,
            days=days
        )
        
        response_data = {
            'website_id': str(website_id),
            'analysis_period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'response_performance': {
                'avg_response_time_ms': performance_data.get('avg_response_time_ms', 0),
                'median_response_time_ms': performance_data.get('median_response_time_ms', 0),
                'p95_response_time_ms': performance_data.get('p95_response_time_ms', 0),
                'slow_responses_count': performance_data.get('slow_responses', 0),
                'response_time_trend': performance_data.get('response_time_trend', 'stable')
            },
            'context_optimization': {
                'avg_context_preparation_time_ms': performance_data.get('avg_context_prep_time_ms', 0),
                'context_optimization_success_rate': performance_data.get('optimization_success_rate', 0),
                'avg_tokens_optimized': performance_data.get('avg_tokens_optimized', 0),
                'compression_ratio': performance_data.get('avg_compression_ratio', 1.0)
            },
            'system_health': {
                'error_rate': performance_data.get('error_rate', 0),
                'timeout_rate': performance_data.get('timeout_rate', 0),
                'cache_hit_rate': performance_data.get('cache_hit_rate', 0),
                'system_status': performance_data.get('system_status', 'healthy')
            },
            'resource_usage': {
                'avg_memory_usage_mb': performance_data.get('avg_memory_usage_mb', 0),
                'peak_concurrent_conversations': performance_data.get('peak_concurrent_conversations', 0),
                'database_query_performance': performance_data.get('avg_db_query_time_ms', 0)
            }
        }
        
        logger.info(f"✅ ANALYTICS: Generated performance analytics for website {website_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating performance analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate performance analytics: {str(e)}"
        )


@router.get("/websites/{website_id}/insights")
async def get_analytics_insights(
    website_id: UUID,
    days: int = Query(7, ge=1, le=90),
    insight_types: Optional[List[str]] = Query(None, description="Types of insights to generate")
) -> Dict[str, Any]:
    """
    Get AI-generated insights and recommendations based on analytics data.
    
    Args:
        website_id: UUID of the website
        days: Number of days to analyze
        insight_types: Specific types of insights to generate
        
    Returns:
        AI-generated insights and actionable recommendations
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Generate insights
        insights_data = await analytics_service.generate_analytics_insights(
            website_id=website_id,
            days=days,
            insight_types=insight_types
        )
        
        response_data = {
            'website_id': str(website_id),
            'analysis_period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'insights': {
                'conversation_patterns': insights_data.get('conversation_patterns', []),
                'user_behavior_insights': insights_data.get('user_behavior', []),
                'performance_recommendations': insights_data.get('performance_recommendations', []),
                'content_optimization_suggestions': insights_data.get('content_suggestions', []),
                'engagement_improvement_tips': insights_data.get('engagement_tips', [])
            },
            'action_items': {
                'high_priority': insights_data.get('high_priority_actions', []),
                'medium_priority': insights_data.get('medium_priority_actions', []),
                'low_priority': insights_data.get('low_priority_actions', [])
            },
            'trends_analysis': {
                'positive_trends': insights_data.get('positive_trends', []),
                'concerning_trends': insights_data.get('concerning_trends', []),
                'neutral_trends': insights_data.get('neutral_trends', [])
            },
            'benchmarking': {
                'performance_vs_average': insights_data.get('performance_benchmark', {}),
                'engagement_vs_average': insights_data.get('engagement_benchmark', {}),
                'quality_vs_average': insights_data.get('quality_benchmark', {})
            },
            'generated_at': datetime.now().isoformat(),
            'insight_confidence': insights_data.get('confidence_score', 0.8)
        }
        
        logger.info(f"✅ ANALYTICS: Generated insights for website {website_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating insights: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics insights: {str(e)}"
        )


@router.get("/websites/{website_id}/export")
async def export_analytics_data(
    website_id: UUID,
    days: int = Query(30, ge=1, le=365),
    format_type: str = Query("json", regex="^(json|csv|excel)$", description="Export format"),
    include_raw_data: bool = Query(False, description="Include raw conversation data")
) -> Dict[str, Any]:
    """
    Export comprehensive analytics data in various formats.
    
    Args:
        website_id: UUID of the website
        days: Number of days to export (1-365)
        format_type: Export format (json, csv, excel)
        include_raw_data: Whether to include raw conversation data
        
    Returns:
        Export information and download link
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Generate export data
        export_data = await analytics_service.export_analytics_data(
            website_id=website_id,
            days=days,
            format_type=format_type,
            include_raw_data=include_raw_data
        )
        
        response_data = {
            'website_id': str(website_id),
            'export_info': {
                'format': format_type,
                'period_days': days,
                'generated_at': datetime.now().isoformat(),
                'file_size_bytes': export_data.get('file_size', 0),
                'record_count': export_data.get('record_count', 0),
                'includes_raw_data': include_raw_data
            },
            'download_info': {
                'download_url': export_data.get('download_url'),
                'expires_at': export_data.get('expires_at'),
                'file_name': export_data.get('file_name')
            },
            'data_summary': {
                'conversations_exported': export_data.get('conversations_count', 0),
                'messages_exported': export_data.get('messages_count', 0),
                'analytics_records': export_data.get('analytics_records', 0),
                'file_sections': export_data.get('sections', [])
            }
        }
        
        logger.info(f"✅ ANALYTICS: Generated export for website {website_id} ({format_type})")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating export: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics export: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/detailed")
async def get_conversation_detailed_analytics(
    conversation_id: UUID,
    include_message_analysis: bool = Query(True, description="Include detailed message analysis")
) -> Dict[str, Any]:
    """
    Get detailed analytics for a specific conversation.
    
    Args:
        conversation_id: UUID of the conversation
        include_message_analysis: Whether to include detailed message analysis
        
    Returns:
        Detailed conversation analytics including message-level insights
    """
    try:
        analytics_service = ConversationAnalyticsService()
        threading_service = ConversationThreadingService()
        
        # Get detailed conversation analysis
        conversation_analysis = await analytics_service.analyze_single_conversation(
            conversation_id=conversation_id,
            include_message_details=include_message_analysis
        )
        
        # Get thread information
        thread_info = await threading_service.get_conversation_thread_info(
            conversation_id=conversation_id
        )
        
        response_data = {
            'conversation_id': str(conversation_id),
            'basic_info': {
                'start_time': conversation_analysis.get('start_time'),
                'end_time': conversation_analysis.get('end_time'),
                'duration_minutes': conversation_analysis.get('duration_minutes', 0),
                'total_messages': conversation_analysis.get('total_messages', 0),
                'visitor_id': conversation_analysis.get('visitor_id'),
                'website_id': conversation_analysis.get('website_id')
            },
            'quality_metrics': {
                'coherence_score': conversation_analysis.get('coherence_score', 0),
                'engagement_score': conversation_analysis.get('engagement_score', 0),
                'satisfaction_score': conversation_analysis.get('satisfaction_score', 0),
                'resolution_score': conversation_analysis.get('resolution_score', 0)
            },
            'conversation_flow': {
                'topic_progression': conversation_analysis.get('topic_progression', []),
                'engagement_patterns': conversation_analysis.get('engagement_patterns', {}),
                'response_times': conversation_analysis.get('response_times', []),
                'conversation_phases': conversation_analysis.get('phases', [])
            },
            'context_optimization': {
                'context_tokens_used': conversation_analysis.get('context_tokens', 0),
                'optimization_applied': conversation_analysis.get('optimization_applied', False),
                'summarization_events': conversation_analysis.get('summarization_events', []),
                'pruning_events': conversation_analysis.get('pruning_events', [])
            },
            'thread_context': {
                'thread_id': thread_info.get('thread_id'),
                'thread_position': thread_info.get('position_in_thread', 1),
                'related_conversations': thread_info.get('related_conversations', []),
                'thread_coherence': thread_info.get('thread_coherence_score', 0)
            }
        }
        
        if include_message_analysis:
            response_data['message_analysis'] = conversation_analysis.get('message_details', [])
        
        logger.info(f"✅ ANALYTICS: Generated detailed analysis for conversation {conversation_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating conversation analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate conversation analysis: {str(e)}"
        )


@router.post("/websites/{website_id}/generate-report")
async def generate_analytics_report(
    website_id: UUID,
    report_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a comprehensive analytics report with custom configuration.
    
    Args:
        website_id: UUID of the website
        report_config: Report configuration including timeframe, metrics, and format
        
    Returns:
        Generated report information and access details
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Validate report configuration
        days = report_config.get('days', 30)
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        metrics_to_include = report_config.get('metrics', ['overview', 'sessions', 'threads', 'performance'])
        report_format = report_config.get('format', 'comprehensive')
        include_charts = report_config.get('include_charts', True)
        include_insights = report_config.get('include_insights', True)
        
        # Generate comprehensive report
        report_data = await analytics_service.generate_comprehensive_report(
            website_id=website_id,
            days=days,
            metrics=metrics_to_include,
            format=report_format,
            include_charts=include_charts,
            include_insights=include_insights
        )
        
        response_data = {
            'website_id': str(website_id),
            'report_info': {
                'report_id': report_data.get('report_id'),
                'generated_at': datetime.now().isoformat(),
                'report_type': report_format,
                'analysis_period': {
                    'days': days,
                    'start_date': report_data.get('start_date'),
                    'end_date': report_data.get('end_date')
                },
                'metrics_included': metrics_to_include,
                'includes_charts': include_charts,
                'includes_insights': include_insights
            },
            'report_summary': {
                'total_pages': report_data.get('total_pages', 0),
                'key_findings': report_data.get('key_findings', []),
                'data_quality_score': report_data.get('data_quality', 0.9),
                'completeness_percentage': report_data.get('completeness', 100)
            },
            'access_info': {
                'download_url': report_data.get('download_url'),
                'view_url': report_data.get('view_url'),
                'expires_at': report_data.get('expires_at'),
                'file_size_mb': report_data.get('file_size_mb', 0)
            },
            'processing_info': {
                'processing_time_seconds': report_data.get('processing_time', 0),
                'data_points_analyzed': report_data.get('data_points', 0),
                'report_status': 'completed'
            }
        }
        
        logger.info(f"✅ ANALYTICS: Generated comprehensive report for website {website_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error generating report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics report: {str(e)}"
        )


@router.get("/system/health")
async def get_analytics_system_health() -> Dict[str, Any]:
    """
    Get analytics system health and performance metrics.
    
    Returns:
        System health status and performance indicators
    """
    try:
        analytics_service = ConversationAnalyticsService()
        
        # Get system health metrics
        health_data = await analytics_service.get_system_health_metrics()
        
        response_data = {
            'system_status': 'healthy',  # healthy, degraded, critical
            'services_status': {
                'analytics_service': health_data.get('analytics_service_status', 'healthy'),
                'database_connection': health_data.get('database_status', 'healthy'),
                'cache_service': health_data.get('cache_status', 'healthy'),
                'threading_service': health_data.get('threading_status', 'healthy')
            },
            'performance_metrics': {
                'avg_query_time_ms': health_data.get('avg_query_time', 0),
                'cache_hit_rate': health_data.get('cache_hit_rate', 0),
                'concurrent_analytics_requests': health_data.get('concurrent_requests', 0),
                'memory_usage_mb': health_data.get('memory_usage', 0)
            },
            'data_integrity': {
                'conversation_data_consistency': health_data.get('conversation_consistency', 100),
                'analytics_data_freshness': health_data.get('data_freshness', 100),
                'thread_link_integrity': health_data.get('thread_integrity', 100)
            },
            'recent_activity': {
                'analytics_requests_last_hour': health_data.get('requests_last_hour', 0),
                'reports_generated_today': health_data.get('reports_today', 0),
                'errors_last_24h': health_data.get('errors_24h', 0)
            },
            'checked_at': datetime.now().isoformat()
        }
        
        # Determine overall system status
        if health_data.get('critical_errors', 0) > 0:
            response_data['system_status'] = 'critical'
        elif health_data.get('warnings', 0) > 5:
            response_data['system_status'] = 'degraded'
        
        logger.info("✅ ANALYTICS: System health check completed")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ ANALYTICS: Error checking system health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check analytics system health: {str(e)}"
        )