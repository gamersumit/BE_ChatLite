"""
Dashboard API endpoints - Basic implementation.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from pydantic import BaseModel

from app.core.database import get_supabase_client, get_supabase_admin_client
from app.core.auth_middleware import get_current_user
from app.core.config import settings
from app.services.registration_scheduler import get_registration_scheduler

router = APIRouter()


class DashboardResponse(BaseModel):
    """Basic dashboard response."""
    status: str = "success"
    data: Dict[str, Any] = {}


# Overview API removed - metrics now included in websites API


@router.get("/websites", response_model=DashboardResponse)
async def get_websites(
    page: int = 1,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin_client)
):
    """Get paginated websites with comprehensive metrics - replaces overview API."""
    try:
        user_id = current_user['id']

        # Query 1: Get all user websites for aggregation metrics
        all_websites_result = supabase.table('websites').select('id, user_id, is_active, created_at').eq('user_id', user_id).execute()
        user_websites_all = all_websites_result.data or []

        # Calculate aggregation metrics
        total_websites_count = len(user_websites_all)
        active_websites_count = len([w for w in user_websites_all if w.get('is_active', True)])

        # Get website IDs for conversation count
        website_ids = [w['id'] for w in user_websites_all]
        total_conversations_count = 0
        active_crawls_count = 0  # TODO: Calculate from crawl status if needed

        # Calculate total pages crawled from latest crawl jobs for each website
        total_pages_crawled = 0
        website_crawl_pages = {}  # Store pages for each website

        if website_ids:
            # Count total conversations across all user websites
            conversations_result = supabase.table('conversations').select('id', count='exact').in_('website_id', website_ids).execute()
            total_conversations_count = conversations_result.count or 0

            # Get the latest successful crawl job for each website to calculate pages and last crawled time
            website_last_crawled = {}  # Store last crawled time for each website
            for website_id in website_ids:
                crawl_jobs_result = supabase.table('crawling_jobs')\
                    .select('crawl_metrics, completed_at')\
                    .eq('website_id', website_id)\
                    .eq('status', 'completed')\
                    .order('completed_at', desc=True)\
                    .limit(1)\
                    .execute()

                if crawl_jobs_result.data:
                    crawl_data = crawl_jobs_result.data[0]
                    crawl_metrics = crawl_data.get('crawl_metrics', {})
                    pages_crawled = crawl_metrics.get('pages_crawled', 0)
                    completed_at = crawl_data.get('completed_at')

                    website_crawl_pages[website_id] = pages_crawled
                    website_last_crawled[website_id] = completed_at
                    total_pages_crawled += pages_crawled
                else:
                    website_crawl_pages[website_id] = 0
                    website_last_crawled[website_id] = None

        # Query 2: Get paginated websites for list display
        offset = (page - 1) * limit
        paginated_websites = user_websites_all[offset:offset + limit]

        # Get detailed data for paginated websites
        websites = []
        paginated_chats_count = 0

        for site in paginated_websites:
            # Get conversation count for this specific website
            conv_result = supabase.table('conversations').select('id', count='exact').eq('website_id', site['id']).execute()
            monthly_chats = conv_result.count or 0
            paginated_chats_count += monthly_chats

            # Get full website data for this paginated item
            full_site_result = supabase.table('websites').select('*').eq('id', site['id']).execute()
            full_site = full_site_result.data[0] if full_site_result.data else site

            websites.append({
                "id": full_site['id'],
                "name": full_site.get('name', full_site.get('domain', 'Unknown')),
                "domain": full_site.get('domain', ''),
                "url": full_site.get('url', f"https://{full_site.get('domain', '')}"),
                "status": "active" if full_site.get('is_active', True) else "inactive",
                "createdAt": full_site.get('created_at', ''),
                "lastCrawled": website_last_crawled.get(site['id']),  # Only show actual crawl dates, not fallback dates
                "totalPages": website_crawl_pages.get(site['id'], 0),
                "monthlyChats": monthly_chats,
                # "responseRate": 92.5,  # Removed hardcoded value - will calculate from actual data later
                "widgetId": full_site.get('widget_id')
            })

        # Calculate total pages for pagination
        total_pages = (total_websites_count + limit - 1) // limit if total_websites_count > 0 else 0

        return DashboardResponse(
            status="success",
            data={
                # Paginated website list
                "websites": websites,

                # Clear pagination info
                "pagination": {
                    "current_page": page,
                    "per_page": limit,
                    "total_items": total_websites_count,
                    "total_pages": total_pages,
                    "has_next_page": page < total_pages,
                    "has_prev_page": page > 1
                },

                # Overall metrics (aggregated from all user websites)
                "metrics": {
                    "total_websites": total_websites_count,
                    "active_websites": active_websites_count,
                    "inactive_websites": total_websites_count - active_websites_count,
                    "total_conversations": total_conversations_count,
                    "total_pages_crawled": total_pages_crawled,
                    "active_crawls": active_crawls_count,
                    "websites_created_this_month": len([w for w in user_websites_all if w.get('created_at', '').startswith('2025-09')])  # Simple month check
                }
            }
        )
    except Exception as e:
        print(f"Websites endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return DashboardResponse(
            status="error",
            data={
                "websites": [],
                "pagination": {
                    "current_page": 1,
                    "per_page": limit,
                    "total_items": 0,
                    "total_pages": 0,
                    "has_next_page": False,
                    "has_prev_page": False
                },
                "metrics": {
                    "total_websites": 0,
                    "active_websites": 0,
                    "inactive_websites": 0,
                    "total_conversations": 0,
                    "total_pages_crawled": 0,
                    "active_crawls": 0,
                    "websites_created_this_month": 0
                },
                "error": str(e)
            }
        )


@router.get("/chat-stats", response_model=DashboardResponse)
async def get_chat_stats(
    period: str = "30d",
    supabase: Client = Depends(get_supabase_client)
):
    """Get chat statistics for the specified period."""
    try:
        # For now, generate some basic stats based on actual conversation data
        # Get conversations from the last 30 days
        from datetime import datetime, timedelta
        
        thirty_days_ago = datetime.now() - timedelta(days=30)
        conversations_result = supabase.table('conversations')\
            .select('created_at')\
            .gte('created_at', thirty_days_ago.isoformat())\
            .execute()
        
        conversations = conversations_result.data or []
        
        # Group by date
        from collections import defaultdict
        stats_by_date = defaultdict(int)
        
        for conv in conversations:
            date_str = conv['created_at'][:10]  # Get YYYY-MM-DD part
            stats_by_date[date_str] += 1
        
        # Create stats array for last 7 days
        stats = []
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            chats = stats_by_date.get(date_str, 0)
            stats.append({
                "date": date_str,
                "chats": chats,
                "responses": chats  # Assuming 1:1 ratio for now
            })
        
        return DashboardResponse(
            status="success",
            data={"stats": stats}
        )
    except Exception as e:
        return DashboardResponse(
            status="error",
            data={"stats": [], "error": str(e)}
        )


@router.get("/analytics", response_model=DashboardResponse)
async def get_analytics(
    supabase: Client = Depends(get_supabase_client)
):
    """Get analytics data."""
    return DashboardResponse(
        status="success", 
        data={"analytics": "available"}
    )


class WebsiteCreateRequest(BaseModel):
    """Website creation request model."""
    name: str
    url: str
    description: str = ""
    category: str = "Technology"
    scrapingFrequency: str = "daily"
    maxPages: int = 100
    primaryColor: str = "#0066CC"
    welcomeMessage: str = "Hi! ask me your queries..."
    placeholder: str = "Ask me anything..."
    title: str = "Chat Support"
    position: str = "bottom-right"
    features: list = []



@router.post("/websites", response_model=DashboardResponse)
async def create_website(
    website_data: WebsiteCreateRequest,
    supabase: Client = Depends(get_supabase_client)
):
    """Create a new website registration."""
    try:
        import uuid
        from datetime import datetime
        from urllib.parse import urlparse

        # Extract domain from URL
        parsed_url = urlparse(website_data.url)
        domain = parsed_url.netloc

        # Generate unique IDs
        website_id = str(uuid.uuid4())
        widget_id = f"widget-{uuid.uuid4().hex[:8]}"

        # Create website record with scraping frequency and other parameters
        website_record = {
            "id": website_id,
            "name": website_data.name,
            "domain": domain,
            "url": website_data.url,
            "widget_id": widget_id,
            "business_description": website_data.description,
            "scraping_enabled": True,
            "scraping_frequency": website_data.scrapingFrequency,
            "max_pages": website_data.maxPages,
            "is_active": True,
            "user_id": current_user['id']  # From the JWT token
        }
        
        # Insert into database
        result = supabase.table('websites').insert(website_record).execute()

        if result.data:
            created_website = result.data[0]

            # Automatically set up crawl scheduling based on registration parameters
            try:
                scheduler = get_registration_scheduler()
                schedule_result = scheduler.setup_website_crawl_schedule(website_id)
                print(f"Setup crawl schedule for website {website_id}: {schedule_result}")

                # Also trigger initial crawl for immediate content indexing
                if website_data.scrapingFrequency != 'manual':
                    initial_crawl_result = scheduler.trigger_initial_crawl(website_id)
                    print(f"Triggered initial crawl for website {website_id}: {initial_crawl_result}")

            except Exception as e:
                print(f"Warning: Failed to setup crawl schedule for website {website_id}: {e}")
                # Don't fail the website creation if scheduling fails

            return DashboardResponse(
                status="success",
                data={
                    "website": created_website,
                    "message": "Website created successfully" + (
                        f" with {website_data.scrapingFrequency} crawling scheduled"
                        if website_data.scrapingFrequency != 'manual'
                        else ""
                    )
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create website"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating website: {str(e)}"
        )