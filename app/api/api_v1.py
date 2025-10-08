from fastapi import APIRouter
from .endpoints import widget, manual_crawl, auth, website_onboarding, dashboard, analytics, websites, health, script_generation, crawling_jobs

api_router = APIRouter()

# Core API Routes - Only the ones actually used by frontend
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(website_onboarding.router, prefix="/onboarding", tags=["website-onboarding"])
api_router.include_router(websites.router, prefix="/websites", tags=["websites"])
api_router.include_router(manual_crawl.router, prefix="/crawl", tags=["crawl"])
api_router.include_router(crawling_jobs.router, prefix="/crawl", tags=["crawling-jobs"])
api_router.include_router(widget.router, prefix="/widget", tags=["widget"])
api_router.include_router(script_generation.router, prefix="/widget", tags=["widget-script"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(health.router, prefix="/health", tags=["health"])