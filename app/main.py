from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
import sys

from .api.api_v1 import api_router
from .core.config import settings
from .services.socket_chat_service import start_socket_server, stop_socket_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crawler.log')
    ]
)

# Set specific loggers to INFO level
logging.getLogger('app.services.crawler_service').setLevel(logging.INFO)
logging.getLogger('app.api.endpoints.crawler').setLevel(logging.INFO)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 LiteChat API starting up...")
    print(f"Environment: {settings.environment}")

    print(f"Database: Supabase ({settings.supabase_url.split('.')[0]}...)")
    print(f"⚙️  Celery: Workers running separately (local or Docker)")

    # Start socket server
    start_socket_server()

    yield

    # Shutdown
    print("🛑 LiteChat API shutting down...")
    stop_socket_server()


# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    description="AI Website Chatbot Plugin API",
    version="1.0.0",
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    docs_url=f"{settings.api_v1_str}/docs",
    redoc_url=f"{settings.api_v1_str}/redoc",
    lifespan=lifespan
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list + ["*"],  # Allow all origins for WebSocket testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.environment == "development" else "An error occurred"
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LiteChat API",
        "version": "1.0.0",
        "timestamp": time.time()
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LiteChat API",
        "version": "1.0.0",
        "docs_url": f"{settings.api_v1_str}/docs",
        "status": "online"
    }


# Include API routes
app.include_router(api_router, prefix=settings.api_v1_str)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level="info"
    )