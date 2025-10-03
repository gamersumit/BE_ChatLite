from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase Configuration (Primary Database)
    supabase_url: str = Field(description="Supabase project URL")
    supabase_anon_key: str = Field(description="Supabase anon key")
    supabase_service_role_key: str = Field(description="Supabase service role key")
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="test-key", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # Authentication & Security
    jwt_secret: str = Field(default="your-jwt-secret-key-change-in-production", description="JWT secret key")
    jwt_expires_in: str = Field(default="24h", description="JWT expiration time")
    auth_cookie_domain: str = Field(default="localhost", description="Authentication cookie domain")
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for application")
    algorithm: str = Field(default="HS256", description="Algorithm for JWT tokens")
    access_token_expire_minutes: int = Field(default=30, description="JWT access token expiration in minutes")
    refresh_token_expire_days: int = Field(default=7, description="JWT refresh token expiration time (7 days)")
    
    # Frontend Configuration  
    frontend_url: str = Field(default="http://localhost:3000", description="Frontend application URL")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @property
    def origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return self.cors_origins

    @property
    def allowed_origins(self) -> str:
        """Backward compatibility alias for cors_origins"""
        return self.cors_origins
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    rate_limit_burst: int = Field(default=10, description="Rate limit burst")
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    
    # Database Configuration (Primary Database)
    database_url: Optional[str] = Field(default=None, description="PostgreSQL database URL")

    # Feature Flags & Limits
    max_pages_per_crawl: int = Field(default=10000, description="Maximum pages per crawl")
    default_crawl_frequency: str = Field(default="weekly", description="Default crawl frequency")
    default_max_pages: int = Field(default=100, description="Default max pages to crawl")
    default_crawl_depth: int = Field(default=3, description="Default crawl depth")

    # System Defaults
    admin_email: str = Field(default="admin@chatlite.com", description="Admin email address")

    # Environment & Debugging
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="info", description="Log level")

    # Optional: Monitoring & Analytics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", description="Celery result backend URL")

    # Cloudinary Configuration (for screenshot storage)
    cloudinary_cloud_name: str = Field(description="Cloudinary cloud name")
    cloudinary_api_key: str = Field(description="Cloudinary API key")
    cloudinary_api_secret: str = Field(description="Cloudinary API secret")

    # API Configuration
    api_base_url: str = Field(default="http://localhost:8001", description="API base URL")
    api_port: int = Field(default=8001, description="API port")
    api_host: str = Field(default="0.0.0.0", description="API host")
    
    # API Configuration
    api_v1_str: str = Field(default="/api/v1", description="API v1 prefix")
    project_name: str = Field(default="LiteChat API", description="Project name")
    
    # Session Management Configuration
    session_duration_days: int = Field(default=7, description="Default session duration in days")
    max_session_duration_days: int = Field(default=30, description="Maximum session duration in days")
    context_window_size: int = Field(default=4000, description="Default context window size in tokens")
    max_context_messages: int = Field(default=20, description="Maximum messages to include in context")
    context_summary_trigger: int = Field(default=15, description="Trigger context summarization after N messages")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

def get_settings() -> Settings:
    """Get application settings."""
    return settings