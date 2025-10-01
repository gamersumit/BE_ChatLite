# Celery Configuration and Deployment Guide

## Overview

This document outlines the Celery configuration for the ChatLite crawling system and provides deployment requirements for running background crawling tasks.

## System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  FastAPI App │────▶│ Redis Broker │◀────│ Celery       │
│              │     │              │     │ Workers      │
└──────────────┘     └──────────────┘     └──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Result Backend   │
                    │     (Redis)       │
                    └───────────────────┘
```

## Prerequisites

### Required Software
- Python 3.10+
- Redis Server 6.0+
- PostgreSQL 14+ (for main database)
- System memory: Minimum 4GB RAM recommended

### Python Dependencies
```bash
celery[redis]==5.3.4
redis==5.0.1
```

## Redis Configuration

### Installation
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

### Configuration Settings
Edit `/etc/redis/redis.conf` or use these settings:

```conf
# Basic settings
bind 127.0.0.1 ::1
port 6379
databases 16

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (optional but recommended)
save 900 1
save 300 10
save 60 10000

# Connection settings
timeout 300
tcp-keepalive 300
tcp-backlog 511
```

### Environment Variables
```bash
# Redis connection
REDIS_URL=redis://localhost:6379/0

# Optional: Redis with password
REDIS_URL=redis://:password@localhost:6379/0
```

## Celery Configuration

### Queue Configuration
The system uses multiple queues for different task types:

- **crawl_queue**: Website crawling tasks
- **process_queue**: Content processing and embedding generation
- **schedule_queue**: Scheduled crawl management
- **monitor_queue**: System monitoring and health checks
- **default**: Fallback queue for unrouted tasks

### Worker Configuration

#### Development Setup
```bash
# Start a worker for all queues
celery -A app.core.celery_config worker --loglevel=info --queues=crawl_queue,process_queue,schedule_queue,monitor_queue

# Or start separate workers for each queue (recommended for production)
celery -A app.core.celery_config worker --loglevel=info --queues=crawl_queue --concurrency=4 -n crawler@%h
celery -A app.core.celery_config worker --loglevel=info --queues=process_queue --concurrency=2 -n processor@%h
celery -A app.core.celery_config worker --loglevel=info --queues=schedule_queue --concurrency=1 -n scheduler@%h
celery -A app.core.celery_config worker --loglevel=info --queues=monitor_queue --concurrency=1 -n monitor@%h
```

#### Production Setup (systemd)
Create `/etc/systemd/system/celery-crawler.service`:

```ini
[Unit]
Description=Celery Crawler Worker
After=network.target redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/path/to/chatlite/backend
Environment="PATH=/path/to/venv/bin"
Environment="REDIS_URL=redis://localhost:6379/0"
ExecStart=/path/to/venv/bin/celery -A app.core.celery_config multi start crawler processor scheduler monitor \
    --pidfile=/var/run/celery/%n.pid \
    --logfile=/var/log/celery/%n%I.log \
    --loglevel=info \
    --queues:crawler=crawl_queue \
    --queues:processor=process_queue \
    --queues:scheduler=schedule_queue \
    --queues:monitor=monitor_queue \
    --concurrency:crawler=4 \
    --concurrency:processor=2 \
    --concurrency:scheduler=1 \
    --concurrency:monitor=1
ExecStop=/path/to/venv/bin/celery multi stopwait crawler processor scheduler monitor --pidfile=/var/run/celery/%n.pid
ExecReload=/path/to/venv/bin/celery multi restart crawler processor scheduler monitor --pidfile=/var/run/celery/%n.pid

[Install]
WantedBy=multi-user.target
```

### Celery Beat (for scheduled tasks)
```bash
# Development
celery -A app.core.celery_config beat --loglevel=info

# Production (systemd)
# Create /etc/systemd/system/celery-beat.service
```

## Configuration Parameters

### Task Settings
```python
# Retry configuration
task_default_retry_delay = 60  # 1 minute
task_max_retries = 3
task_soft_time_limit = 300  # 5 minutes
task_time_limit = 600  # 10 minutes

# Worker settings
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 1000
```

### Performance Tuning

#### Redis Connection Pool
```python
broker_transport_options = {
    'fanout_prefix': True,
    'fanout_patterns': True,
    'retry_on_timeout': True,
    'max_connections': 20,
}
```

#### Concurrency Recommendations
- **Crawl workers**: 2-4 concurrent tasks (network I/O bound)
- **Process workers**: 1-2 concurrent tasks (CPU bound)
- **Schedule workers**: 1 concurrent task
- **Monitor workers**: 1 concurrent task

## Monitoring

### Health Check Endpoints
The system provides health check endpoints:

```bash
# Check Redis connection
GET /api/v1/health/redis

# Check Celery workers
GET /api/v1/health/celery

# Detailed health status
GET /api/v1/health/detailed

# Worker-specific status
GET /api/v1/health/workers
```

### Monitoring Commands
```bash
# Check active tasks
celery -A app.core.celery_config inspect active

# Check registered tasks
celery -A app.core.celery_config inspect registered

# Check worker stats
celery -A app.core.celery_config inspect stats

# Monitor events in real-time
celery -A app.core.celery_config events
```

### Flower (Web-based monitoring)
```bash
# Install Flower
pip install flower

# Start Flower
celery -A app.core.celery_config flower --port=5555

# Access at http://localhost:5555
```

## Deployment Checklist

### Pre-deployment
- [ ] Redis server installed and running
- [ ] Redis connection tested
- [ ] Python dependencies installed
- [ ] Environment variables configured
- [ ] Database migrations completed

### Worker Deployment
- [ ] Worker services configured
- [ ] Log directories created with proper permissions
- [ ] PID file directories created
- [ ] Systemd services enabled
- [ ] Worker health checks passing

### Post-deployment
- [ ] Health endpoints responding
- [ ] Worker registration verified
- [ ] Test crawl task successful
- [ ] Monitoring dashboard accessible
- [ ] Log rotation configured

## Troubleshooting

### Common Issues

#### Workers not starting
```bash
# Check Redis connection
redis-cli ping

# Check Celery configuration
celery -A app.core.celery_config inspect ping

# Check logs
journalctl -u celery-crawler -f
```

#### Tasks not executing
```bash
# Check task registration
celery -A app.core.celery_config inspect registered

# Check active tasks
celery -A app.core.celery_config inspect active

# Purge queue if needed
celery -A app.core.celery_config purge
```

#### Memory issues
```bash
# Monitor Redis memory
redis-cli info memory

# Clear Redis cache if needed
redis-cli FLUSHDB

# Restart workers
systemctl restart celery-crawler
```

## Security Considerations

1. **Redis Security**
   - Bind Redis to localhost only
   - Use strong passwords in production
   - Enable SSL/TLS for remote connections
   - Configure firewall rules

2. **Worker Security**
   - Run workers with limited user privileges
   - Secure log file permissions
   - Monitor for abnormal task execution
   - Implement rate limiting

3. **Task Security**
   - Validate all task inputs
   - Implement task timeouts
   - Monitor for task flooding
   - Use task signatures for sensitive operations

## Performance Optimization

1. **Task Optimization**
   - Batch similar operations
   - Use task chains for sequential operations
   - Implement proper retry logic
   - Monitor task execution times

2. **Redis Optimization**
   - Configure appropriate memory limits
   - Use connection pooling
   - Monitor key expiration
   - Implement proper data serialization

3. **Worker Optimization**
   - Adjust concurrency based on workload
   - Use worker pools for CPU-bound tasks
   - Implement task routing strategies
   - Monitor worker resource usage

## Backup and Recovery

1. **Redis Backup**
   ```bash
   # Manual backup
   redis-cli BGSAVE

   # Backup location
   /var/lib/redis/dump.rdb
   ```

2. **Task Recovery**
   - Enable `task_acks_late` for task persistence
   - Implement task result backend persistence
   - Monitor failed tasks and implement retry logic
   - Log task failures for manual recovery

## Support and Maintenance

For issues or questions:
1. Check system logs: `/var/log/celery/`
2. Review health endpoints
3. Monitor Redis and Celery metrics
4. Consult this documentation

Last updated: 2025-09-17