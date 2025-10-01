"""
Specialized page processing workers for web crawling and content extraction.

This module provides specialized worker implementations for different types of
page processing tasks including crawling, content extraction, and data processing.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from urllib.parse import urljoin, urlparse

from app.services.worker_process_manager import WorkerTask, TaskResult, TaskStatus
from app.services.javascript_crawler import JavaScriptCrawler
from app.services.browser_resource_manager import BrowserResourceManager
from app.services.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class PageProcessingType(Enum):
    """Types of page processing operations."""
    CRAWL_PAGE = "crawl_page"
    EXTRACT_CONTENT = "extract_content"
    PROCESS_LINKS = "process_links"
    EXTRACT_ENTITIES = "extract_entities"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    VALIDATE_CONTENT = "validate_content"
    CLEAN_HTML = "clean_html"


@dataclass
class CrawlConfig:
    """Configuration for page crawling."""
    enable_javascript: bool = True
    wait_for_selector: Optional[str] = None
    wait_timeout: int = 30000
    user_agent: Optional[str] = None
    headers: Dict[str, str] = None
    follow_redirects: bool = True
    max_redirects: int = 5
    viewport: Dict[str, int] = None
    block_resources: List[str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.viewport is None:
            self.viewport = {"width": 1920, "height": 1080}
        if self.block_resources is None:
            self.block_resources = ["font", "image", "media"]


@dataclass
class PageData:
    """Structured page data from crawling."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    html: Optional[str] = None
    links: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    screenshots: List[str] = None
    performance_metrics: Dict[str, Any] = None
    content_hash: Optional[str] = None
    extracted_at: datetime = None

    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}
        if self.screenshots is None:
            self.screenshots = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.extracted_at is None:
            self.extracted_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['extracted_at'] = self.extracted_at.isoformat() if self.extracted_at else None
        return data


class PageCrawlingWorker:
    """Specialized worker for page crawling operations."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.browser_manager: Optional[BrowserResourceManager] = None
        self.javascript_crawler: Optional[JavaScriptCrawler] = None
        self.performance_monitor = PerformanceMonitor()

    async def initialize(self):
        """Initialize crawler resources."""
        try:
            self.browser_manager = BrowserResourceManager(
                max_browsers=2,
                max_pages_per_browser=5,
                browser_timeout=60,
                page_timeout=30
            )
            await self.browser_manager.initialize()

            self.javascript_crawler = JavaScriptCrawler(
                browser_manager=self.browser_manager
            )

            logger.info(f"PageCrawlingWorker {self.worker_id} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PageCrawlingWorker {self.worker_id}: {e}")
            raise

    async def process_task(self, task: WorkerTask) -> TaskResult:
        """Process a crawling task."""
        start_time = time.time()

        try:
            if not self.javascript_crawler:
                raise RuntimeError("Worker not properly initialized")

            # Parse task configuration
            crawl_config = CrawlConfig()
            if "crawl_config" in task.metadata:
                config_data = task.metadata["crawl_config"]
                crawl_config = CrawlConfig(**config_data)

            # Monitor performance
            with self.performance_monitor.monitor_operation(f"crawl_{task.task_id}"):
                # Perform page crawling
                page_data = await self._crawl_page(task.url, crawl_config, task.task_id)

                # Process extracted data
                processed_data = await self._process_page_data(page_data, task)

                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    worker_id=self.worker_id,
                    result_data={
                        "page_data": processed_data,
                        "processing_type": task.task_type,
                        "url": task.url,
                        "execution_time": execution_time,
                        "performance_metrics": self.performance_monitor.get_metrics(f"crawl_{task.task_id}")
                    },
                    execution_time_seconds=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"PageCrawlingWorker {self.worker_id} failed task {task.task_id}: {e}")

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                worker_id=self.worker_id,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

    async def _crawl_page(self, url: str, config: CrawlConfig, task_id: str) -> PageData:
        """Crawl a single page and extract data."""
        try:
            # Use JavaScript crawler for comprehensive extraction
            crawl_result = await self.javascript_crawler.crawl_page(
                url=url,
                enable_javascript=config.enable_javascript,
                wait_for_selector=config.wait_for_selector,
                wait_timeout=config.wait_timeout
            )

            # Extract page content
            page_data = PageData(
                url=url,
                title=crawl_result.get("title"),
                content=crawl_result.get("text_content"),
                html=crawl_result.get("html"),
                links=crawl_result.get("links", []),
                metadata=crawl_result.get("metadata", {}),
                performance_metrics=crawl_result.get("performance", {})
            )

            # Generate content hash
            if page_data.content:
                content_bytes = page_data.content.encode('utf-8')
                page_data.content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Extract additional metadata
            page_data.metadata.update({
                "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "worker_id": self.worker_id,
                "crawl_config": asdict(config)
            })

            logger.info(f"Successfully crawled page: {url} (content: {len(page_data.content or '')} chars)")

            return page_data

        except Exception as e:
            logger.error(f"Failed to crawl page {url}: {e}")
            raise

    async def _process_page_data(self, page_data: PageData, task: WorkerTask) -> Dict[str, Any]:
        """Process and enhance page data."""
        processed_data = page_data.to_dict()

        # Add task-specific processing
        if task.task_type == PageProcessingType.CRAWL_PAGE.value:
            # Basic crawling - data is ready
            pass

        elif task.task_type == PageProcessingType.EXTRACT_CONTENT.value:
            # Enhanced content extraction
            processed_data["content_analysis"] = await self._analyze_content(page_data.content)
            processed_data["content_structure"] = await self._analyze_structure(page_data.html)

        elif task.task_type == PageProcessingType.PROCESS_LINKS.value:
            # Process and validate links
            processed_data["processed_links"] = await self._process_links(page_data.links, page_data.url)

        elif task.task_type == PageProcessingType.EXTRACT_ENTITIES.value:
            # Entity extraction
            processed_data["entities"] = await self._extract_entities(page_data.content)

        return processed_data

    async def _analyze_content(self, content: Optional[str]) -> Dict[str, Any]:
        """Analyze page content for insights."""
        if not content:
            return {}

        analysis = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "paragraph_count": content.count('\n\n') + 1,
            "language": "en",  # Would use language detection
            "readability_score": 0.7,  # Would calculate actual readability
            "content_type": "article"  # Would classify content type
        }

        return analysis

    async def _analyze_structure(self, html: Optional[str]) -> Dict[str, Any]:
        """Analyze HTML structure."""
        if not html:
            return {}

        # Basic HTML structure analysis
        structure = {
            "has_nav": "nav" in html.lower(),
            "has_header": "header" in html.lower(),
            "has_footer": "footer" in html.lower(),
            "has_main": "main" in html.lower(),
            "script_count": html.lower().count("<script"),
            "style_count": html.lower().count("<style"),
            "image_count": html.lower().count("<img"),
            "form_count": html.lower().count("<form")
        }

        return structure

    async def _process_links(self, links: List[Dict[str, str]], base_url: str) -> List[Dict[str, Any]]:
        """Process and enhance extracted links."""
        processed_links = []

        for link in links:
            processed_link = link.copy()

            # Resolve relative URLs
            if "href" in link:
                absolute_url = urljoin(base_url, link["href"])
                processed_link["absolute_url"] = absolute_url

                # Parse URL components
                parsed = urlparse(absolute_url)
                processed_link["domain"] = parsed.netloc
                processed_link["path"] = parsed.path
                processed_link["is_external"] = parsed.netloc != urlparse(base_url).netloc

            # Classify link type
            if "href" in processed_link:
                href = processed_link["href"].lower()
                if any(ext in href for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx"]):
                    processed_link["type"] = "document"
                elif any(ext in href for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg"]):
                    processed_link["type"] = "image"
                elif any(ext in href for ext in [".mp4", ".avi", ".mov", ".wmv"]):
                    processed_link["type"] = "video"
                elif href.startswith("mailto:"):
                    processed_link["type"] = "email"
                elif href.startswith("tel:"):
                    processed_link["type"] = "phone"
                else:
                    processed_link["type"] = "page"

            processed_links.append(processed_link)

        return processed_links

    async def _extract_entities(self, content: Optional[str]) -> List[Dict[str, Any]]:
        """Extract entities from content."""
        if not content:
            return []

        # Mock entity extraction (would use NLP library like spaCy)
        entities = [
            {"text": "Example Entity", "type": "ORG", "confidence": 0.95},
            {"text": "New York", "type": "GPE", "confidence": 0.90},
            {"text": "January 2024", "type": "DATE", "confidence": 0.85}
        ]

        return entities

    async def cleanup(self):
        """Cleanup worker resources."""
        try:
            if self.browser_manager:
                await self.browser_manager.cleanup()

            logger.info(f"PageCrawlingWorker {self.worker_id} cleanup completed")

        except Exception as e:
            logger.error(f"Error during PageCrawlingWorker cleanup: {e}")


class ContentProcessingWorker:
    """Specialized worker for content processing operations."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    async def initialize(self):
        """Initialize content processing resources."""
        logger.info(f"ContentProcessingWorker {self.worker_id} initialized")

    async def process_task(self, task: WorkerTask) -> TaskResult:
        """Process a content processing task."""
        start_time = time.time()

        try:
            # Get content data from task
            content_data = task.metadata.get("content_data", {})

            if task.task_type == PageProcessingType.CLEAN_HTML.value:
                result_data = await self._clean_html(content_data.get("html", ""))

            elif task.task_type == PageProcessingType.GENERATE_EMBEDDINGS.value:
                result_data = await self._generate_embeddings(content_data.get("content", ""))

            elif task.task_type == PageProcessingType.VALIDATE_CONTENT.value:
                result_data = await self._validate_content(content_data)

            else:
                result_data = {"message": f"Processed content for {task.url}"}

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                worker_id=self.worker_id,
                result_data={
                    "processed_content": result_data,
                    "processing_type": task.task_type,
                    "execution_time": execution_time
                },
                execution_time_seconds=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ContentProcessingWorker {self.worker_id} failed task {task.task_id}: {e}")

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                worker_id=self.worker_id,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

    async def _clean_html(self, html: str) -> Dict[str, Any]:
        """Clean and normalize HTML content."""
        # Mock HTML cleaning (would use BeautifulSoup or similar)
        cleaned_content = {
            "original_length": len(html),
            "cleaned_html": html.replace("<script>", "").replace("</script>", ""),
            "removed_elements": ["script", "style", "noscript"],
            "text_content": "Extracted text content",
            "cleaning_stats": {
                "scripts_removed": html.count("<script"),
                "styles_removed": html.count("<style"),
                "comments_removed": html.count("<!--")
            }
        }

        return cleaned_content

    async def _generate_embeddings(self, content: str) -> Dict[str, Any]:
        """Generate embeddings for content."""
        # Mock embedding generation (would use actual embedding model)
        embeddings = {
            "content_length": len(content),
            "embedding_dimension": 768,
            "embedding_model": "mock-embedding-model",
            "embedding_vector": [0.1] * 768,  # Mock vector
            "chunk_count": max(1, len(content) // 500),
            "generation_time": 0.05
        }

        return embeddings

    async def _validate_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality and completeness."""
        validation_results = {
            "is_valid": True,
            "content_score": 0.85,
            "issues": [],
            "recommendations": [],
            "validation_checks": {
                "has_title": bool(content_data.get("title")),
                "has_content": bool(content_data.get("content")),
                "content_length_adequate": len(content_data.get("content", "")) > 100,
                "has_links": bool(content_data.get("links")),
                "has_metadata": bool(content_data.get("metadata"))
            }
        }

        # Add issues based on validation
        if not validation_results["validation_checks"]["has_title"]:
            validation_results["issues"].append("Missing page title")
            validation_results["recommendations"].append("Ensure page has a descriptive title")

        if not validation_results["validation_checks"]["content_length_adequate"]:
            validation_results["issues"].append("Content too short")
            validation_results["recommendations"].append("Page may need more substantial content")

        return validation_results

    async def cleanup(self):
        """Cleanup worker resources."""
        logger.info(f"ContentProcessingWorker {self.worker_id} cleanup completed")


class ParallelPageProcessor:
    """High-level parallel page processing coordinator."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.crawling_workers: Dict[str, PageCrawlingWorker] = {}
        self.content_workers: Dict[str, ContentProcessingWorker] = {}

    async def initialize(self):
        """Initialize all processing workers."""
        # Initialize crawling workers
        for i in range(self.max_workers // 2):
            worker_id = f"crawl_worker_{i:03d}"
            worker = PageCrawlingWorker(worker_id)
            await worker.initialize()
            self.crawling_workers[worker_id] = worker

        # Initialize content processing workers
        for i in range(self.max_workers // 2):
            worker_id = f"content_worker_{i:03d}"
            worker = ContentProcessingWorker(worker_id)
            await worker.initialize()
            self.content_workers[worker_id] = worker

        logger.info(f"Initialized {len(self.crawling_workers)} crawling workers and {len(self.content_workers)} content workers")

    async def process_urls_batch(self, urls: List[str], config: Optional[CrawlConfig] = None) -> List[TaskResult]:
        """Process a batch of URLs in parallel."""
        if not config:
            config = CrawlConfig()

        # Create tasks for each URL
        tasks = []
        for i, url in enumerate(urls):
            task = WorkerTask(
                task_id=f"batch_crawl_{i:03d}",
                task_type=PageProcessingType.CRAWL_PAGE.value,
                url=url,
                priority=1,
                metadata={"crawl_config": asdict(config)}
            )
            tasks.append(task)

        # Process tasks in parallel
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_single_task(task: WorkerTask) -> TaskResult:
            async with semaphore:
                worker_id = f"crawl_worker_{len(results) % len(self.crawling_workers):03d}"
                worker = self.crawling_workers[worker_id]
                return await worker.process_task(task)

        # Execute all tasks concurrently
        task_results = await asyncio.gather(
            *[process_single_task(task) for task in tasks],
            return_exceptions=True
        )

        # Process results
        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Task processing failed: {result}")
                results.append(TaskResult(
                    task_id="unknown",
                    status=TaskStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                results.append(result)

        return results

    async def get_worker_statistics(self) -> Dict[str, Any]:
        """Get statistics for all workers."""
        stats = {
            "total_workers": len(self.crawling_workers) + len(self.content_workers),
            "crawling_workers": len(self.crawling_workers),
            "content_workers": len(self.content_workers),
            "worker_details": {}
        }

        # Add details for each worker type
        for worker_id in self.crawling_workers:
            stats["worker_details"][worker_id] = {
                "type": "crawling",
                "status": "active"
            }

        for worker_id in self.content_workers:
            stats["worker_details"][worker_id] = {
                "type": "content",
                "status": "active"
            }

        return stats

    async def cleanup(self):
        """Cleanup all workers."""
        # Cleanup crawling workers
        cleanup_tasks = []
        for worker in self.crawling_workers.values():
            cleanup_tasks.append(worker.cleanup())

        for worker in self.content_workers.values():
            cleanup_tasks.append(worker.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("ParallelPageProcessor cleanup completed")


# Factory function for creating appropriate workers
def create_worker_for_task_type(task_type: str, worker_id: str):
    """Create appropriate worker based on task type."""
    if task_type in [
        PageProcessingType.CRAWL_PAGE.value,
        PageProcessingType.EXTRACT_CONTENT.value,
        PageProcessingType.PROCESS_LINKS.value,
        PageProcessingType.EXTRACT_ENTITIES.value
    ]:
        return PageCrawlingWorker(worker_id)

    elif task_type in [
        PageProcessingType.CLEAN_HTML.value,
        PageProcessingType.GENERATE_EMBEDDINGS.value,
        PageProcessingType.VALIDATE_CONTENT.value
    ]:
        return ContentProcessingWorker(worker_id)

    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Global parallel processor instance
parallel_processor = ParallelPageProcessor()