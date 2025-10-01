"""
Incremental Update Detection Service

This module provides functionality for detecting when website content has changed
and implementing smart re-crawling logic to avoid unnecessary resource usage.
"""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import asyncio
from urllib.parse import urlparse, urljoin

from app.core.logging import get_logger

logger = get_logger(__name__)


class ChangeType(Enum):
    """Types of content changes detected."""
    CONTENT_MODIFIED = "content_modified"
    STRUCTURE_CHANGED = "structure_changed"
    LINKS_ADDED = "links_added"
    LINKS_REMOVED = "links_removed"
    METADATA_CHANGED = "metadata_changed"
    NO_CHANGE = "no_change"
    FIRST_CRAWL = "first_crawl"


class UpdatePriority(Enum):
    """Priority levels for update processing."""
    CRITICAL = "critical"  # Must be updated immediately
    HIGH = "high"         # Should be updated within hours
    MEDIUM = "medium"     # Can be updated within days
    LOW = "low"          # Can be updated weekly
    DEFERRED = "deferred" # Update only when explicitly requested


@dataclass
class ContentFingerprint:
    """Represents a fingerprint of page content for change detection."""
    url: str
    content_hash: str
    structure_hash: str
    metadata_hash: str
    links_hash: str
    title: Optional[str]
    content_length: int
    last_modified: Optional[datetime]
    etag: Optional[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.last_modified:
            data['last_modified'] = self.last_modified.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentFingerprint':
        """Create instance from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_modified'):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


@dataclass
class ContentChange:
    """Represents a detected change in content."""
    url: str
    change_type: ChangeType
    old_fingerprint: Optional[ContentFingerprint]
    new_fingerprint: ContentFingerprint
    change_details: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    priority: UpdatePriority
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'change_type': self.change_type.value,
            'old_fingerprint': self.old_fingerprint.to_dict() if self.old_fingerprint else None,
            'new_fingerprint': self.new_fingerprint.to_dict(),
            'change_details': self.change_details,
            'confidence_score': self.confidence_score,
            'priority': self.priority.value,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class UpdateSchedule:
    """Represents a schedule for checking updates."""
    url: str
    check_frequency: timedelta
    last_check: datetime
    next_check: datetime
    priority: UpdatePriority
    failure_count: int = 0
    max_failures: int = 5

    def calculate_next_check(self) -> datetime:
        """Calculate next check time based on failure count."""
        if self.failure_count == 0:
            return self.last_check + self.check_frequency

        # Exponential backoff for failures
        backoff_multiplier = min(2 ** self.failure_count, 16)
        extended_frequency = self.check_frequency * backoff_multiplier
        return self.last_check + extended_frequency

    def mark_success(self):
        """Mark successful check and reset failure count."""
        self.last_check = datetime.now(timezone.utc)
        self.next_check = self.calculate_next_check()
        self.failure_count = 0

    def mark_failure(self):
        """Mark failed check and increment failure count."""
        self.failure_count += 1
        self.last_check = datetime.now(timezone.utc)
        self.next_check = self.calculate_next_check()


class ContentHasher:
    """Utility class for generating content hashes."""

    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash of main content."""
        if not content:
            return ""
        # Normalize whitespace and generate hash
        normalized = ' '.join(content.split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    @staticmethod
    def hash_structure(html: str) -> str:
        """Generate hash of HTML structure (tags only)."""
        if not html:
            return ""

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Extract just the tag structure
            structure_tags = []
            for tag in soup.find_all():
                structure_tags.append(tag.name)

            structure_str = ','.join(structure_tags)
            return hashlib.sha256(structure_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing structure: {e}")
            return ""

    @staticmethod
    def hash_metadata(metadata: Dict[str, Any]) -> str:
        """Generate hash of page metadata."""
        if not metadata:
            return ""

        # Sort keys for consistent hashing
        sorted_metadata = json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(sorted_metadata.encode('utf-8')).hexdigest()

    @staticmethod
    def hash_links(links: Set[str]) -> str:
        """Generate hash of page links."""
        if not links:
            return ""

        # Sort links for consistent hashing
        sorted_links = json.dumps(sorted(list(links)))
        return hashlib.sha256(sorted_links.encode('utf-8')).hexdigest()


class ChangeDetector:
    """Detects changes between content fingerprints."""

    @staticmethod
    def detect_changes(
        old_fingerprint: Optional[ContentFingerprint],
        new_fingerprint: ContentFingerprint
    ) -> ContentChange:
        """Detect changes between two fingerprints."""

        if old_fingerprint is None:
            return ContentChange(
                url=new_fingerprint.url,
                change_type=ChangeType.FIRST_CRAWL,
                old_fingerprint=None,
                new_fingerprint=new_fingerprint,
                change_details={'reason': 'First time crawling this URL'},
                confidence_score=1.0,
                priority=UpdatePriority.HIGH,
                detected_at=datetime.now(timezone.utc)
            )

        changes = []
        change_details = {}
        confidence_score = 1.0

        # Check content changes
        if old_fingerprint.content_hash != new_fingerprint.content_hash:
            changes.append(ChangeType.CONTENT_MODIFIED)
            change_details['content_changed'] = True

            # Calculate similarity score
            old_len = old_fingerprint.content_length
            new_len = new_fingerprint.content_length
            size_diff = abs(old_len - new_len) / max(old_len, new_len, 1)
            change_details['content_size_change'] = size_diff

        # Check structure changes
        if old_fingerprint.structure_hash != new_fingerprint.structure_hash:
            changes.append(ChangeType.STRUCTURE_CHANGED)
            change_details['structure_changed'] = True

        # Check link changes
        if old_fingerprint.links_hash != new_fingerprint.links_hash:
            changes.append(ChangeType.LINKS_ADDED)  # We'll refine this later
            change_details['links_changed'] = True

        # Check metadata changes
        if old_fingerprint.metadata_hash != new_fingerprint.metadata_hash:
            changes.append(ChangeType.METADATA_CHANGED)
            change_details['metadata_changed'] = True

        # Determine primary change type and priority
        if not changes:
            primary_change = ChangeType.NO_CHANGE
            priority = UpdatePriority.LOW
        elif ChangeType.CONTENT_MODIFIED in changes:
            primary_change = ChangeType.CONTENT_MODIFIED
            priority = UpdatePriority.HIGH
        elif ChangeType.STRUCTURE_CHANGED in changes:
            primary_change = ChangeType.STRUCTURE_CHANGED
            priority = UpdatePriority.MEDIUM
        else:
            primary_change = changes[0]
            priority = UpdatePriority.MEDIUM

        return ContentChange(
            url=new_fingerprint.url,
            change_type=primary_change,
            old_fingerprint=old_fingerprint,
            new_fingerprint=new_fingerprint,
            change_details=change_details,
            confidence_score=confidence_score,
            priority=priority,
            detected_at=datetime.now(timezone.utc)
        )


class IncrementalUpdateService:
    """Main service for managing incremental updates."""

    def __init__(self, storage_path: str = "/tmp/incremental_updates"):
        self.storage_path = storage_path
        self.fingerprints: Dict[str, ContentFingerprint] = {}
        self.schedules: Dict[str, UpdateSchedule] = {}
        self.pending_changes: List[ContentChange] = []
        self.change_detector = ChangeDetector()
        self.content_hasher = ContentHasher()

        # Initialize storage
        asyncio.create_task(self._initialize_storage())

    async def _initialize_storage(self):
        """Initialize storage directory and load existing data."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        await self.load_fingerprints()
        await self.load_schedules()

    async def create_fingerprint(
        self,
        url: str,
        content: str,
        html: str,
        metadata: Dict[str, Any],
        links: Set[str],
        title: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        etag: Optional[str] = None
    ) -> ContentFingerprint:
        """Create a content fingerprint for the given page."""

        content_hash = self.content_hasher.hash_content(content)
        structure_hash = self.content_hasher.hash_structure(html)
        metadata_hash = self.content_hasher.hash_metadata(metadata)
        links_hash = self.content_hasher.hash_links(links)

        fingerprint = ContentFingerprint(
            url=url,
            content_hash=content_hash,
            structure_hash=structure_hash,
            metadata_hash=metadata_hash,
            links_hash=links_hash,
            title=title,
            content_length=len(content),
            last_modified=last_modified,
            etag=etag,
            timestamp=datetime.now(timezone.utc)
        )

        return fingerprint

    async def detect_changes(
        self,
        url: str,
        new_fingerprint: ContentFingerprint
    ) -> ContentChange:
        """Detect changes for a URL."""
        old_fingerprint = self.fingerprints.get(url)
        change = self.change_detector.detect_changes(old_fingerprint, new_fingerprint)

        # Store the new fingerprint
        self.fingerprints[url] = new_fingerprint

        # Add to pending changes if significant
        if change.change_type != ChangeType.NO_CHANGE:
            self.pending_changes.append(change)
            logger.info(f"Change detected for {url}: {change.change_type.value}")

        return change

    async def schedule_update_check(
        self,
        url: str,
        frequency: timedelta,
        priority: UpdatePriority = UpdatePriority.MEDIUM
    ):
        """Schedule regular update checks for a URL."""
        now = datetime.now(timezone.utc)

        schedule = UpdateSchedule(
            url=url,
            check_frequency=frequency,
            last_check=now,
            next_check=now + frequency,
            priority=priority
        )

        self.schedules[url] = schedule
        logger.info(f"Scheduled update checks for {url} every {frequency}")

    async def get_urls_for_update_check(self) -> List[str]:
        """Get URLs that need to be checked for updates."""
        now = datetime.now(timezone.utc)
        urls_to_check = []

        for url, schedule in self.schedules.items():
            if schedule.next_check <= now and schedule.failure_count < schedule.max_failures:
                urls_to_check.append(url)

        # Sort by priority and next check time
        urls_to_check.sort(key=lambda url: (
            self.schedules[url].priority.value,
            self.schedules[url].next_check
        ))

        return urls_to_check

    async def mark_check_result(self, url: str, success: bool):
        """Mark the result of an update check."""
        if url in self.schedules:
            if success:
                self.schedules[url].mark_success()
            else:
                self.schedules[url].mark_failure()

    async def get_pending_changes(
        self,
        priority: Optional[UpdatePriority] = None,
        limit: Optional[int] = None
    ) -> List[ContentChange]:
        """Get pending changes, optionally filtered by priority."""
        changes = self.pending_changes

        if priority:
            changes = [c for c in changes if c.priority == priority]

        # Sort by priority and detection time
        changes.sort(key=lambda c: (c.priority.value, c.detected_at))

        if limit:
            changes = changes[:limit]

        return changes

    async def clear_processed_changes(self, change_urls: List[str]):
        """Remove processed changes from pending list."""
        self.pending_changes = [
            c for c in self.pending_changes
            if c.url not in change_urls
        ]

    async def save_fingerprints(self):
        """Save fingerprints to storage."""
        fingerprints_file = f"{self.storage_path}/fingerprints.json"
        data = {url: fp.to_dict() for url, fp in self.fingerprints.items()}

        async with aiofiles.open(fingerprints_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def load_fingerprints(self):
        """Load fingerprints from storage."""
        fingerprints_file = f"{self.storage_path}/fingerprints.json"

        try:
            async with aiofiles.open(fingerprints_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            self.fingerprints = {
                url: ContentFingerprint.from_dict(fp_data)
                for url, fp_data in data.items()
            }
            logger.info(f"Loaded {len(self.fingerprints)} fingerprints")

        except FileNotFoundError:
            logger.info("No existing fingerprints found")
        except Exception as e:
            logger.error(f"Error loading fingerprints: {e}")

    async def save_schedules(self):
        """Save schedules to storage."""
        schedules_file = f"{self.storage_path}/schedules.json"
        data = {}

        for url, schedule in self.schedules.items():
            data[url] = {
                'url': schedule.url,
                'check_frequency_seconds': schedule.check_frequency.total_seconds(),
                'last_check': schedule.last_check.isoformat(),
                'next_check': schedule.next_check.isoformat(),
                'priority': schedule.priority.value,
                'failure_count': schedule.failure_count,
                'max_failures': schedule.max_failures
            }

        async with aiofiles.open(schedules_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def load_schedules(self):
        """Load schedules from storage."""
        schedules_file = f"{self.storage_path}/schedules.json"

        try:
            async with aiofiles.open(schedules_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            for url, schedule_data in data.items():
                schedule = UpdateSchedule(
                    url=schedule_data['url'],
                    check_frequency=timedelta(seconds=schedule_data['check_frequency_seconds']),
                    last_check=datetime.fromisoformat(schedule_data['last_check']),
                    next_check=datetime.fromisoformat(schedule_data['next_check']),
                    priority=UpdatePriority(schedule_data['priority']),
                    failure_count=schedule_data.get('failure_count', 0),
                    max_failures=schedule_data.get('max_failures', 5)
                )
                self.schedules[url] = schedule

            logger.info(f"Loaded {len(self.schedules)} update schedules")

        except FileNotFoundError:
            logger.info("No existing schedules found")
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the incremental update system."""
        now = datetime.now(timezone.utc)

        # Count schedules by priority
        priority_counts = {}
        overdue_count = 0

        for schedule in self.schedules.values():
            priority = schedule.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

            if schedule.next_check <= now:
                overdue_count += 1

        # Count changes by type
        change_type_counts = {}
        for change in self.pending_changes:
            change_type = change.change_type.value
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1

        return {
            'fingerprints_stored': len(self.fingerprints),
            'scheduled_urls': len(self.schedules),
            'overdue_checks': overdue_count,
            'pending_changes': len(self.pending_changes),
            'priority_distribution': priority_counts,
            'change_type_distribution': change_type_counts,
            'last_update': now.isoformat()
        }


# Global service instance
incremental_service = IncrementalUpdateService()