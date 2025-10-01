"""
Content Versioning and Diff System

This module provides functionality for tracking content versions over time
and generating detailed diffs between different versions of the same content.
"""

import difflib
import json
import gzip
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import asyncio
from pathlib import Path

from app.core.logging import get_logger
from app.services.incremental_update_service import ContentFingerprint, ChangeType

logger = get_logger(__name__)


class DiffType(Enum):
    """Types of content diffs."""
    TEXT_DIFF = "text_diff"
    HTML_DIFF = "html_diff"
    LINKS_DIFF = "links_diff"
    METADATA_DIFF = "metadata_diff"
    BINARY_DIFF = "binary_diff"


@dataclass
class ContentVersion:
    """Represents a version of content at a specific point in time."""
    url: str
    version_id: str
    fingerprint: ContentFingerprint
    content: str
    html: str
    metadata: Dict[str, Any]
    links: Set[str]
    created_at: datetime
    size_bytes: int
    compression_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['fingerprint'] = self.fingerprint.to_dict()
        data['links'] = list(self.links)  # Convert set to list for JSON
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentVersion':
        """Create instance from dictionary."""
        data = data.copy()
        data['fingerprint'] = ContentFingerprint.from_dict(data['fingerprint'])
        data['links'] = set(data['links'])  # Convert list back to set
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ContentDiff:
    """Represents the difference between two content versions."""
    url: str
    old_version_id: str
    new_version_id: str
    diff_type: DiffType
    changes_summary: Dict[str, Any]
    detailed_diff: str
    similarity_score: float  # 0.0 to 1.0
    change_magnitude: int  # Number of changed lines/elements
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['diff_type'] = self.diff_type.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class VersionHistory:
    """Represents the version history for a URL."""
    url: str
    versions: List[ContentVersion]
    total_versions: int
    oldest_version: datetime
    newest_version: datetime
    total_size_bytes: int

    def get_version(self, version_id: str) -> Optional[ContentVersion]:
        """Get a specific version by ID."""
        for version in self.versions:
            if version.version_id == version_id:
                return version
        return None

    def get_latest_version(self) -> Optional[ContentVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    def get_versions_in_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ContentVersion]:
        """Get versions within a date range."""
        return [
            v for v in self.versions
            if start_date <= v.created_at <= end_date
        ]


class TextDiffer:
    """Utility class for generating text diffs."""

    @staticmethod
    def generate_unified_diff(
        old_text: str,
        new_text: str,
        context_lines: int = 3
    ) -> str:
        """Generate unified diff between two text strings."""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='old',
            tofile='new',
            n=context_lines
        )

        return ''.join(diff)

    @staticmethod
    def generate_html_diff(old_text: str, new_text: str) -> str:
        """Generate HTML diff for web display."""
        differ = difflib.HtmlDiff()
        return differ.make_file(
            old_text.splitlines(),
            new_text.splitlines(),
            fromdesc='Previous Version',
            todesc='Current Version'
        )

    @staticmethod
    def calculate_similarity(old_text: str, new_text: str) -> float:
        """Calculate similarity ratio between two texts."""
        return difflib.SequenceMatcher(None, old_text, new_text).ratio()

    @staticmethod
    def get_change_summary(old_text: str, new_text: str) -> Dict[str, Any]:
        """Get summary of changes between two texts."""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        # Use SequenceMatcher to get detailed change information
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        opcodes = matcher.get_opcodes()

        lines_added = 0
        lines_deleted = 0
        lines_modified = 0

        for op, i1, i2, j1, j2 in opcodes:
            if op == 'delete':
                lines_deleted += i2 - i1
            elif op == 'insert':
                lines_added += j2 - j1
            elif op == 'replace':
                lines_modified += max(i2 - i1, j2 - j1)

        return {
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'lines_modified': lines_modified,
            'total_changes': lines_added + lines_deleted + lines_modified,
            'old_line_count': len(old_lines),
            'new_line_count': len(new_lines),
            'similarity_ratio': matcher.ratio()
        }


class LinksDiffer:
    """Utility class for analyzing link changes."""

    @staticmethod
    def analyze_link_changes(old_links: Set[str], new_links: Set[str]) -> Dict[str, Any]:
        """Analyze changes in page links."""
        added_links = new_links - old_links
        removed_links = old_links - new_links
        unchanged_links = old_links & new_links

        return {
            'added_links': list(added_links),
            'removed_links': list(removed_links),
            'unchanged_links': list(unchanged_links),
            'total_added': len(added_links),
            'total_removed': len(removed_links),
            'total_unchanged': len(unchanged_links),
            'change_ratio': len(added_links | removed_links) / max(len(old_links | new_links), 1)
        }


class MetadataDiffer:
    """Utility class for analyzing metadata changes."""

    @staticmethod
    def analyze_metadata_changes(
        old_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze changes in page metadata."""
        old_keys = set(old_metadata.keys())
        new_keys = set(new_metadata.keys())

        added_keys = new_keys - old_keys
        removed_keys = old_keys - new_keys
        common_keys = old_keys & new_keys

        changed_values = {}
        for key in common_keys:
            if old_metadata[key] != new_metadata[key]:
                changed_values[key] = {
                    'old_value': old_metadata[key],
                    'new_value': new_metadata[key]
                }

        return {
            'added_keys': list(added_keys),
            'removed_keys': list(removed_keys),
            'changed_values': changed_values,
            'total_changes': len(added_keys) + len(removed_keys) + len(changed_values)
        }


class ContentVersioningService:
    """Main service for managing content versions and diffs."""

    def __init__(self, storage_path: str = "/tmp/content_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.version_histories: Dict[str, VersionHistory] = {}
        self.version_cache: Dict[str, ContentVersion] = {}

        # Configuration
        self.max_versions_per_url = 50
        self.max_version_age_days = 365
        self.compression_enabled = True
        self.compression_threshold_bytes = 1024  # Compress content larger than 1KB

        # Initialize storage
        asyncio.create_task(self._initialize_storage())

    async def _initialize_storage(self):
        """Initialize storage and load existing data."""
        await self.load_version_histories()

    async def store_version(
        self,
        url: str,
        fingerprint: ContentFingerprint,
        content: str,
        html: str,
        metadata: Dict[str, Any],
        links: Set[str]
    ) -> ContentVersion:
        """Store a new version of content."""

        # Generate version ID
        version_id = f"{fingerprint.timestamp.strftime('%Y%m%d_%H%M%S')}_{fingerprint.content_hash[:8]}"

        # Calculate content size
        content_size = len(content.encode('utf-8'))

        # Create version
        version = ContentVersion(
            url=url,
            version_id=version_id,
            fingerprint=fingerprint,
            content=content,
            html=html,
            metadata=metadata,
            links=links,
            created_at=fingerprint.timestamp,
            size_bytes=content_size
        )

        # Store version data to disk
        await self._store_version_data(version)

        # Update version history
        if url not in self.version_histories:
            self.version_histories[url] = VersionHistory(
                url=url,
                versions=[],
                total_versions=0,
                oldest_version=version.created_at,
                newest_version=version.created_at,
                total_size_bytes=0
            )

        history = self.version_histories[url]
        history.versions.append(version)
        history.total_versions += 1
        history.newest_version = version.created_at
        history.total_size_bytes += content_size

        # Clean up old versions if needed
        await self._cleanup_old_versions(url)

        # Cache the version
        self.version_cache[version_id] = version

        logger.info(f"Stored version {version_id} for URL {url}")
        return version

    async def _store_version_data(self, version: ContentVersion):
        """Store version data to disk with optional compression."""
        version_dir = self.storage_path / self._url_to_path(version.url)
        version_dir.mkdir(parents=True, exist_ok=True)

        version_file = version_dir / f"{version.version_id}.json"

        # Prepare data for storage
        version_data = {
            'url': version.url,
            'version_id': version.version_id,
            'fingerprint': version.fingerprint.to_dict(),
            'metadata': version.metadata,
            'links': list(version.links),
            'created_at': version.created_at.isoformat(),
            'size_bytes': version.size_bytes
        }

        # Store content separately if compression is enabled
        if self.compression_enabled and version.size_bytes > self.compression_threshold_bytes:
            # Store compressed content
            content_file = version_dir / f"{version.version_id}_content.gz"
            html_file = version_dir / f"{version.version_id}_html.gz"

            async with aiofiles.open(content_file, 'wb') as f:
                compressed_content = gzip.compress(version.content.encode('utf-8'))
                await f.write(compressed_content)

            async with aiofiles.open(html_file, 'wb') as f:
                compressed_html = gzip.compress(version.html.encode('utf-8'))
                await f.write(compressed_html)

            version_data['content_compressed'] = True
            version_data['compression_ratio'] = len(compressed_content) / version.size_bytes
        else:
            # Store uncompressed content inline
            version_data['content'] = version.content
            version_data['html'] = version.html
            version_data['content_compressed'] = False

        # Write version metadata
        async with aiofiles.open(version_file, 'w') as f:
            await f.write(json.dumps(version_data, indent=2))

    async def _load_version_data(self, url: str, version_id: str) -> Optional[ContentVersion]:
        """Load version data from disk."""
        version_dir = self.storage_path / self._url_to_path(url)
        version_file = version_dir / f"{version_id}.json"

        try:
            async with aiofiles.open(version_file, 'r') as f:
                content = await f.read()
                version_data = json.loads(content)

            # Load content based on compression
            if version_data.get('content_compressed', False):
                # Load compressed content
                content_file = version_dir / f"{version_id}_content.gz"
                html_file = version_dir / f"{version_id}_html.gz"

                async with aiofiles.open(content_file, 'rb') as f:
                    compressed_content = await f.read()
                    content = gzip.decompress(compressed_content).decode('utf-8')

                async with aiofiles.open(html_file, 'rb') as f:
                    compressed_html = await f.read()
                    html = gzip.decompress(compressed_html).decode('utf-8')
            else:
                content = version_data['content']
                html = version_data['html']

            # Create version object
            version = ContentVersion(
                url=version_data['url'],
                version_id=version_data['version_id'],
                fingerprint=ContentFingerprint.from_dict(version_data['fingerprint']),
                content=content,
                html=html,
                metadata=version_data['metadata'],
                links=set(version_data['links']),
                created_at=datetime.fromisoformat(version_data['created_at']),
                size_bytes=version_data['size_bytes'],
                compression_ratio=version_data.get('compression_ratio', 1.0)
            )

            return version

        except FileNotFoundError:
            logger.warning(f"Version file not found: {version_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading version {version_id}: {e}")
            return None

    def _url_to_path(self, url: str) -> str:
        """Convert URL to safe filesystem path."""
        import urllib.parse
        import re

        # Parse URL and create safe path
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.replace(':', '_')
        path = parsed.path.replace('/', '_')

        # Remove invalid characters
        safe_path = re.sub(r'[^\w\-_.]', '_', f"{domain}{path}")
        return safe_path[:100]  # Limit length

    async def get_version(self, url: str, version_id: str) -> Optional[ContentVersion]:
        """Get a specific version."""
        # Check cache first
        if version_id in self.version_cache:
            return self.version_cache[version_id]

        # Load from disk
        version = await self._load_version_data(url, version_id)
        if version:
            self.version_cache[version_id] = version

        return version

    async def get_version_history(self, url: str) -> Optional[VersionHistory]:
        """Get version history for a URL."""
        return self.version_histories.get(url)

    async def generate_diff(
        self,
        url: str,
        old_version_id: str,
        new_version_id: str,
        diff_type: DiffType = DiffType.TEXT_DIFF
    ) -> Optional[ContentDiff]:
        """Generate diff between two versions."""

        old_version = await self.get_version(url, old_version_id)
        new_version = await self.get_version(url, new_version_id)

        if not old_version or not new_version:
            return None

        # Generate diff based on type
        if diff_type == DiffType.TEXT_DIFF:
            detailed_diff = TextDiffer.generate_unified_diff(
                old_version.content,
                new_version.content
            )
            changes_summary = TextDiffer.get_change_summary(
                old_version.content,
                new_version.content
            )
            similarity_score = changes_summary['similarity_ratio']
            change_magnitude = changes_summary['total_changes']

        elif diff_type == DiffType.HTML_DIFF:
            detailed_diff = TextDiffer.generate_html_diff(
                old_version.html,
                new_version.html
            )
            changes_summary = TextDiffer.get_change_summary(
                old_version.html,
                new_version.html
            )
            similarity_score = changes_summary['similarity_ratio']
            change_magnitude = changes_summary['total_changes']

        elif diff_type == DiffType.LINKS_DIFF:
            changes_summary = LinksDiffer.analyze_link_changes(
                old_version.links,
                new_version.links
            )
            detailed_diff = json.dumps(changes_summary, indent=2)
            similarity_score = 1.0 - changes_summary['change_ratio']
            change_magnitude = changes_summary['total_added'] + changes_summary['total_removed']

        elif diff_type == DiffType.METADATA_DIFF:
            changes_summary = MetadataDiffer.analyze_metadata_changes(
                old_version.metadata,
                new_version.metadata
            )
            detailed_diff = json.dumps(changes_summary, indent=2)
            similarity_score = 1.0 - (changes_summary['total_changes'] / max(
                len(old_version.metadata) + len(new_version.metadata), 1))
            change_magnitude = changes_summary['total_changes']

        else:
            return None

        return ContentDiff(
            url=url,
            old_version_id=old_version_id,
            new_version_id=new_version_id,
            diff_type=diff_type,
            changes_summary=changes_summary,
            detailed_diff=detailed_diff,
            similarity_score=similarity_score,
            change_magnitude=change_magnitude,
            created_at=datetime.now(timezone.utc)
        )

    async def _cleanup_old_versions(self, url: str):
        """Clean up old versions based on configured limits."""
        history = self.version_histories.get(url)
        if not history:
            return

        # Sort versions by creation time (newest first)
        history.versions.sort(key=lambda v: v.created_at, reverse=True)

        # Apply version count limit
        if len(history.versions) > self.max_versions_per_url:
            versions_to_remove = history.versions[self.max_versions_per_url:]
            history.versions = history.versions[:self.max_versions_per_url]

            # Remove files for deleted versions
            for version in versions_to_remove:
                await self._remove_version_files(version)

        # Apply age limit
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_version_age_days)
        old_versions = [v for v in history.versions if v.created_at < cutoff_date]

        if old_versions:
            # Keep at least one version
            if len(history.versions) - len(old_versions) >= 1:
                for version in old_versions:
                    await self._remove_version_files(version)
                    history.versions.remove(version)

        # Update history metadata
        if history.versions:
            history.total_versions = len(history.versions)
            history.oldest_version = min(v.created_at for v in history.versions)
            history.newest_version = max(v.created_at for v in history.versions)
            history.total_size_bytes = sum(v.size_bytes for v in history.versions)

    async def _remove_version_files(self, version: ContentVersion):
        """Remove version files from disk."""
        version_dir = self.storage_path / self._url_to_path(version.url)
        version_file = version_dir / f"{version.version_id}.json"

        try:
            # Remove main version file
            if version_file.exists():
                version_file.unlink()

            # Remove compressed content files if they exist
            content_file = version_dir / f"{version.version_id}_content.gz"
            html_file = version_dir / f"{version.version_id}_html.gz"

            if content_file.exists():
                content_file.unlink()
            if html_file.exists():
                html_file.unlink()

            # Remove from cache
            if version.version_id in self.version_cache:
                del self.version_cache[version.version_id]

        except Exception as e:
            logger.error(f"Error removing version files for {version.version_id}: {e}")

    async def save_version_histories(self):
        """Save version histories metadata to disk."""
        histories_file = self.storage_path / "version_histories.json"

        data = {}
        for url, history in self.version_histories.items():
            data[url] = {
                'url': history.url,
                'total_versions': history.total_versions,
                'oldest_version': history.oldest_version.isoformat(),
                'newest_version': history.newest_version.isoformat(),
                'total_size_bytes': history.total_size_bytes,
                'version_ids': [v.version_id for v in history.versions]
            }

        async with aiofiles.open(histories_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def load_version_histories(self):
        """Load version histories metadata from disk."""
        histories_file = self.storage_path / "version_histories.json"

        try:
            async with aiofiles.open(histories_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            for url, history_data in data.items():
                # Create minimal version objects for the history
                versions = []
                for version_id in history_data['version_ids']:
                    # We'll load full version data on demand
                    versions.append(type('MinimalVersion', (), {
                        'version_id': version_id,
                        'created_at': datetime.now()  # Will be updated when loaded
                    })())

                self.version_histories[url] = VersionHistory(
                    url=history_data['url'],
                    versions=versions,
                    total_versions=history_data['total_versions'],
                    oldest_version=datetime.fromisoformat(history_data['oldest_version']),
                    newest_version=datetime.fromisoformat(history_data['newest_version']),
                    total_size_bytes=history_data['total_size_bytes']
                )

            logger.info(f"Loaded {len(self.version_histories)} version histories")

        except FileNotFoundError:
            logger.info("No existing version histories found")
        except Exception as e:
            logger.error(f"Error loading version histories: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the versioning system."""
        total_versions = sum(h.total_versions for h in self.version_histories.values())
        total_size = sum(h.total_size_bytes for h in self.version_histories.values())

        return {
            'total_urls_tracked': len(self.version_histories),
            'total_versions_stored': total_versions,
            'total_storage_bytes': total_size,
            'cached_versions': len(self.version_cache),
            'compression_enabled': self.compression_enabled,
            'max_versions_per_url': self.max_versions_per_url,
            'max_version_age_days': self.max_version_age_days
        }


# Global service instance
versioning_service = ContentVersioningService()