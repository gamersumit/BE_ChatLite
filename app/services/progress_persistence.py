"""
Progress Persistence and Recovery Service

This module provides persistence and recovery capabilities for progress tracking,
ensuring that progress data survives system restarts and failures.
"""

import asyncio
import json
import pickle
import sqlite3
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import aiofiles
import aiofiles.os

from app.services.progress_tracking_service import (
    JobProgressTracker,
    JobProgressSnapshot,
    ProgressEvent,
    ProgressAlert,
    JobStatus,
    ProgressPhase,
    AlertLevel
)

logger = logging.getLogger(__name__)


class ProgressPersistenceService:
    """Service for persisting and recovering progress data."""

    def __init__(self, data_dir: str = "data/progress"):
        self.data_dir = Path(data_dir)
        self.snapshots_db = self.data_dir / "progress_snapshots.db"
        self.events_db = self.data_dir / "progress_events.db"
        self.alerts_db = self.data_dir / "progress_alerts.db"
        self.checkpoints_dir = self.data_dir / "checkpoints"

        self._initialized = False
        self._checkpoint_interval = 300  # 5 minutes
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize the persistence service."""
        if self._initialized:
            return

        # Create directories
        await aiofiles.os.makedirs(self.data_dir, exist_ok=True)
        await aiofiles.os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Initialize databases
        await self._init_databases()

        # Start periodic checkpointing
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        self._initialized = True
        logger.info("Progress persistence service initialized")

    async def shutdown(self):
        """Shutdown the persistence service."""
        self._shutdown_event.set()

        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        logger.info("Progress persistence service shutdown")

    async def _init_databases(self):
        """Initialize SQLite databases for persistence."""

        # Snapshots database
        async with aiofiles.open(self.snapshots_db, 'a'):
            pass  # Create file if not exists

        def create_snapshots_schema():
            conn = sqlite3.connect(str(self.snapshots_db))
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_snapshots (
                    job_id TEXT,
                    snapshot_time TEXT,
                    status TEXT,
                    current_phase TEXT,
                    snapshot_data TEXT,
                    PRIMARY KEY (job_id, snapshot_time)
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_job_snapshots_job_id
                ON job_snapshots(job_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_job_snapshots_time
                ON job_snapshots(snapshot_time)
            ''')

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, create_snapshots_schema)

        # Events database
        async with aiofiles.open(self.events_db, 'a'):
            pass

        def create_events_schema():
            conn = sqlite3.connect(str(self.events_db))
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_events (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    event_data TEXT
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_progress_events_job_id
                ON progress_events(job_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_progress_events_timestamp
                ON progress_events(timestamp)
            ''')

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, create_events_schema)

        # Alerts database
        async with aiofiles.open(self.alerts_db, 'a'):
            pass

        def create_alerts_schema():
            conn = sqlite3.connect(str(self.alerts_db))
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_alerts (
                    alert_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    level TEXT,
                    message TEXT,
                    timestamp TEXT,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_progress_alerts_job_id
                ON progress_alerts(job_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_progress_alerts_timestamp
                ON progress_alerts(timestamp)
            ''')

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, create_alerts_schema)

    async def save_job_snapshot(self, snapshot: JobProgressSnapshot):
        """Save a job progress snapshot to persistent storage."""
        def save_snapshot():
            conn = sqlite3.connect(str(self.snapshots_db))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO job_snapshots
                (job_id, snapshot_time, status, current_phase, snapshot_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                snapshot.job_id,
                snapshot.snapshot_time.isoformat(),
                snapshot.status.value,
                snapshot.current_phase.value,
                json.dumps(snapshot.to_dict())
            ))

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, save_snapshot)

    async def load_job_snapshot(self, job_id: str) -> Optional[JobProgressSnapshot]:
        """Load the latest job snapshot from persistent storage."""
        def load_snapshot():
            conn = sqlite3.connect(str(self.snapshots_db))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT snapshot_data FROM job_snapshots
                WHERE job_id = ?
                ORDER BY snapshot_time DESC
                LIMIT 1
            ''', (job_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])
            return None

        snapshot_data = await asyncio.get_event_loop().run_in_executor(None, load_snapshot)

        if snapshot_data:
            return self._deserialize_snapshot(snapshot_data)
        return None

    async def save_progress_event(self, event: ProgressEvent):
        """Save a progress event to persistent storage."""
        def save_event():
            conn = sqlite3.connect(str(self.events_db))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO progress_events
                (event_id, job_id, event_type, timestamp, event_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.job_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                json.dumps(event.to_dict())
            ))

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, save_event)

    async def load_progress_events(self, job_id: str, limit: int = 1000) -> List[ProgressEvent]:
        """Load progress events for a job from persistent storage."""
        def load_events():
            conn = sqlite3.connect(str(self.events_db))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT event_data FROM progress_events
                WHERE job_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (job_id, limit))

            results = cursor.fetchall()
            conn.close()

            return [json.loads(row[0]) for row in results]

        events_data = await asyncio.get_event_loop().run_in_executor(None, load_events)

        events = []
        for event_data in events_data:
            try:
                event = self._deserialize_event(event_data)
                events.append(event)
            except Exception as e:
                logger.error(f"Error deserializing event: {e}")

        return events

    async def save_progress_alert(self, alert: ProgressAlert):
        """Save a progress alert to persistent storage."""
        def save_alert():
            conn = sqlite3.connect(str(self.alerts_db))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO progress_alerts
                (alert_id, job_id, level, message, timestamp, details, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.job_id,
                alert.level.value,
                alert.message,
                alert.timestamp.isoformat(),
                json.dumps(alert.details),
                1 if alert.acknowledged else 0
            ))

            conn.commit()
            conn.close()

        await asyncio.get_event_loop().run_in_executor(None, save_alert)

    async def load_progress_alerts(self, job_id: Optional[str] = None, limit: int = 1000) -> List[ProgressAlert]:
        """Load progress alerts from persistent storage."""
        def load_alerts():
            conn = sqlite3.connect(str(self.alerts_db))
            cursor = conn.cursor()

            if job_id:
                cursor.execute('''
                    SELECT alert_id, job_id, level, message, timestamp, details, acknowledged
                    FROM progress_alerts
                    WHERE job_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (job_id, limit))
            else:
                cursor.execute('''
                    SELECT alert_id, job_id, level, message, timestamp, details, acknowledged
                    FROM progress_alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

            results = cursor.fetchall()
            conn.close()

            return results

        alerts_data = await asyncio.get_event_loop().run_in_executor(None, load_alerts)

        alerts = []
        for row in alerts_data:
            try:
                alert = ProgressAlert(
                    alert_id=row[0],
                    job_id=row[1],
                    level=AlertLevel(row[2]),
                    message=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    details=json.loads(row[5]),
                    acknowledged=bool(row[6])
                )
                alerts.append(alert)
            except Exception as e:
                logger.error(f"Error deserializing alert: {e}")

        return alerts

    async def create_checkpoint(self, job_tracker: JobProgressTracker):
        """Create a checkpoint for job progress."""
        checkpoint_file = self.checkpoints_dir / f"{job_tracker.job_id}_checkpoint.pkl"

        try:
            # Serialize the job tracker
            checkpoint_data = {
                'job_id': job_tracker.job_id,
                'metrics': job_tracker.metrics,
                'phases': job_tracker.phases,
                'current_phase': job_tracker.current_phase,
                'status': job_tracker.status,
                'started_at': job_tracker.started_at,
                'completed_at': job_tracker.completed_at,
                'error_messages': job_tracker.error_messages,
                'warnings': job_tracker.warnings,
                'checkpoint_time': datetime.now(timezone.utc)
            }

            async with aiofiles.open(checkpoint_file, 'wb') as f:
                await f.write(pickle.dumps(checkpoint_data))

            logger.debug(f"Created checkpoint for job {job_tracker.job_id}")

        except Exception as e:
            logger.error(f"Error creating checkpoint for job {job_tracker.job_id}: {e}")

    async def load_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a job."""
        checkpoint_file = self.checkpoints_dir / f"{job_id}_checkpoint.pkl"

        try:
            if await aiofiles.os.path.exists(checkpoint_file):
                async with aiofiles.open(checkpoint_file, 'rb') as f:
                    data = await f.read()
                    checkpoint_data = pickle.loads(data)

                logger.info(f"Loaded checkpoint for job {job_id}")
                return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint for job {job_id}: {e}")

        return None

    async def restore_job_tracker(self, job_id: str) -> Optional[JobProgressTracker]:
        """Restore a job tracker from checkpoint and persistent data."""
        try:
            # Load checkpoint data
            checkpoint_data = await self.load_checkpoint(job_id)
            if not checkpoint_data:
                logger.info(f"No checkpoint found for job {job_id}")
                return None

            # Create new tracker
            tracker = JobProgressTracker(
                job_id=job_id,
                total_urls=checkpoint_data['metrics'].total_urls
            )

            # Restore state
            tracker.metrics = checkpoint_data['metrics']
            tracker.phases = checkpoint_data['phases']
            tracker.current_phase = checkpoint_data['current_phase']
            tracker.status = checkpoint_data['status']
            tracker.started_at = checkpoint_data['started_at']
            tracker.completed_at = checkpoint_data['completed_at']
            tracker.error_messages = checkpoint_data['error_messages']
            tracker.warnings = checkpoint_data['warnings']

            logger.info(f"Restored job tracker for job {job_id} from checkpoint")
            return tracker

        except Exception as e:
            logger.error(f"Error restoring job tracker for job {job_id}: {e}")
            return None

    async def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old progress data."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()

        def cleanup_database(db_path: str, table: str, timestamp_column: str):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(f'''
                DELETE FROM {table}
                WHERE {timestamp_column} < ?
            ''', (cutoff_iso,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            return deleted_count

        try:
            # Cleanup snapshots
            snapshots_deleted = await asyncio.get_event_loop().run_in_executor(
                None, cleanup_database, str(self.snapshots_db), "job_snapshots", "snapshot_time"
            )

            # Cleanup events
            events_deleted = await asyncio.get_event_loop().run_in_executor(
                None, cleanup_database, str(self.events_db), "progress_events", "timestamp"
            )

            # Cleanup alerts
            alerts_deleted = await asyncio.get_event_loop().run_in_executor(
                None, cleanup_database, str(self.alerts_db), "progress_alerts", "timestamp"
            )

            # Cleanup old checkpoint files
            checkpoints_deleted = 0
            if await aiofiles.os.path.exists(self.checkpoints_dir):
                async for checkpoint_file in aiofiles.os.listdir(self.checkpoints_dir):
                    file_path = self.checkpoints_dir / checkpoint_file
                    try:
                        stat = await aiofiles.os.stat(file_path)
                        file_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                        if file_time < cutoff_date:
                            await aiofiles.os.remove(file_path)
                            checkpoints_deleted += 1
                    except Exception:
                        pass

            logger.info(f"Cleanup completed: {snapshots_deleted} snapshots, "
                       f"{events_deleted} events, {alerts_deleted} alerts, "
                       f"{checkpoints_deleted} checkpoints deleted")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        def get_db_stats(db_path: str, table: str):
            if not os.path.exists(db_path):
                return {"count": 0, "size_mb": 0}

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            conn.close()

            size_bytes = os.path.getsize(db_path)
            size_mb = size_bytes / (1024 * 1024)

            return {"count": count, "size_mb": round(size_mb, 2)}

        try:
            snapshots_stats = await asyncio.get_event_loop().run_in_executor(
                None, get_db_stats, str(self.snapshots_db), "job_snapshots"
            )

            events_stats = await asyncio.get_event_loop().run_in_executor(
                None, get_db_stats, str(self.events_db), "progress_events"
            )

            alerts_stats = await asyncio.get_event_loop().run_in_executor(
                None, get_db_stats, str(self.alerts_db), "progress_alerts"
            )

            # Count checkpoint files
            checkpoint_count = 0
            checkpoint_size = 0
            if await aiofiles.os.path.exists(self.checkpoints_dir):
                async for checkpoint_file in aiofiles.os.listdir(self.checkpoints_dir):
                    file_path = self.checkpoints_dir / checkpoint_file
                    try:
                        stat = await aiofiles.os.stat(file_path)
                        checkpoint_count += 1
                        checkpoint_size += stat.st_size
                    except Exception:
                        pass

            return {
                "snapshots": snapshots_stats,
                "events": events_stats,
                "alerts": alerts_stats,
                "checkpoints": {
                    "count": checkpoint_count,
                    "size_mb": round(checkpoint_size / (1024 * 1024), 2)
                },
                "total_size_mb": round(
                    snapshots_stats["size_mb"] +
                    events_stats["size_mb"] +
                    alerts_stats["size_mb"] +
                    (checkpoint_size / (1024 * 1024)), 2
                )
            }

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    async def _checkpoint_loop(self):
        """Periodic checkpointing loop."""
        from app.services.progress_tracking_service import progress_service

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._checkpoint_interval)

                # Create checkpoints for all active jobs
                for job_id, tracker in progress_service.job_trackers.items():
                    if tracker.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
                        await self.create_checkpoint(tracker)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")

    def _deserialize_snapshot(self, data: Dict[str, Any]) -> JobProgressSnapshot:
        """Deserialize a progress snapshot from stored data."""
        # This would need to properly reconstruct the snapshot object
        # For now, returning a mock implementation
        from app.services.progress_tracking_service import JobProgressSnapshot, ProgressMetrics, PhaseProgress

        # Implementation would deserialize the complete snapshot
        # This is a simplified version
        snapshot = JobProgressSnapshot(
            job_id=data["job_id"],
            snapshot_time=datetime.fromisoformat(data["snapshot_time"]),
            status=JobStatus(data["status"]),
            current_phase=ProgressPhase(data["current_phase"]),
            metrics=ProgressMetrics(),  # Would deserialize metrics
            phases={},  # Would deserialize phases
            error_messages=data.get("error_messages", []),
            warnings=data.get("warnings", []),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            estimated_completion=datetime.fromisoformat(data["estimated_completion"]) if data.get("estimated_completion") else None
        )

        return snapshot

    def _deserialize_event(self, data: Dict[str, Any]) -> ProgressEvent:
        """Deserialize a progress event from stored data."""
        from app.services.progress_tracking_service import ProgressEvent, ProgressEventType

        return ProgressEvent(
            event_id=data["event_id"],
            job_id=data["job_id"],
            event_type=ProgressEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {})
        )


# Global persistence service instance
persistence_service = ProgressPersistenceService()