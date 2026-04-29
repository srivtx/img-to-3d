"""In-memory job queue with status tracking."""

import asyncio
import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field
from app.core.models import JobStatus


@dataclass
class Job:
    job_id: str
    status: JobStatus
    image_path: str
    preview_path: Optional[str] = None
    final_path: Optional[str] = None
    progress_percent: int = 0
    message: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    semaphore: Optional[asyncio.Semaphore] = None


class JobQueue:
    """Simple in-memory job queue. Swap for Redis/Celery later."""
    
    def __init__(self, max_concurrent: int = 2):
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
    
    async def create_job(self, image_path: str) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            image_path=image_path,
            message="Queued for processing",
            semaphore=self._semaphore,
        )
        async with self._lock:
            self.jobs[job_id] = job
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        async with self._lock:
            return self.jobs.get(job_id)
    
    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        image_path: Optional[str] = None,
        preview_path: Optional[str] = None,
        final_path: Optional[str] = None,
        progress_percent: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update job fields."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            if status:
                job.status = status
            if image_path:
                job.image_path = image_path
            if preview_path:
                job.preview_path = preview_path
            if final_path:
                job.final_path = final_path
            if progress_percent is not None:
                job.progress_percent = progress_percent
            if message:
                job.message = message
            if error:
                job.error = error
            job.updated_at = time.time()
    
    async def cleanup_old_jobs(self, ttl_seconds: int = 3600):
        """Remove jobs older than ttl."""
        now = time.time()
        async with self._lock:
            old = [jid for jid, j in self.jobs.items() if now - j.created_at > ttl_seconds]
            for jid in old:
                del self.jobs[jid]
    
    def get_stats(self) -> Dict:
        """Return queue statistics."""
        active = sum(1 for j in self.jobs.values() if j.status in {JobStatus.PROCESSING_COARSE, JobStatus.REFINING})
        queued = sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)
        return {"active": active, "queued": queued, "total": len(self.jobs)}


# Global queue instance
queue = JobQueue()
