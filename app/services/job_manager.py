"""Job Management Service.

This module provides the JobManager service for handling asynchronous generation jobs.
It implements the "Fail-Fast" architecture using Semaphores to enforce concurrency limits.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Coroutine

from app.config import Settings
from app.models.job import JobInfo, JobResult, JobStatus
from app.services.model_manager import ModelMode

logger = logging.getLogger(__name__)


class JobManager:
    """Manages asynchronous generation jobs and concurrency limits."""

    def __init__(self, settings: Settings):
        """Initialize the JobManager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Concurrency Semaphores
        # These are lazily initialized since we need an event loop
        self._image_semaphore: Optional[asyncio.Semaphore] = None
        self._video_semaphore: Optional[asyncio.Semaphore] = None
        
        # In-memory job store (job_id -> JobInfo)
        # Note: In a production cluster, this should be Redis/Database
        # For a single GPU worker, in-memory is sufficient
        self._jobs: Dict[str, JobInfo] = {}
        
        # Limit the size of history to prevent memory leaks
        self.MAX_HISTORY = 1000

    @property
    def image_semaphore(self) -> asyncio.Semaphore:
        """Get or initialize image semaphore."""
        if self._image_semaphore is None:
            limit = self.settings.max_concurrent_image_generations
            logger.info(f"Initializing Image Semaphore with limit: {limit}")
            self._image_semaphore = asyncio.Semaphore(limit)
        return self._image_semaphore

    @property
    def video_semaphore(self) -> asyncio.Semaphore:
        """Get or initialize video semaphore."""
        if self._video_semaphore is None:
            limit = self.settings.max_concurrent_video_generations
            logger.info(f"Initializing Video Semaphore with limit: {limit}")
            self._video_semaphore = asyncio.Semaphore(limit)
        return self._video_semaphore

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information by ID."""
        return self._jobs.get(job_id)

    async def list_active_jobs(self) -> list[JobInfo]:
        """List currently processing jobs."""
        return [
            job for job in self._jobs.values() 
            if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
        ]

    def _cleanup_old_jobs(self):
        """Cleanup old completed/failed jobs to free memory."""
        if len(self._jobs) <= self.MAX_HISTORY:
            return
            
        # Sort by creation time
        sorted_jobs = sorted(
            self._jobs.items(), 
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest, keeping MAX_HISTORY
        # Only remove completed/failed jobs
        to_remove = []
        keep_count = 0
        target = len(self._jobs) - self.MAX_HISTORY
        
        for job_id, job in sorted_jobs:
            if keep_count >= target:
                break
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                to_remove.append(job_id)
                keep_count += 1
                
        for job_id in to_remove:
            del self._jobs[job_id]

    async def try_submit_job(
        self,
        job_id: str,
        mode: ModelMode,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        **kwargs
    ) -> bool:
        """Try to submit a job for execution.
        
        Implements "Fail-Fast" logic:
        1. Check if semaphore is available
        2. If avail, acquire slot and start background task
        3. If full, return False immediately
        
        Args:
            job_id: Unique job ID
            mode: Target mode (IMAGE or VIDEO)
            task_func: Async function to execute
            **kwargs: Arguments for task_func
            
        Returns:
            True if accepted, False if busy
        """
        semaphore = (
            self.image_semaphore if mode == ModelMode.IMAGE 
            else self.video_semaphore
        )
        
        if semaphore.locked():
            logger.warning(f"Rejecting job {job_id}: {mode} semaphore full")
            return False
            
        # Initialize job record
        self._jobs[job_id] = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
        )
        self._cleanup_old_jobs()
        
        # Start the background wrapper
        # We don't await this; it runs in the background
        asyncio.create_task(
            self._run_job_wrapper(job_id, semaphore, task_func, **kwargs)
        )
        
        return True

    async def _run_job_wrapper(
        self,
        job_id: str,
        semaphore: asyncio.Semaphore,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        **kwargs
    ):
        """Wrapper to run the task within the semaphore context."""
        async with semaphore:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            
            try:
                logger.info(f"Starting job {job_id}")
                
                # Execute the actual generation task
                # The task_func is expected to return a JobResult
                result = await task_func(**kwargs)
                
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.result = result
                
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                logger.exception(f"Job {job_id} failed: {e}")
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error_message = str(e)
