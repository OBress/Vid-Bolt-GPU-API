"""Job Management Service.

This module provides the JobManager service for handling asynchronous generation jobs.
It implements the "Fail-Fast" architecture using Semaphores to enforce concurrency limits.

Features:
- Semaphore-based concurrency control
- OOM (Out of Memory) error handling with GPU cleanup
- Job timeout enforcement
- Periodic stale job cleanup
- Progress tracking for long-running jobs
"""

import asyncio
import gc
import logging
import time
from typing import Any, Callable, Dict, Optional, Coroutine

from app.config import Settings, InferenceConfig
from app.models.job import JobInfo, JobResult, JobStatus
from app.services.model_manager import ModelMode

logger = logging.getLogger(__name__)


class JobManager:
    """Manages asynchronous generation jobs and concurrency limits."""

    # Job retention settings (hours)
    COMPLETED_JOB_RETENTION_HOURS = 24
    FAILED_JOB_RETENTION_HOURS = 48
    
    # Cleanup interval (seconds)
    CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour

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
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

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

    def update_job_progress(
        self, 
        job_id: str, 
        progress_percent: int, 
        stage: str
    ) -> None:
        """Update job progress for client polling.
        
        Args:
            job_id: The job ID to update
            progress_percent: Progress percentage (0-100)
            stage: Current stage description (e.g., "generating", "upscaling")
        """
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PROCESSING:
            job.progress_percent = min(100, max(0, progress_percent))
            job.progress_stage = stage
            logger.debug(f"Job {job_id} progress: {progress_percent}% - {stage}")

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

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started periodic job cleanup task")

    def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Stopped periodic job cleanup task")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up stale jobs."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                self._cleanup_stale_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Job cleanup error: {e}")

    def _cleanup_stale_jobs(self) -> None:
        """Remove jobs older than retention period."""
        now = time.time()
        completed_cutoff = now - (self.COMPLETED_JOB_RETENTION_HOURS * 3600)
        failed_cutoff = now - (self.FAILED_JOB_RETENTION_HOURS * 3600)
        
        to_remove = []
        for job_id, job in self._jobs.items():
            if job.status == JobStatus.COMPLETED:
                if job.completed_at and job.completed_at < completed_cutoff:
                    to_remove.append(job_id)
            elif job.status == JobStatus.FAILED:
                if job.completed_at and job.completed_at < failed_cutoff:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self._jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} stale jobs")

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
            
        # Try to acquire semaphore immediately (non-blocking / very short timeout)
        # This prevents race conditions where multiple requests pass the .locked() check
        # before the background task starts and acquires the semaphore.
        try:
            # We use a tiny timeout to ensure we don't block the event loop for long
            # but allow a moment to acquire if available.
            await asyncio.wait_for(semaphore.acquire(), timeout=0.01)
        except (asyncio.TimeoutError, asyncio.CancelledError):
             logger.warning(f"Rejecting job {job_id}: {mode} semaphore full (acquisition failed)")
             return False
        
        # Determine timeout based on mode
        timeout_seconds = (
            InferenceConfig.IMAGE_JOB_TIMEOUT if mode == ModelMode.IMAGE
            else InferenceConfig.VIDEO_JOB_TIMEOUT
        )
            
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
            self._run_job_wrapper(
                job_id, 
                semaphore, 
                task_func, 
                timeout_seconds=timeout_seconds,
                already_acquired=True,
                **kwargs
            )
        )
        
        return True

    async def _run_job_wrapper(
        self,
        job_id: str,
        semaphore: asyncio.Semaphore,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        timeout_seconds: float = 600,
        already_acquired: bool = False,
        **kwargs
    ):
        """Wrapper to run the task within the semaphore context.
        
        Handles:
        - Semaphore-based concurrency
        - Job timeout enforcement
        - OOM error detection and GPU cleanup
        - Generic exception handling
        """
        try:
            if not already_acquired:
                await semaphore.acquire()
            
            # The actual work logic
            await self._run_job_internal(job_id, task_func, timeout_seconds, **kwargs)
            
        finally:
            # Always release the semaphore
            semaphore.release()

    async def _run_job_internal(
        self,
        job_id: str,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        timeout_seconds: float = 600,
        **kwargs
    ):
        """Internal job execution logic."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job.progress_percent = 0
        job.progress_stage = "starting"
        
        try:
            logger.info(f"Starting job {job_id} with {timeout_seconds}s timeout")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                task_func(**kwargs),
                timeout=timeout_seconds
            )
            
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            job.progress_percent = 100
            job.progress_stage = "completed"
            
            logger.info(f"Job {job_id} completed successfully")
            
            # Clean up GPU memory after every job to prevent VRAM accumulation
            self._cleanup_gpu_memory()
            
        except asyncio.TimeoutError:
            logger.error(f"Job {job_id} timed out after {timeout_seconds}s")
            job.status = JobStatus.FAILED
            job.completed_at = time.time()
            job.error_message = f"Job timed out after {timeout_seconds} seconds"
            job.error_code = "JOB_TIMEOUT"
            job.progress_stage = "timeout"
            
            # Clean up GPU memory after timeout
            self._cleanup_gpu_memory()
            
        except Exception as e:
            error_message = str(e)
            error_code = "GENERATION_FAILED"
            
            # Check for OOM errors
            is_oom = self._check_oom_error(e, error_message)
            
            if is_oom:
                error_message = "GPU out of memory. Try reducing resolution or duration."
                error_code = "GPU_OUT_OF_MEMORY"
                logger.error(f"Job {job_id} failed with OOM: {e}")
            else:
                logger.exception(f"Job {job_id} failed: {e}")
            
            job.status = JobStatus.FAILED
            job.completed_at = time.time()
            job.error_message = error_message
            job.error_code = error_code
            job.progress_stage = "failed"
            
            # Clean up GPU memory for ALL exceptions
            self._cleanup_gpu_memory()

    def _check_oom_error(self, exception: Exception, error_message: str) -> bool:
        """Check if an exception is an OOM error.
        
        Args:
            exception: The caught exception
            error_message: String representation of the error
            
        Returns:
            True if this is an OOM error
        """
        try:
            import torch
            if isinstance(exception, torch.cuda.OutOfMemoryError):
                return True
        except ImportError:
            pass
        
        # Also check error message for OOM indicators
        oom_indicators = [
            "out of memory",
            "CUDA out of memory",
            "CUDA error: out of memory",
            "OutOfMemoryError",
        ]
        
        error_lower = error_message.lower()
        return any(indicator.lower() in error_lower for indicator in oom_indicators)

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory after an OOM error."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleaned up after OOM")
        except ImportError:
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to clean up GPU memory: {e}")
