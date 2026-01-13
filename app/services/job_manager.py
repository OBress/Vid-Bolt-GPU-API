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
    """Manages asynchronous generation jobs with dynamic queue scheduling."""

    # Job retention settings (hours)
    COMPLETED_JOB_RETENTION_HOURS = 24
    FAILED_JOB_RETENTION_HOURS = 48
    
    # Cleanup interval (seconds)
    CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour

    def __init__(self, settings: Settings, model_manager: Optional["ModelManager"] = None):
        """Initialize the JobManager.
        
        Args:
            settings: Application settings
            model_manager: ModelManager instance (for mode awareness)
        """
        self.settings = settings
        self._model_manager = model_manager
        
        # Job Queue
        # We use a simple list + condition variable instead of asyncio.Queue
        # so we can peek and select jobs based on scheduling logic (Grouping vs FIFO)
        self._pending_jobs: list[str] = []
        self._condition = asyncio.Condition()
        
        # In-memory job store (job_id -> JobInfo)
        self._jobs: Dict[str, JobInfo] = {}
        
        # Limit the size of history to prevent memory leaks
        self.MAX_HISTORY = 1000
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def set_model_manager(self, model_manager: "ModelManager"):
        """Set the ModelManager instance (circular dependency check)."""
        self._model_manager = model_manager

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
        """Update job progress for client polling."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PROCESSING:
            job.progress_percent = min(100, max(0, progress_percent))
            job.progress_stage = stage
            logger.debug(f"Job {job_id} progress: {progress_percent}% - {stage}")

    def _cleanup_old_jobs(self):
        """Cleanup old completed/failed jobs to free memory."""
        if len(self._jobs) <= self.MAX_HISTORY:
            return
            
        # Remove oldest completed/failed jobs
        to_remove = []
        keep_count = 0
        target = len(self._jobs) - self.MAX_HISTORY
        
        # Sort by creation time
        sorted_jobs = sorted(self._jobs.items(), key=lambda x: x[1].created_at)
        
        for job_id, job in sorted_jobs:
            if keep_count >= target:
                break
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                to_remove.append(job_id)
                keep_count += 1
                
        for job_id in to_remove:
            del self._jobs[job_id]

    def start(self) -> None:
        """Start background tasks (Worker + Cleanup)."""
        self._running = True
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started periodic job cleanup task")
            
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Started job worker task")

    def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Stopped periodic job cleanup task")
            
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            logger.info("Stopped job worker task")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up stale jobs."""
        while self._running:
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

    async def submit_job(
        self,
        job_id: str,
        mode: ModelMode,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        **kwargs
    ) -> bool:
        """Submit a job to the queue.
        
        Args:
            job_id: Unique job ID
            mode: Target mode (IMAGE or VIDEO)
            task_func: Async function to execute
            **kwargs: Arguments for task_func
            
        Returns:
            True (always accepted now that we have a queue)
        """
        # Initialize job record
        # Initialize job record
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
        )
        # Store execution details (PrivateAttrs must be set after init)
        job._task_func = task_func
        job._kwargs = kwargs
        job._mode = mode
        
        self._jobs[job_id] = job
        self._cleanup_old_jobs()
        
        # Add to queue and notify worker
        async with self._condition:
            self._pending_jobs.append(job_id)
            self._condition.notify()
            
        logger.info(f"Job {job_id} queued for {mode.value} mode. Queue length: {len(self._pending_jobs)}")
        return True

    # Compatibility alias for existing code
    async def try_submit_job(self, *args, **kwargs) -> bool:
        return await self.submit_job(*args, **kwargs)

    async def _worker_loop(self) -> None:
        """Main worker loop handling job execution and scheduling."""
        from app.services.model_manager import VRAMLoadMode, ModelMode

        while self._running:
            try:
                job_id = None
                
                # 1. Wait for jobs
                async with self._condition:
                    await self._condition.wait_for(lambda: len(self._pending_jobs) > 0 or not self._running)
                    
                    if not self._running:
                        break
                        
                    # 2. Select next job based on scheduling logic
                    job_id = self._select_next_job()
                    
                    if job_id:
                        self._pending_jobs.remove(job_id)
                
                if not job_id:
                    continue
                    
                # 3. Process the job
                await self._process_job(job_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)  # Prevent tight loops on error

    def _select_next_job(self) -> Optional[str]:
        """Select the next job to run based on current mode and configuration."""
        from app.services.model_manager import VRAMLoadMode, ModelMode

        if not self._pending_jobs:
            return None
            
        # Get current state
        if not self._model_manager:
            # Fallback if manager not linked yet
            return self._pending_jobs[0]

        vram_mode = self._model_manager.vram_mode
        current_mode = self._model_manager.current_mode
        
        # Strategy 1: STATIC Mode (Strict FIFO)
        if vram_mode == VRAMLoadMode.STATIC:
            return self._pending_jobs[0]
            
        # Strategy 2: DYNAMIC Mode (Grouping)
        # Prioritize jobs matching the current mode
        
        # If we are in NO mode (startup), just pick the first one
        if current_mode == ModelMode.NONE or current_mode == ModelMode.SWITCHING:
            return self._pending_jobs[0]
            
        # Try to find a job matching current mode
        for job_id in self._pending_jobs:
            job_info = self._jobs.get(job_id)
            if job_info and job_info._mode == current_mode:
                return job_id
                
        # If no jobs match current mode, switch mode (pick first available)
        return self._pending_jobs[0]

    async def _process_job(self, job_id: str):
        """Execute a single job."""
        from app.config import InferenceConfig
        from app.services.model_manager import ModelMode
        
        job = self._jobs.get(job_id)
        if not job:
            return

        try:
            # 1. Ensure correct model mode
            if self._model_manager:
                await self._model_manager.ensure_mode(job._mode)
            
            # 2. Determine timeout
            timeout_seconds = (
                InferenceConfig.IMAGE_JOB_TIMEOUT if job._mode == ModelMode.IMAGE
                else InferenceConfig.VIDEO_JOB_TIMEOUT
            )
            
            # 3. Update Status
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            job.progress_percent = 0
            job.progress_stage = "starting"
            
            # 4. Execute
            logger.info(f"Starting job {job_id} [{job._mode}] with {timeout_seconds}s timeout")
            
            result = await asyncio.wait_for(
                job._task_func(**job._kwargs),
                timeout=timeout_seconds
            )
            
            # 5. Success
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            job.progress_percent = 100
            job.progress_stage = "completed"
            logger.info(f"Job {job_id} completed successfully")
            
        except asyncio.TimeoutError:
            self._handle_job_error(job, f"Job timed out after {timeout_seconds} seconds", "JOB_TIMEOUT")
        except Exception as e:
            self._handle_job_error(job, str(e), "GENERATION_FAILED")
        finally:
            # Cleanup GPU memory after every job
            self._cleanup_gpu_memory()

    def _handle_job_error(self, job: JobInfo, error_msg: str, error_code: str):
        """Handle job failure."""
        logger.error(f"Job {job.job_id} failed: {error_msg}")
        
        # Check for OOM
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
             if self._check_oom_error(None, error_msg):
                 error_msg = "GPU out of memory. Try reducing resolution or duration."
                 error_code = "GPU_OUT_OF_MEMORY"

        job.status = JobStatus.FAILED
        job.completed_at = time.time()
        job.error_message = error_msg
        job.error_code = error_code
        job.progress_stage = "failed"

    def _check_oom_error(self, exception: Optional[Exception], error_message: str) -> bool:
        """Check if an error is OOM."""
        oom_indicators = [
            "out of memory",
            "CUDA out of memory",
            "CUDA error: out of memory",
            "OutOfMemoryError",
        ]
        return any(indicator.lower() in error_message.lower() for indicator in oom_indicators)

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to clean up GPU memory: {e}")

