"""Job Management Service.

This module provides the JobManager service for handling asynchronous generation jobs.
It implements intelligent batch processing with dynamic VRAM-based sizing.

Features:
- Resolution-based job bucketing for efficient batching
- Dynamic batch sizing based on available VRAM
- OOM (Out of Memory) error handling with GPU cleanup
- Job timeout enforcement
- Periodic stale job cleanup
- Progress tracking for long-running jobs
"""

import asyncio
import gc
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Coroutine, Tuple

from app.config import Settings, InferenceConfig
from app.models.job import JobInfo, JobResult, JobStatus
from app.models.webhook import WebhookPayload
from app.models.internal import (
    ImageGenerationResult,
    ImageEditResult,
    VideoGenerationResult,
)
from app.services.model_manager import VRAMLoadMode, JobType, ModelMode

logger = logging.getLogger(__name__)


class JobManager:
    """Manages asynchronous generation jobs with intelligent batch scheduling."""

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
        
        # Job Queue - Resolution-based buckets for batch processing
        # Key: (width, height, job_type), Value: list of job_ids
        self._pending_buckets: Dict[Tuple[int, int, JobType], List[str]] = defaultdict(list)
        self._pending_jobs_set: set[str] = set()  # For O(1) membership check
        self._condition = asyncio.Condition()
        
        # In-memory job store (job_id -> JobInfo)
        self._jobs: Dict[str, JobInfo] = {}
        
        # Limit the size of history to prevent memory leaks
        self.MAX_HISTORY = 1000
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Webhook service (set during startup)
        self._webhook_service = None

    def set_model_manager(self, model_manager: "ModelManager"):
        """Set the ModelManager instance (circular dependency check)."""
        self._model_manager = model_manager

    def set_webhook_service(self, webhook_service):
        """Set the WebhookService instance."""
        self._webhook_service = webhook_service

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information by ID."""
        return self._jobs.get(job_id)

    async def list_active_jobs(self) -> list[JobInfo]:
        """List currently processing jobs."""
        return [
            job for job in self._jobs.values() 
            if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
        ]

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get the current queue position for a pending job (1-based).
        
        Returns:
            Position (1, 2, 3...) if in queue, None otherwise.
        """
        if job_id not in self._pending_jobs_set:
            return None
        
        # Count jobs ahead of this one (by bucket order)
        position = 0
        job = self._jobs.get(job_id)
        if not job:
            return None
            
        job_created_at = job.created_at
        
        for bucket_jobs in self._pending_buckets.values():
            for other_id in bucket_jobs:
                other_job = self._jobs.get(other_id)
                if other_job and other_job.created_at < job_created_at:
                    position += 1
                    
        return position + 1

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
        job_type: JobType,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        webhook_url: str,
        item_id: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Submit a job to the queue.
        
        Args:
            job_id: Unique job ID
            job_type: Job type (IMAGE_GENERATION, IMAGE_EDITING, VIDEO_GENERATION)
            task_func: Async function to execute
            webhook_url: REQUIRED - URL to POST when job completes
            item_id: Optional client identifier (defaults to job_id in webhook)
            webhook_secret: Optional HMAC signing secret
            **kwargs: Arguments for task_func (must include 'params' with width/height)
            
        Returns:
            True (always accepted now that we have a queue)
        """
        # Initialize job record
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
            item_id=item_id,
        )
        # Store execution details (PrivateAttrs must be set after init)
        job._task_func = task_func
        job._kwargs = kwargs
        job._job_type = job_type
        job._webhook_url = webhook_url
        job._webhook_secret = webhook_secret
        
        # Extract dimensions for bucketing (default to standard size)
        params = kwargs.get("params")
        width = getattr(params, "width", 1024) if params else 1024
        height = getattr(params, "height", 1024) if params else 1024
        job._bucket_key = (width, height, job_type)
        
        self._jobs[job_id] = job
        self._cleanup_old_jobs()
        
        # Add to appropriate bucket and notify worker
        async with self._condition:
            bucket_key = (width, height, job_type)
            self._pending_buckets[bucket_key].append(job_id)
            self._pending_jobs_set.add(job_id)
            self._condition.notify()
            
        logger.info(
            f"Job {job_id} queued for {job_type.value} ({width}x{height}). "
            f"Total pending: {len(self._pending_jobs_set)}"
        )
        return True

    # Compatibility alias for existing code
    async def try_submit_job(self, *args, **kwargs) -> bool:
        return await self.submit_job(*args, **kwargs)

    async def requeue_job(self, job_id: str) -> bool:
        """Requeue a failed job to the back of its bucket queue.
        
        This is used by BatchManager for retry-on-failure. The job is moved
        to the end of its original bucket, giving priority to other jobs.
        
        Args:
            job_id: Job ID to requeue
            
        Returns:
            True if successfully requeued, False if job not found
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Cannot requeue job {job_id}: not found")
            return False
        
        # Reset job state for retry
        job.status = JobStatus.PENDING
        job.started_at = None
        job.completed_at = None
        job.error_message = None
        job.error_code = None
        job.progress_percent = None
        job.progress_stage = None
        
        # Get bucket key from job's stored params
        bucket_key = getattr(job, '_bucket_key', None)
        if not bucket_key:
            logger.warning(f"Cannot requeue job {job_id}: no bucket key")
            return False
        
        # Add back to queue at the END (other jobs get priority)
        async with self._condition:
            self._pending_buckets[bucket_key].append(job_id)
            self._pending_jobs_set.add(job_id)
            self._condition.notify()
        
        logger.info(f"Requeued job {job_id} to back of {bucket_key} queue")
        return True

    async def _worker_loop(self) -> None:
        """Main worker loop handling job execution and batch scheduling."""
        from app.services.model_manager import VRAMLoadMode, ModelMode
        from app.services import vram_estimator

        while self._running:
            try:
                batch_job_ids: List[str] = []
                bucket_key: Optional[Tuple[int, int, JobType]] = None
                
                # 1. Wait for jobs and select a batch
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: len(self._pending_jobs_set) > 0 or not self._running
                    )
                    
                    if not self._running:
                        break
                    
                    # 2. Select batch using smart scheduling
                    batch_job_ids, bucket_key = self._select_batch()
                    
                    # Remove selected jobs from pending
                    for job_id in batch_job_ids:
                        self._pending_jobs_set.discard(job_id)
                    if bucket_key and bucket_key in self._pending_buckets:
                        # Remove processed jobs from bucket
                        remaining = [
                            jid for jid in self._pending_buckets[bucket_key]
                            if jid not in batch_job_ids
                        ]
                        if remaining:
                            self._pending_buckets[bucket_key] = remaining
                        else:
                            del self._pending_buckets[bucket_key]
                
                if not batch_job_ids:
                    continue
                
                # 3. Process the batch
                if len(batch_job_ids) == 1:
                    # Single job - use legacy single-job processing
                    await self._process_job(batch_job_ids[0])
                else:
                    # Batch processing
                    await self._process_batch(batch_job_ids, bucket_key)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Prevent tight loops on error

    def _select_batch(self) -> Tuple[List[str], Optional[Tuple[int, int, JobType]]]:
        """Select a batch of jobs to process together.
        
        Uses smart bucketing:
        1. Find the bucket with the OLDEST job (prevents starvation)
        2. Calculate max batch size based on available VRAM
        3. Return up to max_batch jobs from that bucket
        
        Returns:
            Tuple of (list of job_ids, bucket_key)
        """
        from app.services import vram_estimator
        
        if not self._pending_buckets:
            return [], None
        
        # Find bucket with oldest job
        oldest_time = float('inf')
        oldest_bucket_key: Optional[Tuple[int, int, JobType]] = None
        
        for bucket_key, job_ids in self._pending_buckets.items():
            if not job_ids:
                continue
            # Get creation time of first job in bucket
            first_job = self._jobs.get(job_ids[0])
            if first_job and first_job.created_at < oldest_time:
                oldest_time = first_job.created_at
                oldest_bucket_key = bucket_key
        
        if not oldest_bucket_key:
            return [], None
        
        width, height, job_type = oldest_bucket_key
        job_ids = self._pending_buckets[oldest_bucket_key]
        
        # Calculate max batch size based on job type and VRAM
        if job_type == JobType.IMAGE_GENERATION:
            # Z-Image: true vectorized batching
            max_batch = vram_estimator.calculate_max_batch_size(width, height)
        elif job_type == JobType.IMAGE_EDITING:
            # LightX2V: sequential batching with shared model state
            # Check if other models are loaded (ALL mode)
            other_models_loaded = False
            if self._model_manager:
                from app.services.model_manager import VRAMLoadMode
                other_models_loaded = self._model_manager.current_mode == VRAMLoadMode.ALL
            
            max_batch = vram_estimator.calculate_lightx2v_max_batch_size(
                width, height,
                other_models_loaded=other_models_loaded
            )
        elif job_type == JobType.VIDEO_GENERATION:
            # LTX-2: sequential batching with warm model
            # Return all pending jobs - they will be processed sequentially
            max_batch = len(job_ids)  # Process all pending (sequential anyway)
        else:
            # Unknown job type - return single job as fallback
            logger.warning(f"Unknown job type {job_type}, processing single job")
            return [job_ids[0]], oldest_bucket_key
        
        # Select up to max_batch jobs
        batch_size = min(len(job_ids), max_batch)
        selected_jobs = job_ids[:batch_size]
        
        logger.info(
            f"Selected batch of {len(selected_jobs)} {job_type.value} jobs for "
            f"{width}x{height} (max allowed: {max_batch}, pending: {len(job_ids)})"
        )
        
        return selected_jobs, oldest_bucket_key

    async def _process_batch(
        self, 
        job_ids: List[str],
        bucket_key: Optional[Tuple[int, int, JobType]]
    ):
        """Execute a batch of jobs together.
        
        Supports both:
        - IMAGE_GENERATION (Z-Image): True vectorized batch processing
        - IMAGE_EDITING (LightX2V): Sequential batch processing with warm model
        """
        from app.config import InferenceConfig
        
        jobs = [self._jobs.get(jid) for jid in job_ids]
        jobs = [j for j in jobs if j is not None]
        
        if not jobs:
            return
        
        job_type = jobs[0]._job_type if hasattr(jobs[0], '_job_type') else None
        timeout_seconds = InferenceConfig.IMAGE_JOB_TIMEOUT
        
        try:
            # 1. Ensure correct model mode
            if self._model_manager and job_type:
                await self._model_manager.ensure_mode_for_job(job_type)
            
            # 2. Update all jobs to PROCESSING
            for job in jobs:
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()
                job.progress_percent = 0
                job.progress_stage = "batching"
            
            logger.info(
                f"Starting {job_type.value if job_type else 'unknown'} batch of "
                f"{len(jobs)} jobs with {timeout_seconds}s timeout"
            )
            
            # 3. Get appropriate generator and execute batch
            if not self._model_manager:
                raise RuntimeError("ModelManager not available")
            
            # Collect params from all jobs
            params_list = [job._kwargs.get("params") for job in jobs]
            params_list = [p for p in params_list if p is not None]
            
            if len(params_list) != len(jobs):
                raise RuntimeError("Mismatch between jobs and params")
            
            # Route to appropriate generator based on job type
            if job_type == JobType.IMAGE_GENERATION:
                # Z-Image: true vectorized batch
                generator = self._model_manager.get_image_generator()
                results = await asyncio.wait_for(
                    generator.generate_batch(params_list),
                    timeout=timeout_seconds
                )
            elif job_type == JobType.IMAGE_EDITING:
                # LightX2V: sequential batch with warm model
                generator = self._model_manager.get_image_editor()
                results = await asyncio.wait_for(
                    generator.edit_batch(params_list),
                    timeout=timeout_seconds
                )
            elif job_type == JobType.VIDEO_GENERATION:
                # LTX-2: concurrent batch with shared pipeline
                generator = self._model_manager.get_video_generator()
                # Scale timeout by number of videos (even concurrent has limits)
                batch_timeout = InferenceConfig.VIDEO_JOB_TIMEOUT * len(jobs)
                # Use concurrent batch if enabled, otherwise sequential
                if hasattr(generator, 'concurrent_enabled') and generator.concurrent_enabled:
                    results = await asyncio.wait_for(
                        generator.generate_concurrent_batch(params_list),
                        timeout=batch_timeout
                    )
                else:
                    results = await asyncio.wait_for(
                        generator.generate_batch(params_list),
                        timeout=batch_timeout
                    )
            else:
                raise RuntimeError(f"Unsupported job type for batching: {job_type}")
            
            # 4. Complete jobs with their results
            # Handle partial success (generators may return empty data for failed jobs)
            for job, result in zip(jobs, results):
                # Check if this is a failed result (empty data)
                is_failed = False
                if hasattr(result, 'image_data'):
                    is_failed = result.image_data is not None and len(result.image_data) == 0
                elif hasattr(result, 'video_data'):
                    is_failed = result.video_data is not None and len(result.video_data) == 0
                
                if is_failed:
                    error_msg = "Video generation failed" if job_type == JobType.VIDEO_GENERATION else "Image edit failed"
                    self._handle_job_error(
                        job, 
                        f"{error_msg} within batch", 
                        "BATCH_ITEM_FAILED"
                    )
                    # Send failure webhook
                    await self._send_webhook(job)
                else:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = time.time()
                    job.result = result
                    job.progress_percent = 100
                    job.progress_stage = "completed"
                    # Send success webhook
                    await self._send_webhook(job)
            
            # Count successes
            completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
            logger.info(
                f"Batch complete: {completed}/{len(jobs)} jobs succeeded"
            )
            
        except asyncio.TimeoutError:
            for job in jobs:
                self._handle_job_error(job, f"Batch timed out after {timeout_seconds}s", "JOB_TIMEOUT")
                await self._send_webhook(job)
        except Exception as e:
            for job in jobs:
                self._handle_job_error(job, str(e), "GENERATION_FAILED")
                await self._send_webhook(job)
        finally:
            self._cleanup_gpu_memory()

    async def _process_job(self, job_id: str):
        """Execute a single job (legacy path for non-batchable jobs)."""
        from app.config import InferenceConfig
        
        job = self._jobs.get(job_id)
        if not job:
            return

        try:
            # 1. Ensure correct model mode for this job type
            if self._model_manager and hasattr(job, '_job_type'):
                await self._model_manager.ensure_mode_for_job(job._job_type)
            
            # 2. Determine timeout based on job type
            job_type = getattr(job, '_job_type', None)
            if job_type == JobType.VIDEO_GENERATION:
                timeout_seconds = InferenceConfig.VIDEO_JOB_TIMEOUT
            else:
                timeout_seconds = InferenceConfig.IMAGE_JOB_TIMEOUT
            
            # 3. Update Status
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            job.progress_percent = 0
            job.progress_stage = "starting"
            
            # 4. Execute
            logger.info(f"Starting job {job_id} with {timeout_seconds}s timeout")
            
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
            
            # Send success webhook
            await self._send_webhook(job)
            
        except asyncio.TimeoutError:
            self._handle_job_error(job, f"Job timed out after {timeout_seconds} seconds", "JOB_TIMEOUT")
            await self._send_webhook(job)
        except Exception as e:
            self._handle_job_error(job, str(e), "GENERATION_FAILED")
            await self._send_webhook(job)
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
        """Clean up GPU memory aggressively after job completion."""
        try:
            import torch
            gc.collect()  # Force Python GC first to release tensor references
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Release cached memory to CUDA
                torch.cuda.ipc_collect()  # Clean up IPC handles from multiprocessing
                torch.cuda.synchronize()  # Wait for all GPU operations
                # Log VRAM state for debugging
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(f"GPU cleanup complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except ImportError:
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to clean up GPU memory: {e}")

    async def _send_webhook(self, job: JobInfo) -> None:
        """Send webhook for a completed/failed job and schedule deletion."""
        if not self._webhook_service:
            logger.warning(f"WebhookService not available, skipping webhook for {job.job_id}")
            return
        
        webhook_url = getattr(job, '_webhook_url', None)
        if not webhook_url:
            logger.warning(f"No webhook URL for job {job.job_id}")
            return
        
        # Convert internal dataclass result to JobResult Pydantic model
        job_result: Optional[JobResult] = None
        is_success = job.status == JobStatus.COMPLETED
        
        if is_success and job.result is not None:
            raw_result = job.result
            
            if isinstance(raw_result, ImageGenerationResult):
                job_result = JobResult(
                    save_url=None,  # Set by upload step later
                    generation_time=None,
                    metadata={
                        "width": raw_result.width,
                        "height": raw_result.height,
                        "seed": raw_result.seed,
                        "image_size_bytes": len(raw_result.image_data) if raw_result.image_data else 0,
                    }
                )
            elif isinstance(raw_result, ImageEditResult):
                job_result = JobResult(
                    save_url=None,
                    generation_time=None,
                    metadata={
                        "width": raw_result.width,
                        "height": raw_result.height,
                        "original_width": raw_result.original_width,
                        "original_height": raw_result.original_height,
                        "seed": raw_result.seed,
                        "image_size_bytes": len(raw_result.image_data) if raw_result.image_data else 0,
                    }
                )
            elif isinstance(raw_result, VideoGenerationResult):
                job_result = JobResult(
                    save_url=None,
                    generation_time=None,
                    duration_seconds=raw_result.duration_seconds,
                    has_audio=raw_result.has_audio,
                    metadata={
                        "width": raw_result.width,
                        "height": raw_result.height,
                        "frame_rate": raw_result.frame_rate,
                        "seed": raw_result.seed,
                        "video_size_bytes": len(raw_result.video_data) if raw_result.video_data else 0,
                        "upscale_info": raw_result.upscale_info,
                    }
                )
            elif isinstance(raw_result, JobResult):
                # Already a JobResult, use directly
                job_result = raw_result
            else:
                # Fallback: try to convert dict-like objects
                logger.warning(f"Unknown result type {type(raw_result)}, attempting dict conversion")
                try:
                    if hasattr(raw_result, '__dict__'):
                        job_result = JobResult(metadata=raw_result.__dict__)
                    else:
                        job_result = JobResult(metadata={"raw": str(raw_result)})
                except Exception as e:
                    logger.error(f"Failed to convert result: {e}")
                    job_result = JobResult(metadata={"conversion_error": str(e)})
        
        # Build payload
        payload = WebhookPayload(
            event="generation.completed" if is_success else "generation.failed",
            job_id=job.job_id,
            item_id=job.item_id or job.job_id,
            batch_id=job.batch_id,
            status="completed" if is_success else "failed",
            completed_at=job.completed_at or time.time(),
            generation_type=job._job_type.value if job._job_type else "unknown",
            result=job_result,
            error_message=job.error_message if not is_success else None,
            error_code=job.error_code if not is_success else None,
            retry_count=0,  # TBD: integrate with BatchManager retry count
        )
        
        # Callback to delete job after successful delivery
        async def cleanup_job():
            self._jobs.pop(job.job_id, None)
            logger.debug(f"Deleted job {job.job_id} after webhook delivery")
        
        await self._webhook_service.deliver(
            webhook_url=webhook_url,
            payload=payload,
            secret=getattr(job, '_webhook_secret', None),
            on_success=cleanup_job,
        )
