"""Batch Manager Service.

This module provides the BatchManager for coordinating batch job submissions.
It handles:
- Batch lifecycle management (create, track, delete)
- Automatic retry-on-failure (requeue once before marking as failed)
- 5-minute auto-expiry for uncollected batches
- Aggregating per-item status into batch-level status
"""

import asyncio
import io
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from PIL import Image

from app.config import Settings
from app.models.batch import (
    AsyncBatchResponse,
    BatchInfo,
    BatchItemState,
    BatchItemStatus,
    BatchStatus,
)
from app.models.batch_image_generation import BatchImageGenerateItem
from app.models.batch_image_editing import BatchImageEditItem
from app.models.batch_video_generation import BatchVideoGenerateItem
from app.models.common import get_dimensions
from app.models.internal import ImageGenerationParams, ImageEditParams, VideoGenerationParams
from app.models.job import JobResult, JobStatus

if TYPE_CHECKING:
    from app.services.job_manager import JobManager
    from app.services.model_manager import ModelManager
    from app.services.storage import StorageService

logger = logging.getLogger(__name__)


class BatchManager:
    """Manages batch job submissions and aggregates status.
    
    Features:
    - Submit multiple items as a single batch
    - Automatic retry-on-failure (items requeued once before permanently failing)
    - 5-minute auto-expiry for uncollected batches
    - Collect endpoint for retrieving and deleting in one call
    """
    
    MAX_IMAGE_BATCH_SIZE = 500
    MAX_VIDEO_BATCH_SIZE = 100
    BATCH_RETENTION_SECONDS = 300  # 5 minutes auto-expiry
    CLEANUP_INTERVAL_SECONDS = 60   # Check for expired batches every minute
    
    def __init__(self, settings: Settings, job_manager: "JobManager"):
        """Initialize the BatchManager.
        
        Args:
            settings: Application settings
            job_manager: JobManager instance for submitting individual jobs
        """
        self._settings = settings
        self._job_manager = job_manager
        
        # Batch tracking
        self._batch_metadata: Dict[str, Dict[str, Any]] = {}  # batch_id -> {batch_type, created_at}
        self._batch_to_jobs: Dict[str, List[str]] = {}        # batch_id -> [job_ids]
        self._job_to_batch: Dict[str, str] = {}               # job_id -> batch_id
        self._retry_counts: Dict[str, int] = {}               # job_id -> retry count (0 or 1)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    def start(self) -> None:
        """Start background cleanup task."""
        self._running = True
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("BatchManager started with 5-minute auto-expiry")
    
    def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("BatchManager stopped")
    
    async def _periodic_cleanup(self) -> None:
        """Periodically remove expired batches."""
        while self._running:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_expired_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch cleanup error: {e}")
    
    async def _cleanup_expired_batches(self) -> None:
        """Remove batches older than BATCH_RETENTION_SECONDS."""
        now = time.time()
        expired = [
            batch_id for batch_id, metadata in self._batch_metadata.items()
            if now - metadata["created_at"] > self.BATCH_RETENTION_SECONDS
        ]
        for batch_id in expired:
            logger.info(f"Auto-expiring batch {batch_id} (5 minute retention exceeded)")
            self._delete_batch(batch_id)
    
    # =========================================================================
    # Batch Submission Methods
    # =========================================================================
    
    async def submit_image_generation_batch(
        self,
        batch_id: str,
        items: List[BatchImageGenerateItem],
        generator: Any,
        storage: "StorageService",
        webhook_url: str,
        webhook_secret: Optional[str] = None,
    ) -> BatchInfo:
        """Submit a batch of image generation requests.
        
        Args:
            batch_id: Client-provided unique batch identifier
            items: List of image generation items
            generator: Image generator instance
            storage: Storage service for uploads
            webhook_url: URL to POST when each item completes
            webhook_secret: Optional HMAC signing secret
            
        Returns:
            BatchInfo with initial pending status
        """
        from app.services.model_manager import JobType
        from app.routers.image_generation import _run_image_generation
        
        if batch_id in self._batch_to_jobs:
            raise ValueError(f"Batch {batch_id} already exists")
        
        job_ids = []
        for idx, item in enumerate(items):
            job_id = f"{batch_id}__item_{idx}"
            
            # Resolve dimensions
            if item.width and item.height:
                width, height = item.width, item.height
            else:
                width, height = get_dimensions(item.aspect_ratio)
            
            # Create params matching ImageGenerationParams
            params = ImageGenerationParams(
                job_id=job_id,
                prompt=item.prompt,
                width=width,
                height=height,
                seed=item.seed,
                num_inference_steps=item.num_inference_steps,
                lora_name=item.lora_name if item.lora_name and item.lora_name.lower() != "none" else None,
            )
            
            # Submit to JobManager with webhook config
            await self._job_manager.submit_job(
                job_id=job_id,
                job_type=JobType.IMAGE_GENERATION,
                task_func=_run_image_generation,
                webhook_url=webhook_url,
                item_id=item.item_id,
                webhook_secret=webhook_secret,
                generator=generator,
                storage=storage,
                params=params,
                save_url=item.save_url,
            )
            
            # Update job with batch linkage
            job = self._job_manager.get_job(job_id)
            if job:
                job.batch_id = batch_id
                job.batch_index = idx
            
            job_ids.append(job_id)
            self._retry_counts[job_id] = 0
            self._job_to_batch[job_id] = batch_id
        
        # Store batch metadata
        self._batch_to_jobs[batch_id] = job_ids
        self._batch_metadata[batch_id] = {
            "batch_type": "image_generation",
            "created_at": time.time(),
        }
        
        logger.info(f"Submitted image generation batch {batch_id} with {len(items)} items")
        return self.get_batch(batch_id)
    
    async def submit_image_editing_batch(
        self,
        batch_id: str,
        items: List[BatchImageEditItem],
        generator: Any,
        storage: "StorageService",
        webhook_url: str,
        webhook_secret: Optional[str] = None,
    ) -> BatchInfo:
        """Submit a batch of image editing requests.
        
        Args:
            batch_id: Client-provided unique batch identifier
            items: List of image editing items
            generator: Image editor instance
            storage: Storage service for uploads
            webhook_url: URL to POST when each item completes
            webhook_secret: Optional HMAC signing secret
            
        Returns:
            BatchInfo with initial pending status
        """
        from app.services.model_manager import JobType
        from app.routers.image_editing import _run_image_edit, _validate_image_magic_bytes
        
        if batch_id in self._batch_to_jobs:
            raise ValueError(f"Batch {batch_id} already exists")
        
        job_ids = []
        for idx, item in enumerate(items):
            job_id = f"{batch_id}__item_{idx}"
            
            # Pre-download and validate input image
            input_image_data = await storage.download_from_url(item.input_image_url)
            if not _validate_image_magic_bytes(input_image_data):
                raise ValueError(f"Item {idx}: input_image_url is not a valid image")
            
            # Download mask if provided
            mask_data = None
            if item.mask_image_url:
                mask_data = await storage.download_from_url(item.mask_image_url)
                if not _validate_image_magic_bytes(mask_data):
                    raise ValueError(f"Item {idx}: mask_image_url is not a valid image")
            
            # Extract dimensions from input image
            input_image = Image.open(io.BytesIO(input_image_data))
            width, height = input_image.size
            input_image.close()
            
            # Create params
            params = ImageEditParams(
                job_id=job_id,
                input_image_data=input_image_data,
                prompt=item.prompt,
                width=width,
                height=height,
                mask_data=mask_data,
                seed=item.seed,
                # Dynamic LoRA support (per-item)
                lora_name=item.lora_name,
                lora_strength=item.lora_strength,
            )
            
            # Submit to JobManager with webhook config
            await self._job_manager.submit_job(
                job_id=job_id,
                job_type=JobType.IMAGE_EDITING,
                task_func=_run_image_edit,
                webhook_url=webhook_url,
                item_id=item.item_id,
                webhook_secret=webhook_secret,
                generator=generator,
                storage=storage,
                params=params,
                save_url=item.save_url,
            )
            
            # Update job with batch linkage
            job = self._job_manager.get_job(job_id)
            if job:
                job.batch_id = batch_id
                job.batch_index = idx
            
            job_ids.append(job_id)
            self._retry_counts[job_id] = 0
            self._job_to_batch[job_id] = batch_id
        
        # Store batch metadata
        self._batch_to_jobs[batch_id] = job_ids
        self._batch_metadata[batch_id] = {
            "batch_type": "image_editing",
            "created_at": time.time(),
        }
        
        logger.info(f"Submitted image editing batch {batch_id} with {len(items)} items")
        return self.get_batch(batch_id)
    
    async def submit_video_generation_batch(
        self,
        batch_id: str,
        items: List[BatchVideoGenerateItem],
        generator: Any,
        storage: "StorageService",
        webhook_url: str,
        webhook_secret: Optional[str] = None,
    ) -> BatchInfo:
        """Submit a batch of video generation requests.
        
        Args:
            batch_id: Client-provided unique batch identifier
            items: List of video generation items
            generator: Video generator instance
            storage: Storage service for uploads
            webhook_url: URL to POST when each item completes
            webhook_secret: Optional HMAC signing secret
            
        Returns:
            BatchInfo with initial pending status
        """
        from app.services.model_manager import JobType
        from app.routers.ltx2_generation import _run_ltx2_generation, _validate_image_magic_bytes
        
        if batch_id in self._batch_to_jobs:
            raise ValueError(f"Batch {batch_id} already exists")
        
        job_ids = []
        for idx, item in enumerate(items):
            job_id = f"{batch_id}__item_{idx}"
            
            # Pre-download and validate start frame
            start_frame_data = await storage.download_from_url(item.start_frame_url)
            if not _validate_image_magic_bytes(start_frame_data):
                raise ValueError(f"Item {idx}: start_frame_url is not a valid image")
            
            # Download end frame if provided
            end_frame_data = None
            if item.end_frame_url:
                end_frame_data = await storage.download_from_url(item.end_frame_url)
                if not _validate_image_magic_bytes(end_frame_data):
                    raise ValueError(f"Item {idx}: end_frame_url is not a valid image")
            
            # Resolve dimensions
            width, height = get_dimensions(item.aspect_ratio)
            if item.width is not None and item.height is not None:
                width, height = item.width, item.height
            
            # Create params
            params = VideoGenerationParams(
                job_id=job_id,
                prompt=item.prompt,
                negative_prompt=item.negative_prompt,
                start_frame_data=start_frame_data,
                end_frame_data=end_frame_data,
                duration_seconds=item.duration_seconds,
                frame_rate=item.frame_rate,
                width=width,
                height=height,
                seed=item.seed,
                enhance_prompt=item.enhance_prompt,
            )
            
            # Submit to JobManager with webhook config
            await self._job_manager.submit_job(
                job_id=job_id,
                job_type=JobType.VIDEO_GENERATION,
                task_func=_run_ltx2_generation,
                webhook_url=webhook_url,
                item_id=item.item_id,
                webhook_secret=webhook_secret,
                generator=generator,
                storage=storage,
                params=params,
                save_url=item.save_url,
            )
            
            # Update job with batch linkage
            job = self._job_manager.get_job(job_id)
            if job:
                job.batch_id = batch_id
                job.batch_index = idx
            
            job_ids.append(job_id)
            self._retry_counts[job_id] = 0
            self._job_to_batch[job_id] = batch_id
        
        # Store batch metadata
        self._batch_to_jobs[batch_id] = job_ids
        self._batch_metadata[batch_id] = {
            "batch_type": "video_generation",
            "created_at": time.time(),
        }
        
        logger.info(f"Submitted video generation batch {batch_id} with {len(items)} items")
        return self.get_batch(batch_id)
    
    # =========================================================================
    # Retry Handling
    # =========================================================================
    
    def handle_job_failure(self, job_id: str) -> bool:
        """Handle a failed job - requeue once, then mark as permanently failed.
        
        This is called by JobManager when a job fails. If this is the first
        failure, the job is requeued to the back of the queue for retry.
        
        Args:
            job_id: The failed job ID
            
        Returns:
            True if job was requeued for retry, False if permanently failed
        """
        if job_id not in self._retry_counts:
            return False
        
        if self._retry_counts[job_id] == 0:
            # First failure - requeue to back of queue
            self._retry_counts[job_id] = 1
            logger.info(f"Requeueing failed job {job_id} for retry (attempt 2/2)")
            # The actual requeue is handled by JobManager
            return True
        else:
            # Already retried once - permanent failure
            logger.warning(f"Job {job_id} failed after retry, marking as permanently failed")
            return False
    
    def get_retry_count(self, job_id: str) -> int:
        """Get the current retry count for a job."""
        return self._retry_counts.get(job_id, 0)
    
    def is_batch_job(self, job_id: str) -> bool:
        """Check if a job belongs to a batch."""
        return job_id in self._job_to_batch
    
    def get_batch_for_job(self, job_id: str) -> Optional[str]:
        """Get the batch ID for a job, if it belongs to a batch."""
        return self._job_to_batch.get(job_id)
    
    # =========================================================================
    # Status Retrieval
    # =========================================================================
    
    def get_batch(self, batch_id: str) -> Optional[BatchInfo]:
        """Get current batch status by aggregating job statuses.
        
        Args:
            batch_id: The batch ID to look up
            
        Returns:
            BatchInfo with aggregated status, or None if not found
        """
        job_ids = self._batch_to_jobs.get(batch_id)
        metadata = self._batch_metadata.get(batch_id)
        
        if not job_ids or not metadata:
            return None
        
        # Aggregate from JobManager
        items: List[BatchItemStatus] = []
        completed_count = 0
        failed_count = 0
        pending_count = 0
        processing_count = 0
        retrying_count = 0
        
        for idx, job_id in enumerate(job_ids):
            job = self._job_manager.get_job(job_id)
            retry_count = self._retry_counts.get(job_id, 0)
            
            if job:
                # Map JobStatus to BatchItemState
                if job.status == JobStatus.PENDING:
                    if retry_count > 0:
                        state = BatchItemState.RETRYING
                        retrying_count += 1
                    else:
                        state = BatchItemState.PENDING
                        pending_count += 1
                elif job.status == JobStatus.PROCESSING:
                    state = BatchItemState.PROCESSING
                    processing_count += 1
                elif job.status == JobStatus.COMPLETED:
                    state = BatchItemState.COMPLETED
                    completed_count += 1
                elif job.status == JobStatus.FAILED:
                    state = BatchItemState.FAILED
                    failed_count += 1
                else:
                    state = BatchItemState.PENDING
                    pending_count += 1
                
                items.append(BatchItemStatus(
                    item_index=idx,
                    item_id=job.item_id or job_id,
                    job_id=job_id,
                    status=state,
                    retry_count=retry_count,
                    error_message=job.error_message,
                ))
            else:
                # Job not found (already cleaned up after webhook) - mark as completed
                items.append(BatchItemStatus(
                    item_index=idx,
                    item_id=job_id,  # Fallback to job_id
                    job_id=job_id,
                    status=BatchItemState.COMPLETED,
                    retry_count=retry_count,
                ))
                completed_count += 1
        
        # Determine overall batch status
        all_done = completed_count + failed_count == len(job_ids)
        any_processing = processing_count > 0 or retrying_count > 0
        
        if all_done:
            batch_status = BatchStatus.COMPLETED
            completed_at = time.time()
        elif any_processing:
            batch_status = BatchStatus.PROCESSING
            completed_at = None
        else:
            batch_status = BatchStatus.PENDING
            completed_at = None
        
        return BatchInfo(
            batch_id=batch_id,
            status=batch_status,
            batch_type=metadata["batch_type"],
            total_items=len(job_ids),
            completed_items=completed_count,
            failed_items=failed_count,
            pending_items=pending_count,
            processing_items=processing_count,
            retrying_items=retrying_count,
            created_at=metadata["created_at"],
            completed_at=completed_at,
            items=items,
        )
    
    def collect_batch(self, batch_id: str) -> Optional[BatchInfo]:
        """Get batch results and immediately delete the batch.
        
        This is the preferred method for final retrieval. Returns the batch
        status and then immediately deletes all batch data.
        
        Args:
            batch_id: The batch ID to collect
            
        Returns:
            BatchInfo with final status, or None if not found
        """
        batch_info = self.get_batch(batch_id)
        if batch_info:
            logger.info(f"Collecting and deleting batch {batch_id}")
            self._delete_batch(batch_id)
        return batch_info
    
    def _delete_batch(self, batch_id: str) -> None:
        """Remove batch and all associated tracking data.
        
        Note: This does NOT delete the underlying jobs from JobManager.
        Those will be cleaned up by JobManager's own retention policy.
        """
        job_ids = self._batch_to_jobs.pop(batch_id, [])
        for job_id in job_ids:
            self._job_to_batch.pop(job_id, None)
            self._retry_counts.pop(job_id, None)
        self._batch_metadata.pop(batch_id, None)
        
        logger.debug(f"Deleted batch {batch_id} ({len(job_ids)} jobs)")
    
    def list_batches(self) -> List[str]:
        """List all active batch IDs."""
        return list(self._batch_to_jobs.keys())
