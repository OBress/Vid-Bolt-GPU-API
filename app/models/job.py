"""Job models for async processing."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, PrivateAttr


class JobStatus(str, Enum):
    """Status of an asynchronous job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """Result of a completed job."""
    save_url: Optional[str] = None
    generation_time: Optional[float] = None
    # For video, we might have duration/audio info
    duration_seconds: Optional[float] = None
    has_audio: Optional[bool] = None
    # Generic extras
    metadata: Optional[Dict[str, Any]] = None


class JobInfo(BaseModel):
    """Information about a specific job."""
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None  # e.g., "GPU_OUT_OF_MEMORY", "JOB_TIMEOUT"
    result: Optional[JobResult] = None
    # Progress tracking for long-running jobs
    progress_percent: Optional[int] = None  # 0-100
    progress_stage: Optional[str] = None    # "loading", "generating", "upscaling", "uploading"
    queue_position: Optional[int] = None    # Current position in queue (1-based), only if status is PENDING
    
    # Internal execution details (not serialized)
    _task_func: Any = PrivateAttr(default=None)
    _kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _mode: Any = PrivateAttr(default=None)



class AsyncJobResponse(BaseModel):
    """Immediate response when a job is accepted."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    status_url: str
    message: str = "Job accepted for processing"
