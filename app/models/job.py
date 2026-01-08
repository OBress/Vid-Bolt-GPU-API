"""Job models for async processing."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel


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
    result: Optional[JobResult] = None


class AsyncJobResponse(BaseModel):
    """Immediate response when a job is accepted."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    status_url: str
    message: str = "Job accepted for processing"
