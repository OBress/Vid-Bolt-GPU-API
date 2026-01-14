"""Batch job models for multi-item submissions.

This module provides models for batch job management, allowing multiple
generation requests to be submitted as a single batch with one status poll.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.job import JobResult


class BatchStatus(str, Enum):
    """Status of an entire batch."""
    PENDING = "pending"        # All items pending
    PROCESSING = "processing"  # At least one item processing
    COMPLETED = "completed"    # All items finished (some may have failed)


class BatchItemState(str, Enum):
    """Status of a single item within a batch."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"      # Failed once, requeued for retry


class BatchItemStatus(BaseModel):
    """Status of a single item within a batch."""
    item_index: int = Field(..., description="0-based position in original request")
    item_id: str = Field(..., description="Client-provided identifier for this item")
    job_id: str = Field(..., description="Internal job ID for this item")
    status: BatchItemState = Field(..., description="Current state of this item")
    retry_count: int = Field(default=0, ge=0, le=1, description="Number of retry attempts (0 or 1)")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    # Note: result is not included here - results are delivered via webhook only


class BatchInfo(BaseModel):
    """Aggregate status for a batch of jobs."""
    batch_id: str = Field(..., description="Unique batch identifier")
    status: BatchStatus = Field(..., description="Overall batch status")
    batch_type: str = Field(..., description="Type: 'image_generation', 'image_editing', 'video_generation'")
    total_items: int = Field(..., ge=1, description="Total number of items in batch")
    completed_items: int = Field(default=0, ge=0, description="Number of successfully completed items")
    failed_items: int = Field(default=0, ge=0, description="Number of permanently failed items")
    pending_items: int = Field(default=0, ge=0, description="Number of pending items")
    processing_items: int = Field(default=0, ge=0, description="Number of currently processing items")
    retrying_items: int = Field(default=0, ge=0, description="Number of items being retried")
    created_at: float = Field(..., description="Unix timestamp when batch was created")
    completed_at: Optional[float] = Field(default=None, description="Unix timestamp when batch completed")
    items: List[BatchItemStatus] = Field(default_factory=list, description="Per-item status details")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "batch-abc123",
                    "status": "processing",
                    "batch_type": "image_generation",
                    "total_items": 10,
                    "completed_items": 5,
                    "failed_items": 1,
                    "pending_items": 2,
                    "processing_items": 2,
                    "retrying_items": 0,
                    "created_at": 1715420000.0,
                    "items": [
                        {"item_index": 0, "job_id": "batch-abc123__item_0", "status": "completed", "retry_count": 0},
                        {"item_index": 1, "job_id": "batch-abc123__item_1", "status": "processing", "retry_count": 0},
                    ]
                }
            ]
        }
    }


class AsyncBatchResponse(BaseModel):
    """Immediate response when a batch is accepted."""
    batch_id: str = Field(..., description="Unique batch identifier")
    status: BatchStatus = Field(default=BatchStatus.PENDING, description="Initial batch status")
    total_items: int = Field(..., ge=1, description="Total number of items in batch")
    status_url: str = Field(..., description="URL to poll for batch status")
    message: str = Field(default="Batch accepted for processing", description="Human-readable message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "batch-abc123",
                    "status": "pending",
                    "total_items": 100,
                    "status_url": "/api/v1/batch/batch-abc123",
                    "message": "Batch accepted for processing"
                }
            ]
        }
    }
