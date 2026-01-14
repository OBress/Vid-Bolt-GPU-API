"""Webhook payload models.

This module defines the payload structure for webhook callbacks
sent when generation tasks complete (success or failure).
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.models.job import JobResult


class WebhookPayload(BaseModel):
    """Payload sent to client webhook URL when a job completes."""
    
    event: str = Field(
        ...,
        description="Event type: 'generation.completed' or 'generation.failed'"
    )
    
    # Identifiers
    job_id: str = Field(..., description="Internal job ID")
    item_id: str = Field(..., description="Client-provided identifier (or job_id if not provided)")
    batch_id: Optional[str] = Field(default=None, description="Parent batch ID (if part of batch)")
    
    # Status
    status: str = Field(..., description="'completed' or 'failed'")
    completed_at: float = Field(..., description="Unix timestamp when job finished")
    generation_type: str = Field(
        ...,
        description="Type: 'image_generation', 'image_editing', 'video_generation'"
    )
    
    # Success fields (only present if status == "completed")
    result: Optional[JobResult] = Field(
        default=None,
        description="Generation result including save_url and metadata"
    )
    
    # Error fields (only present if status == "failed")
    error_message: Optional[str] = Field(default=None, description="Error description")
    error_code: Optional[str] = Field(default=None, description="Error code")
    retry_count: int = Field(default=0, description="Number of retry attempts (0 or 1)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "event": "generation.completed",
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "item_id": "scene_001_image",
                    "batch_id": None,
                    "status": "completed",
                    "completed_at": 1715420015.0,
                    "generation_type": "image_generation",
                    "result": {
                        "save_url": "https://storage.example.com/output.png",
                        "generation_time": 2.5,
                        "metadata": {"seed": 12345, "width": 1920, "height": 1080}
                    }
                },
                {
                    "event": "generation.failed",
                    "job_id": "550e8400-e29b-41d4-a716-446655440001",
                    "item_id": "scene_002_image",
                    "batch_id": "batch-abc123",
                    "status": "failed",
                    "completed_at": 1715420020.0,
                    "generation_type": "image_generation",
                    "error_message": "GPU out of memory",
                    "error_code": "GPU_OUT_OF_MEMORY",
                    "retry_count": 1
                }
            ]
        }
    }


class WebhookConfig(BaseModel):
    """Webhook configuration for a job."""
    url: str = Field(..., description="Webhook URL to POST to")
    secret: Optional[str] = Field(default=None, description="HMAC-SHA256 signing secret")
    item_id: Optional[str] = Field(default=None, description="Client identifier (defaults to job_id)")
