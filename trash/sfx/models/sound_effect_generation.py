"""Pydantic models for sound effect generation API."""

from typing import Optional

from pydantic import BaseModel, Field


class SoundEffectGenerateRequest(BaseModel):
    """Request body for sound effect generation endpoint."""

    job_id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(..., description="Sound effect description", min_length=1)
    duration_seconds: float = Field(
        5.0, ge=1.0, le=30.0, description="Duration in seconds (1s-30s)"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # Webhook and storage
    webhook_url: Optional[str] = Field(None, description="URL for job completion webhook")
    save_url: str = Field(..., description="Pre-signed URL to save audio file")
    item_id: Optional[str] = Field(None, description="Optional item ID for tracking")
    webhook_secret: Optional[str] = Field(None, description="Secret for webhook authentication")

    model_config = {"extra": "forbid"}


class SoundEffectGenerateResponse(BaseModel):
    """Response for sound effect generation endpoint (202 Accepted)."""

    job_id: str
    status: str = "queued"
    message: str = "Sound effect generation job queued"
