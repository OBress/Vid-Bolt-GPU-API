"""Pydantic models for music generation API."""

from typing import Optional

from pydantic import BaseModel, Field


class MusicGenerateRequest(BaseModel):
    """Request body for music generation endpoint."""

    job_id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(..., description="Music style/genre description", min_length=1)
    lyrics: Optional[str] = Field(
        None, description="Optional lyrics for vocal generation"
    )
    duration_seconds: float = Field(
        30.0, ge=10.0, le=600.0, description="Duration in seconds (10s-10min)"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # Webhook and storage
    webhook_url: Optional[str] = Field(None, description="URL for job completion webhook")
    save_url: str = Field(..., description="Pre-signed URL to save audio file")
    item_id: Optional[str] = Field(None, description="Optional item ID for tracking")
    webhook_secret: Optional[str] = Field(None, description="Secret for webhook authentication")

    model_config = {"extra": "forbid"}


class MusicGenerateResponse(BaseModel):
    """Response for music generation endpoint (202 Accepted)."""

    job_id: str
    status: str = "queued"
    message: str = "Music generation job queued"
