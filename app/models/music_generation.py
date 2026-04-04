"""Pydantic models for music generation API."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


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

    # ACE-Step 1.5 metadata controls (optional — auto-detected if omitted)
    bpm: Optional[int] = Field(
        None, ge=30, le=300, description="Tempo in BPM (30-300)"
    )
    key_scale: Optional[str] = Field(
        None, description="Musical key/scale, e.g. 'C Major', 'Am', 'F# minor'"
    )
    time_signature: Optional[str] = Field(
        None, description="Time signature: '2' (2/4), '3' (3/4), '4' (4/4), '6' (6/8)"
    )
    vocal_language: Optional[str] = Field(
        None, description="Vocal language code (ISO 639-1), e.g. 'en', 'zh', 'ja', 'es'"
    )

    # Webhook and storage
    webhook_url: Optional[str] = Field(None, description="URL for job completion webhook")
    save_url: str = Field(..., description="Pre-signed URL to save audio file")
    item_id: Optional[str] = Field(None, description="Optional item ID for tracking")
    webhook_secret: Optional[str] = Field(None, description="Secret for webhook authentication")

    @field_validator("lyrics", mode="before")
    @classmethod
    def normalize_lyrics(cls, value):
        """Accept legacy list payloads and normalize them to ACE-Step's string format."""
        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None

        if isinstance(value, (list, tuple)):
            normalized_lines = []
            for line in value:
                if line is None:
                    continue
                if not isinstance(line, str):
                    raise TypeError("lyrics must be a string or a list of strings")
                normalized_lines.append(line)

            normalized = "\n".join(normalized_lines).strip()
            return normalized or None

        raise TypeError("lyrics must be a string or a list of strings")

    model_config = {"extra": "forbid"}


class MusicGenerateResponse(BaseModel):
    """Response for music generation endpoint (202 Accepted)."""

    job_id: str
    status: str = "queued"
    message: str = "Music generation job queued"
