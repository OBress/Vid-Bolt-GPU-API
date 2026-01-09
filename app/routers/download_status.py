"""Download status and management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from app.config import Settings, get_settings
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/v1/download", tags=["Download Status"])


class ModelProgress(BaseModel):
    """Progress for a single model."""
    model_name: str
    status: str  # pending, downloading, completed, failed, skipped
    progress_percent: float
    error: Optional[str] = None


class DownloadStatusResponse(BaseModel):
    """Overall download status response."""
    status: str = Field(..., description="Overall status: pending, downloading, completed, failed")
    ready: bool = Field(..., description="Whether API is ready for generation requests")
    total_models: int = Field(..., description="Total number of models to download")
    completed_models: int = Field(..., description="Number of completed/skipped models")
    current_model: Optional[str] = Field(None, description="Currently downloading model")
    models: dict[str, ModelProgress] = Field(default_factory=dict, description="Per-model status")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "downloading",
                    "ready": False,
                    "total_models": 7,
                    "completed_models": 3,
                    "current_model": "ltx2-checkpoint",
                    "models": {
                        "z-image-turbo": {
                            "model_name": "z-image-turbo",
                            "status": "completed",
                            "progress_percent": 100.0,
                            "error": None
                        }
                    }
                }
            ]
        }
    }


@router.get(
    "/status",
    response_model=DownloadStatusResponse,
    summary="Get Download Status",
    description="Get the current status of model downloads. No auth required.",
)
async def get_download_status(
    settings: Settings = Depends(get_settings),
) -> DownloadStatusResponse:
    """Get current model download status.
    
    This endpoint is used by clients to poll download progress and know
    when the API is ready for generation requests.
    """
    from app.services.model_downloader import get_model_downloader
    
    downloader = get_model_downloader()
    
    # If no downloader (mock mode or not initialized)
    if downloader is None:
        return DownloadStatusResponse(
            status="completed",
            ready=True,
            total_models=0,
            completed_models=0,
            current_model=None,
            models={},
        )
    
    status = downloader.get_status()
    
    # Convert internal status to response
    models = {}
    for name, progress in status.models.items():
        models[name] = ModelProgress(
            model_name=progress.model_name,
            status=progress.status.value,
            progress_percent=progress.progress_percent,
            error=progress.error,
        )
    
    return DownloadStatusResponse(
        status=status.status,
        ready=status.ready,
        total_models=status.total_models,
        completed_models=status.completed_models,
        current_model=status.current_model,
        models=models,
        started_at=status.started_at,
        completed_at=status.completed_at,
        error=status.error,
    )


@router.post(
    "/retry",
    response_model=DownloadStatusResponse,
    summary="Retry Failed Downloads",
    description="Retry downloading failed models. Requires authentication.",
    dependencies=[Depends(verify_api_key)],
)
async def retry_downloads(
    settings: Settings = Depends(get_settings),
) -> DownloadStatusResponse:
    """Retry downloading any failed models."""
    from app.services.model_downloader import get_model_downloader
    
    downloader = get_model_downloader()
    
    if downloader is None:
        raise HTTPException(status_code=400, detail="No download manager available")
    
    # Start download again (will skip already completed models)
    downloader.start_download()
    
    return await get_download_status(settings)
