"""Mode management router - API endpoints for switching between Image and Video modes."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import APIKeyDep, get_model_manager
from app.services.model_manager import ModelManager, ModelMode

router = APIRouter(prefix="/api/v1/mode", tags=["mode"])


class ModeStatusResponse(BaseModel):
    """Response model for current mode status."""
    mode: str
    is_busy: bool
    active_job_id: str | None
    loaded_models: list[str]
    # Switching progress fields
    is_switching: bool = False
    switching_target: str | None = None
    switching_step: str | None = None
    switching_progress: float | None = None  # 0.0-1.0


class ModeSwitchRequest(BaseModel):
    """Request model for switching modes."""
    target_mode: Literal["image", "video"]


class ModeSwitchResponse(BaseModel):
    """Response model for mode switch operation."""
    status: str
    previous_mode: str
    current_mode: str
    message: str


@router.get("", response_model=ModeStatusResponse)
async def get_mode_status(
    _api_key: APIKeyDep,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> ModeStatusResponse:
    """Get the current mode status.
    
    Returns the current mode (image/video/switching/none), whether the system
    is busy processing a job, and which models are currently loaded.
    
    When is_switching is true, check switching_target, switching_step, and
    switching_progress for mode switch status.
    """
    status = model_manager.get_status()
    return ModeStatusResponse(
        mode=status.mode.value,
        is_busy=status.is_busy,
        active_job_id=status.active_job_id,
        loaded_models=status.loaded_models,
        is_switching=status.is_switching,
        switching_target=status.switching_target,
        switching_step=status.switching_step,
        switching_progress=status.switching_progress,
    )


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    request: ModeSwitchRequest,
    _api_key: APIKeyDep,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> ModeSwitchResponse:
    """Switch between Image Mode and Video Mode.
    
    This endpoint unloads the current mode's models and loads the target mode's models.
    
    - **Image Mode**: Loads Z-Image Turbo (text-to-image) + Qwen-Image-Edit (image editing)
    - **Video Mode**: Loads LTX-2 19B (video generation) + Stream-DiffVSR (upscaling)
    
    Note: This operation takes ~30-60 seconds as models are unloaded and loaded.
    
    Raises:
        503 Service Unavailable: If system is currently busy with a job
    """
    previous_mode = model_manager.current_mode.value
    
    # Check if we're already switching or busy
    if model_manager.current_mode == ModelMode.SWITCHING:
        raise HTTPException(
            status_code=503,
            detail="Mode switch already in progress"
        )
    
    if model_manager.is_busy:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot switch modes while a job is in progress (job: {model_manager.active_job_id})"
        )
    
    # Check if jobs are queued (prevents race between batch jobs)
    if model_manager._job_manager and model_manager._job_manager.has_pending_or_active_jobs():
        raise HTTPException(
            status_code=503,
            detail="Cannot switch modes while jobs are queued or processing"
        )
    
    try:
        if request.target_mode == "image":
            await model_manager.switch_to_image_mode()
        else:
            await model_manager.switch_to_video_mode()
        
        current_mode = model_manager.current_mode.value
        
        return ModeSwitchResponse(
            status="success",
            previous_mode=previous_mode,
            current_mode=current_mode,
            message=f"Successfully switched from {previous_mode} to {current_mode} mode"
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch mode: {str(e)}"
        )
