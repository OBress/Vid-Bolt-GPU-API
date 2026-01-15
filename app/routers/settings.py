"""Settings router - API endpoints for runtime configuration."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import APIKeyDep, get_model_manager
from app.services.model_manager import ModelManager, VRAMLoadMode

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class VRAMModeResponse(BaseModel):
    """Response model for VRAM mode."""
    mode: str
    description: str


class VRAMModeRequest(BaseModel):
    """Request model for setting VRAM mode."""
    mode: Literal["image_generation", "image_editing", "video_generation"]


# Mode descriptions
MODE_DESCRIPTIONS = {
    VRAMLoadMode.IMAGE_GENERATION: "Image Generation - Z-Image Turbo only (~16GB VRAM)",
    VRAMLoadMode.IMAGE_EDITING: "Image Editing - LightX2V only (~40GB VRAM)",
    VRAMLoadMode.VIDEO_GENERATION: "Video Generation - LTX-2 DistilledPipeline only (~40GB VRAM)",
    VRAMLoadMode.ALL: "All Models - Disabled (requires ~100GB+ VRAM)",
}


@router.get("/vram-mode", response_model=VRAMModeResponse)
async def get_vram_mode(
    _api_key: APIKeyDep,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> VRAMModeResponse:
    """Get the current VRAM loading mode.
    
    Available modes:
    - **image_generation**: Z-Image Turbo only (~16GB)
    - **image_editing**: LightX2V (Qwen-Image-Edit) only (~40GB)
    - **video_generation**: LTX-2 DistilledPipeline (~40GB, supports start + optional end frame)
    """
    mode = model_manager.vram_mode
    description = MODE_DESCRIPTIONS.get(mode, f"Unknown mode: {mode.value}")
    
    return VRAMModeResponse(
        mode=mode.value,
        description=description,
    )


@router.post("/vram-mode", response_model=VRAMModeResponse)
async def set_vram_mode(
    request: VRAMModeRequest,
    _api_key: APIKeyDep,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> VRAMModeResponse:
    """Set the VRAM loading mode.
    
    Each mode loads only specific models:
    - **image_generation**: Unloads all, loads Z-Image Turbo (~16GB)
    - **image_editing**: Unloads all, loads LightX2V (~40GB)
    - **video_generation**: Unloads all, loads LTX-2 DistilledPipeline (~40GB)
    
    Note: ALL mode is disabled (requires ~100GB VRAM). Video generation supports
    1 start frame with optional end frame (1-2 keyframes total).
    
    Raises:
        503 Service Unavailable: If system is currently busy with a job
    """
    try:
        target_mode = VRAMLoadMode(request.mode)
        await model_manager.set_vram_mode(target_mode)
        
        mode = model_manager.vram_mode
        description = MODE_DESCRIPTIONS.get(mode, f"Mode set to: {mode.value}")
        
        return VRAMModeResponse(
            mode=mode.value,
            description=description,
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set VRAM mode: {str(e)}"
        )
