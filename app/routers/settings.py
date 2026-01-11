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
    mode: Literal["dynamic", "static"]


@router.get("/vram-mode", response_model=VRAMModeResponse)
async def get_vram_mode(
    _api_key: APIKeyDep,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> VRAMModeResponse:
    """Get the current VRAM loading mode.
    
    - **dynamic**: Loads/unloads models as needed to save VRAM (default)
    - **static**: Keeps all models loaded in VRAM for instant switching
    """
    mode = model_manager.vram_mode
    description = (
        "Dynamic loading - saves VRAM" 
        if mode == VRAMLoadMode.DYNAMIC 
        else "Static loading - instant switching, higher VRAM usage"
    )
    
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
    
    Switching to 'static' triggers loading of all models immediately.
    Switching to 'dynamic' unloads models not currently needed.
    
    Raises:
        503 Service Unavailable: If system is currently busy with a job
    """
    try:
        target_mode = VRAMLoadMode(request.mode)
        await model_manager.set_vram_mode(target_mode)
        
        mode = model_manager.vram_mode
        description = (
            "Dynamic loading - saves VRAM" 
            if mode == VRAMLoadMode.DYNAMIC 
            else "Static loading - instant switching, higher VRAM usage"
        )
        
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
