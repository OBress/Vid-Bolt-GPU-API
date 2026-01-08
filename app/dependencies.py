"""FastAPI dependency injection functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings
from app.exceptions import InvalidAPIKeyError, MissingAPIKeyError
from app.services.storage import StorageService
from app.services.mock_generator import MockGenerator

if TYPE_CHECKING:
    from app.services.video_upscaler import StreamDiffVSRUpscaler
    from app.services.model_manager import ModelManager


# Global instances (set during startup)
_generator_instance: Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator", None] = None
_upscaler_instance: "StreamDiffVSRUpscaler | None" = None
_model_manager_instance: "ModelManager | None" = None


def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify the API key from request headers.

    Args:
        x_api_key: API key from X-API-Key header
        settings: Application settings

    Returns:
        The validated API key

    Raises:
        MissingAPIKeyError: If API key is not provided
        InvalidAPIKeyError: If API key is invalid
    """
    if not x_api_key:
        raise MissingAPIKeyError()

    if x_api_key != settings.api_key:
        raise InvalidAPIKeyError()

    return x_api_key


def get_storage_service(
    settings: Settings = Depends(get_settings),
) -> StorageService:
    """Get storage service instance.

    Args:
        settings: Application settings

    Returns:
        StorageService instance
    """
    return StorageService(settings)


def set_generator_instance(
    instance: Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator"]
) -> None:
    """Set the global generator instance (called during startup).
    
    Args:
        instance: The generator instance to use (MockGenerator, ZImageGenerator, LightX2VImageEditGenerator, or LTX2Generator)
    """
    global _generator_instance
    _generator_instance = instance


def set_upscaler_instance(instance: "StreamDiffVSRUpscaler") -> None:
    """Set the global upscaler instance (called during startup).

    Args:
        instance: The StreamDiffVSRUpscaler instance to use for video upscaling
    """
    global _upscaler_instance
    _upscaler_instance = instance


def set_model_manager_instance(instance: "ModelManager") -> None:
    """Set the global ModelManager instance (called during startup).
    
    Args:
        instance: The ModelManager instance for mode switching
    """
    global _model_manager_instance
    _model_manager_instance = instance


def get_model_manager() -> "ModelManager":
    """Get the ModelManager instance.
    
    Returns:
        ModelManager instance
        
    Raises:
        RuntimeError: If ModelManager is not initialized
    """
    if _model_manager_instance is None:
        raise RuntimeError("ModelManager not initialized. Server startup may have failed.")
    return _model_manager_instance


def get_upscaler() -> "StreamDiffVSRUpscaler | None":
    """Get the upscaler instance, if available.

    Returns:
        StreamDiffVSRUpscaler instance or None if not initialized
    """
    return _upscaler_instance


def get_generator(
    settings: Settings = Depends(get_settings),
) -> Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator"]:
    """Get generator service instance.

    Args:
        settings: Application settings

    Returns:
        Generator instance (MockGenerator, ZImageGenerator, LightX2VImageEditGenerator, or LTX2Generator)
    """
    global _generator_instance
    
    # Return cached instance if available
    if _generator_instance is not None:
        return _generator_instance
    
    # Fallback: create new instance (shouldn't happen if startup ran correctly)
    if settings.mock_mode:
        return MockGenerator(settings)
    
    # Import here to avoid circular imports
    from app.services.zimage_generator import ZImageGenerator
    return ZImageGenerator(settings)


def require_image_mode() -> None:
    """Dependency that requires Image Mode to be active.
    
    Raises:
        HTTPException 503: If not in Image Mode or system is busy
    """
    if _model_manager_instance is None:
        # In mock mode or legacy mode, allow all requests
        return
    
    from app.services.model_manager import ModelMode
    
    if _model_manager_instance.current_mode == ModelMode.SWITCHING:
        raise HTTPException(
            status_code=503,
            detail="Mode switch in progress. Please wait and try again."
        )
    
    if _model_manager_instance.current_mode != ModelMode.IMAGE:
        raise HTTPException(
            status_code=503,
            detail=f"Image models not loaded. Current mode: {_model_manager_instance.current_mode.value}. "
                   f"Switch to image mode using POST /api/v1/mode/switch"
        )
    
    if _model_manager_instance.is_busy:
        raise HTTPException(
            status_code=503,
            detail=f"System is busy with job: {_model_manager_instance.active_job_id}. Please try again later."
        )


def require_video_mode() -> None:
    """Dependency that requires Video Mode to be active.
    
    Raises:
        HTTPException 503: If not in Video Mode or system is busy
    """
    if _model_manager_instance is None:
        # In mock mode or legacy mode, allow all requests
        return
    
    from app.services.model_manager import ModelMode
    
    if _model_manager_instance.current_mode == ModelMode.SWITCHING:
        raise HTTPException(
            status_code=503,
            detail="Mode switch in progress. Please wait and try again."
        )
    
    if _model_manager_instance.current_mode != ModelMode.VIDEO:
        raise HTTPException(
            status_code=503,
            detail=f"Video models not loaded. Current mode: {_model_manager_instance.current_mode.value}. "
                   f"Switch to video mode using POST /api/v1/mode/switch"
        )
    
    if _model_manager_instance.is_busy:
        raise HTTPException(
            status_code=503,
            detail=f"System is busy with job: {_model_manager_instance.active_job_id}. Please try again later."
        )


# Type aliases for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageService, Depends(get_storage_service)]
GeneratorDep = Annotated[
    Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator"],
    Depends(get_generator)
]
ImageModeDep = Annotated[None, Depends(require_image_mode)]
VideoModeDep = Annotated[None, Depends(require_video_mode)]
