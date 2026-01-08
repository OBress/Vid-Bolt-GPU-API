"""FastAPI dependency injection functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union, Optional
from app.services.interfaces import BaseModelGenerator, ImageGenerator, ImageEditor, VideoGenerator, Upscaler

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings
from app.exceptions import InvalidAPIKeyError, MissingAPIKeyError
from app.services.storage import StorageService
from app.services.mock_generator import MockGenerator

if TYPE_CHECKING:
    from app.services.video_upscaler import StreamDiffVSRUpscaler
    from app.services.model_manager import ModelManager
    from app.services.job_manager import JobManager
    from app.services.zimage_generator import ZImageGenerator
    from app.services.ltx2_generator import LTX2Generator
    from app.services.lightx2v_generator import LightX2VImageEditGenerator

# Global instances (set during startup)
_generator_instance: Union[MockGenerator, BaseModelGenerator, None] = None
_upscaler_instance: Optional[Upscaler] = None
_model_manager_instance: Optional["ModelManager"] = None
_job_manager_instance: Optional["JobManager"] = None


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
    instance: Union[MockGenerator, BaseModelGenerator]
) -> None:
    """Set the global generator instance (called during startup).
    
    Args:
        instance: The generator instance to use
    """
    global _generator_instance
    _generator_instance = instance


def set_upscaler_instance(instance: Upscaler) -> None:
    """Set the global upscaler instance (called during startup).

    Args:
        instance: The Upscaler instance to use for video upscaling
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


def set_job_manager_instance(instance: "JobManager") -> None:
    """Set the global JobManager instance (called during startup)."""
    global _job_manager_instance
    _job_manager_instance = instance


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


def get_job_manager() -> "JobManager":
    """Get the JobManager instance.
    
    Returns:
        JobManager instance
        
    Raises:
        RuntimeError: If JobManager is not initialized
    """
    if _job_manager_instance is None:
        raise RuntimeError("JobManager not initialized. Server startup may have failed.")
    return _job_manager_instance


def get_upscaler() -> Optional[Upscaler]:
    """Get the upscaler instance, if available.

    Returns:
        Upscaler instance or None if not initialized
    """
    return _upscaler_instance


def get_generator(
    settings: Settings = Depends(get_settings),
) -> Union[MockGenerator, BaseModelGenerator]:
    """Get generator service instance.

    Args:
        settings: Application settings

    Returns:
        Generator instance
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


# Type aliases for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageService, Depends(get_storage_service)]
GeneratorDep = Annotated[
    Union[MockGenerator, BaseModelGenerator],
    Depends(get_generator)
]
JobManagerDep = Annotated["JobManager", Depends(get_job_manager)]
ModelManagerDep = Annotated["ModelManager", Depends(get_model_manager)]
