"""FastAPI dependency injection functions."""

from typing import Annotated, Union

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.exceptions import InvalidAPIKeyError, MissingAPIKeyError
from app.services.storage import StorageService
from app.services.mock_generator import MockGenerator


# Global generator instance (set during startup)
_generator_instance: Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator", None] = None


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


# Type aliases for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageService, Depends(get_storage_service)]
GeneratorDep = Annotated[
    Union[MockGenerator, "ZImageGenerator", "LightX2VImageEditGenerator", "LTX2Generator"],
    Depends(get_generator)
]

