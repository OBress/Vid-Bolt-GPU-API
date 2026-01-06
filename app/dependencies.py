"""FastAPI dependency injection functions."""

from typing import Annotated

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.exceptions import InvalidAPIKeyError, MissingAPIKeyError
from app.services.storage import StorageService
from app.services.mock_generator import MockGenerator


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


def get_generator(
    settings: Settings = Depends(get_settings),
) -> MockGenerator:
    """Get generator service instance.

    Args:
        settings: Application settings

    Returns:
        MockGenerator instance (will be replaced with real generator when MOCK_MODE=false)
    """
    # For now, always return mock generator
    # In the future, this will check settings.mock_mode and return appropriate generator
    return MockGenerator(settings)


# Type aliases for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageService, Depends(get_storage_service)]
GeneratorDep = Annotated[MockGenerator, Depends(get_generator)]
