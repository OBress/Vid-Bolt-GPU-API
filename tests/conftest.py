"""Pytest configuration and fixtures."""

import io
import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient
from PIL import Image

# Set test environment variables before importing app
os.environ["MOCK_MODE"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"
os.environ["LOG_LEVEL"] = "WARNING"

from app.main import app
from app.dependencies import get_storage_service


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    # Use TestClient with a context manager to ensure startup/shutdown events run
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> Generator[httpx.AsyncClient, None, None]:
    """Create an async test client."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client



@pytest.fixture
def api_key_headers() -> dict[str, str]:
    """Return headers with valid API key."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def invalid_api_key_headers() -> dict[str, str]:
    """Return headers with invalid API key."""
    return {"X-API-Key": "invalid-key"}


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a sample PNG image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Create a sample JPEG image for testing."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_storage(sample_image_bytes) -> Generator[MagicMock, None, None]:
    """Mock the StorageService to avoid actual HTTP calls."""
    storage_instance = MagicMock()
    
    # Async methods
    storage_instance.download_from_url = AsyncMock(return_value=sample_image_bytes)
    
    # upload_to_url should return the URL that was passed in (matching real behavior)
    async def mock_upload_to_url(data, url, content_type):
        return url
    storage_instance.upload_to_url = AsyncMock(side_effect=mock_upload_to_url)
    
    # Apply override using the imported function
    app.dependency_overrides[get_storage_service] = lambda: storage_instance
    
    yield storage_instance
    
    # Cleanup
    app.dependency_overrides.pop(get_storage_service, None)


@pytest.fixture
def sample_job_id() -> str:
    """Return a sample job ID for testing."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def mock_job_manager() -> Generator[MagicMock, None, None]:
    """Create a fresh JobManager for each test."""
    from app.services.job_manager import JobManager
    from app.config import get_settings
    from app.dependencies import get_job_manager
    
    settings = get_settings()
    # Ensure fresh semaphores
    manager = JobManager(settings)
    
    app.dependency_overrides[get_job_manager] = lambda: manager
    yield manager
    app.dependency_overrides.pop(get_job_manager, None)


@pytest.fixture
def mock_model_manager() -> Generator[MagicMock, None, None]:
    """Create a ModelManager for tests."""
    from app.services.model_manager import ModelManager, ModelMode
    from app.config import get_settings
    from app.dependencies import get_model_manager
    
    settings = get_settings()
    # In mock mode, this is lightweight
    manager = ModelManager(settings)
    # Force default mode
    import asyncio
    # We can't easily run async init here, but ModelManager.__init__ doesn't do async stuff.
    # It just sets up locks.
    # We might need to manually set the initial mode if logic depends on it.
    manager._mode = ModelMode.IMAGE
    
    app.dependency_overrides[get_model_manager] = lambda: manager
    yield manager
    app.dependency_overrides.pop(get_model_manager, None)


