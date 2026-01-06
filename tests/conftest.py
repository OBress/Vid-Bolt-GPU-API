"""Pytest configuration and fixtures."""

import io
import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Set test environment variables before importing app
os.environ["MOCK_MODE"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"
os.environ["LOG_LEVEL"] = "WARNING"

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


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
    # We patch the dependency type in routers
    with patch("app.dependencies.get_storage_service") as mock_get_storage:
        storage_instance = MagicMock()
        
        # Async methods
        storage_instance.download_from_url = AsyncMock(return_value=sample_image_bytes)
        storage_instance.upload_to_url = AsyncMock(return_value="https://custom-output.com/result.png")
        
        # Apply override
        app.dependency_overrides[app.dependencies.get_storage_service] = lambda: storage_instance
        
        yield storage_instance
        
        # Cleanup
        app.dependency_overrides.pop(app.dependencies.get_storage_service, None)


@pytest.fixture
def sample_job_id() -> str:
    """Return a sample job ID for testing."""
    return "550e8400-e29b-41d4-a716-446655440000"
