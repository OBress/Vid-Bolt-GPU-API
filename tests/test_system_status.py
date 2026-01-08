"""Tests for system status and GPU monitoring endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestSystemStatusEndpoint:
    """Tests for GET /api/v1/system/status endpoint."""

    def test_system_status_requires_auth(self, client: TestClient):
        """Test that system status endpoint requires authentication."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 401

    def test_system_status_returns_info(
        self, client: TestClient, api_key_headers: dict
    ):
        """Test that system status returns expected structure."""
        response = client.get("/api/v1/system/status", headers=api_key_headers)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "system" in data
        assert "mock_mode" in data
        assert "concurrency_limits" in data
        
        # Check system info structure
        system = data["system"]
        assert "os" in system
        assert "os_version" in system
        assert "python_version" in system
        assert "cpu_count" in system
        assert "hostname" in system
        
        # Check concurrency limits
        limits = data["concurrency_limits"]
        assert "max_concurrent_image_generations" in limits
        assert "max_concurrent_video_generations" in limits
        assert limits["max_concurrent_image_generations"] == 2
        assert limits["max_concurrent_video_generations"] == 1

    def test_system_status_mock_mode_flag(
        self, client: TestClient, api_key_headers: dict
    ):
        """Test that mock_mode flag is correctly reported."""
        response = client.get("/api/v1/system/status", headers=api_key_headers)
        assert response.status_code == 200
        
        data = response.json()
        # In tests, MOCK_MODE is set to true
        assert data["mock_mode"] is True
        
        # GPU info should be None in mock mode
        assert data["gpu"] is None

    def test_system_status_no_mode_info_in_mock(
        self, client: TestClient, api_key_headers: dict
    ):
        """Test that mode info is None in mock mode (no ModelManager)."""
        response = client.get("/api/v1/system/status", headers=api_key_headers)
        assert response.status_code == 200
        
        data = response.json()
        # In mock mode, ModelManager isn't initialized
        assert data["mode"] is None


class TestConcurrencyLimitsConfig:
    """Tests for concurrency limits configuration."""

    def test_image_concurrency_limit_value(self):
        """Test that image concurrency limit is correctly configured."""
        from app.config import InferenceConfig, get_settings
        
        assert InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS == 2
        
        settings = get_settings()
        assert settings.max_concurrent_image_generations == 2

    def test_video_concurrency_limit_value(self):
        """Test that video concurrency limit is correctly configured."""
        from app.config import InferenceConfig, get_settings
        
        assert InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS == 1
        
        settings = get_settings()
        assert settings.max_concurrent_video_generations == 1

    def test_concurrency_limits_are_positive(self):
        """Test that concurrency limits are positive integers."""
        from app.config import InferenceConfig
        
        assert InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS > 0
        assert InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS > 0
        assert isinstance(InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS, int)
        assert isinstance(InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS, int)
