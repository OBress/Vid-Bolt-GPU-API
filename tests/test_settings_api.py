"""Tests for Settings API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

from unittest.mock import MagicMock, AsyncMock

from app.dependencies import get_model_manager
from app.services.model_manager import ModelManager, VRAMLoadMode

client = TestClient(app)


class TestSettingsAPI:
    """Tests for settings configuration endpoints."""

    def setup_method(self):
        """Setup mock dependency."""
        self.mock_manager = MagicMock(spec=ModelManager)
        self.mock_manager.vram_mode = VRAMLoadMode.IMAGE_GENERATION
        self.mock_manager.set_vram_mode = AsyncMock()
        
        # Override the dependency
        app.dependency_overrides[get_model_manager] = lambda: self.mock_manager

    def teardown_method(self):
        """Clean up overrides."""
        app.dependency_overrides = {}

    def test_get_vram_mode_default(self):
        """Test getting default VRAM mode."""
        headers = {"X-API-Key": "test-api-key-12345"}
        response = client.get("/api/v1/settings/vram-mode", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "image_generation"
        assert "description" in data

    def test_set_vram_mode_image_generation(self):
        """Test setting VRAM mode to image_generation."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "image_generation"}
        
        # Update mock implementation to change property
        async def side_effect(mode):
            self.mock_manager.vram_mode = mode
            
        self.mock_manager.set_vram_mode.side_effect = side_effect
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "image_generation"
        self.mock_manager.set_vram_mode.assert_called_with(VRAMLoadMode.IMAGE_GENERATION)

    def test_set_vram_mode_video_generation(self):
        """Test setting VRAM mode to video_generation."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "video_generation"}
        
        # Update mock implementation to change property
        async def side_effect(mode):
            self.mock_manager.vram_mode = mode
            
        self.mock_manager.set_vram_mode.side_effect = side_effect
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "video_generation"
        self.mock_manager.set_vram_mode.assert_called_with(VRAMLoadMode.VIDEO_GENERATION)

    def test_set_vram_mode_all(self):
        """Test setting VRAM mode to all."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "all"}
        
        # Update mock implementation to change property
        async def side_effect(mode):
            self.mock_manager.vram_mode = mode
            
        self.mock_manager.set_vram_mode.side_effect = side_effect
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "all"
        self.mock_manager.set_vram_mode.assert_called_with(VRAMLoadMode.ALL)

    def test_set_vram_mode_invalid(self):
        """Test setting invalid VRAM mode."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "invalid_mode"}
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        # Custom validation exception handler returns 400, not 422
        assert response.status_code == 400

    def test_get_vram_mode_unauthorized(self):
        """Test access without API key."""
        # Clear override for this test to ensure auth runs first (though auth is separate dep)
        # Auth usually returns 401 for missing key
        response = client.get("/api/v1/settings/vram-mode")
        assert response.status_code in [401, 403]
