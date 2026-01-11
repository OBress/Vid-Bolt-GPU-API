"""Tests for Settings API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

from unittest.mock import MagicMock, AsyncMock

from app.dependencies import get_model_manager
from app.services.model_manager import ModelManager, VRAMLoadMode, ModelMode

client = TestClient(app)


class TestSettingsAPI:
    """Tests for settings configuration endpoints."""

    def setup_method(self):
        """Setup mock dependency."""
        self.mock_manager = MagicMock(spec=ModelManager)
        self.mock_manager.vram_mode = VRAMLoadMode.DYNAMIC
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
        assert data["mode"] == "dynamic"
        assert "description" in data

    def test_set_vram_mode_dynamic(self):
        """Test setting VRAM mode to dynamic."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "dynamic"}
        
        # Update mock implementation to change property
        async def side_effect(mode):
            self.mock_manager.vram_mode = mode
            
        self.mock_manager.set_vram_mode.side_effect = side_effect
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "dynamic"
        self.mock_manager.set_vram_mode.assert_called_with(VRAMLoadMode.DYNAMIC)

    def test_set_vram_mode_static(self):
        """Test setting VRAM mode to static."""
        headers = {"X-API-Key": "test-api-key-12345"}
        payload = {"mode": "static"}
        
        # Update mock implementation to change property
        async def side_effect(mode):
            self.mock_manager.vram_mode = mode
            
        self.mock_manager.set_vram_mode.side_effect = side_effect
        
        response = client.post("/api/v1/settings/vram-mode", json=payload, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "static"
        self.mock_manager.set_vram_mode.assert_called_with(VRAMLoadMode.STATIC)

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
