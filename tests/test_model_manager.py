"""Tests for ModelManager and mode switching functionality."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestModelManagerBasicFunctionality:
    """Tests for ModelManager basic operations."""

    def test_model_manager_initialization(self):
        """Test ModelManager initializes with correct default state."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, ModelMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        assert manager.current_mode == ModelMode.NONE
        assert manager.is_busy is False
        assert manager.active_job_id is None

    def test_model_manager_get_status(self):
        """Test ModelManager status reporting."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, ModelMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        status = manager.get_status()
        
        assert status.mode == ModelMode.NONE
        assert status.is_busy is False
        assert status.active_job_id is None
        assert status.loaded_models == []

    def test_model_manager_status_to_dict(self):
        """Test ModeStatus serialization."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        status_dict = manager.get_status().to_dict()
        
        assert isinstance(status_dict, dict)
        assert "mode" in status_dict
        assert "is_busy" in status_dict
        assert "active_job_id" in status_dict
        assert "loaded_models" in status_dict


class TestModelManagerJobLock:
    """Tests for ModelManager job locking mechanism."""

    @pytest.mark.asyncio
    async def test_acquire_job_lock_success(self):
        """Test successfully acquiring job lock."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        acquired = await manager.acquire_job_lock("job-123")
        
        assert acquired is True
        assert manager.is_busy is True
        assert manager.active_job_id == "job-123"

    @pytest.mark.asyncio
    async def test_acquire_job_lock_fails_when_busy(self):
        """Test that second job lock acquisition fails."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # First lock should succeed
        await manager.acquire_job_lock("job-1")
        
        # Second lock should fail
        acquired = await manager.acquire_job_lock("job-2")
        
        assert acquired is False
        assert manager.active_job_id == "job-1"

    @pytest.mark.asyncio
    async def test_release_job_lock(self):
        """Test releasing job lock."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        await manager.acquire_job_lock("job-123")
        await manager.release_job_lock("job-123")
        
        assert manager.is_busy is False
        assert manager.active_job_id is None

    @pytest.mark.asyncio
    async def test_release_wrong_job_id_ignored(self):
        """Test that releasing with wrong job ID is ignored."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        await manager.acquire_job_lock("job-123")
        await manager.release_job_lock("wrong-job")
        
        # Should still be busy with original job
        assert manager.is_busy is True
        assert manager.active_job_id == "job-123"

    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self):
        """Test concurrent lock acquisition attempts."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Simulate concurrent lock attempts
        results = await asyncio.gather(
            manager.acquire_job_lock("job-1"),
            manager.acquire_job_lock("job-2"),
            manager.acquire_job_lock("job-3"),
        )
        
        # Only one should succeed
        assert sum(results) == 1
        assert manager.is_busy is True


class TestModelManagerModeGuards:
    """Tests for mode-specific generator access guards."""

    def test_get_image_generator_fails_when_not_in_image_mode(self):
        """Test that getting image generator fails in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        with pytest.raises(RuntimeError, match="Not in Image Mode"):
            manager.get_image_generator()

    def test_get_video_generator_fails_when_not_in_video_mode(self):
        """Test that getting video generator fails in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        with pytest.raises(RuntimeError, match="Not in Video Mode"):
            manager.get_video_generator()

    def test_get_image_editor_fails_when_not_in_image_mode(self):
        """Test that getting image editor fails in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        with pytest.raises(RuntimeError, match="Not in Image Mode"):
            manager.get_image_editor()

    def test_get_upscaler_returns_none_when_not_in_video_mode(self):
        """Test that upscaler returns None in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Should return None, not raise
        result = manager.get_upscaler()
        assert result is None


class TestModeEnum:
    """Tests for ModelMode enum values."""

    def test_mode_enum_values(self):
        """Test ModelMode enum has expected values."""
        from app.services.model_manager import ModelMode
        
        assert ModelMode.NONE.value == "none"
        assert ModelMode.IMAGE.value == "image"
        assert ModelMode.VIDEO.value == "video"
        assert ModelMode.SWITCHING.value == "switching"

    def test_mode_enum_string_inheritance(self):
        """Test ModelMode inherits from str."""
        from app.services.model_manager import ModelMode
        
        # Should be usable as string
        assert isinstance(ModelMode.IMAGE.value, str)
        assert str(ModelMode.IMAGE) == "ModelMode.IMAGE"
