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


class TestVRAMMode:
    """Tests for VRAM loading mode functionality."""

    @pytest.mark.asyncio
    async def test_set_vram_mode_static_loads_all_models(self):
        """Test valid VRAM mode transition to static."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Mock loading methods
        manager._load_image_models = AsyncMock()
        manager._load_video_models = AsyncMock()
        
        await manager.set_vram_mode(VRAMLoadMode.STATIC)
        
        assert manager.vram_mode == VRAMLoadMode.STATIC
        manager._load_image_models.assert_called_once()
        manager._load_video_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_vram_mode_dynamic_unloads_unused(self):
        """Test valid VRAM mode transition to dynamic unloads unused."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode, ModelMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        manager._mode = ModelMode.IMAGE
        manager._vram_mode = VRAMLoadMode.STATIC # Start in static
        
        # Mock unloading methods
        manager._unload_video_models = AsyncMock()
        manager._unload_image_models = AsyncMock()
        
        # Switch to dynamic while in Image mode -> should unload video
        await manager.set_vram_mode(VRAMLoadMode.DYNAMIC)
        
        assert manager.vram_mode == VRAMLoadMode.DYNAMIC
        manager._unload_video_models.assert_called_once()
        manager._unload_image_models.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_mode_static_prevents_unloading(self):
        """Test that switching modes in static VRAM mode prevents unloading."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode, ModelMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        manager._vram_mode = VRAMLoadMode.STATIC
        manager._mode = ModelMode.IMAGE
        
        # Mock methods
        manager._unload_video_models = AsyncMock()
        manager._load_video_models = AsyncMock()
        manager._unload_image_models = AsyncMock()
        manager._load_image_models = AsyncMock()
        
        # Switch to video mode
        # Since we are static, verify we DO NOT unload image models
        # We still "load" video models (idempotent, effectively checks they are loaded)
        
        # NOTE: logic in switch_to_video_mode does:
        # if static: skips unload
        # calls load_video_models
        
        await manager.switch_to_video_mode()
        
        manager._unload_image_models.assert_not_called()
        manager._load_video_models.assert_called_once()
        assert manager.current_mode == ModelMode.VIDEO
