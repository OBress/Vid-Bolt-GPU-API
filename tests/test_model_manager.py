"""Tests for ModelManager and mode switching functionality."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestModelManagerBasicFunctionality:
    """Tests for ModelManager basic operations."""

    def test_model_manager_initialization(self):
        """Test ModelManager initializes with correct default state."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Default mode is IMAGE_GENERATION
        assert manager.current_mode == VRAMLoadMode.IMAGE_GENERATION
        assert manager.is_busy is False
        assert manager.active_job_id is None

    def test_model_manager_get_status(self):
        """Test ModelManager status reporting."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        status = manager.get_status()
        
        assert status.mode == VRAMLoadMode.IMAGE_GENERATION
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

    def test_get_image_generator_fails_when_not_loaded(self):
        """Test that getting image generator fails when not loaded."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Mode is IMAGE_GENERATION by default, but generator not loaded
        with pytest.raises(RuntimeError, match="Z-Image generator not loaded"):
            manager.get_image_generator()

    def test_get_video_generator_fails_when_wrong_mode(self):
        """Test that getting video generator fails in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Default mode is IMAGE_GENERATION, not VIDEO_GENERATION
        with pytest.raises(RuntimeError, match="Not in valid mode for video generation"):
            manager.get_video_generator()

    def test_get_image_editor_fails_when_wrong_mode(self):
        """Test that getting image editor fails in wrong mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Default mode is IMAGE_GENERATION, not IMAGE_EDITING
        with pytest.raises(RuntimeError, match="Not in valid mode for image editing"):
            manager.get_image_editor()


class TestVRAMLoadModeEnum:
    """Tests for VRAMLoadMode enum values."""

    def test_vram_mode_enum_values(self):
        """Test VRAMLoadMode enum has expected values."""
        from app.services.model_manager import VRAMLoadMode
        
        assert VRAMLoadMode.IMAGE_GENERATION.value == "image_generation"
        assert VRAMLoadMode.IMAGE_EDITING.value == "image_editing"
        assert VRAMLoadMode.VIDEO_GENERATION.value == "video_generation"
        assert VRAMLoadMode.ALL.value == "all"

    def test_job_type_enum_values(self):
        """Test JobType enum has expected values."""
        from app.services.model_manager import JobType
        
        assert JobType.IMAGE_GENERATION.value == "image_generation"
        assert JobType.IMAGE_EDITING.value == "image_editing"
        assert JobType.VIDEO_GENERATION.value == "video_generation"


class TestVRAMMode:
    """Tests for VRAM loading mode functionality."""

    @pytest.mark.asyncio
    async def test_set_vram_mode_all_loads_all_models(self):
        """Test valid VRAM mode transition to ALL.
        
        Note: In ALL mode, Z-Image is NOT loaded initially to provide VRAM headroom
        for video generation. It loads dynamically when an image gen request comes in.
        """
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Mock loading methods
        manager._load_zimage = AsyncMock()
        manager._unload_zimage = AsyncMock()
        manager._load_lightx2v = AsyncMock()
        manager._load_ltx2 = AsyncMock()
        
        await manager.set_vram_mode(VRAMLoadMode.ALL)
        
        assert manager.vram_mode == VRAMLoadMode.ALL
        # Z-Image should NOT be loaded in ALL mode (deferred for VRAM headroom)
        manager._load_zimage.assert_not_called()
        # But LightX2V and LTX-2 should be loaded
        manager._load_lightx2v.assert_called_once()
        manager._load_ltx2.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_vram_mode_image_generation(self):
        """Test switching to image_generation mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        manager._mode = VRAMLoadMode.VIDEO_GENERATION  # Start in different mode
        
        # Mock methods
        manager._load_zimage = AsyncMock()
        manager._unload_lightx2v = AsyncMock()
        manager._unload_ltx2 = AsyncMock()
        
        await manager.set_vram_mode(VRAMLoadMode.IMAGE_GENERATION)
        
        assert manager.vram_mode == VRAMLoadMode.IMAGE_GENERATION
        manager._load_zimage.assert_called_once()
        manager._unload_lightx2v.assert_called_once()
        manager._unload_ltx2.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_vram_mode_video_generation(self):
        """Test switching to video_generation mode."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode
        
        settings = get_settings()
        manager = ModelManager(settings)
        
        # Mock methods
        manager._load_ltx2 = AsyncMock()
        manager._unload_zimage = AsyncMock()
        manager._unload_lightx2v = AsyncMock()
        
        await manager.set_vram_mode(VRAMLoadMode.VIDEO_GENERATION)
        
        assert manager.vram_mode == VRAMLoadMode.VIDEO_GENERATION
        manager._load_ltx2.assert_called_once()
        manager._unload_zimage.assert_called_once()
        manager._unload_lightx2v.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_mode_for_job_in_all_mode(self):
        """Test that ALL mode can handle any job type."""
        from app.config import get_settings
        from app.services.model_manager import ModelManager, VRAMLoadMode, JobType
        
        settings = get_settings()
        manager = ModelManager(settings)
        manager._mode = VRAMLoadMode.ALL
        manager._loaded = True
        
        # ALL mode should accept any job type without switching
        result = await manager.ensure_mode_for_job(JobType.VIDEO_GENERATION)
        assert result is True
        assert manager._mode == VRAMLoadMode.ALL  # Mode unchanged
