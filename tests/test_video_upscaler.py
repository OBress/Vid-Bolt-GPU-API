"""Tests for Stream-DiffVSR Video Upscaler."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables before importing modules
os.environ["MOCK_MODE"] = "true"
os.environ["STREAM_DIFFVSR_DRY_RUN"] = "true"
os.environ["STREAM_DIFFVSR_ENABLED"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"


class TestStreamDiffVSRUpscaler:
    """Test suite for StreamDiffVSRUpscaler."""

    @pytest.fixture
    def settings(self):
        """Create test settings with dry-run enabled."""
        from app.config import Settings
        return Settings(
            mock_mode=False,
            stream_diffvsr_enabled=True,
            stream_diffvsr_dry_run=True,
        )

    @pytest.fixture
    def sample_video_bytes(self) -> bytes:
        """Create minimal sample video bytes for testing.
        
        In dry-run mode, the actual content doesn't matter since
        the video is passed through unchanged.
        """
        # Minimal bytes representing a "video" for testing
        return b"FAKE_VIDEO_DATA_FOR_TESTING" * 100

    def test_init(self, settings):
        """Test upscaler initialization."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler
        
        upscaler = StreamDiffVSRUpscaler(settings)
        
        assert upscaler.is_loaded is False
        assert upscaler.dry_run is True
        assert upscaler.components is None

    def test_load_models_dry_run(self, settings):
        """Test model loading in dry-run mode."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler
        
        upscaler = StreamDiffVSRUpscaler(settings)
        upscaler.load_models()
        
        assert upscaler.is_loaded is True
        # Components remain None in dry-run
        assert upscaler.components is None
        # Temp directory should be created
        assert upscaler._temp_dir is not None

    @pytest.mark.asyncio
    async def test_upscale_video_dry_run(self, settings, sample_video_bytes):
        """Test video upscaling in dry-run mode."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler, UpscaleParams
        
        upscaler = StreamDiffVSRUpscaler(settings)
        upscaler.load_models()
        
        params = UpscaleParams(
            job_id="test-upscale-001",
            video_data=sample_video_bytes,
            preserve_audio=True,
        )
        
        result = await upscaler.upscale_video(params)
        
        # In dry-run, video passes through unchanged
        assert result.video_data == sample_video_bytes
        assert result.was_upscaled is False
        # Dry-run assumes 720p -> 1080p
        assert result.original_width == 1280
        assert result.original_height == 720
        assert result.upscaled_width == 1920
        assert result.upscaled_height == 1080
        assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_upscale_video_not_loaded_raises(self, settings, sample_video_bytes):
        """Test that calling upscale without loading raises error."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler, UpscaleParams
        
        upscaler = StreamDiffVSRUpscaler(settings)
        # Don't call load_models()
        
        params = UpscaleParams(
            job_id="test-upscale-002",
            video_data=sample_video_bytes,
        )
        
        with pytest.raises(RuntimeError, match="not loaded"):
            await upscaler.upscale_video(params)

    def test_get_status(self, settings):
        """Test status retrieval."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler
        
        upscaler = StreamDiffVSRUpscaler(settings)
        upscaler.load_models()
        
        status = upscaler.get_status()
        
        assert status["is_loaded"] is True
        assert status["dry_run"] is True
        assert status["enabled"] is True
        assert status["service_type"] == "StreamDiffVSRUpscaler"
        assert "model_id" in status
        assert "device" in status
        assert "num_inference_steps" in status

    def test_get_status_not_loaded(self, settings):
        """Test status before loading."""
        from app.services.video_upscaler import StreamDiffVSRUpscaler
        
        upscaler = StreamDiffVSRUpscaler(settings)
        
        status = upscaler.get_status()
        
        assert status["is_loaded"] is False
        assert status["dry_run"] is True


class TestUpscaleParams:
    """Test UpscaleParams dataclass."""

    def test_defaults(self):
        """Test default parameter values."""
        from app.services.video_upscaler import UpscaleParams
        
        params = UpscaleParams(
            job_id="test-001",
            video_data=b"fake_video",
        )
        
        assert params.job_id == "test-001"
        assert params.video_data == b"fake_video"
        assert params.preserve_audio is True


class TestUpscaleResult:
    """Test UpscaleResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        from app.services.video_upscaler import UpscaleResult
        
        result = UpscaleResult(
            video_data=b"upscaled_video",
            original_width=1280,
            original_height=720,
            upscaled_width=1920,
            upscaled_height=1080,
            frame_count=96,
            processing_time_seconds=5.5,
            was_upscaled=True,
        )
        
        assert result.original_width == 1280
        assert result.original_height == 720
        assert result.upscaled_width == 1920
        assert result.upscaled_height == 1080
        assert result.frame_count == 96
        assert result.was_upscaled is True

    def test_9_16_aspect_ratio(self):
        """Test result with 9:16 aspect ratio (portrait)."""
        from app.services.video_upscaler import UpscaleResult
        
        result = UpscaleResult(
            video_data=b"upscaled_video",
            original_width=720,
            original_height=1280,
            upscaled_width=1080,
            upscaled_height=1920,
            frame_count=96,
            processing_time_seconds=5.5,
            was_upscaled=True,
        )
        
        # Verify portrait aspect ratio is preserved
        assert result.original_width == 720
        assert result.original_height == 1280
        assert result.upscaled_width == 1080
        assert result.upscaled_height == 1920


class TestLTX2UpscalerIntegration:
    """Test LTX-2 generator integration with upscaler."""

    @pytest.fixture
    def settings_with_upscaler(self):
        """Create test settings with both LTX-2 and upscaler dry-run enabled."""
        from app.config import Settings
        return Settings(
            mock_mode=False,
            ltx2_dry_run=True,
            stream_diffvsr_enabled=True,
            stream_diffvsr_dry_run=True,
        )

    def test_set_upscaler(self, settings_with_upscaler):
        """Test connecting upscaler to LTX-2 generator."""
        from app.services.ltx2_generator import LTX2Generator
        from app.services.video_upscaler import StreamDiffVSRUpscaler
        
        generator = LTX2Generator(settings_with_upscaler)
        upscaler = StreamDiffVSRUpscaler(settings_with_upscaler)
        
        # Initially no upscaler
        assert generator._upscaler is None
        
        # Connect upscaler
        generator.set_upscaler(upscaler)
        
        assert generator._upscaler is upscaler

    def test_generator_has_set_upscaler_method(self, settings_with_upscaler):
        """Test that LTX2Generator has set_upscaler method for DI."""
        from app.services.ltx2_generator import LTX2Generator
        
        generator = LTX2Generator(settings_with_upscaler)
        
        assert hasattr(generator, "set_upscaler")
        assert callable(generator.set_upscaler)
