"""Tests for LTX-2 Video Generator."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Set test environment variables before importing modules
os.environ["MOCK_MODE"] = "true"
os.environ["LTX2_DRY_RUN"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"


class TestLTX2Generator:
    """Test suite for LTX2Generator."""

    @pytest.fixture
    def settings(self):
        """Create test settings with dry-run enabled."""
        from app.config import Settings
        return Settings(
            mock_mode=False,
            ltx2_dry_run_override=True,
        )

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Create a sample PNG image for testing."""
        img = Image.new("RGB", (512, 512), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    def test_init(self, settings):
        """Test generator initialization."""
        from app.services.ltx2_generator import LTX2Generator
        
        generator = LTX2Generator(settings)
        
        assert generator.is_loaded is False
        assert generator.dry_run is True
        assert generator.components is None

    def test_load_models_dry_run(self, settings):
        """Test model loading in dry-run mode."""
        from app.services.ltx2_generator import LTX2Generator
        
        generator = LTX2Generator(settings)
        generator.load_models()
        
        assert generator.is_loaded is True
        # Components remain None in dry-run
        assert generator.components is None
        # Temp directory should be created
        assert generator._temp_dir is not None

    @pytest.mark.asyncio
    async def test_generate_video_dry_run(self, settings, sample_image_bytes):
        """Test I2V video generation in dry-run mode."""
        from app.services.ltx2_generator import LTX2Generator, LTX2VideoParams
        
        generator = LTX2Generator(settings)
        generator.load_models()
        
        params = LTX2VideoParams(
            job_id="test-i2v-001",
            prompt="A calm ocean scene",
            negative_prompt="",
            input_image_data=sample_image_bytes,
            end_image_data=None,
            duration_seconds=3.0,
            frame_rate=24.0,
            width=1024,
            height=576,
            seed=42,
        )
        
        result = await generator.generate_video(params)
        
        assert result.video_data is not None
        assert len(result.video_data) > 0
        assert result.width == 1024
        assert result.height == 576
        assert result.seed == 42
        assert result.duration_seconds == 3.0
        # Dry-run doesn't generate audio
        assert result.has_audio is False

    @pytest.mark.asyncio
    async def test_generate_keyframe_video_dry_run(self, settings, sample_image_bytes):
        """Test keyframe interpolation in dry-run mode."""
        from app.services.ltx2_generator import LTX2Generator, KeyframeInterpolationParams
        
        generator = LTX2Generator(settings)
        generator.load_models()
        
        params = KeyframeInterpolationParams(
            job_id="test-keyframe-001",
            prompt="A person walking",
            negative_prompt="",
            keyframes=[
                (sample_image_bytes, 0, 1.0),
                (sample_image_bytes, 72, 1.0),  # ~3 seconds at 24fps
            ],
            duration_seconds=3.0,
            frame_rate=24.0,
            width=1024,
            height=576,
            seed=123,
        )
        
        result = await generator.generate_keyframe_video(params)
        
        assert result.video_data is not None
        assert len(result.video_data) > 0
        assert result.width == 1024
        assert result.height == 576
        assert result.seed == 123
        assert result.duration_seconds == 3.0

    def test_get_status(self, settings):
        """Test status retrieval."""
        from app.services.ltx2_generator import LTX2Generator
        
        generator = LTX2Generator(settings)
        generator.load_models()
        
        status = generator.get_status()
        
        assert status["is_loaded"] is True
        assert status["dry_run"] is True
        assert status["generator_type"] == "LTX2Generator"
        assert "checkpoint_path" in status
        assert "spatial_upsampler_path" in status
        assert "device" in status


class TestFrameRounding:
    """Test frame rounding utility function."""

    def test_round_up_to_valid_frames_already_valid(self):
        """Test with already valid frame counts."""
        from app.models.ltx2_generation import round_up_to_valid_frames
        
        # 8k+1 values should remain unchanged
        assert round_up_to_valid_frames(9) == 9
        assert round_up_to_valid_frames(17) == 17
        assert round_up_to_valid_frames(121) == 121
        assert round_up_to_valid_frames(241) == 241

    def test_round_up_to_valid_frames_round_up(self):
        """Test rounding up to nearest valid value."""
        from app.models.ltx2_generation import round_up_to_valid_frames
        
        assert round_up_to_valid_frames(10) == 17
        assert round_up_to_valid_frames(15) == 17
        assert round_up_to_valid_frames(16) == 17
        assert round_up_to_valid_frames(100) == 105
        assert round_up_to_valid_frames(120) == 121

    def test_round_up_to_valid_frames_minimum(self):
        """Test minimum frame count enforcement."""
        from app.models.ltx2_generation import round_up_to_valid_frames
        
        assert round_up_to_valid_frames(1) == 9
        assert round_up_to_valid_frames(5) == 9
        assert round_up_to_valid_frames(8) == 9
