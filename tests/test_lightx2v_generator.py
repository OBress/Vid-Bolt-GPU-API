"""Tests for LightX2V image editing generator."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Set test environment variables before importing app modules
os.environ["MOCK_MODE"] = "true"
os.environ["LIGHTX2V_DRY_RUN"] = "true"


class TestLightX2VImageEditGenerator:
    """Test suite for LightX2VImageEditGenerator."""

    @pytest.fixture
    def settings(self):
        """Create mock settings for testing."""
        from app.config import Settings
        
        settings = Settings(
            mock_mode=False,
            lightx2v_dry_run_override=True,  # Always use dry-run for tests
        )
        return settings

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Create a sample PNG image for testing."""
        img = Image.new("RGB", (512, 512), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    def test_init(self, settings):
        """Test generator initialization."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        generator = LightX2VImageEditGenerator(settings)
        
        assert generator.settings == settings
        assert generator.components is None
        assert generator.is_loaded is False
        assert generator.dry_run is True

    def test_load_models_dry_run(self, settings):
        """Test loading models in dry-run mode."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        assert generator.is_loaded is True
        # In dry-run mode, components should remain None
        assert generator.components is None

    def test_get_status(self, settings):
        """Test getting generator status."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        status = generator.get_status()
        
        assert status["generator_type"] == "LightX2VImageEditGenerator"
        assert status["model"] == "Qwen-Image-Edit-2511"
        assert status["is_loaded"] is True
        assert status["dry_run"] is True
        assert status["infer_steps"] == 8
        assert status["guidance_scale"] == 1.0

    @pytest.mark.asyncio
    async def test_edit_image_dry_run(self, settings, sample_image_bytes):
        """Test image editing in dry-run mode."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        from app.services.mock_generator import ImageEditParams
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        params = ImageEditParams(
            job_id="test-job-123",
            input_image_data=sample_image_bytes,
            prompt="Make the image more vibrant",
            width=512,
            height=512,
            mask_data=None,
            seed=42,
        )
        
        result = await generator.edit_image(params)
        
        assert result.image_data is not None
        assert len(result.image_data) > 0
        assert result.original_width == 512
        assert result.original_height == 512
        assert result.seed == 42
        
        # Verify the result is a valid image
        img = Image.open(io.BytesIO(result.image_data))
        assert img.size == (512, 512)

    @pytest.mark.asyncio
    async def test_edit_image_random_seed(self, settings, sample_image_bytes):
        """Test image editing with random seed generation."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        from app.services.mock_generator import ImageEditParams
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        params = ImageEditParams(
            job_id="test-job-456",
            input_image_data=sample_image_bytes,
            prompt="Add a sunset effect",
            width=1024,
            height=768,
            mask_data=None,
            seed=None,  # Random seed
        )
        
        result = await generator.edit_image(params)
        
        assert result.image_data is not None
        assert result.seed is not None
        assert result.seed >= 0

    @pytest.mark.asyncio
    async def test_edit_image_not_loaded(self, settings, sample_image_bytes):
        """Test that editing fails when models are not loaded."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        from app.services.mock_generator import ImageEditParams
        
        generator = LightX2VImageEditGenerator(settings)
        # Don't call load_models()
        
        params = ImageEditParams(
            job_id="test-job-789",
            input_image_data=sample_image_bytes,
            prompt="Test prompt",
            width=512,
            height=512,
            mask_data=None,
            seed=42,
        )
        
        with pytest.raises(RuntimeError, match="not loaded"):
            await generator.edit_image(params)

    @pytest.mark.asyncio
    async def test_generate_image_not_implemented(self, settings):
        """Test that generate_image raises NotImplementedError."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        from app.services.mock_generator import ImageGenerationParams
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        params = ImageGenerationParams(
            job_id="test-job",
            prompt="A test prompt",
            width=512,
            height=512,
            seed=42,
            num_inference_steps=8,
        )
        
        with pytest.raises(NotImplementedError, match="image editing only"):
            await generator.generate_image(params)

    @pytest.mark.asyncio
    async def test_generate_video_not_implemented(self, settings):
        """Test that generate_video raises NotImplementedError."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        generator = LightX2VImageEditGenerator(settings)
        generator.load_models()
        
        # Create a minimal mock params object
        class MockVideoParams:
            job_id = "test"
        
        with pytest.raises(NotImplementedError, match="does not support video"):
            await generator.generate_video(MockVideoParams())


class TestLightX2VIntegration:
    """Integration tests for LightX2V with the API."""

    @pytest.fixture
    def client(self):
        """Create a test client with LightX2V dry-run mode."""
        # Set environment for LightX2V dry-run
        os.environ["MOCK_MODE"] = "false"
        os.environ["LIGHTX2V_DRY_RUN_OVERRIDE"] = "true"
        os.environ["API_KEY"] = "test-api-key-12345"
        
        # Need to reimport after setting env vars
        from importlib import reload
        import app.config as app_config
        reload(app_config)
        
        from app.main import app
        from fastapi.testclient import TestClient
        
        with TestClient(app) as test_client:
            yield test_client
        
        # Reset to mock mode
        os.environ["MOCK_MODE"] = "true"
        reload(app_config)

    @pytest.fixture
    def api_key_headers(self) -> dict[str, str]:
        """Return headers with valid API key."""
        return {"X-API-Key": "test-api-key-12345"}

    def test_health_check_with_lightx2v(self, client):
        """Test health endpoint when LightX2V is loaded."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
