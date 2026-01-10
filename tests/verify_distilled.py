import os
import pytest
from app.config import get_settings
from app.services.ltx2_generator import LTX2Generator, KeyframeInterpolationParams

@pytest.fixture
def generator():
    settings = get_settings()
    # Force mock mode for standalone testing if not on GPU
    settings.mock_mode = True 
    return LTX2Generator(settings)

@pytest.mark.asyncio
async def test_distilled_workflow_integration(generator):
    """Verify that the generator is correctly configured for the distilled pipeline."""
    generator.load_models()
    
    assert generator.is_loaded
    # Check if the correct checkpoint is configured
    assert "distilled-fp8" in generator.settings.ltx2_checkpoint_path
    
    # Test I2V params formatting
    params = KeyframeInterpolationParams(
        job_id="test-distilled",
        prompt="A futuristic city",
        negative_prompt="",
        keyframes=[(b"fake-image-data", 0, 1.0)],
        duration_seconds=1.0,
        frame_rate=24.0,
        width=1024,
        height=576,
        seed=42
    )
    
    # In mock mode, this should return successfully with dummy data
    result = await generator.generate_keyframe_video(params)
    
    assert result.video_data is not None
    assert isinstance(result.video_data, bytes)
    print("\n✅ Distilled workflow integration verified (Dry Run)")

if __name__ == "__main__":
    pytest.main([__file__])
