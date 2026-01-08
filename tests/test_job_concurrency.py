"""Tests for JobManager concurrency limits (Fail-Fast)."""

import pytest
from unittest.mock import patch, MagicMock
from app.services.job_manager import JobManager
from app.config import Settings

@pytest.fixture
def restricted_job_manager():
    """Create a JobManager with very strictly limited concurrency."""
    from app.config import InferenceConfig, get_settings
    
    # Store original values
    orig_image_limit = InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS
    orig_video_limit = InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS
    
    # Patch the class attributes directly since Settings properties read from here
    InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS = 1
    InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS = 1
    
    settings = get_settings()
    manager = JobManager(settings)
    
    yield manager
    
    # Restore original values
    InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS = orig_image_limit
    InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS = orig_video_limit

# We need to mock the dependency injection to use our restricted manager
@pytest.fixture
async def async_client_with_limits(restricted_job_manager):
    from app.main import app
    from app.dependencies import get_job_manager
    import httpx
    
    app.dependency_overrides[get_job_manager] = lambda: restricted_job_manager
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        client.app = app # Attach app to client for test access
        yield client 
    app.dependency_overrides.pop(get_job_manager, None)

@pytest.mark.asyncio
async def test_fail_fast_concurrency(async_client_with_limits, api_key_headers, mock_model_manager):
    """Test that the 2nd request fails immediately with 429 when limit is 1."""
    
    # We need to simulate a "long running" job.
    # Since we can't easily make the background task sleep in the actual app 
    # without modifying the generator code, we'll manually acquire the semaphore 
    # on the manager instance *before* making the request.
    
    # Access the manager instance used by the client
    # (It's the one from our fixture)
    
    from app.dependencies import get_job_manager
    # Access app from client
    manager = async_client_with_limits.app.dependency_overrides[get_job_manager]()
    
    # Manually lock the semaphore to simulate a busy state
    await manager.image_semaphore.acquire()
    
    # Now try to generating an image
    payload = {
        "job_id": "job-rejected-1",
        "prompt": "test",
        "aspect_ratio": "1:1",
        "save_url": "http://g.com/1.png"
    }
    
    response = await async_client_with_limits.post(
        "/api/v1/image/generate", 
        headers=api_key_headers, 
        json=payload
    )
        
    # Should be 429 because we held the only slot
    assert response.status_code == 429
    assert "System busy" in response.json()["detail"]
    
    # Release to cleanup
    manager.image_semaphore.release()
    
    # Now it should work
    response = await async_client_with_limits.post(
        "/api/v1/image/generate", 
        headers=api_key_headers, 
        json=payload
    )
    assert response.status_code == 202

