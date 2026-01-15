import pytest
import asyncio
from app.config import InferenceConfig, get_settings
from app.services.job_manager import JobManager

@pytest.fixture
def limited_job_manager():
    """Create a JobManager with concurrency limit of 1."""
    # Store original values
    orig_image_limit = InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS
    InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS = 1
    
    settings = get_settings()
    manager = JobManager(settings)
    
    yield manager
    
    # Restore
    InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS = orig_image_limit

@pytest.fixture
async def async_client_race(limited_job_manager):
    from app.main import app
    from app.dependencies import get_job_manager
    import httpx
    
    app.dependency_overrides[get_job_manager] = lambda: limited_job_manager
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.pop(get_job_manager, None)

@pytest.mark.asyncio
async def test_race_condition_concurrency_queuing(async_client_race, api_key_headers):
    """Test that concurrent requests are all queued successfully (202 Accepted).
    
    The new system uses an unbounded queue (or very large queue), so we expect
    all requests to be accepted (202), not rejected (429).
    The concurrency limit is enforced at execution time, not submission time.
    """
    
    # Fire 5 requests concurrently
    tasks = []
    for i in range(5):
        payload = {
            "job_id": f"race-job-{i}",
            "prompt": "test",
            "aspect_ratio": "1:1",
            "save_url": f"http://g.com/{i}.png",
            "webhook_url": "http://webhook.test"
        }
        tasks.append(
            async_client_race.post(
                "/api/v1/image/generate", 
                headers=api_key_headers, 
                json=payload
            )
        )
        
    responses = await asyncio.gather(*tasks)
    
    # Check results
    status_codes = [r.status_code for r in responses]
    accepted = status_codes.count(202)
    rejected = status_codes.count(429)
    
    # Expect ALL accepted
    assert accepted == 5
    assert rejected == 0
