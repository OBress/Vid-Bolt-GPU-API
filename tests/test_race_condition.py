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
async def test_race_condition_concurrency(async_client_race, api_key_headers, mock_model_manager):
    """Test that concurrent requests obey the limit (1) despite race conditions."""
    
    # We need to ensure the "generation" takes some time so the lock is held
    # The MockGenerator usually returns instantly.
    # We can mock the _run_job_internal method of the manager to simply sleep
    # But wait, we want to test the full flow.
    # A better way is to patch the generator's generate_image to sleep.
    
    # Retrieve the generator instance from the app
    # In tests/conftest.py, verify_distilled.py etc, we see patterns.
    # Since we are using an async client which starts the app, the generator should be initialized.
    # However, get_generator() is a dependency function.
    
    from app.config import get_settings
    from app.dependencies import get_generator, set_generator_instance
    from app.main import app
    
    settings = get_settings()
    # We need to manually initialize/get it if we want to patch it
    generator = get_generator(settings)
    
    # We need to make sure the app uses THIS generator instance
    # The async_client uses the app. The app uses get_generator dependency.
    # We should override the dependency to return our instance that we will patch.
    
    app.dependency_overrides[get_generator] = lambda: generator
    
    # We need to make the generator sleep a bit to hold the lock
    original_generate = generator.generate_image
    
    async def slow_generate(*args, **kwargs):
        # Sleep to ensure all requests arrive while lock is held
        await asyncio.sleep(0.5) 
        return await original_generate(*args, **kwargs)
        
    with pytest.MonkeyPatch.context() as m:
        m.setattr(generator, "generate_image", slow_generate)
        
        # Fire 5 requests concurrently
        tasks = []
        for i in range(5):
            payload = {
                "job_id": f"race-job-{i}",
                "prompt": "test",
                "aspect_ratio": "1:1",
                "save_url": f"http://g.com/{i}.png"
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
        
        print(f"Accepted: {accepted}, Rejected: {rejected}")
        print(f"Status codes: {status_codes}")
        
        # With limit 1 and 5 requests, exactly 1 should be accepted
        assert accepted == 1
        assert rejected == 4
