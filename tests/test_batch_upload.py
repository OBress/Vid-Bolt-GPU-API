
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
import time

from app.services.job_manager import JobManager
from app.services.model_manager import JobType, VRAMLoadMode
from app.models.internal import ImageGenerationParams, ImageGenerationResult
from app.config import Settings

@pytest.mark.asyncio
async def test_batch_upload_flow():
    """Verify that batch processing uploads images to R2."""
    
    # 1. Setup Mocks
    settings = MagicMock(spec=Settings)
    settings.mock_mode = True
    
    # Mock Storage
    mock_storage = AsyncMock()
    mock_storage.upload_to_url.return_value = "https://r2.example.com/uploaded.png"
    
    # Mock Generator
    mock_generator = AsyncMock()
    # Mock generate_batch to return 2 results
    mock_generator.generate_batch.return_value = [
        ImageGenerationResult(
            image_data=b"fake_image_1",
            width=1024,
            height=1024,
            seed=123
        ),
        ImageGenerationResult(
            image_data=b"fake_image_2",
            width=1024,
            height=1024,
            seed=456
        )
    ]
    
    # Mock Model Manager
    mock_model_manager = MagicMock()
    mock_model_manager.get_image_generator.return_value = mock_generator
    mock_model_manager.ensure_mode_for_job = AsyncMock()
    
    # Mock Webhook Service
    mock_webhook = AsyncMock()
    
    # 2. Initialize JobManager
    job_manager = JobManager(settings, mock_model_manager)
    job_manager.set_webhook_service(mock_webhook)
    
    # 3. Submit 2 Jobs
    # Note: We must pass 'storage' and 'save_url' in kwargs as the routers do
    params = ImageGenerationParams(
        job_id="test-1",
        prompt="test",
        width=1024,
        height=1024,
        seed=123,
        num_inference_steps=20
    )
    
    await job_manager.submit_job(
        job_id="job-1",
        job_type=JobType.IMAGE_GENERATION,
        task_func=AsyncMock(),
        webhook_url="http://webhook.com",
        # Router args:
        params=params,
        storage=mock_storage,
        save_url="https://r2.example.com/upload-1"
    )
    
    await job_manager.submit_job(
        job_id="job-2",
        job_type=JobType.IMAGE_GENERATION,
        task_func=AsyncMock(),
        webhook_url="http://webhook.com",
        # Router args:
        params=params, 
        storage=mock_storage,
        save_url="https://r2.example.com/upload-2"
    )
    
    # 4. Trigger Batch Processing Manually
    # Instead of running the full worker loop, we'll manually call _process_batch
    # to avoid race conditions and timeouts in tests
    
    batch_job_ids = ["job-1", "job-2"]
    bucket_key = (1024, 1024, JobType.IMAGE_GENERATION)
    
    await job_manager._process_batch(batch_job_ids, bucket_key)
    
    # 5. Verify Uploads
    assert mock_storage.upload_to_url.call_count == 2
    
    # Verify call arguments
    calls = mock_storage.upload_to_url.call_args_list
    
    # Check first upload
    assert calls[0].kwargs['url'] == "https://r2.example.com/upload-1"
    assert calls[0].kwargs['data'] == b"fake_image_1"
    
    # Check second upload
    assert calls[1].kwargs['url'] == "https://r2.example.com/upload-2"
    assert calls[1].kwargs['data'] == b"fake_image_2"
    
    # 6. Verify Job Results contain the uploaded URL
    job1 = job_manager.get_job("job-1")
    assert job1.result.save_url == "https://r2.example.com/uploaded.png"
    
    job2 = job_manager.get_job("job-2")
    assert job2.result.save_url == "https://r2.example.com/uploaded.png"

    # 7. Verify Webhooks
    assert mock_webhook.deliver.call_count == 2
