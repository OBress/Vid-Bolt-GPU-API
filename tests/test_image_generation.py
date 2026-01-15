"""Tests for image generation endpoint."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_generate_image_success(async_client, api_key_headers, mock_storage, mock_job_manager, mock_model_manager):
    """Test successful image generation request (Async)."""
    payload = {
        "job_id": "test-job-1",
        "prompt": "A beautiful sunset",
        "aspect_ratio": "16:9",
        "save_url": "https://r2.example.com/output.png",
        "webhook_url": "http://webhook.test"
    }
    
    # 1. Initiate Generation
    response = await async_client.post("/api/v1/image/generate", headers=api_key_headers, json=payload)
    
    # Assert accepted
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "test-job-1"
    assert "status_url" in data
    
    # 2. Poll for completion
    import asyncio
    max_retries = 50 
    
    status_data = None
    for _ in range(max_retries):
        status_response = await async_client.get(data["status_url"], headers=api_key_headers)
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        if status_data["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.2)
        
    assert status_data["status"] == "completed"
    assert "generation_time" in status_data["result"]
    assert status_data["result"]["save_url"] == "https://r2.example.com/output.png"


@pytest.mark.asyncio
async def test_generate_image_with_output_url(async_client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test image generation with a custom output URL."""
    custom_url = "https://custom-storage.com/upload/here?token=123"
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "prompt": "A beautiful sunset",
            "save_url": custom_url,
            "webhook_url": "http://webhook.test",
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == sample_job_id
    
    # Poll for completion to verify upload
    import asyncio
    max_retries = 60
    for _ in range(max_retries):
        status_response = await async_client.get(data["status_url"], headers=api_key_headers)
        if status_response.json()["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.5)

    final_status = (await async_client.get(data["status_url"], headers=api_key_headers)).json()
    assert final_status["status"] == "completed"

    mock_storage.upload_to_url.assert_called_once()
    call_args = mock_storage.upload_to_url.call_args
    assert call_args.kwargs["url"] == custom_url


@pytest.mark.asyncio
async def test_generate_image_custom_dimensions(async_client, api_key_headers, mock_storage, sample_job_id, mock_job_manager, mock_model_manager):
    """Test image generation with custom width and height."""
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "prompt": "A beautiful sunset",
            "width": 512,
            "height": 512,
            "save_url": "https://example.com/save.png",
            "webhook_url": "http://webhook.test",
        },
    )

    assert response.status_code == 202
    data = response.json()
    
    # Poll
    import asyncio
    max_retries = 50
    for _ in range(max_retries):
        status_response = await async_client.get(data["status_url"], headers=api_key_headers)
        if status_response.json()["status"] == "completed":
            break
        await asyncio.sleep(0.2)
        
    status = (await async_client.get(data["status_url"], headers=api_key_headers)).json()
    assert status["status"] == "completed"


@pytest.mark.asyncio
async def test_generate_image_custom_dimensions_validation(async_client, api_key_headers, mock_model_manager):
    """Test validation of custom dimensions."""
    # Test width too small
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-width-too-small",
            "save_url": "http://save.com",
            "prompt": "A beautiful sunset",
            "width": 64,
            "height": 512,
            "webhook_url": "http://webhook.test",
        },
    )
    assert response.status_code == 400
    assert "width" in response.text.lower()

    # Case 2: Only width provided
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "test",
            "width": 512,
            "save_url": "http://save.com",
            "webhook_url": "http://webhook.test",
        },
    )
    assert response.status_code == 400

    # Case 3: Out of range (both too small)
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "test",
            "width": 64,
            "height": 64,
            "save_url": "http://save.com",
            "webhook_url": "http://webhook.test",
        },
    )
    assert response.status_code == 400

    # Case 4: Too large
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "test",
            "width": 4096,
            "height": 4096,
            "save_url": "http://save.com",
            "webhook_url": "http://webhook.test",
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_generate_image_invalid_aspect_ratio(async_client, api_key_headers, mock_model_manager):
    """Test validation of aspect ratio."""
    response = await async_client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-invalid-aspect",
            "prompt": "A beautiful sunset",
            "aspect_ratio": "invalid",
            "save_url": "http://save.com",
            "webhook_url": "http://webhook.test",
        },
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "failed"
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_generate_image_unauthorized(async_client):
    """Test handling of unauthorized requests (missing API key)."""
    response = await async_client.post(
        "/api/v1/image/generate",
        json={"prompt": "test"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "MISSING_API_KEY"


def test_generate_image_invalid_key(client, invalid_api_key_headers):
    """Test access with invalid API key."""
    response = client.post(
        "/api/v1/image/generate",
        headers=invalid_api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "sunset",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "INVALID_API_KEY"
