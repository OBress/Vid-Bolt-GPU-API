"""Tests for image generation endpoint."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_generate_image_success(async_client, api_key_headers, mock_storage, mock_job_manager, mock_model_manager):
    """Test successful image generation submission (async job flow).
    
    Note: Verifies submission only. Worker doesn't run in pytest.
    """
    payload = {
        "job_id": "test-job-1",
        "prompt": "A beautiful sunset",
        "aspect_ratio": "16:9",
        "save_url": "https://r2.example.com/output.png",
        "webhook_url": "http://webhook.test"
    }
    
    # Initiate Generation
    response = await async_client.post("/api/v1/image/generate", headers=api_key_headers, json=payload)
    
    # Verify submission accepted
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "test-job-1"
    assert data["status"] == "pending"
    assert "status_url" in data
    
    # Verify job can be queried
    status_response = await async_client.get(data["status_url"], headers=api_key_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["pending", "processing"]


@pytest.mark.asyncio
async def test_generate_image_with_output_url(async_client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test image generation submission with a custom output URL.
    
    Note: Verifies submission only. Worker doesn't run in pytest.
    """
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

    # Verify submission accepted
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == sample_job_id
    assert "status_url" in data
    
    # Verify job can be queried
    status_response = await async_client.get(data["status_url"], headers=api_key_headers)
    assert status_response.status_code == 200


@pytest.mark.asyncio
async def test_generate_image_custom_dimensions(async_client, api_key_headers, mock_storage, sample_job_id, mock_job_manager, mock_model_manager):
    """Test image generation submission with custom width and height.
    
    Note: Verifies submission only. Worker doesn't run in pytest.
    """
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

    # Verify submission accepted
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == sample_job_id
    assert "status_url" in data
    
    # Verify job can be queried
    status_response = await async_client.get(data["status_url"], headers=api_key_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["pending", "processing"]


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
