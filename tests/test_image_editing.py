"""Tests for image editing endpoint."""

import pytest
import asyncio
from app.models.job import JobStatus


@pytest.mark.asyncio
async def test_edit_image_success(async_client, api_key_headers, mock_storage, sample_job_id, mock_job_manager, mock_model_manager):
    """Test successful image editing submission (async job flow).
    
    Note: This test verifies job submission only. The worker doesn't run in pytest,
    so we can't test actual completion. Use integration tests for end-to-end flow.
    """
    input_url = "https://example.com/source.png"
    response = await async_client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Make it look vintage",
            "aspect_ratio": "16:9",
            "save_url": "https://example.com/save.png",
            "webhook_url": "http://webhook.test",
        },
    )

    # Verify submission accepted
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "pending"
    assert data["job_id"] == sample_job_id
    assert "status_url" in data
    
    # Verify storage download called immediately (sync validation)
    mock_storage.download_from_url.assert_called_with(input_url)
    
    # Verify job is in the queue (can be polled)
    status_response = await async_client.get(data["status_url"], headers=api_key_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["pending", "processing"]


async def test_edit_image_with_output_url(async_client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test image editing submission with a custom output URL.
    
    Note: Verifies submission only. Worker doesn't run in pytest.
    """
    input_url = "https://example.com/source.png"
    output_url = "https://custom-storage.com/result.png?token=sig"
    
    response = await async_client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Change color to blue",
            "save_url": output_url,
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


async def test_edit_image_inpaint_with_mask(async_client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test editing with a mask URL."""
    input_url = "https://example.com/source.png"
    mask_url = "https://example.com/mask.png"
    
    response = await async_client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "mask_image_url": mask_url,
            "prompt": "Fill the hole",
            "save_url": "https://example.com/save.png",
            "webhook_url": "http://webhook.test",
        },
    )

    assert response.status_code == 202
    # Verify both URLs were downloaded
    assert mock_storage.download_from_url.call_count == 2


async def test_edit_image_unauthorized(async_client):
    """Test unauthorized access."""
    response = await async_client.post("/api/v1/image/edit", json={"job_id": "test", "save_url": "https://example.com/save.png"})
    assert response.status_code == 401


async def test_edit_image_download_error(async_client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test handling of download failures."""
    from app.exceptions import ValidationError
    mock_storage.download_from_url.side_effect = ValidationError("Download failed")
    
    response = await async_client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": "https://bad-url.com/img.png",
            "prompt": "edit",
            "save_url": "https://example.com/save.png",
            "webhook_url": "http://webhook.test",
        },
    )

    assert response.status_code == 400
    assert "Download failed" in response.json()["detail"]
