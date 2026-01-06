"""Tests for image editing endpoint."""

import pytest


def test_edit_image_success(client, api_key_headers, mock_storage, sample_job_id):
    """Test successful image editing using URLs."""
    input_url = "https://example.com/source.png"
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Make it look vintage",
            "aspect_ratio": "16:9",
            "save_url": "https://example.com/save.png",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "save_url" in data
    assert data["save_url"] == "https://example.com/save.png"
    assert "generation_time" in data

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_to_url.assert_called()


def test_edit_image_with_output_url(client, api_key_headers, mock_storage, sample_job_id):
    """Test image editing with a custom output URL."""
    input_url = "https://example.com/source.png"
    output_url = "https://custom-storage.com/result.png?token=sig"
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Change color to blue",
            "save_url": output_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["save_url"] == output_url

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_to_url.assert_called()
    assert mock_storage.upload_to_url.call_args.kwargs["url"] == output_url


def test_edit_image_inpaint_with_mask(client, api_key_headers, mock_storage, sample_job_id):
    """Test editing with a mask URL."""
    input_url = "https://example.com/source.png"
    mask_url = "https://example.com/mask.png"
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "mask_image_url": mask_url,
            "prompt": "Fill the hole",
            "save_url": "https://example.com/save.png",
        },
    )

    assert response.status_code == 200
    # Verify both URLs were downloaded
    assert mock_storage.download_from_url.call_count == 2


def test_edit_image_unauthorized(client):
    """Test unauthorized access."""
    response = client.post("/api/v1/image/edit", json={"job_id": "test", "save_url": "https://example.com/save.png"})
    assert response.status_code == 401


def test_edit_image_download_error(client, api_key_headers, mock_storage, sample_job_id):
    """Test handling of download failures."""
    from app.exceptions import ValidationError
    mock_storage.download_from_url.side_effect = ValidationError("Download failed")
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": "https://bad-url.com/img.png",
            "prompt": "edit",
            "save_url": "https://example.com/save.png",
        },
    )

    assert response.status_code == 400
    assert "Download failed" in response.json()["error_message"]
