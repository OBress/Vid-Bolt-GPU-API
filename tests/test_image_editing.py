"""Tests for image editing endpoint."""

import pytest


def test_edit_image_success(client, api_key_headers, mock_storage, sample_job_id):
    """Test successful image editing using URLs."""
    input_url = "https://example.com/source.png"
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        data={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Make it look vintage",
            "edit_type": "style_transfer",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "r2_url" in data
    assert "input_r2_key" in data

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_image.assert_called()


def test_edit_image_with_output_url(client, api_key_headers, mock_storage, sample_job_id):
    """Test image editing with a custom output URL."""
    input_url = "https://example.com/source.png"
    output_url = "https://custom-storage.com/result.png?token=sig"
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        data={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Change color to blue",
            "output_url": output_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["r2_url"] == output_url
    assert data["r2_key"] is None

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_to_url.assert_called()
    assert mock_storage.upload_to_url.call_args.kwargs["url"] == output_url


def test_edit_image_inpaint_with_mask(client, api_key_headers, mock_storage, sample_job_id):
    """Test inpainting which requires a mask URL."""
    input_url = "https://example.com/source.png"
    mask_url = "https://example.com/mask.png"
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        data={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "mask_image_url": mask_url,
            "prompt": "Fill the hole",
            "edit_type": "inpaint",
        },
    )

    assert response.status_code == 200
    # Verify both URLs were downloaded
    assert mock_storage.download_from_url.call_count == 2


def test_edit_image_inpaint_missing_mask(client, api_key_headers, sample_job_id):
    """Test inpainting validation error when mask is missing."""
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        data={
            "job_id": sample_job_id,
            "input_image_url": "https://example.com/img.png",
            "prompt": "Fill",
            "edit_type": "inpaint",
            # missing mask_image_url
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_edit_image_unauthorized(client):
    """Test unauthorized access."""
    response = client.post("/api/v1/image/edit", data={"job_id": "test"})
    assert response.status_code == 401


def test_edit_image_download_error(client, api_key_headers, mock_storage, sample_job_id):
    """Test handling of download failures."""
    from app.exceptions import ValidationError
    mock_storage.download_from_url.side_effect = ValidationError("Download failed")
    
    response = client.post(
        "/api/v1/image/edit",
        headers=api_key_headers,
        data={
            "job_id": sample_job_id,
            "input_image_url": "https://bad-url.com/img.png",
            "prompt": "edit",
        },
    )

    assert response.status_code == 400
    assert "Download failed" in response.json()["error_message"]
