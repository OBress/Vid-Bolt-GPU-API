"""Regression tests for segmentation playback URLs."""

from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.dependencies import get_model_manager, get_storage_service
from app.main import app
from app.models.internal import ImageAnimationResult
from app.services.storage import StorageService


class _TestStorage(StorageService):
    """Storage service that skips network I/O but keeps URL resolution behavior."""

    def __init__(self, sample_image_bytes: bytes):
        super().__init__(Settings(public_asset_base_url="https://assets.vidbolt.app"))
        self._sample_image_bytes = sample_image_bytes

    async def download_from_url(self, url: str) -> bytes:
        return self._sample_image_bytes

    async def upload_to_url(
        self,
        data: bytes,
        url: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self.resolve_public_url(url)


class _FakeSegmenter:
    async def animate_image(self, params):
        return ImageAnimationResult(
            video_data=b"fake-h264-video",
            width=100,
            height=100,
            duration_seconds=params.duration_seconds,
            fps=params.fps,
            frame_count=int(params.duration_seconds * params.fps),
            object_count=1,
            labels=["bear"],
        )


class _FakeModelManager:
    def get_segmenter(self):
        return _FakeSegmenter()


@pytest.mark.asyncio
async def test_segment_animate_job_status_and_webhook_use_public_cdn_url(
    async_client,
    api_key_headers,
    mock_job_manager,
    sample_image_bytes,
):
    storage = _TestStorage(sample_image_bytes)
    webhook_service = AsyncMock()
    mock_job_manager.set_webhook_service(webhook_service)

    app.dependency_overrides[get_storage_service] = lambda: storage
    app.dependency_overrides[get_model_manager] = lambda: _FakeModelManager()

    save_url = (
        "https://vidbolt.681bbaa2698eb1b1ab4aabcfab93bfd4.r2.cloudflarestorage.com/"
        "temporary/gpu-api-test/job-123/segmentation.mp4"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test"
    )
    expected_public_url = "https://assets.vidbolt.app/temporary/gpu-api-test/job-123/segmentation.mp4"
    job_id = "seg-animate-job-123"

    try:
        response = await async_client.post(
            "/api/v1/segment/animate",
            headers=api_key_headers,
            json={
                "job_id": job_id,
                "input_image_url": "https://example.com/input.png",
                "text_prompt": "bear",
                "operations": [{"type": "outline", "color": [255, 0, 0, 255]}],
                "save_url": save_url,
                "webhook_url": "https://example.com/webhook",
            },
        )

        assert response.status_code == 202

        await mock_job_manager._process_job(job_id)

        status_response = await async_client.get(
            f"/api/v1/jobs/{job_id}",
            headers=api_key_headers,
        )
        assert status_response.status_code == 200
        assert status_response.json()["result"]["save_url"] == expected_public_url

        job = mock_job_manager.get_job(job_id)
        assert job is not None
        assert job.result.save_url == expected_public_url

        payload = webhook_service.deliver.await_args.kwargs["payload"]
        assert payload.result.save_url == expected_public_url
    finally:
        app.dependency_overrides.pop(get_storage_service, None)
        app.dependency_overrides.pop(get_model_manager, None)
