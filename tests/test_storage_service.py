"""Tests for storage URL resolution."""

from app.config import Settings
from app.services.storage import StorageService


def test_resolve_public_url_maps_r2_upload_to_cdn_url():
    storage = StorageService(Settings(public_asset_base_url="https://assets.vidbolt.app"))

    presigned_url = (
        "https://vidbolt.681bbaa2698eb1b1ab4aabcfab93bfd4.r2.cloudflarestorage.com/"
        "temporary/gpu-api-test/job-123/output.mp4"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test"
    )

    assert storage.resolve_public_url(presigned_url) == (
        "https://assets.vidbolt.app/temporary/gpu-api-test/job-123/output.mp4"
    )


def test_resolve_public_url_falls_back_to_upload_url_when_public_base_unset():
    storage = StorageService(Settings(public_asset_base_url=""))

    presigned_url = (
        "https://vidbolt.681bbaa2698eb1b1ab4aabcfab93bfd4.r2.cloudflarestorage.com/"
        "temporary/gpu-api-test/job-123/output.mp4"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test"
    )

    assert storage.resolve_public_url(presigned_url) == (
        "https://vidbolt.681bbaa2698eb1b1ab4aabcfab93bfd4.r2.cloudflarestorage.com/"
        "temporary/gpu-api-test/job-123/output.mp4"
    )
