
import pytest
import respx
import httpx
from app.services.storage import StorageService
from app.config import Settings
from app.exceptions import ValidationError

@pytest.mark.asyncio
async def test_ssrf_bypass_via_redirect():
    """
    Test that the current implementation is vulnerable to SSRF via redirects.
    We simulate a safe URL redirecting to a private IP.
    """
    settings = Settings()
    storage = StorageService(settings)

    safe_url = "http://safe.example.com/image.jpg"
    redirect_target = "http://169.254.169.254/latest/meta-data/"

    # Mock the HTTP requests
    async with respx.mock(base_url=None) as respx_mock:
        # 1. First request to safe URL returns a 302 Redirect to the metadata service
        respx_mock.get(safe_url).respond(
            status_code=302,
            headers={"Location": redirect_target}
        )

        # Now this should raise ValidationError because we check the redirect target
        with pytest.raises(ValidationError) as exc_info:
            await storage.download_from_url(safe_url)

        assert "metadata" in str(exc_info.value).lower() or \
               "private" in str(exc_info.value).lower() or \
               "loopback" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_valid_redirect():
    """
    Test that valid redirects are still followed.
    """
    settings = Settings()
    storage = StorageService(settings)

    initial_url = "http://example.com/start"
    redirect_url = "http://example.com/end"

    async with respx.mock(base_url=None) as respx_mock:
        # 1. Initial URL redirects to final URL
        respx_mock.get(initial_url).respond(
            status_code=302,
            headers={"Location": redirect_url}
        )

        # 2. Final URL returns content
        respx_mock.get(redirect_url).respond(
            status_code=200,
            content=b"valid-image-content"
        )

        content = await storage.download_from_url(initial_url)
        assert content == b"valid-image-content"
