import pytest
import httpx
import respx
from app.services.storage import StorageService
from app.config import get_settings
from app.exceptions import ValidationError

@pytest.mark.asyncio
async def test_ssrf_redirect_bypass_blocked():
    """
    Verify that the fix prevents bypassing SSRF checks via redirects.
    This test expects ValidationError when redirecting to a blocked URL.
    """
    settings = get_settings()
    service = StorageService(settings)

    safe_url = "http://safe.com/image.png"
    blocked_url = "http://169.254.169.254/latest/meta-data/"
    secret_content = b"secret-aws-key"

    with respx.mock(base_url=None, assert_all_called=False) as respx_mock:
        # 1. Initial request to safe URL redirects to blocked URL
        respx_mock.get(safe_url).respond(302, headers={"Location": blocked_url})

        # 2. Blocked URL returns secret content (should not be reached/returned)
        blocked_route = respx_mock.get(blocked_url).respond(200, content=secret_content)

        # 3. Attempt download
        # Should raise ValidationError due to metadata IP in redirect
        with pytest.raises(ValidationError) as exc_info:
            await service.download_from_url(safe_url)

        assert "metadata" in str(exc_info.value).lower()
        assert not blocked_route.called

@pytest.mark.asyncio
async def test_valid_redirect_works():
    """Ensure that valid redirects between public URLs still work."""
    settings = get_settings()
    service = StorageService(settings)

    url1 = "http://example.com/start"
    url2 = "http://example.com/end"
    expected_content = b"valid-image-data"

    with respx.mock(base_url=None) as respx_mock:
        respx_mock.get(url1).respond(302, headers={"Location": url2})
        respx_mock.get(url2).respond(200, content=expected_content)

        content = await service.download_from_url(url1)
        assert content == expected_content
