import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.storage import StorageService
from app.config import Settings
from app.exceptions import ValidationError

@pytest.mark.asyncio
async def test_ssrf_redirect_bypass():
    """
    Test that redirects to blocked IPs are caught by the StorageService.

    The current implementation (vulnerable) allows httpx to follow redirects automatically,
    bypassing the initial validation.
    The fixed implementation should manually handle redirects and validate each URL.
    """
    settings = Settings(api_key="test")
    service = StorageService(settings)

    initial_url = "http://example.com/safe"
    # This URL is blocked by validate_external_url
    blocked_url = "http://169.254.169.254/latest"

    # 1. Redirect response
    r1 = MagicMock()
    r1.status_code = 302
    r1.headers = {"Location": blocked_url}
    r1.content = b"redirecting..."
    r1.raise_for_status = MagicMock()

    # 2. Blocked content response (should never be reached if validation works)
    r2 = MagicMock()
    r2.status_code = 200
    r2.content = b"SECRET_DATA"
    r2.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Setup mock to return the redirect first, then the secret
        mock_client.get = AsyncMock(side_effect=[r1, r2])

        with pytest.raises(ValidationError) as excinfo:
            await service.download_from_url(initial_url)

        # Verify that the error is indeed about the blocked IP/metadata
        assert "metadata" in str(excinfo.value).lower() or "reserved" in str(excinfo.value).lower()
