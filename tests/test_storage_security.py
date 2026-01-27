
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
from app.services.storage import StorageService
from app.config import Settings
from app.exceptions import ValidationError

@pytest.mark.asyncio
async def test_ssrf_redirect_validation_called():
    """
    Test that StorageService validates the redirect target URL.
    """
    settings = Settings()
    service = StorageService(settings)

    initial_url = "http://example.com/image.png"
    redirect_url = "http://example.com/redirected.png"

    # We mock validate_external_url to verify it's called
    with patch("app.services.storage.validate_external_url") as mock_validate:
        # Mock httpx client
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # Setup responses
            # 1. Redirect response
            response1 = MagicMock()
            response1.status_code = 302
            response1.headers = {"Location": redirect_url}

            # 2. Success response
            response2 = MagicMock()
            response2.status_code = 200
            response2.content = b"data"

            # Use side_effect to return different responses
            mock_client.get.side_effect = [response1, response2]

            await service.download_from_url(initial_url)

            # Verify validate_external_url was called for BOTH URLs
            assert mock_validate.call_count == 2
            mock_validate.assert_has_calls([
                call(initial_url),
                call(redirect_url)
            ])

            # Verify follow_redirects=False was used
            # First call
            mock_client.get.assert_any_call(initial_url, follow_redirects=False)
            # Second call
            mock_client.get.assert_any_call(redirect_url, follow_redirects=False)

@pytest.mark.asyncio
async def test_ssrf_redirect_blocks_internal():
    """
    Test that if redirect points to internal IP, it raises ValidationError.
    """
    settings = Settings()
    service = StorageService(settings)

    initial_url = "http://example.com/image.png"
    # This URL should be blocked by validate_external_url
    internal_url = "http://localhost/secret"

    # We use the REAL validate_external_url here (or mock it to raise)
    # Since we are in unit test, let's mock validate_external_url to raise for the second call

    with patch("app.services.storage.validate_external_url") as mock_validate:
        # First call passes, second call raises
        mock_validate.side_effect = [None, ValidationError("Loopback not allowed")]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            response1 = MagicMock()
            response1.status_code = 302
            response1.headers = {"Location": internal_url}

            mock_client.get.return_value = response1

            with pytest.raises(ValidationError) as exc:
                await service.download_from_url(initial_url)

            assert "Loopback not allowed" in str(exc.value)

            # Ensure we didn't make the second request (to localhost)
            # The loop should continue after validation, but validation raised exception
            assert mock_client.get.call_count == 1
