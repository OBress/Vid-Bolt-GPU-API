
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import httpx
from app.services.storage import StorageService
from app.exceptions import ValidationError
from app.config import Settings

@pytest.mark.asyncio
async def test_ssrf_redirect_prevention():
    settings = Settings()
    service = StorageService(settings)

    # Mock validate_external_url to behave as the real one does
    # We can use the real one, but we need to mock socket.getaddrinfo to be sure
    # However, validate_external_url checks the string first.
    # "http://127.0.0.1/admin" will be caught by validate_external_url without DNS mocking if passed directly.

    # Scenario:
    # 1. User provides "http://safe.com/image.png"
    # 2. Service calls validate_external_url("http://safe.com/image.png") -> PASS
    # 3. Service calls GET "http://safe.com/image.png"
    # 4. Server responds 302 to "http://127.0.0.1/admin"
    # 5. Service MUST check "http://127.0.0.1/admin" and RAISE ValidationError

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        # Return a redirect response
        mock_get.return_value = httpx.Response(
            302,
            headers={"Location": "http://127.0.0.1/admin"},
            request=httpx.Request("GET", "http://safe.com/image.png")
        )

        # Expectation: The service should catch the redirect and validate the new URL
        with pytest.raises(ValidationError) as exc:
             await service.download_from_url("http://safe.com/image.png")

        # Verify the error message relates to the blocked IP
        assert "loopback" in str(exc.value).lower() or "private" in str(exc.value).lower()
