import pytest
import respx
from httpx import Response
from app.services.storage import StorageService
from app.config import Settings
from app.exceptions import ValidationError

# Mock Settings
class MockSettings(Settings):
    max_image_size_mb: int = 10

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

@pytest.mark.asyncio
async def test_ssrf_redirect_bypass():
    """
    Test that StorageService handles redirects safely and blocks SSRF attempts
    where a safe URL redirects to a blocked internal IP.
    """
    settings = MockSettings()
    service = StorageService(settings)

    # URL that initially looks safe (using example.com to pass real DNS check if needed,
    # but validate_external_url mocks are not here so it will do real DNS.
    # example.com is safe.
    safe_url = "http://example.com/image.png"
    # Target that should be blocked
    target_url = "http://169.254.169.254/latest/meta-data/"

    async with respx.mock(base_url=None, assert_all_called=False) as respx_mock:
        # 1. Initial request to safe URL -> Redirects to internal IP
        respx_mock.get(safe_url).mock(return_value=Response(302, headers={"Location": target_url}))

        # 2. Request to internal IP -> Returns sensitive data
        # If the code follows redirect blindly, it will hit this.
        # If the code checks the redirect location, it should block it before hitting this.
        target_route = respx_mock.get(target_url).mock(return_value=Response(200, text="secret-key"))

        # We expect a ValidationError because the redirect points to a blocked IP
        with pytest.raises(ValidationError) as excinfo:
            await service.download_from_url(safe_url)

        assert "not allowed" in str(excinfo.value) or "metadata" in str(excinfo.value) or "private" in str(excinfo.value)

        # Verify that the target route was NOT called (i.e., blocked before request)
        assert not target_route.called
