"""Health check endpoints."""

from fastapi import APIRouter, Depends

from app import __version__
from app.config import get_settings, Settings
from app.dependencies import verify_api_key
from app.models.common import HealthResponse, StatusResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Basic health check endpoint. No authentication required.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return basic health status.

    This endpoint does not require authentication and can be used
    for load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        mock_mode=settings.mock_mode,
    )


@router.get(
    "/api/v1/status",
    response_model=StatusResponse,
    summary="Detailed Status",
    description="Detailed service status. Requires authentication.",
    dependencies=[Depends(verify_api_key)],
)
async def detailed_status(
    settings: Settings = Depends(get_settings),
) -> StatusResponse:
    """Return detailed service status.

    This endpoint requires authentication and provides more
    detailed information about the service status.
    """
    return StatusResponse(
        status="healthy",
        version=__version__,
        mock_mode=settings.mock_mode,
    )

