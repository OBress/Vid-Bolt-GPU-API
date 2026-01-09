"""Health check endpoints."""

from fastapi import APIRouter, Depends

from app import __version__
from app.config import get_settings, Settings
from app.dependencies import verify_api_key
from app.models.common import HealthResponse, StatusResponse, ReadinessResponse

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
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Check",
    description="Check if API is fully ready for generation requests. No auth required.",
)
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Return readiness status for VM provisioning.

    This endpoint checks whether models are loaded and the API
    is ready to accept generation requests. Used by external
    orchestrators to know when a freshly started VM is ready.
    
    Returns ready=True only when:
    - In mock mode (always ready), OR
    - Models downloaded AND ModelManager initialized with active generators
    """
    from app.dependencies import _model_manager_instance, _generator_instance
    from app.services.model_downloader import get_model_downloader
    
    # Mock mode is always ready
    if settings.mock_mode:
        return ReadinessResponse(
            ready=True,
            status="ready",
            version=__version__,
            mock_mode=True,
            current_mode="mock",
            models_loaded=True,
        )
    
    # Check if model download is in progress
    downloader = get_model_downloader()
    if downloader is not None:
        dl_status = downloader.get_status()
        if dl_status.status == "downloading":
            return ReadinessResponse(
                ready=False,
                status=f"downloading_models ({dl_status.completed_models}/{dl_status.total_models})",
                version=__version__,
                mock_mode=False,
                current_mode=None,
                models_loaded=False,
            )
        elif dl_status.status == "failed":
            return ReadinessResponse(
                ready=False,
                status="download_failed",
                version=__version__,
                mock_mode=False,
                current_mode=None,
                models_loaded=False,
            )
    
    # Check if ModelManager is initialized
    if _model_manager_instance is None:
        return ReadinessResponse(
            ready=False,
            status="starting",
            version=__version__,
            mock_mode=False,
            current_mode=None,
            models_loaded=False,
        )
    
    # Check if models are loaded
    try:
        current_mode = _model_manager_instance.current_mode
        has_generators = (
            _model_manager_instance._image_generator is not None or
            _model_manager_instance._video_generator is not None
        )
        
        if has_generators:
            return ReadinessResponse(
                ready=True,
                status="ready",
                version=__version__,
                mock_mode=False,
                current_mode=current_mode,
                models_loaded=True,
            )
        else:
            return ReadinessResponse(
                ready=False,
                status="loading_models",
                version=__version__,
                mock_mode=False,
                current_mode=current_mode,
                models_loaded=False,
            )
    except Exception:
        return ReadinessResponse(
            ready=False,
            status="error",
            version=__version__,
            mock_mode=False,
            current_mode=None,
            models_loaded=False,
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

