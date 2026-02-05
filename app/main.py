"""Vid-Bolt GPU API - FastAPI application."""

# CRITICAL: Set LightX2V device BEFORE any imports
# This must be at the very top because LightX2V modules cache AI_DEVICE at import time.
# Without this, models load on "meta" device causing: "Cannot copy out of meta tensor"
try:
    import lightx2v_platform.base.global_var
    lightx2v_platform.base.global_var.AI_DEVICE = "cuda"
except ImportError:
    pass  # LightX2V not installed - that's fine

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app import __version__
from app.config import get_settings
from app.exceptions import APIError
from app.models.common import ErrorResponse
from app.routers import health, image_generation, image_editing, ltx2_generation, mode, system, lora_management, jobs, gpu, download_status, settings as settings_router_module, batch as batch_router, music_generation, sound_effect_generation
from app.utils.logging import setup_logging

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    from app.dependencies import set_generator_instance, set_model_manager_instance
    from app.services.mock_generator import MockGenerator
    from pathlib import Path

    # Startup
    logger.info(
        f"Starting Vid-Bolt GPU API v{__version__}",
        extra={
            "mock_mode": settings.mock_mode,
            "log_level": settings.log_level,
            "default_mode": settings.default_model_mode,
        },
    )

    # Initialize based on mode
    if settings.mock_mode:
        # Mock mode: use MockGenerator for all requests
        logger.info("Running in MOCK MODE - no actual GPU processing")
        generator = MockGenerator(settings)
        set_generator_instance(generator)
        logger.info("Mock generator initialized successfully")

        # Initialize ModelManager (Dry Run) for API compatibility
        # This ensures /mode endpoints work even in mock mode
        from app.services.model_manager import ModelManager
        logger.info("Initializing ModelManager (Dry Run) for API compatibility...")
        model_manager = ModelManager(settings)
        set_model_manager_instance(model_manager)

        # Set initial mode (dry run)
        if settings.default_model_mode == "image":
            await model_manager.switch_to_image_mode()
        else:
            await model_manager.switch_to_video_mode()
    else:
        # Production mode: check for models and download if missing
        from app.services.model_downloader import init_model_downloader
        from app.services.model_manager import ModelManager
        
        # Initialize model downloader
        base_path = Path(__file__).parent.parent  # Project root
        downloader = init_model_downloader(base_path)
        
        # Check if models exist
        if downloader.check_all_models_exist():
            logger.info("All models found locally")
        else:
            logger.info("Missing models detected - starting background download...")
            downloader.start_download()
            
            # Wait for downloads to complete before loading models
            logger.info("Waiting for model downloads to complete...")
            while not downloader.is_ready():
                status = downloader.get_status()
                if status.status == "failed":
                    logger.error(f"Model download failed: {status.error}")
                    break
                await asyncio.sleep(5)  # Check every 5 seconds
                logger.info(f"Download progress: {status.completed_models}/{status.total_models} models")
            
            if not downloader.is_ready():
                logger.error("Model downloads did not complete successfully")
                # Continue anyway - ModelManager will fail gracefully if models missing
        
        logger.info("Initializing ModelManager for dynamic mode switching...")
        model_manager = ModelManager(settings)
        set_model_manager_instance(model_manager)
        
        # Load image generation mode by default (can switch dynamically via /mode endpoints)
        from app.services.model_manager import VRAMLoadMode
        logger.info("Loading IMAGE_GENERATION mode (Z-Image Turbo)...")
        await model_manager.set_vram_mode(VRAMLoadMode.IMAGE_GENERATION)
        
        logger.info(f"ModelManager initialized in {settings.default_model_mode} mode")

    # Initialize JobManager
    from app.services.job_manager import JobManager
    from app.dependencies import set_job_manager_instance
    
    logger.info("Initializing JobManager...")
    job_manager = JobManager(settings)
    set_job_manager_instance(job_manager)
    
    # Link JobManager to ModelManager for batch processing
    # This is required for _process_batch() to access generators
    job_manager.set_model_manager(model_manager)
    
    # Initialize WebhookService for callback notifications
    from app.services.webhook_service import WebhookService, set_webhook_service_instance
    
    logger.info("Initializing WebhookService...")
    webhook_service = WebhookService(settings)
    await webhook_service.start()
    set_webhook_service_instance(webhook_service)
    job_manager.set_webhook_service(webhook_service)
    logger.info("WebhookService started (1 retry, 30s delay)")
    
    # Start background tasks (Worker + Cleanup)
    job_manager.start()
    logger.info("JobManager started (Queue System Active)")

    # Initialize BatchManager for batch job submissions
    from app.services.batch_manager import BatchManager
    from app.routers.batch import set_batch_manager_instance
    
    logger.info("Initializing BatchManager...")
    batch_manager = BatchManager(settings, job_manager)
    set_batch_manager_instance(batch_manager)
    batch_manager.start()
    logger.info("BatchManager started (5-minute auto-expiry active)")

    yield

    # Shutdown
    logger.info("Shutting down Vid-Bolt GPU API...")
    
    # Stop WebhookService
    await webhook_service.stop()
    
    # Stop BatchManager
    batch_manager.stop()
    
    # Stop JobManager
    job_manager.stop()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Vid-Bolt GPU API",
    description="""
GPU-powered image and video generation API.

This API provides endpoints for:
- **Image Generation**: Generate images from text prompts
- **Image Editing**: Edit existing images (inpaint, outpaint, style transfer, etc.)
- **Video Generation**: Create videos from images with AI-powered motion
- **Music Generation**: Generate music tracks from text prompts
- **Sound Effect Generation**: Generate sound effects from text descriptions

All generated outputs are uploaded to Cloudflare R2 and accessible via CDN URLs.

## Authentication

All `/api/v1/*` endpoints require authentication via the `X-API-Key` header.
The `/health` endpoint does not require authentication.

## Mock Mode

When running in mock mode (default for development), the API simulates generation
with placeholder outputs instead of using actual GPU processing.
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
# Parse allowed origins from comma-separated string
allowed_origins = [
    origin.strip()
    for origin in settings.cors_allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for origins: {allowed_origins}")


# Exception Handlers


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors."""
    logger.warning(
        f"API error: {exc.error_code}",
        extra={
            "error_code": exc.error_code,
            "error_msg": exc.message,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="failed",
            error_code=exc.error_code,
            error_message=exc.message,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle FastAPI request validation errors."""
    # Extract the first error message for user-friendly response
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        message = first_error.get("msg", "Validation error")
        error_message = f"{field}: {message}" if field else message
    else:
        error_message = "Validation error"

    logger.warning(
        f"Validation error",
        extra={
            "path": request.url.path,
            "errors": errors,
        },
    )

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            status="failed",
            error_code="VALIDATION_ERROR",
            error_message=error_message,
        ).model_dump(),
    )


@app.exception_handler(PydanticValidationError)
async def pydantic_validation_error_handler(
    request: Request,
    exc: PydanticValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        message = first_error.get("msg", "Validation error")
        error_message = f"{field}: {message}" if field else message
    else:
        error_message = "Validation error"

    logger.warning(
        f"Pydantic validation error",
        extra={
            "path": request.url.path,
            "errors": errors,
        },
    )

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            status="failed",
            error_code="VALIDATION_ERROR",
            error_message=error_message,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        f"Unexpected error: {exc}",
        extra={"path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="failed",
            error_code="INTERNAL_ERROR",
            error_message="An unexpected error occurred",
        ).model_dump(),
    )


# Include routers
app.include_router(health.router)
app.include_router(mode.router)
app.include_router(system.router)
app.include_router(image_generation.router)
app.include_router(image_editing.router)
app.include_router(ltx2_generation.router)
app.include_router(lora_management.router)
app.include_router(jobs.router)
app.include_router(gpu.router)
app.include_router(download_status.router)
app.include_router(settings_router_module.router)
app.include_router(batch_router.router)
app.include_router(music_generation.router)
app.include_router(sound_effect_generation.router)
