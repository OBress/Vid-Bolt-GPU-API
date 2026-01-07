"""Vid-Bolt GPU API - FastAPI application."""

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
from app.routers import health, image_generation, image_editing, video_generation
from app.utils.logging import setup_logging

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    from app.dependencies import set_generator_instance
    from app.services.mock_generator import MockGenerator

    # Startup
    logger.info(
        f"Starting Vid-Bolt GPU API v{__version__}",
        extra={
            "mock_mode": settings.mock_mode,
            "log_level": settings.log_level,
        },
    )

    # Initialize generator based on mode
    if settings.mock_mode:
        logger.info("Running in MOCK MODE - no actual GPU processing")
        generator = MockGenerator(settings)
    else:
        # Import Z-Image generator only when needed
        from app.services.zimage_generator import ZImageGenerator

        generator = ZImageGenerator(settings)

        if settings.zimage_dry_run:
            logger.info("Running in DRY RUN MODE - workflow testing without models")
        else:
            logger.info("Loading Z-Image models (this may take a moment)...")

        # Load models (or validate dry-run configuration)
        generator.load_models()

    # Set the global generator instance
    set_generator_instance(generator)
    logger.info("Generator initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Vid-Bolt GPU API")


# Create FastAPI application
app = FastAPI(
    title="Vid-Bolt GPU API",
    description="""
GPU-powered image and video generation API.

This API provides endpoints for:
- **Image Generation**: Generate images from text prompts
- **Image Editing**: Edit existing images (inpaint, outpaint, style transfer, etc.)
- **Video Generation**: Create videos from images with AI-powered motion

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
app.include_router(image_generation.router)
app.include_router(image_editing.router)
app.include_router(video_generation.router)
