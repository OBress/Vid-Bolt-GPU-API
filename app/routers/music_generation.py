"""Music generation endpoint."""

import logging
import time

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.models.common import ErrorResponse
from app.models.music_generation import MusicGenerateRequest, MusicGenerateResponse
from app.models.job import JobResult
from app.models.internal import MusicGenerationParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/music",
    tags=["Music Generation"],
)


@router.post(
    "/generate",
    response_model=MusicGenerateResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Audio mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Generate Music",
    description="Generate music from a text prompt and optional lyrics using ACE-Step 1.5.",
)
async def generate_music(
    request: Request,
    body: MusicGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> MusicGenerateResponse:
    """Generate music from a text prompt (Async).

    Returns 202 Accepted if job is queued, or 429 if busy.
    """
    params = MusicGenerationParams(
        job_id=body.job_id,
        prompt=body.prompt,
        lyrics=body.lyrics,
        duration_seconds=body.duration_seconds,
        seed=body.seed,
        bpm=body.bpm,
        key_scale=body.key_scale,
        time_signature=body.time_signature,
        vocal_language=body.vocal_language,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.MUSIC_GENERATION,
        task_func=_run_music_generation,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        model_manager=model_manager,
        storage=storage,
        params=params,
        save_url=body.save_url,
    )

    if not submitted:
        raise HTTPException(
            status_code=429,
            detail="System busy: Maximum concurrent audio generations reached."
        )

    return MusicGenerateResponse(
        job_id=body.job_id,
        status="queued",
        message="Music generation job queued",
    )


async def _run_music_generation(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: MusicGenerationParams,
    save_url: str,
) -> JobResult:
    """Background task for music generation."""
    start_time = time.time()
    
    generator = model_manager.get_music_generator()
    result = await generator.generate_music(params)

    final_url = await storage.upload_to_url(
        data=result.audio_data,
        url=save_url,
        content_type="audio/wav",
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "seed": result.seed,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
        }
    )
