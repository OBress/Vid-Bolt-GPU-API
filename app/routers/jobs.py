"""Job status API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from app.dependencies import APIKeyDep, JobManagerDep
from app.models.common import ErrorResponse
from app.models.job import JobInfo

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["Jobs"],
)


@router.get(
    "/{job_id}",
    response_model=JobInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
    },
    summary="Get Job Status",
    description="Get the current status and result of a background generation job.",
)
async def get_job_status(
    job_id: Annotated[str, Path(description="The unique job ID")],
    api_key: APIKeyDep,
    job_manager: JobManagerDep,
) -> JobInfo:
    """Get job status."""
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404, 
            detail=f"Job {job_id} not found. It may have expired or never existed."
        )

    # Return a copy to avoid mutating the cached object in JobManager
    # (though Pydantic models are mutable, we want to be safe with this calculated field)
    response_job = job.model_copy()
    
    if response_job.status == "pending":
         position = job_manager.get_queue_position(job_id)
         if position is not None:
             response_job.queue_position = position

    return response_job
