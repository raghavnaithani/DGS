from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.jobs import JobRecord

router = APIRouter()


@router.get("/{job_id}", response_model=JobRecord)
async def get_job(job_id: str, request: Request) -> JobRecord:
    job_store = request.app.state.job_store
    try:
        return job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
