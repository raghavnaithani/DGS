from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.jobs import IngestionRequest, JobSubmission

router = APIRouter()


@router.post("/ingest", response_model=JobSubmission, status_code=202)
async def ingest_knowledge(payload: IngestionRequest, request: Request) -> JobSubmission:
    ingestion_service = request.app.state.ingestion_service
    job = await ingestion_service.submit(payload)
    return JobSubmission(job_id=job.id, status="queued")
