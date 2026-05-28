from __future__ import annotations

from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..database.jobs_store import get_job_store
from ..models.jobs import JobSubmission
from ..engines.simulation import generate_initial_graph
from ..models.schemas import UserIntent
from ..database.connection import get_connection
from ..config import settings

router = APIRouter()


class StartSimRequest(BaseModel := __import__("pydantic").base.BaseModel):
    model_config = __import__("pydantic").base.ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_intent_id: str
    persona: str | None = None


def _fetch_user_intent(intent_id: str) -> dict:
    # fetch from sqlite 'user_intents' table if exists, else raise
    conn = get_connection()
    row = conn.execute("SELECT json FROM user_intents WHERE id = ?", (intent_id,)).fetchone()
    if not row:
        raise KeyError(intent_id)
    return __import__("json").loads(row[0])


def _run_simulation(job_id: str, request: dict):
    try:
        intent = request.get("user_intent")
        result = generate_initial_graph(intent, top_k=10)
        store = get_job_store()
        store.update_job(job_id, status="completed", progress=100, result=result)
    except Exception as exc:
        store = get_job_store()
        store.update_job(job_id, status="failed", error_message=str(exc))


@router.post("/start")
def start_simulation(payload: dict) -> JobSubmission:
    # payload must contain user_intent_id and optional persona
    user_intent_id = payload.get("user_intent_id")
    if not user_intent_id:
        raise HTTPException(status_code=400, detail="user_intent_id required")
    try:
        user_intent = _fetch_user_intent(user_intent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="UserIntent not found")

    job_store = get_job_store()
    request = {"user_intent": user_intent, "persona": payload.get("persona")}
    job = job_store.create_simulation_job(request)
    # run simulation synchronously for now (could be background task)
    try:
        _run_simulation(job.id, request)
    except Exception:
        pass
    return JobSubmission(job_id=job.id, status="queued")


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    store = get_job_store()
    try:
        job = store.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error_message": job.error_message,
    }
from fastapi import APIRouter

router = APIRouter()
