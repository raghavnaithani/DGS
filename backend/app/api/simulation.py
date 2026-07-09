from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ..auth.middleware import AuthenticatedUser, get_optional_user
from ..database.connection import get_connection
from ..database.jobs_store import get_job_store
from ..engines.simulation import generate_initial_graph
from ..models.jobs import JobSubmission
from ..services.simulation_worker import SimulationJobWorker
from ..services.usage_service import check_and_increment_graph_counter

router = APIRouter()


class StartSimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_intent_id: str
    disable_scraping: bool = False
    persona: str | None = None
    webhook_url: HttpUrl | None = None
    depth: int | None = Field(default=None, ge=1)
    branching_factor: int | None = Field(default=None, ge=1)
    mode: str | None = Field(default="quick")
    target_nodes: int | None = Field(default=None, ge=1)


class BranchSimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str
    parent_node_id: str
    action_description: str
    persona: str | None = None
    webhook_url: HttpUrl | None = None
    depth: int | None = Field(default=1, ge=1)
    branching_factor: int | None = Field(default=2, ge=1)


def _fetch_user_intent(intent_id: str, db_path: str | None = None) -> dict:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT id, original_prompt, domain, horizon_months, risk_tolerance, constraints_json, personal_context, clarified_entities_json, ambiguities_remaining_json, created_at FROM user_intents WHERE id = ?",
        (intent_id,),
    ).fetchone()
    if not row:
        raise KeyError(intent_id)
    return {
        "id": row["id"],
        "original_prompt": row["original_prompt"],
        "domain": row["domain"],
        "horizon_months": int(row["horizon_months"]),
        "risk_tolerance": int(row["risk_tolerance"]),
        "constraints": json.loads(row["constraints_json"] or "[]"),
        "personal_context": row["personal_context"],
        "clarified_entities": json.loads(row["clarified_entities_json"] or "[]"),
        "ambiguities_remaining": json.loads(row["ambiguities_remaining_json"] or "[]"),
        "created_at": row["created_at"],
    }


def _simulation_worker(request: Request) -> SimulationJobWorker:
    worker = getattr(request.app.state, "simulation_worker", None)
    current_job_store = getattr(request.app.state, "job_store", None)
    current_vector_store = getattr(request.app.state, "vector_store", None)
    if worker is not None and getattr(worker, "job_store", None) is current_job_store and getattr(worker, "vector_store", None) is current_vector_store:
        return worker

    job_store = current_job_store
    if job_store is None:
        job_store = get_job_store()
        request.app.state.job_store = job_store

    vector_store = current_vector_store
    if vector_store is None:
        from ..database.vector_store import get_vector_store

        vector_store = get_vector_store()
        request.app.state.vector_store = vector_store

    worker = SimulationJobWorker(job_store=job_store, vector_store=vector_store)
    request.app.state.simulation_worker = worker
    worker.start()
    return worker


@router.post("/start")
def start_simulation(payload: StartSimRequest, request: Request, user: AuthenticatedUser | None = Depends(get_optional_user)) -> JobSubmission:
    worker = _simulation_worker(request)
    try:
        _fetch_user_intent(payload.user_intent_id, db_path=str(worker.job_store.db_path))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="UserIntent not found") from exc

    if user:
        check_and_increment_graph_counter(str(worker.job_store.db_path), user.user_id)

    job = worker.enqueue_start(
        user_intent_id=payload.user_intent_id,
        user_id=user.user_id if user else None,
        disable_scraping=payload.disable_scraping,
        persona=payload.persona,
        webhook_url=str(payload.webhook_url) if payload.webhook_url else None,
        depth=payload.depth,
        branching_factor=payload.branching_factor,
        mode=payload.mode,
        target_nodes=payload.target_nodes,
    )
    return JobSubmission(job_id=job.id, status="queued")


@router.post("/branch")
def branch_simulation(payload: BranchSimRequest, request: Request) -> JobSubmission:
    worker = _simulation_worker(request)

    with get_connection(worker.job_store.db_path) as connection:
        session = connection.execute("SELECT id FROM sessions WHERE id = ?", (payload.session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        node = connection.execute(
            "SELECT id FROM nodes WHERE session_id = ? AND id = ?",
            (payload.session_id, payload.parent_node_id),
        ).fetchone()
        if node is None:
            raise HTTPException(status_code=404, detail="Parent node not found")

    job = worker.enqueue_branch(
        session_id=payload.session_id,
        parent_node_id=payload.parent_node_id,
        action_description=payload.action_description,
        persona=payload.persona,
        webhook_url=str(payload.webhook_url) if payload.webhook_url else None,
        depth=payload.depth,
        branching_factor=payload.branching_factor,
    )
    return JobSubmission(job_id=job.id, status="queued")


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    store = getattr(request.app.state, "job_store", None) or get_job_store()
    try:
        job = store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error_message": job.error_message,
    }