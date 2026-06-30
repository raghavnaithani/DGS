from __future__ import annotations

import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import intake, simulation
from app.database.connection import get_connection
from app.database.jobs_store import SQLiteJobStore
from app.database.vector_store import LanceChunkStore
from app.main import app
from app.services import simulation_worker as simulation_worker_module


client = TestClient(app)


def _wait_for_job(job_id: str, *, expected_status: str = "completed", attempts: int = 50, delay_seconds: float = 0.05) -> dict:
    for _ in range(attempts):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["status"] == expected_status:
            return payload
        time.sleep(delay_seconds)
    raise AssertionError(f"Job {job_id} did not reach {expected_status}")


def _seed_intent(monkeypatch) -> dict:
    monkeypatch.setattr(intake, "_require_groq_key", lambda: "test-key")
    monkeypatch.setattr(
        intake,
        "_chat_completion",
        lambda api_key, prompt: json.dumps(
            {
                "domain": "career planning",
                "horizon_months": 6,
                "risk_tolerance": 7,
                "constraints": ["budget", "family commitments"],
                "personal_context": "Professional exploring a career transition.",
                "clarified_entities": ["career move", "timeline"],
                "ambiguities_remaining": ["salary target"],
            }
        ),
    )

    response = client.post(
        "/v1/intake/build-intent",
        json={"prompt": "Should I change my career?", "answers": {"q1": "Career growth"}},
    )
    assert response.status_code == 200
    return response.json()


def _stub_graph():
    return {
        "nodes": [
            {
                "id": "n1",
                "title": "Step 1",
                "summary": "Step 1 summary.",
                "description": "Step 1 description.",
                "time_step": 0,
                "created_by_engine": "test",
                "alternatives": [],
                "risks": [{"id": "r1", "description": "risk", "severity": "High", "likelihood": "Low"}],
                "source_citations": ["speculative"],
                "confidence_score": 0.9,
                "speculative": True,
                "created_at": "2026-05-29T00:00:00+00:00",
            },
            {
                "id": "n2",
                "title": "Step 2",
                "summary": "Step 2 summary.",
                "description": "Step 2 description.",
                "time_step": 1,
                "created_by_engine": "test",
                "alternatives": [],
                "risks": [{"id": "r2", "description": "risk", "severity": "High", "likelihood": "Low"}],
                "source_citations": ["speculative"],
                "confidence_score": 0.8,
                "speculative": True,
                "created_at": "2026-05-29T00:00:00+00:00",
            },
        ],
        "edges": [
            {"source": "n1", "target": "n2", "action_description": "step 1 to step 2"},
        ],
    }


def test_graph_and_share_endpoints(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "graph.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "lancedb")
    app.state.simulation_worker = simulation_worker_module.SimulationJobWorker(
        job_store=app.state.job_store,
        vector_store=app.state.vector_store,
    )

    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "_refresh_evidence", lambda self, query, disable_scraping=False: None)
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_graph())
    intent_payload = _seed_intent(monkeypatch)

    start_response = client.post(
        "/v1/simulate/start",
        json={"user_intent_id": intent_payload["id"], "persona": "skeptical_investor"},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    app.state.simulation_worker.process_pending_jobs_once()

    job_response = _wait_for_job(job_id)
    assert job_response["status"] == "completed"

    graph_response = client.get(f"/v1/graph/{intent_payload['id']}")
    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert graph_payload["session_id"] == intent_payload["id"]
    assert len(graph_payload["nodes"]) == 2
    assert len(graph_payload["edges"]) == 1

    share_response = client.post(f"/v1/graph/{intent_payload['id']}/share")
    assert share_response.status_code == 200
    public_id = share_response.json()["public_id"]
    assert public_id

    public_response = client.get(f"/v1/share/{public_id}")
    assert public_response.status_code == 200
    assert public_response.json()["session_id"] == intent_payload["id"]