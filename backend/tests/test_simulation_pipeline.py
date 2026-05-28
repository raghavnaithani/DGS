from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.api import intake, simulation
from app.database.connection import get_connection
from app.database.jobs_store import SQLiteJobStore
from app.database.vector_store import LanceChunkStore
from app.main import app
from app.services import simulation_worker as simulation_worker_module
import time


client = TestClient(app)


def _wait_for_job(job_id: str, *, expected_status: str = "completed", attempts: int = 50, delay_seconds: float = 0.05) -> dict:
    for _ in range(attempts):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["status"] == expected_status:
            return payload
        time.sleep(delay_seconds)
    raise AssertionError(f"Job {job_id} did not reach {expected_status}")


def test_intent_to_three_step_simulation_pipeline(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "pipeline.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "lancedb")

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

    captured_request: dict[str, object] = {}

    def fake_generate_initial_graph(user_intent, top_k=10):
        captured_request["user_intent"] = user_intent
        captured_request["top_k"] = top_k
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
                {
                    "id": "n3",
                    "title": "Step 3",
                    "summary": "Step 3 summary.",
                    "description": "Step 3 description.",
                    "time_step": 2,
                    "created_by_engine": "test",
                    "alternatives": [],
                    "risks": [{"id": "r3", "description": "risk", "severity": "High", "likelihood": "Low"}],
                    "source_citations": ["speculative"],
                    "confidence_score": 0.7,
                    "speculative": True,
                    "created_at": "2026-05-29T00:00:00+00:00",
                },
            ],
            "edges": [
                {"source": "n1", "target": "n2", "action_description": "step 1 to step 2"},
                {"source": "n2", "target": "n3", "action_description": "step 2 to step 3"},
            ],
        }

    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", fake_generate_initial_graph)

    build_response = client.post(
        "/v1/intake/build-intent",
        json={"prompt": "Should I change my career?", "answers": {"q1": "Career growth"}},
    )
    assert build_response.status_code == 200
    intent_payload = build_response.json()

    connection = get_connection(app.state.job_store.db_path)
    persisted_intent = connection.execute("SELECT * FROM user_intents WHERE id = ?", (intent_payload["id"],)).fetchone()
    assert persisted_intent is not None
    assert persisted_intent["original_prompt"] == "Should I change my career?"

    simulation_response = client.post(
        "/v1/simulate/start",
        json={"user_intent_id": intent_payload["id"], "persona": "skeptical_investor"},
    )
    assert simulation_response.status_code == 200
    job_id = simulation_response.json()["job_id"]

    app.state.simulation_worker.process_pending_jobs_once()

    job_payload = _wait_for_job(job_id)
    assert job_payload["status"] == "completed"
    assert job_payload["progress"] == 100
    assert job_payload["result"]["session_id"] == intent_payload["id"]
    assert len(job_payload["result"]["nodes"]) == 3
    assert len(job_payload["result"]["edges"]) == 2
    assert captured_request["top_k"] == 10
    assert captured_request["user_intent"]["id"] == intent_payload["id"]
    assert captured_request["user_intent"]["original_prompt"] == "Should I change my career?"

    connection = get_connection(app.state.job_store.db_path)
    session_row = connection.execute("SELECT * FROM sessions WHERE id = ?", (intent_payload["id"],)).fetchone()
    node_count = connection.execute("SELECT COUNT(*) AS count FROM nodes WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    edge_count = connection.execute("SELECT COUNT(*) AS count FROM edges WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    assert session_row is not None
    assert node_count == 3
    assert edge_count == 2
