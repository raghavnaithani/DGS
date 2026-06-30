from __future__ import annotations

import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import intake
from app.database.connection import get_connection
from app.database.jobs_store import SQLiteJobStore
from app.database.vector_store import LanceChunkStore
from app.main import app
from app.services import simulation_worker as simulation_worker_module
from app.services.simulation_worker import SimulationJobWorker


client = TestClient(app)


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


def _stub_initial_graph():
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


def _wait_for_job(job_id: str, *, expected_status: str = "completed", attempts: int = 50, delay_seconds: float = 0.05) -> dict:
    for _ in range(attempts):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["status"] == expected_status:
            return payload
        time.sleep(delay_seconds)
    raise AssertionError(f"Job {job_id} did not reach {expected_status}")


def test_branch_job_persists_child_node_and_webhook(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "branch.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "lancedb")

    webhook_calls: list[dict[str, object]] = []

    class FakeHttpResponse:
        status_code = 204

    class FakeHttpClient:
        def __init__(self, timeout=10.0):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            webhook_calls.append({"url": url, "json": json, "timeout": self.timeout})
            return FakeHttpResponse()

    monkeypatch.setattr(simulation_worker_module.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "_refresh_evidence", lambda self, query, disable_scraping=False: None)
    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_initial_graph())
    monkeypatch.setattr(
        simulation_worker_module,
        "generate_branch_node",
        lambda **kwargs: {
            "node": {
                "id": "branch-1",
                "title": "Branch step",
                "summary": "Branch summary.",
                "description": "Branch description.",
                "time_step": 1,
                "created_by_engine": "test-branch",
                "alternatives": [],
                "risks": [{"id": "r4", "description": "risk", "severity": "High", "likelihood": "Low"}],
                "source_citations": ["speculative"],
                "confidence_score": 0.85,
                "speculative": True,
                "created_at": "2026-05-29T00:00:00+00:00",
            },
            "evidence_chunks": [],
        },
    )

    intent_payload = _seed_intent(monkeypatch)

    start_response = client.post(
        "/v1/simulate/start",
        json={"user_intent_id": intent_payload["id"], "persona": "skeptical_investor"},
    )
    assert start_response.status_code == 200
    start_job_id = start_response.json()["job_id"]
    app.state.simulation_worker.process_pending_jobs_once()

    start_job = _wait_for_job(start_job_id)
    assert start_job["status"] == "completed"
    parent_node_id = start_job["result"]["nodes"][0]["id"]

    branch_response = client.post(
        "/v1/simulate/branch",
        json={
            "session_id": intent_payload["id"],
            "parent_node_id": parent_node_id,
            "action_description": "take the safer option",
            "persona": "cautious_regulator",
            "webhook_url": "https://example.com/webhook",
        },
    )
    assert branch_response.status_code == 200
    branch_job_id = branch_response.json()["job_id"]
    app.state.simulation_worker.process_pending_jobs_once()

    branch_job = _wait_for_job(branch_job_id)
    assert branch_job["status"] == "completed"
    assert branch_job["result"]["node"]["id"] == "branch-1"

    connection = get_connection(app.state.job_store.db_path)
    node_count = connection.execute("SELECT COUNT(*) AS count FROM nodes WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    edge_count = connection.execute("SELECT COUNT(*) AS count FROM edges WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    assert node_count == 4
    assert edge_count == 3
    assert webhook_calls
    assert webhook_calls[0]["json"]["status"] == "completed"
    assert webhook_calls[0]["json"]["workflow"] == "branch"


def test_branch_job_depth_two_expands_subtree(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "branch-depth.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "branch-depth-lancedb")

    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "_refresh_evidence", lambda self, query, disable_scraping=False: None)
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_initial_graph())
    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))

    def fake_generate_branch_node(**kwargs):
        parent_node = kwargs["parent_node"]
        action_description = kwargs["action_description"]
        time_step = int(parent_node.get("time_step", 0)) + 1
        if time_step == 1:
            alternatives = [
                {"id": "alt-a", "description": "Branch deeper one", "action_type": "Research"},
                {"id": "alt-b", "description": "Branch deeper two", "action_type": "Execution"},
            ]
        else:
            alternatives = []

        return {
            "node": {
                "id": f"{action_description.replace(' ', '-')}-{time_step}",
                "title": f"Branch {action_description}",
                "summary": "Branch summary.",
                "description": "Branch description.",
                "time_step": time_step,
                "created_by_engine": "test-branch",
                "alternatives": alternatives,
                "risks": [{"id": f"risk-{time_step}", "description": "risk", "severity": "High", "likelihood": "Low"}],
                "source_citations": ["https://example.com/evidence"],
                "confidence_score": 0.85,
                "speculative": False,
                "created_at": "2026-05-29T00:00:00+00:00",
            },
            "evidence_chunks": [],
        }

    monkeypatch.setattr(simulation_worker_module, "generate_branch_node", fake_generate_branch_node)

    intent_payload = _seed_intent(monkeypatch)

    start_response = client.post(
        "/v1/simulate/start",
        json={"user_intent_id": intent_payload["id"], "persona": "skeptical_investor"},
    )
    assert start_response.status_code == 200
    start_job_id = start_response.json()["job_id"]
    app.state.simulation_worker.process_pending_jobs_once()

    start_job = _wait_for_job(start_job_id)
    parent_node_id = start_job["result"]["nodes"][0]["id"]

    branch_response = client.post(
        "/v1/simulate/branch",
        json={
            "session_id": intent_payload["id"],
            "parent_node_id": parent_node_id,
            "action_description": "take the safer option",
            "persona": "cautious_regulator",
            "depth": 2,
            "branching_factor": 2,
        },
    )
    assert branch_response.status_code == 200
    branch_job_id = branch_response.json()["job_id"]
    app.state.simulation_worker.process_pending_jobs_once()

    branch_job = _wait_for_job(branch_job_id)
    assert branch_job["status"] == "completed"

    connection = get_connection(app.state.job_store.db_path)
    node_count = connection.execute("SELECT COUNT(*) AS count FROM nodes WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    edge_count = connection.execute("SELECT COUNT(*) AS count FROM edges WHERE session_id = ?", (intent_payload["id"],)).fetchone()["count"]
    assert node_count == 6
    assert edge_count == 5


def test_queued_simulation_job_is_claimed_after_restart(monkeypatch, tmp_path):
    job_store = SQLiteJobStore(tmp_path / "resume.sqlite3")
    vector_store = LanceChunkStore(path=tmp_path / "resume-lancedb")
    app.state.job_store = job_store
    app.state.vector_store = vector_store
    worker = SimulationJobWorker(job_store=job_store, vector_store=vector_store)

    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "_refresh_evidence", lambda self, query, disable_scraping=False: None)
    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_initial_graph())
    intent_payload = _seed_intent(monkeypatch)

    job_store.create_simulation_job({"workflow": "start", "user_intent_id": intent_payload["id"], "persona": "skeptical_investor", "webhook_url": None})

    processed = worker.process_pending_jobs_once()
    assert processed == 1

    connection = get_connection(job_store.db_path)
    row = connection.execute(
        "SELECT status, progress, result_json FROM jobs WHERE job_type = 'simulation' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "completed"
    assert row["progress"] == 100
    assert json.loads(row["result_json"])["session_id"] == intent_payload["id"]


def test_start_simulation_refreshes_live_evidence_when_enabled(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "refresh.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "refresh-lancedb")

    refresh_calls: list[dict[str, object]] = []

    class FakeIngestionService:
        def __init__(self, *, job_store, vector_store):
            self.job_store = job_store
            self.vector_store = vector_store

        async def _run_job(self, job_id, payload):
            refresh_calls.append({"job_id": job_id, "query": payload.query})

    monkeypatch.setattr(simulation_worker_module, "IngestionService", FakeIngestionService)
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_initial_graph())
    monkeypatch.setattr(simulation_worker_module.settings, "simulation_enable_live_refresh", True)
    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))

    intent_payload = _seed_intent(monkeypatch)

    start_response = client.post(
        "/v1/simulate/start",
        json={"user_intent_id": intent_payload["id"], "persona": "skeptical_investor", "disable_scraping": False},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    app.state.simulation_worker.process_pending_jobs_once()

    job = _wait_for_job(job_id)
    assert job["status"] == "completed"
    assert refresh_calls and refresh_calls[0]["query"] == "Should I change my career?"