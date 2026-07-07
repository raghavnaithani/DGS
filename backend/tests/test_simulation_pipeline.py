from __future__ import annotations

import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import intake
from app.database.connection import get_connection
from app.database.jobs_store import SQLiteJobStore
from app.database.vector_store import LanceChunkStore
from app.engines import simulation as simulation_engine
from app.main import app
from app.services import simulation_worker as simulation_worker_module


client = TestClient(app)


def _wait_for_job(job_id: str, *, expected_status: str = "completed", attempts: int = 50, delay_seconds: float = 0.05) -> dict:
    for _ in range(attempts):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["status"] == expected_status:
            return payload
        time.sleep(delay_seconds)
    raise AssertionError(f"Job {job_id} did not reach {expected_status}. Last payload: {payload}")


def test_intent_to_three_step_simulation_pipeline(monkeypatch, tmp_path):
    app.state.job_store = SQLiteJobStore(tmp_path / "pipeline.sqlite3")
    app.state.vector_store = LanceChunkStore(path=tmp_path / "lancedb")

    class FakeHttpClient:
        def __init__(self, timeout=10.0):
            self.timeout = timeout
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def get(self, url, **kwargs):
            raise Exception("timeout")
        def post(self, url, **kwargs):
            raise Exception("timeout")

    monkeypatch.setattr(simulation_worker_module.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(simulation_worker_module, "assemble_evidence", lambda query, top_k=10: SimpleNamespace(evidence=[]))
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "start", lambda self: None)
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

    def fake_generate_initial_graph(user_intent, top_k=10, **kwargs):
        captured_request["user_intent"] = user_intent
        captured_request["top_k"] = top_k
        captured_request["kwargs"] = kwargs
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
    assert captured_request["top_k"] == 8
    assert not captured_request.get("kwargs")
    assert captured_request["user_intent"]["id"] == intent_payload["id"]
    assert captured_request["user_intent"]["original_prompt"] == "Should I change my career?"


def test_generate_initial_graph_branches_from_root(monkeypatch):
    monkeypatch.setattr(
        simulation_engine,
        "assemble_evidence",
        lambda query, top_k=10: SimpleNamespace(
            evidence=[
                SimpleNamespace(
                    id="chunk-1",
                    content="Evidence about the decision topic.",
                    source_url="https://example.com/evidence",
                    source_title="Example Evidence",
                    chunk_index=0,
                    dense_similarity=0.91,
                    bm25_score=0.88,
                )
            ]
        ),
    )

    class FakeGenerator:
        def generate_node(self, *, user_intent, evidence_chunks, parent_summary=None, persona_prompt=None, time_step=0, max_retries=None):
            if time_step == 0:
                return (
                    {
                        "id": "root-node",
                        "title": "Root decision",
                        "summary": "Root summary.",
                        "description": "Root description.",
                        "time_step": 0,
                        "created_by_engine": "test",
                        "alternatives": [
                            {"id": "alt-1", "description": "Compare options", "action_type": "Research"},
                            {"id": "alt-2", "description": "Take action now", "action_type": "Execution"},
                            {"id": "alt-3", "description": "Wait and reassess", "action_type": "Research"},
                        ],
                        "risks": [{"id": "risk-1", "description": "risk", "severity": "High", "likelihood": "Low"}],
                        "source_citations": ["https://example.com/evidence"],
                        "confidence_score": 0.9,
                        "speculative": False,
                        "created_at": "2026-05-29T00:00:00+00:00",
                    },
                    "root-raw",
                )

            action = "branch"
            if parent_summary and "Chosen action:" in parent_summary:
                action = parent_summary.split("Chosen action:", 1)[1].strip() or "branch"

            return (
                {
                    "id": f"node-{action.replace(' ', '-').lower()}",
                    "title": f"Branch {action}",
                    "summary": f"Summary for {action}.",
                    "description": f"Description for {action}.",
                    "time_step": 1,
                    "created_by_engine": "test-branch",
                    "alternatives": [],
                    "risks": [{"id": f"risk-{action}", "description": "risk", "severity": "High", "likelihood": "Low"}],
                    "source_citations": ["https://example.com/evidence"],
                    "confidence_score": 0.8,
                    "speculative": False,
                    "created_at": "2026-05-29T00:00:00+00:00",
                },
                f"raw-{action}",
            )

    monkeypatch.setattr(simulation_engine, "NodeGenerator", FakeGenerator)

    graph = simulation_engine.generate_initial_graph({"original_prompt": "Should I study now?"}, top_k=5)

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2
    assert graph["nodes"][0]["id"] == "root-node"


def test_generate_initial_graph_recursively_expands_child_nodes(monkeypatch):
    monkeypatch.setattr(
        simulation_engine,
        "assemble_evidence",
        lambda query, top_k=10: SimpleNamespace(
            evidence=[
                SimpleNamespace(
                    id="chunk-1",
                    content="Evidence about the decision topic.",
                    source_url="https://example.com/evidence",
                    source_title="Example Evidence",
                    chunk_index=0,
                    dense_similarity=0.91,
                    bm25_score=0.88,
                )
            ]
        ),
    )

    class RecursiveGenerator:
        def generate_node(self, *, user_intent, evidence_chunks, parent_summary=None, persona_prompt=None, time_step=0, max_retries=None):
            if time_step == 0:
                return (
                    {
                        "id": "root-node",
                        "title": "Root decision",
                        "summary": "Root summary.",
                        "description": "Root description.",
                        "time_step": 0,
                        "created_by_engine": "test",
                        "alternatives": [
                            {"id": "alt-1", "description": "Compare options", "action_type": "Research"},
                        ],
                        "risks": [{"id": "risk-1", "description": "risk", "severity": "High", "likelihood": "Low"}],
                        "source_citations": ["https://example.com/evidence"],
                        "confidence_score": 0.9,
                        "speculative": False,
                        "created_at": "2026-05-29T00:00:00+00:00",
                    },
                    "root-raw",
                )

            chosen_action = "branch"
            if parent_summary and "Chosen action:" in parent_summary:
                chosen_action = parent_summary.split("Chosen action:", 1)[1].strip() or "branch"

            alternatives = []
            if chosen_action == "Compare options":
                alternatives = [{"id": "alt-3", "description": "Deepen research", "action_type": "Research"}]

            return (
                {
                    "id": f"node-{chosen_action.replace(' ', '-').lower()}",
                    "title": f"Branch {chosen_action}",
                    "summary": f"Summary for {chosen_action}.",
                    "description": f"Description for {chosen_action}.",
                    "time_step": 1 if chosen_action != "Deepen research" else 2,
                    "created_by_engine": "test-branch",
                    "alternatives": alternatives,
                    "risks": [{"id": f"risk-{chosen_action}", "description": "risk", "severity": "High", "likelihood": "Low"}],
                    "source_citations": ["https://example.com/evidence"],
                    "confidence_score": 0.8,
                    "speculative": False,
                    "created_at": "2026-05-29T00:00:00+00:00",
                },
                f"raw-{chosen_action}",
            )

    monkeypatch.setattr(simulation_engine, "NodeGenerator", RecursiveGenerator)

    graph = simulation_engine.generate_initial_graph({"original_prompt": "Should I study now?"}, top_k=5)

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2
