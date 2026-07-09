from __future__ import annotations

from datetime import datetime, timezone
from app.config import settings
import time
from app.models.schemas import DecisionNode
from app.api import intake
import pytest
from app.database.vector_store import LanceChunkStore
import httpx
from types import SimpleNamespace
from app.services.simulation_worker import SimulationJobWorker
from app.database.jobs_store import SQLiteJobStore
from fastapi.testclient import TestClient
from app.main import app
from app.api import intake, simulation
from app.engines.reasoning import NodeGenerator, NodeGenerationError
from app.services import simulation_worker as simulation_worker_module
import json
from app.engines.simulation import generate_llm_skeleton
from app.engines import simulation as simulation_engine
from app.database.connection import get_connection

def _valid_node_json(id: str = "n1") -> str:
    now = datetime.now(timezone.utc).isoformat()
    summary = " ".join(["A grounded summary with evidence and decision context."] * 4)
    description = " ".join(["A grounded description with evidence, tradeoffs, and concrete outcomes."] * 16)
    node = {
        "id": id,
        "title": "Test",
        "summary": summary,
        "description": description,
        "time_step": 0,
        "created_by_engine": "test",
        "alternatives": [
            {"id": "alt-1", "description": "Choose a careful option.", "action_type": "Research"},
            {"id": "alt-2", "description": "Take a faster path.", "action_type": "Execution"},
            {"id": "alt-3", "description": "Wait and gather more evidence.", "action_type": "Research"},
        ],
        "risks": [{"id": "r1", "description": "Minor risk.", "severity": "Low", "likelihood": "Low"}],
        "source_citations": ["https://example.com/source"],
        "confidence_score": 0.5,
        "speculative": True,
        "created_at": now,
    }
    return json.dumps(node, ensure_ascii=False)

def test_reasoning_pipeline(monkeypatch):
    gen = NodeGenerator()

    # 1. Generate node success
    monkeypatch.setattr(gen, "_chat_completion", lambda *args, **kwargs: _valid_node_json("node-1"))
    node, raw = gen.generate_node(user_intent={"id": "i1", "original_prompt": "test", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=[{"id":"c1","dense_similarity":0.8}], time_step=0)
    assert isinstance(node, dict)
    assert node["id"] == "node-1"
    assert node["speculative"] is True

    # 2. Generate node validation retry
    invalid = '{"id":"bad","title":"Bad","summary":"s","description":"d","time_step":0,"created_by_engine":"test","alternatives":[],"risks":[],"source_citations":[],"confidence_score":0.5,"speculative":true,"created_at":"2020-01-01T00:00:00Z"}'
    calls = {"n": 0}
    def fake_chat(s, u, **kwargs):
        if calls["n"] == 0:
            calls["n"] += 1
            return invalid
        return _valid_node_json("node-final")
    monkeypatch.setattr(gen, "_chat_completion", fake_chat)
    node, raw = gen.generate_node(user_intent={"id": "i2", "original_prompt": "test2", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=[{"id":"c1","dense_similarity":0.6}], time_step=0, max_retries=2)
    assert node["id"] == "node-final"

    # 3. Generate node no API key fallback
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u, **kwargs: (_ for _ in ()).throw(Exception("no key")))
    fallback_node, raw = gen.generate_node(
        user_intent={"original_prompt": "test", "domain": "test", "horizon_months": 3, "risk_tolerance": 5, "constraints": []},
        evidence_chunks=[],
        max_retries=1
    )
    assert fallback_node["speculative"] is True
    assert fallback_node["confidence_score"] == 0.20

    # 4. Marks unsupported claims speculative
    raw = """
    {
      "id": "node-grounding",
      "title": "Test",
      "summary": "The company will double revenue next quarter.",
      "description": "The company will double revenue next quarter.",
      "time_step": 0,
      "created_by_engine": "test",
      "alternatives": [],
      "risks": [{"id": "r1", "description": "There is execution risk.", "severity": "High", "likelihood": "Medium"}],
      "source_citations": [],
      "confidence_score": 0.5,
      "speculative": false,
      "created_at": "2026-05-29T00:00:00+00:00"
    }
    """
    monkeypatch.setattr(gen, "_chat_completion", lambda *args, **kwargs: raw)
    node, _ = gen.generate_node(
        user_intent={"id": "i4", "original_prompt": "test4", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []},
        evidence_chunks=[],
        time_step=0,
    )
    assert node["speculative"] is True
    assert "[Source: speculative]" in node["summary"]

    # 5. Generate node inserts citation
    raw_cite = """
{
  "id": "node-cite",
  "title": "Test",
    "summary": "A grounded summary with enough detail to stay specific and decision-focused.",
        "description": "A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement. A grounded description with evidence, tradeoffs, risks, and concrete outcomes repeated enough times to satisfy the stricter evidence-backed length requirement.",
  "time_step": 0,
  "created_by_engine": "test",
    "alternatives": [{"id": "alt-1", "description": "Carefully expand the plan.", "action_type": "Research"}, {"id": "alt-2", "description": "Move faster with limited scope.", "action_type": "Execution"}, {"id": "alt-3", "description": "Wait for more evidence.", "action_type": "Research"}],
    "risks": [{"id": "r1", "description": "Execution risk remains present.", "severity": "High", "likelihood": "Medium"}],
    "source_citations": [],
  "confidence_score": 0.5,
  "speculative": false,
  "created_at": "2026-05-29T00:00:00+00:00"
}
"""
    monkeypatch.setattr(gen, "_chat_completion", lambda *args, **kwargs: raw_cite)
    evidence = [{"id": "c1", "dense_similarity": 0.82, "content": "AI job market trends 2026 forecast", "source_url": "https://bls.gov/report"}]
    node, _ = gen.generate_node(
        user_intent={"id": "ix", "original_prompt": "test-cite", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []},
        evidence_chunks=evidence,
        time_step=0,
        max_retries=0,
    )
    assert node.get("speculative") is not True
    assert any("Source:" in s for s in [node.get("summary", ""), node.get("description", "")])
    assert node.get("source_citations") and len(node.get("source_citations")) >= 1

    # 6. Generate node prompt emphasizes branch distinctness
    captured = {}
    def fake_chat_branch(system_prompt, user_message, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_message"] = user_message
        return _valid_node_json("node-branch")
    monkeypatch.setattr(gen, "_chat_completion", fake_chat_branch)
    gen.generate_node(
        user_intent={"id": "i5", "original_prompt": "test5", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []},
        evidence_chunks=[{"id": "c1", "dense_similarity": 0.7, "content": "supporting evidence", "source_url": "https://example.com"}],
        parent_summary="Parent node summary.",
        time_step=1,
    )
    assert "child branch" in captured["system_prompt"]
    assert "distinct action, tradeoff, or consequence" in captured["system_prompt"]
    assert "meaningfully different from the parent branch" in captured["user_message"]

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
    raise AssertionError(f"Job {job_id} did not reach {expected_status}. Last payload: {payload}")

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
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "start", lambda self: None)
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
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "_refresh_evidence", lambda self, query, disable_scraping=False: None)
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "start", lambda self: None)
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
    monkeypatch.setattr(simulation_worker_module, "IngestionService", FakeIngestionService)
    monkeypatch.setattr(simulation_worker_module.SimulationJobWorker, "start", lambda self: None)
    monkeypatch.setattr(simulation_worker_module, "generate_initial_graph", lambda user_intent, top_k=10, **kwargs: _stub_initial_graph())
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

def test_ollama_skeleton_pipeline(monkeypatch):
    # 1. Chat completion success
    class FakeResponseSuccess:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '{"test": "value"}'}}]}
            
    monkeypatch.setattr(httpx.Client, "post", lambda *args, **kwargs: FakeResponseSuccess())
    monkeypatch.setattr("app.engines.reasoning.settings.groq_api_key", "fake-key")
    gen = NodeGenerator()
    res = gen._chat_completion("sys", "user")
    assert res == '{"test": "value"}'

    # 2. Chat completion error
    class FakeResponseError:
        status_code = 500
        text = "Internal Server Error"
        
    monkeypatch.setattr(httpx.Client, "post", lambda *args, **kwargs: FakeResponseError())
    with pytest.raises(NodeGenerationError):
        gen._chat_completion("sys", "user")

    # 3. Generate LLM skeleton success
    fake_skeleton = {
        "nodes": [
            {
                "id": "n1",
                "title": "Root decision",
                "summary": "Root decision summary",
                "alternatives": [{"id": "a1", "description": "Alt 1"}],
                "time_step": 0
            }
        ],
        "edges": []
    }
    
    monkeypatch.setattr(NodeGenerator, "_chat_completion", lambda s, sys, user: json.dumps(fake_skeleton))
    class FakeUserIntent:
        original_prompt = "Switching careers to AI"
        
    res = generate_llm_skeleton(user_intent=FakeUserIntent(), target_nodes=1, horizon_months=3)
    assert len(res["nodes"]) == 1
    assert res["nodes"][0]["title"] == "Root decision"
    assert res["nodes"][0]["speculative"] is True
    assert "created_at" in res["nodes"][0]

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
