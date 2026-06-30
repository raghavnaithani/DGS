from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone

from app.engines.reasoning import NodeGenerator, NodeGenerationError
from app.models.schemas import DecisionNode


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


def test_generate_node_success(monkeypatch):
    gen = NodeGenerator()
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u: _valid_node_json("node-1"))
    node, raw = gen.generate_node(user_intent={"id": "i1", "original_prompt": "test", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=[{"id":"c1","dense_similarity":0.8}], time_step=0)
    assert isinstance(node, dict)
    assert node["id"] == "node-1"
    assert node["speculative"] is True


def test_generate_node_validation_retry(monkeypatch):
    gen = NodeGenerator()
    # first return invalid JSON (no risks), then valid
    invalid = '{"id":"bad","title":"Bad","summary":"s","description":"d","time_step":0,"created_by_engine":"test","alternatives":[],"risks":[],"source_citations":[],"confidence_score":0.5,"speculative":true,"created_at":"2020-01-01T00:00:00Z"}'
    calls = {"n": 0}
    def fake_chat(s, u):
        if calls["n"] == 0:
            calls["n"] += 1
            return invalid
        return _valid_node_json("node-final")

    monkeypatch.setattr(gen, "_chat_completion", fake_chat)
    node, raw = gen.generate_node(user_intent={"id": "i2", "original_prompt": "test2", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=[{"id":"c1","dense_similarity":0.6}], time_step=0, max_retries=2)
    assert node["id"] == "node-final"


def test_generate_node_no_api_key(monkeypatch):
    gen = NodeGenerator()
    # ensure missing key triggers NodeGenerationError, which should now return a fallback node
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u, **kwargs: (_ for _ in ()).throw(Exception("no key")))
    
    # We now expect a fallback node with speculative=True and confidence=0.20
    fallback_node, raw = gen.generate_node(
        user_intent={"original_prompt": "test", "domain": "test", "horizon_months": 3, "risk_tolerance": 5, "constraints": []},
        evidence_chunks=[],
        max_retries=1
    )
    
    assert fallback_node["speculative"] is True
    assert fallback_node["confidence_score"] == 0.20


def test_generate_node_marks_unsupported_claims_speculative(monkeypatch):
    gen = NodeGenerator()
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
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u: raw)

    node, _ = gen.generate_node(
        user_intent={"id": "i4", "original_prompt": "test4", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []},
        evidence_chunks=[],
        time_step=0,
    )

    assert node["speculative"] is True
    assert "[Source: speculative]" in node["summary"]


def test_generate_node_inserts_citation(monkeypatch):
    gen = NodeGenerator()
    # minimal valid node JSON without risks/alternatives to avoid risk-based speculative flags
    raw = """
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
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u: raw)

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


def test_generate_node_prompt_emphasizes_branch_distinctness(monkeypatch):
    gen = NodeGenerator()
    captured = {}

    def fake_chat(system_prompt, user_message):
        captured["system_prompt"] = system_prompt
        captured["user_message"] = user_message
        return _valid_node_json("node-branch")

    monkeypatch.setattr(gen, "_chat_completion", fake_chat)

    gen.generate_node(
        user_intent={"id": "i5", "original_prompt": "test5", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []},
        evidence_chunks=[{"id": "c1", "dense_similarity": 0.7, "content": "supporting evidence", "source_url": "https://example.com"}],
        parent_summary="Parent node summary.",
        time_step=1,
    )

    assert "child branch" in captured["system_prompt"]
    assert "distinct action, tradeoff, or consequence" in captured["system_prompt"]
    assert "meaningfully different from the parent branch" in captured["user_message"]
