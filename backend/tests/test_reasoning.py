from __future__ import annotations

import pytest
from datetime import datetime

from app.engines.reasoning import NodeGenerator, NodeGenerationError
from app.models.schemas import DecisionNode


def _valid_node_json(id: str = "n1") -> str:
    now = datetime.utcnow().isoformat()
    return f"{{\n  \"id\": \"{id}\",\n  \"title\": \"Test\",\n  \"summary\": \"A summary. [Source: speculative]\",\n  \"description\": \"A description. [Source: speculative]\",\n  \"time_step\": 0,\n  \"created_by_engine\": \"test\",\n  \"alternatives\": [],\n  \"risks\": [{{\"id\": \"r1\", \"description\": \"Minor risk. [Source: speculative]\", \"severity\": \"Low\", \"likelihood\": \"Low\"}}],\n  \"source_citations\": [],\n  \"confidence_score\": 0.5,\n  \"speculative\": true,\n  \"created_at\": \"{now}\"\n}}"


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
    # ensure missing key triggers NodeGenerationError
    monkeypatch.setattr(gen, "_chat_completion", lambda s, u: (_ for _ in ()).throw(Exception("no key")))
    with pytest.raises(NodeGenerationError):
        gen.generate_node(user_intent={"id": "i3", "original_prompt": "test3", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=[], time_step=0, max_retries=0)
