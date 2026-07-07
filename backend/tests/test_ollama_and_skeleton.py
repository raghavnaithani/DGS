from __future__ import annotations

import pytest
import json
import httpx
from app.engines.reasoning import NodeGenerator, NodeGenerationError
from app.engines.simulation import generate_llm_skeleton
from app.config import settings


def test_ollama_chat_completion_success(monkeypatch):
    class FakeResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '{"test": "value"}'}}]}
            
    def fake_post(*args, **kwargs):
        return FakeResponse()
        
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr("app.engines.reasoning.settings.groq_api_key", "fake-key")
    
    gen = NodeGenerator()
    res = gen._chat_completion("sys", "user")
    assert res == '{"test": "value"}'


def test_ollama_chat_completion_error(monkeypatch):
    class FakeResponse:
        status_code = 500
        text = "Internal Server Error"
        
    def fake_post(*args, **kwargs):
        return FakeResponse()
        
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    
    gen = NodeGenerator()
    with pytest.raises(NodeGenerationError):
        gen._chat_completion("sys", "user")


def test_generate_llm_skeleton_success(monkeypatch):
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
