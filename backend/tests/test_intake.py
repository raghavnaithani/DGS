from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.api import intake
from app.main import app


client = TestClient(app)


def test_clarify_returns_valid_questions(monkeypatch):
    monkeypatch.setattr(intake, "_require_groq_key", lambda: "test-key")
    monkeypatch.setattr(
        intake,
        "_chat_completion",
        lambda api_key, prompt: json.dumps(
            [
                {"id": "q1", "text": "What is your goal?", "type": "text"},
                {"id": "q2", "text": "What timeline are you working with?", "type": "choice", "choices": ["3", "6", "12"]},
                {"id": "q3", "text": "What is your risk tolerance?", "type": "number"},
                {"id": "q4", "text": "What constraints exist?", "type": "text"},
                {"id": "q5", "text": "What entities matter?", "type": "text"},
            ]
        ),
    )

    response = client.post("/v1/intake/clarify", json={"prompt": "Should I switch careers?"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["questions"]) == 5
    assert payload["questions"][1]["type"] == "choice"
    assert payload["questions"][2]["type"] == "number"


def test_build_intent_returns_valid_user_intent(monkeypatch):
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
    payload = response.json()
    assert payload["original_prompt"] == "Should I change my career?"
    assert payload["domain"] == "career planning"
    assert payload["horizon_months"] == 6
    assert payload["risk_tolerance"] == 7


def test_build_intent_rejects_missing_answers():
    response = client.post("/v1/intake/build-intent", json={"prompt": "Invest or wait?", "answers": {}})

    assert response.status_code == 400
    assert "answers are required" in response.json()["detail"]


def test_mock_endpoints_return_static_data():
    clarify_response = client.get("/v1/intake/mock-clarify", params={"prompt": "test"})
    build_response = client.post("/v1/intake/mock-build-intent", json={"prompt": "test", "answers": {"q1": "yes"}})

    assert clarify_response.status_code == 200
    assert len(clarify_response.json()["questions"]) == 5

    assert build_response.status_code == 200
    payload = build_response.json()
    assert payload["domain"] == "general decision making"
    assert payload["risk_tolerance"] == 5
