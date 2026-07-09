from __future__ import annotations

import pytest
import json
import uuid
from app.tests.fixtures.auth_fixtures import auth_headers
from app.database.jobs_store import SQLiteJobStore
from fastapi.testclient import TestClient
import sqlite3
from app.main import app
from app.database.connection import get_connection
from app.api import intake
from datetime import datetime, timedelta, timezone

def test_e2e_auth_lifecycle(client, auth_headers, test_db_path):
    # 1. P0.4: GET /v1/profile without JWT returns 401
    response = client.get("/v1/profile")
    assert response.status_code == 401
    assert "Authorization header" in response.json()["detail"] or "token" in response.json()["detail"].lower()

    # 2. P0.3: GET /v1/profile with valid JWT returns 404 (profile not created yet)
    response = client.get("/v1/profile", headers=auth_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

    # 3. P0.7: New user_profiles table exists in SQLite after server start
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles'").fetchone()
        assert row is not None

    # 4. P0.8: New sessions columns (user_id, status, domain, node_count) exist
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        assert "user_id" in columns
        assert "status" in columns
        assert "domain" in columns
        assert "node_count" in columns

def test_e2e_profile_lifecycle(client, auth_headers, test_db_path):
    # 1. Profile upsert sets onboarding_complete
    payload = {
        "expertise_level": "beginner",
        "risk_tolerance": 8,
        "values": ["Financial growth", "Freedom"],
        "life_situation": "Testing profile upsert"
    }
    res = client.post("/v1/profile", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["onboarding_complete"] is True
    assert data["expertise_level"] == "beginner"

    # 2. Profile roundtrip
    payload = {
        "expertise_level": "expert",
        "risk_tolerance": 2,
        "values": ["Stability", "Family"],
        "life_situation": "Stable job"
    }
    client.post("/v1/profile", json=payload, headers=auth_headers)
    res = client.get("/v1/profile", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["expertise_level"] == "expert"
    assert data["risk_tolerance"] == 2
    assert data["values"] == ["Stability", "Family"]
    assert data["life_situation"] == "Stable job"

    # 3. Patch partial update
    res = client.patch("/v1/profile", json={"risk_tolerance": 9}, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tolerance"] == 9
    assert data["expertise_level"] == "expert"

def test_e2e_usage_lifecycle(client, auth_headers, test_db_path):
    # 1. 402 on limit exceeded
    payload = {
        "expertise_level": "beginner",
        "risk_tolerance": 5,
        "values": [],
        "life_situation": ""
    }
    client.post("/v1/profile", json=payload, headers=auth_headers)
    
    with sqlite3.connect(test_db_path) as conn:
        conn.execute("UPDATE user_profiles SET graphs_this_month = 5 WHERE id = 'user-123'")
        conn.commit()

    intent_id = str(uuid.uuid4())
    with sqlite3.connect(test_db_path) as conn:
        conn.execute(
            "INSERT INTO user_intents (id, original_prompt, domain, horizon_months, risk_tolerance, personal_context) VALUES (?, ?, ?, ?, ?, ?)",
            (intent_id, "test prompt", "career", 12, 5, "context")
        )
        conn.commit()

    start_payload = {
        "user_intent_id": intent_id,
        "mode": "quick",
        "disable_scraping": True
    }
    res = client.post("/v1/simulate/start", json=start_payload, headers=auth_headers)
    assert res.status_code == 402
    assert "detail" in res.json()
    assert res.json()["detail"]["error"] == "Payment Required"

    # 2. Counter resets on new month
    last_month = (datetime.now(timezone.utc) - timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    with sqlite3.connect(test_db_path) as conn:
        conn.execute("UPDATE user_profiles SET graphs_this_month = 5, month_reset_at = ? WHERE id = 'user-123'", (last_month,))
        conn.commit()

    intent_id = str(uuid.uuid4())
    with sqlite3.connect(test_db_path) as conn:
        conn.execute(
            "INSERT INTO user_intents (id, original_prompt, domain, horizon_months, risk_tolerance, personal_context) VALUES (?, ?, ?, ?, ?, ?)",
            (intent_id, "test prompt", "career", 12, 5, "context")
        )
        conn.commit()

    start_payload["user_intent_id"] = intent_id
    res = client.post("/v1/simulate/start", json=start_payload, headers=auth_headers)
    assert res.status_code == 200

    # Verify counter is now 1 in DB
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT graphs_this_month FROM user_profiles WHERE id = 'user-123'").fetchone()
        assert row["graphs_this_month"] == 1

def test_e2e_sessions_lifecycle(client, auth_headers, test_db_path):
    # 1. Create a user profile
    payload = {
        "expertise_level": "beginner",
        "risk_tolerance": 5,
        "values": [],
        "life_situation": ""
    }
    client.post("/v1/profile", json=payload, headers=auth_headers)

    # 2. Add an intent
    intent_id = str(uuid.uuid4())
    with sqlite3.connect(test_db_path) as conn:
        conn.execute(
            """INSERT INTO user_intents (id, original_prompt, domain, horizon_months, risk_tolerance, personal_context)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (intent_id, "test auth prompt", "career", 12, 5, "context")
        )
        conn.commit()

    # 3. Call start_simulation
    start_payload = {
        "user_intent_id": intent_id,
        "mode": "quick",
        "disable_scraping": True
    }
    res = client.post("/v1/simulate/start", json=start_payload, headers=auth_headers)
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    
    # Process job synchronously
    from app.services.simulation_worker import SimulationJobWorker
    from app.database.jobs_store import SQLiteJobStore
    from app.database.vector_store import LanceChunkStore
    worker = SimulationJobWorker(job_store=SQLiteJobStore(test_db_path), vector_store=LanceChunkStore(test_db_path.replace(".db", ".lance")))
    worker._process_job(job_id, {"workflow": "start", "user_intent_id": intent_id, "disable_scraping": True, "mode": "quick", "user_id": "user-123"})
    
    # 4. Verify session appears in history
    session_id = str(uuid.uuid4())
    with sqlite3.connect(test_db_path) as conn:
        conn.execute("INSERT INTO sessions (id, intent_id, user_id, title, status) VALUES (?, ?, ?, ?, ?)", (session_id, intent_id, "user-123", "My Session", "active"))
        conn.execute("INSERT INTO nodes (id, session_id, title, summary, description, time_step, confidence_score, created_by_engine, speculative, created_at) VALUES ('n1', ?, 'N1', 'summary', 'desc', 0, 0.9, 'test', 0, CURRENT_TIMESTAMP)", (session_id,))
        conn.execute("UPDATE sessions SET node_count = (SELECT COUNT(*) FROM nodes WHERE session_id = ?) WHERE id = ?", (session_id, session_id))
        conn.commit()

    res = client.get("/v1/sessions", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    session_res = next((s for s in data if s["id"] == session_id), None)
    assert session_res is not None
    assert session_res["title"] == "My Session"
    assert session_res["node_count"] == 1

    # 5. Rename session
    res = client.patch(f"/v1/sessions/{session_id}", json={"title": "New Title"}, headers=auth_headers)
    assert res.status_code == 200
    res2 = client.get("/v1/sessions", headers=auth_headers)
    session_res = next((s for s in res2.json() if s["id"] == session_id), None)
    assert session_res["title"] == "New Title"

    # 6. Cannot delete others session
    other_session_id = str(uuid.uuid4())
    with sqlite3.connect(test_db_path) as conn:
        conn.execute("INSERT INTO sessions (id, intent_id, user_id, title, status) VALUES (?, 'intent2', 'other-user', 'Not Yours', 'active')", (other_session_id,))
        conn.commit()
    res = client.delete(f"/v1/sessions/{other_session_id}", headers=auth_headers)
    assert res.status_code == 404

    # 7. Soft delete hidden from list
    res = client.delete(f"/v1/sessions/{session_id}", headers=auth_headers)
    assert res.status_code == 200
    res2 = client.get("/v1/sessions", headers=auth_headers)
    data = res2.json()
    assert not any(s["id"] == session_id for s in data)

    # 8. Deleted session graph 404
    res = client.get(f"/v1/graph/{session_id}", headers=auth_headers)
    assert res.status_code == 404

client = TestClient(app)

def test_intake_pipeline(monkeypatch, tmp_path):
    # 1. Clarify returns valid questions
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

    # 2. Build intent rejects missing answers
    response = client.post("/v1/intake/build-intent", json={"prompt": "Invest or wait?", "answers": {}})
    assert response.status_code == 400
    assert "answers are required" in response.json()["detail"]

    # 3. Build intent returns valid user intent
    app.state.job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
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
    assert payload["domain"] == "career planning"
    assert payload["horizon_months"] == 6

    # 4. Mock endpoints return static data
    clarify_response = client.get("/v1/intake/mock-clarify", params={"prompt": "test"})
    build_response = client.post("/v1/intake/mock-build-intent", json={"prompt": "test", "answers": {"q1": "yes"}})
    assert clarify_response.status_code == 200
    assert build_response.status_code == 200
