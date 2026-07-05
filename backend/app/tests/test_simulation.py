import sqlite3
import pytest

def test_start_without_auth(client, test_db_path):
    """P0.5: POST /v1/simulate/start without JWT still works (v0.1 compatibility)"""
    # 1. Inject a fake intent to satisfy _fetch_user_intent
    import uuid
    intent_id = str(uuid.uuid4())
    
    with sqlite3.connect(test_db_path) as conn:
        conn.execute(
            """INSERT INTO user_intents (id, original_prompt, domain, horizon_months, risk_tolerance, personal_context)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (intent_id, "test prompt", "career", 12, 5, "some context")
        )
        conn.commit()

    # 2. Call the endpoint without auth headers
    payload = {
        "user_intent_id": intent_id,
        "mode": "quick"
    }
    response = client.post("/v1/simulate/start", json=payload)
    
    # 3. Assert success
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert response.json()["status"] == "queued"
