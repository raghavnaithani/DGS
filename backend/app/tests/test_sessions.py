import pytest
import sqlite3
import uuid

def test_session_linked_to_user(client, auth_headers, test_db_path):
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
    
    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute("SELECT * FROM sessions WHERE intent_id = ?", (intent_id,)).fetchone()
        assert session is not None
        assert session["user_id"] == "user-123"
