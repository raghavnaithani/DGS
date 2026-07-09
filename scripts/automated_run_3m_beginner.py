import httpx
import json
import time
import sys
import os
import subprocess
import threading
import sqlite3
import jwt
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytz

ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime("%Y%m%d_%H%M%S")

BASE_URL = "http://127.0.0.1:8000"
PROMPT = "wanna start selling handmade stuff online idk candles or something got maybe 500 bucks. I want a 3-month plan."
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SIM_RUNS_DIR = PROJECT_ROOT / "SIMULATIONS_RUNS"
SIM_RUNS_DIR.mkdir(exist_ok=True)
OUT_FILE = SIM_RUNS_DIR / f"sim_run_3m_{current_time_ist}_IST.json"

# Database path based on Phase 1 settings
DB_PATH = os.path.join(BACKEND_DIR, "app", "database", "dgs_phase1.sqlite3")

def log(msg, prefix="TEST"):
    t = datetime.now(ist).strftime("%I:%M:%S %p")
    print(f"[{t} IST] [{prefix}] {msg}", flush=True)

def stream_logs(pipe, prefix):
    for line in iter(pipe.readline, b''):
        line = line.decode('utf-8', errors='replace').rstrip()
        if line:
            log(line, prefix)

def get_env_var(key):
    env_path = os.path.join(BACKEND_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.strip().split('=', 1)[1]
    return None

def run():
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
    
    # 1. Start Backend Subprocess
    log("Starting backend server subprocess...")
    env = os.environ.copy()
    
    # Load .env into subprocess env
    env_path = os.path.join(BACKEND_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        env[k] = v

    python_exe = os.path.join(BACKEND_DIR, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--port", "8000", "--host", "127.0.0.1"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    
    threading.Thread(target=stream_logs, args=(backend_proc.stdout, "BACKEND"), daemon=True).start()

    # 2. Wait for Backend
    log("Waiting for backend to become healthy...")
    for _ in range(60):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                log("Backend is UP and healthy!")
                break
        except Exception:
            time.sleep(1)
    else:
        log("Backend failed to start or is not reachable.")
        backend_proc.terminate()
        sys.exit(1)

    # 3. Generate Mock JWT for Phase 1 Authentication
    jwt_secret = get_env_var("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        log("ERROR: SUPABASE_JWT_SECRET not found in backend/.env")
        backend_proc.terminate()
        sys.exit(1)
        
    log("Generating JWT for user 'test-user-123'...")
    now = datetime.now(timezone.utc)
    token_payload = {
        "sub": "test-user-123",
        "email": "automated_test@example.com",
        "aud": "authenticated",
        "iat": now,
        "exp": now + timedelta(minutes=60)
    }
    token = jwt.encode(token_payload, jwt_secret, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 4. Create User Profile (Phase 1 Logic)
    log("POST /v1/profile - Creating user profile...")
    profile_payload = {
        "expertise_level": "beginner",
        "risk_tolerance": 2,
        "values": ["Stability", "Health"],
        "life_situation": "Testing Phase 1 with beginner constraints"
    }
    r = httpx.post(f"{BASE_URL}/v1/profile", json=profile_payload, headers=headers, timeout=10)
    if r.status_code != 200:
        log(f"Failed to create profile: {r.text}")
        backend_proc.terminate()
        sys.exit(1)
    log(f"Profile created/updated: {r.json()}")

    # 5. Clarify Intent
    log(f"POST /v1/intake/clarify - Prompt: '{PROMPT}'")
    r = httpx.post(f"{BASE_URL}/v1/intake/clarify", json={"prompt": PROMPT}, headers=headers, timeout=60)
    questions = r.json().get("questions", [])
    
    answers = {}
    crude_answers = [
        "not sure yet, just starting", "no experience", "just weekends",
        "want to make some extra money", "no idea really", "maybe etsy or instagram",
        "just myself", "no",
    ]
    for i, q in enumerate(questions):
        qid = q.get("id") or q.get("question_id") or str(i)
        answers[qid] = crude_answers[i] if i < len(crude_answers) else "not sure"

    # 6. Build Intent
    log("POST /v1/intake/build-intent...")
    r = httpx.post(f"{BASE_URL}/v1/intake/build-intent", json={"prompt": PROMPT, "answers": answers}, headers=headers, timeout=90)
    intent_id = r.json()["id"]
    log(f"Intent ID created: {intent_id}")

    # 7. Start Simulation
    log("POST /v1/simulate/start...")
    r = httpx.post(f"{BASE_URL}/v1/simulate/start", json={"user_intent_id": intent_id, "disable_scraping": False}, headers=headers, timeout=30)
    job_id = r.json()["job_id"]
    log(f"Job started successfully: {job_id}")

    # 8. Monitor Job & SQLite Database
    start_time = time.time()
    log("Monitoring job progress and SQLite nodes...")
    while True:
        try:
            r = httpx.get(f"{BASE_URL}/v1/simulate/jobs/{job_id}", headers=headers, timeout=10)
            data = r.json()
            status = data.get("status")
            progress = data.get("progress")
            
            # Query SQLite for real-time stats
            node_count = 0
            if os.path.exists(DB_PATH):
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE session_id = ?", (intent_id,)).fetchone()[0]
                except Exception as e:
                    pass
            
            if status == "completed":
                end_time = time.time()
                log(f"Job COMPLETE in {end_time - start_time:.2f} seconds!")
                result = data.get("result", {})
                with open(OUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                
                nodes = result.get("nodes", [])
                alts = [len(n.get("alternatives", [])) for n in nodes]
                log(f"Final Output: {len(nodes)} nodes. Alternatives per node: {alts}")
                
                # Check DB auto-linking
                with sqlite3.connect(DB_PATH) as conn:
                    session_row = conn.execute("SELECT user_id, node_count FROM sessions WHERE intent_id = ?", (intent_id,)).fetchone()
                    if session_row:
                        log(f"DB Validation: session.user_id = {session_row[0]} (Expected: test-user-123), session.node_count = {session_row[1]}")
                    else:
                        log("DB Validation: Session row not found!")
                
                # Phase 2 History and Graph Delete Tests
                log("--- START PHASE 2 FEATURE VERIFICATION ---")
                
                log("Testing GET /v1/sessions ...")
                r_hist = httpx.get(f"{BASE_URL}/v1/sessions", headers=headers, timeout=10)
                if r_hist.status_code == 200:
                    sessions = r_hist.json() # Endpoint returns a list directly
                    if isinstance(sessions, dict):
                        sessions = sessions.get("sessions", [])
                    log(f"History Success: Found {len(sessions)} sessions.")
                    if not any(s.get("id") == intent_id for s in sessions):
                        log("ERROR: Session not found in history!")
                else:
                    log(f"History Fetch Error: {r_hist.status_code} {r_hist.text}")
                    
                log("Testing PATCH /v1/sessions/{id} ...")
                r_patch = httpx.patch(f"{BASE_URL}/v1/sessions/{intent_id}", json={"title": "Automated Rename 3M Beginner"}, headers=headers, timeout=10)
                if r_patch.status_code == 200:
                    log("Rename Success: Graph renamed to 'Automated Rename 3M Beginner'")
                else:
                    log(f"Rename Error: {r_patch.status_code} {r_patch.text}")
                
                log("Testing DELETE /v1/sessions/{id} ...")
                r_del = httpx.delete(f"{BASE_URL}/v1/sessions/{intent_id}", headers=headers, timeout=10)
                if r_del.status_code == 200:
                    log("Delete Success: Session soft-deleted.")
                else:
                    log(f"Delete Error: {r_del.status_code} {r_del.text}")
                
                log("Testing GET /v1/graph/{id} on soft-deleted session ...")
                r_graph = httpx.get(f"{BASE_URL}/v1/graph/{intent_id}", headers=headers, timeout=10)
                if r_graph.status_code == 404:
                    log("Graph Load Success: 404 Not Found returned successfully for deleted graph.")
                else:
                    log(f"Graph Load Error: Expected 404, got {r_graph.status_code}")
                    
                log("--- END PHASE 2 FEATURE VERIFICATION ---")
                break
            elif status in ("failed", "error"):
                log(f"Job failed! {data}")
                break
            else:
                log(f"Progress: {progress}% - Step: {data.get('current_step')} - DB Nodes Generated: {node_count}")
        except Exception as e:
            log(f"Monitor error: {e}")
        time.sleep(5)
    
    log("Shutting down backend...")
    backend_proc.terminate()
    backend_proc.wait()

if __name__ == "__main__":
    run()
