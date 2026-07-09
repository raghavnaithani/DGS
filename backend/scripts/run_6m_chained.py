from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
import httpx
import jwt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.engines.simulation import merge_graphs

DEFAULT_DB = ROOT / "backend" / "app" / "database" / "dgs_phase1.sqlite3"
DEFAULT_OUT_DIR = ROOT / "backend" / "simulation_runs"
DEFAULT_JOB_FILE = DEFAULT_OUT_DIR / "current_6m_chained_job.txt"
DEFAULT_PROMPT = "I want to launch a high-end tech startup in the AI space and secure venture capital."
DEFAULT_BASE = "http://127.0.0.1:8000"

TERMINAL_STATUSES = {"completed", "failed"}

def ist_stamp() -> str:
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.now(ist).strftime("%Y%m%d_%H%M%S_IST")

def get_json(client: httpx.Client, url: str, headers: dict = None) -> dict[str, Any]:
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def post_json(client: httpx.Client, url: str, payload: dict[str, Any], headers: dict = None) -> dict[str, Any]:
    response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def wait_for_server(client: httpx.Client, base: str, timeout_seconds: int) -> None:
    print(f"Waiting for backend at {base} ...")
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{base}/health")
            if response.status_code == 200:
                print(f"Backend UP: {response.json()}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Backend not reachable after {timeout_seconds}s. Last error: {last_error}")

def read_job_from_db(db_path: Path, job_id: str) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, job_type, status, progress, current_step, error_message,
                   request_json, result_json, created_at, updated_at
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("request_json", "result_json"):
        if data.get(key):
            data[key.replace("_json", "")] = json.loads(data[key])
        data.pop(key, None)
    return data

def run_simulation(client, api_base, db_path, out_dir, job_file, payload, args, phase_name, headers) -> tuple[dict, str]:
    print(f"\n[{phase_name}] Starting simulation job...")
    job = post_json(client, f"{api_base}/simulate/start", payload, headers=headers)
    job_id = str(job["job_id"])
    job_file.write_text(job_id, encoding="utf-8")
    print(f"[{phase_name}] Job ID: {job_id}")

    print(f"\n[{phase_name}] Monitoring automatically")
    started = time.monotonic()
    last_line = ""
    latest_api_job: dict[str, Any] = {}
    latest_db_job: dict[str, Any] | None = None
    while True:
        try:
            latest_api_job = get_json(client, f"{api_base}/jobs/{job_id}", headers=headers)
            latest_db_job = read_job_from_db(db_path, job_id)
            status = latest_api_job.get("status")
            progress = latest_api_job.get("progress")
            step = (latest_db_job or {}).get("current_step") or latest_api_job.get("current_step") or "unknown"
            elapsed = int(time.monotonic() - started)
            line = f"[+{elapsed:>5}s] status={status} progress={progress}% step={step}"
            if line != last_line:
                print(line, flush=True)
                last_line = line
        except Exception as e:
            print(f"[WARN] Failed to poll job status: {e}")
            
        if latest_api_job.get("status") in TERMINAL_STATUSES:
            break
        if elapsed > args.max_wait:
            raise TimeoutError(f"Simulation did not finish within {args.max_wait}s")
        time.sleep(args.poll_interval)

    if latest_api_job.get("status") != "completed":
        error = latest_api_job.get("error_message") or (latest_db_job or {}).get("error_message")
        print(f"\n[{phase_name}] Simulation failed. Error: {error}")
        raise RuntimeError(f"Simulation failed: {error}")

    result = latest_api_job.get("result") or (latest_db_job or {}).get("result") or {}
    session_id = str(result.get("session_id") or payload.get("user_intent_id"))
    print(f"\n[{phase_name}] Fetching graph for session: {session_id}")
    graph = get_json(client, f"{api_base}/graph/{session_id}", headers=headers)
    return graph, session_id

def run(args: argparse.Namespace) -> int:
    base = args.base.rstrip("/")
    api_base = f"{base}/v1"
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    job_file = Path(args.job_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    import subprocess
    env = os.environ.copy()
    env_path = ROOT / "backend" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v

    python_exe = os.path.join(ROOT, "backend", ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print("Starting backend server subprocess...")
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--port", "8000", "--host", "127.0.0.1"],
        cwd=ROOT / "backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    
    import threading
    def stream_logs(pipe, prefix):
        for line in iter(pipe.readline, b''):
            line = line.decode('utf-8', errors='replace').rstrip()
            if line:
                print(f"[{prefix}] {line}", flush=True)

    threading.Thread(target=stream_logs, args=(backend_proc.stdout, "BACKEND"), daemon=True).start()

    with httpx.Client(timeout=httpx.Timeout(args.http_timeout)) as client:
        wait_for_server(client, base, args.server_wait)

        # ----------------------------------------------------
        # GENERATE MOCK JWT & PROFILE (PHASE 1)
        # ----------------------------------------------------
        env_path = ROOT / "backend" / ".env"
        jwt_secret = None
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("SUPABASE_JWT_SECRET="):
                    jwt_secret = line.split("=", 1)[1].strip()
                    break
        if not jwt_secret:
            print("ERROR: SUPABASE_JWT_SECRET not found in backend/.env")
            return 1
            
        print("Generating JWT for user 'test-user-123'...")
        now = dt.datetime.now(dt.timezone.utc)
        token_payload = {
            "sub": "test-user-123",
            "email": "automated_test@example.com",
            "aud": "authenticated",
            "iat": now,
            "exp": now + dt.timedelta(minutes=60)
        }
        token = jwt.encode(token_payload, jwt_secret, algorithm="HS256")
        headers = {"Authorization": f"Bearer {token}"}
        
        print("POST /v1/profile - Creating user profile...")
        profile_payload = {
            "expertise_level": "intermediate",
            "risk_tolerance": 5,
            "values": ["Sustainable growth", "Work-life balance"],
            "life_situation": "Testing Phase 1 with $15,000 budget, moderate risk-taker, looking for steady progress without burning out"
        }
        post_json(client, f"{api_base}/profile", profile_payload, headers=headers)
        
        # ----------------------------------------------------
        # RUN 1: MONTHS 0-3
        # ----------------------------------------------------
        print(f"\n--- PHASE 1: MONTHS 0-3 ---")
        print(f"[Phase1] Building live intent for prompt: {args.prompt!r}")
        intent1 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": args.prompt, "answers": {"q1": "make extra money", "q2": "3", "q3": "4", "q4": "$500 budget"}}, headers=headers)
        intent1_id = str(intent1["id"])
        
        sim1_payload = {
            "user_intent_id": intent1_id,
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
        }
        graph1, session1_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim1_payload, args, "Phase1", headers)
        
        # Extract terminal node summaries
        g1_nodes = graph1.get("nodes", [])
        g1_edges = graph1.get("edges", [])
        sources_in_g1 = set(edge.get("source") for edge in g1_edges)
        leaf_nodes_g1 = [node for node in g1_nodes if node.get("id") not in sources_in_g1]
        if not leaf_nodes_g1 and g1_nodes:
            leaf_nodes_g1 = [g1_nodes[-1]]
            
        summaries = [n.get("summary") or n.get("title") for n in leaf_nodes_g1]
        state_summary = "; ".join(filter(None, summaries))
        
        
        print(f"\n[Sleep] Waiting 15 seconds to clear Groq TPM rate limits before Phase 2...")
        time.sleep(15)

        # ----------------------------------------------------
        # RUN 2: MONTHS 4-6
        # ----------------------------------------------------
        print(f"\n--- PHASE 2: MONTHS 4-6 ---")
        continuation_prompt = (
            f"Original Goal: {args.prompt}. "
            f"You have completed the first 3 months. Current state: {state_summary}. "
            f"Now plan the next 3 months (Months 4-6). You MUST include Watch Points and long-term pivot risks. "
            f"After the actionable steps, include a 'watchpoints' section as a JSON array of objects, each with 'id' and 'text' fields. "
            f"List 2-3 specific trends, risks, or market changes the user should monitor over Months 4-6. "
            f"For each, suggest how to adapt if it occurs. Cite evidence where possible."
        )
        print(f"[Phase2] Building live continuation intent: {continuation_prompt!r}")
        intent2 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": continuation_prompt, "answers": {"q1": "continue simulation", "q2": "3", "q3": "4", "q4": "no constraints"}}, headers=headers)
        intent2_id = str(intent2["id"])
        
        sim2_payload = {
            "user_intent_id": intent2_id,
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
        }
        graph2, session2_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim2_payload, args, "Phase2", headers)
        
        # ----------------------------------------------------
        # MERGE GRAPHS
        # ----------------------------------------------------
        print(f"\n--- MERGING GRAPHS ---")
        merged_graph = merge_graphs(graph1, graph2)
        
        stamp = ist_stamp()
        graph_path = out_dir / f"run_output_6m_chained_graph_{stamp}.json"
        graph_path.write_text(json.dumps(merged_graph, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # Phase 2 History and Graph Delete Tests
        print("\n--- START PHASE 2 FEATURE VERIFICATION ---")
        
        print("Testing GET /v1/sessions ...")
        r_hist = client.get(f"{api_base}/sessions", headers=headers)
        if r_hist.status_code == 200:
            sessions = r_hist.json()
            if isinstance(sessions, dict):
                sessions = sessions.get("sessions", [])
            print(f"History Success: Found {len(sessions)} sessions.")
            if not any(s.get("id") == intent2_id for s in sessions):
                print("ERROR: Session 2 not found in history!")
        else:
            print(f"History Fetch Error: {r_hist.status_code} {r_hist.text}")
            
        print("Testing PATCH /v1/sessions/{id} ...")
        r_patch = client.patch(f"{api_base}/sessions/{intent2_id}", json={"title": "Automated Rename 6M Mid"}, headers=headers)
        if r_patch.status_code == 200:
            print("Rename Success: Graph renamed to 'Automated Rename 6M Mid'")
        else:
            print(f"Rename Error: {r_patch.status_code} {r_patch.text}")
        
        print("Testing DELETE /v1/sessions/{id} ...")
        r_del = client.delete(f"{api_base}/sessions/{intent2_id}", headers=headers)
        if r_del.status_code == 200:
            print("Delete Success: Session soft-deleted.")
        else:
            print(f"Delete Error: {r_del.status_code} {r_del.text}")
        
        print("Testing GET /v1/graph/{id} on soft-deleted session ...")
        r_graph = client.get(f"{api_base}/graph/{intent2_id}", headers=headers)
        if r_graph.status_code == 404:
            print("Graph Load Success: 404 Not Found returned successfully for deleted graph.")
        else:
            print(f"Graph Load Error: Expected 404, got {r_graph.status_code}")
            
        print("--- END PHASE 2 FEATURE VERIFICATION ---")
        
        print("\nCompleted Chained 6-Month Simulation")
        print(f"Graph dump: {graph_path}")
        print(f"Total Nodes: {len(merged_graph.get('nodes', []))}")
        print(f"Total Edges: {len(merged_graph.get('edges', []))}")
        
        backend_proc.terminate()
        return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 6-month chained live DGS simulation.")
    parser.add_argument("--base", default=os.environ.get("DGS_BASE", DEFAULT_BASE))
    parser.add_argument("--prompt", default=os.environ.get("DGS_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--db", default=os.environ.get("DGS_DB", str(DEFAULT_DB)))
    parser.add_argument("--out-dir", default=os.environ.get("DGS_OUT_DIR", str(DEFAULT_OUT_DIR)))
    parser.add_argument("--job-file", default=os.environ.get("DGS_JOB_FILE", str(DEFAULT_JOB_FILE)))
    parser.add_argument("--depth", type=int, default=int(os.environ.get("DGS_SIM_DEPTH", "2")))
    parser.add_argument("--branching-factor", type=int, default=int(os.environ.get("DGS_SIM_BRANCHING", "3")))
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("DGS_POLL_INTERVAL", "5")))
    parser.add_argument("--server-wait", type=int, default=int(os.environ.get("DGS_SERVER_WAIT", "90")))
    parser.add_argument("--max-wait", type=int, default=int(os.environ.get("DGS_MAX_WAIT", "7200")))
    parser.add_argument("--http-timeout", type=float, default=float(os.environ.get("DGS_HTTP_TIMEOUT", "120")))
    return parser.parse_args()

if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
