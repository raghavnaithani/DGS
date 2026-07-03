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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.engines.simulation import merge_graphs

DEFAULT_DB = ROOT / "backend" / "app" / "database" / "dgs_phase1.sqlite3"
DEFAULT_OUT_DIR = ROOT / "backend" / "simulation_runs"
DEFAULT_JOB_FILE = DEFAULT_OUT_DIR / "current_6m_chained_job.txt"
DEFAULT_PROMPT = "wanna start selling handmade stuff online idk candles or something got maybe 500 bucks"
DEFAULT_BASE = "http://127.0.0.1:8000"

TERMINAL_STATUSES = {"completed", "failed"}

def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def get_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    return response.json()

def post_json(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
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

def run_simulation(client, api_base, db_path, out_dir, job_file, payload, args, phase_name) -> tuple[dict, str]:
    print(f"\n[{phase_name}] Starting simulation job...")
    job = post_json(client, f"{api_base}/simulate/start", payload)
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
            latest_api_job = get_json(client, f"{api_base}/simulate/jobs/{job_id}")
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
    graph = get_json(client, f"{api_base}/graph/{session_id}")
    return graph, session_id

def run(args: argparse.Namespace) -> int:
    base = args.base.rstrip("/")
    api_base = f"{base}/v1"
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    job_file = Path(args.job_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=httpx.Timeout(args.http_timeout)) as client:
        wait_for_server(client, base, args.server_wait)

        # ----------------------------------------------------
        # RUN 1: MONTHS 0-3
        # ----------------------------------------------------
        print(f"\n--- PHASE 1: MONTHS 0-3 ---")
        print(f"[Phase1] Building live intent for prompt: {args.prompt!r}")
        intent1 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": args.prompt, "answers": {"q1": "make extra money", "q2": "3", "q3": "4", "q4": "$500 budget"}})
        intent1_id = str(intent1["id"])
        
        sim1_payload = {
            "user_intent_id": intent1_id,
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
        }
        graph1, session1_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim1_payload, args, "Phase1")
        
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
        print(f"\n