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
DEFAULT_JOB_FILE = DEFAULT_OUT_DIR / "current_12m_chained_job.txt"
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
            "target_nodes": 5,
        }
        graph1, session1_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim1_payload, args, "Phase1")
        
        def get_state_summary(graph):
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            sources = set(edge.get("source") for edge in edges)
            leaf_nodes = [node for node in nodes if node.get("id") not in sources]
            if not leaf_nodes and nodes:
                leaf_nodes = [nodes[-1]]
            summaries = [n.get("summary") or n.get("title") for n in leaf_nodes]
            return "; ".join(filter(None, summaries))
            
        print(f"\n[Sleep] Waiting 15 seconds to clear Groq TPM rate limits before Phase 2...")
        time.sleep(15)

        # ----------------------------------------------------
        # RUN 2: MONTHS 4-6
        # ----------------------------------------------------
        print(f"\n--- PHASE 2: MONTHS 4-6 ---")
        state_summary = get_state_summary(graph1)
        continuation_prompt2 = (
            f"Original Goal: {args.prompt}. "
            f"You have completed the first 3 months. Current state: {state_summary}. "
            f"Now plan the next 3 months (Months 4-6). You MUST include Watch Points and long-term pivot risks. "
            f"After the actionable steps, include a 'watchpoints' section as a JSON array of objects, each with 'id' and 'text' fields. "
            f"List 2-3 specific trends, risks, or market changes the user should monitor over Months 4-6. "
            f"For each, suggest how to adapt if it occurs. Cite evidence where possible."
        )
        intent2 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": continuation_prompt2, "answers": {"q1": "continue simulation", "q2": "3", "q3": "4", "q4": "no constraints"}})
        
        sim2_payload = {
            "user_intent_id": str(intent2["id"]),
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
            "target_nodes": 5,
        }
        graph2, session2_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim2_payload, args, "Phase2")

        print(f"\n[Sleep] Waiting 15 seconds before Phase 3...")
        time.sleep(15)

        # ----------------------------------------------------
        # RUN 3: MONTHS 7-9
        # ----------------------------------------------------
        print(f"\n--- PHASE 3: MONTHS 7-9 ---")
        state_summary = get_state_summary(graph2)
        continuation_prompt3 = (
            f"Original Goal: {args.prompt}. "
            f"You have completed 6 months. Current state: {state_summary}. "
            f"Now plan the next 3 months (Months 7-9). You MUST include Watch Points and long-term pivot risks. "
            f"After the actionable steps, include a 'watchpoints' section as a JSON array of objects, each with 'id' and 'text' fields. "
            f"List 2-3 specific trends, risks, or market changes the user should monitor over Months 7-9. "
            f"For each, suggest how to adapt if it occurs. Cite evidence where possible."
        )
        intent3 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": continuation_prompt3, "answers": {"q1": "continue simulation", "q2": "3", "q3": "4", "q4": "no constraints"}})
        
        sim3_payload = {
            "user_intent_id": str(intent3["id"]),
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
            "target_nodes": 5,
        }
        graph3, session3_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim3_payload, args, "Phase3")

        print(f"\n[Sleep] Waiting 15 seconds before Phase 4...")
        time.sleep(15)
        
        # ----------------------------------------------------
        # RUN 4: MONTHS 10-12
        # ----------------------------------------------------
        print(f"\n--- PHASE 4: MONTHS 10-12 ---")
        state_summary = get_state_summary(graph3)
        continuation_prompt4 = (
            f"Original Goal: {args.prompt}. "
            f"You have completed 9 months. Current state: {state_summary}. "
            f"Now plan the final 3 months (Months 10-12). You MUST include Watch Points and long-term pivot risks. "
            f"After the actionable steps, include a 'watchpoints' section as a JSON array of objects, each with 'id' and 'text' fields. "
            f"List 2-3 specific trends, risks, or market changes the user should monitor over Months 10-12. "
            f"For each, suggest how to adapt if it occurs. Cite evidence where possible."
        )
        intent4 = post_json(client, f"{api_base}/intake/build-intent", {"prompt": continuation_prompt4, "answers": {"q1": "continue simulation", "q2": "3", "q3": "4", "q4": "no constraints"}})
        
        sim4_payload = {
            "user_intent_id": str(intent4["id"]),
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
            "target_nodes": 5,
        }
        graph4, session4_id = run_simulation(client, api_base, db_path, out_dir, job_file, sim4_payload, args, "Phase4")

        # ----------------------------------------------------
        # MERGE GRAPHS
        # ----------------------------------------------------
        print(f"\n--- MERGING GRAPHS ---")
        merged = merge_graphs(graph1, graph2, namespace_prefix="g2_", time_offset=3)
        merged = merge_graphs(merged, graph3, namespace_prefix="g3_", time_offset=6)
        merged = merge_graphs(merged, graph4, namespace_prefix="g4_", time_offset=9)
        
        stamp = utc_stamp()
        graph_path = out_dir / f"run_output_12m_chained_graph_{stamp}.json"
        graph_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print("\nCompleted Chained 12-Month Simulation")
        print(f"Graph dump: {graph_path}")
        print(f"Total Nodes: {len(merged.get('nodes', []))}")
        print(f"Total Edges: {len(merged.get('edges', []))}")
        return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 12-month chained live DGS simulation.")
    parser.add_argument("--base", default=os.environ.get("DGS_BASE", DEFAULT_BASE))
    parser.add_argument("--prompt", default=os.environ.get("DGS_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--db", default=os.environ.get("DGS_DB", str(DEFAULT_DB)))
    parser.add_argument("--out-dir", default=os.environ.get("DGS_OUT_DIR", str(DEFAULT_OUT_DIR)))
    parser.add_argument("--job-file", default=os.environ.get("DGS_JOB_FILE", str(DEFAULT_JOB_FILE)))
    parser.add_argument("--depth", type=int, default=int(os.environ.get("DGS_SIM_DEPTH", "2")))
    parser.add_argument("--branching-factor", type=int, default=int(os.environ.get("DGS_SIM_BRANCHING", "2")))
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
