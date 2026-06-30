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
DEFAULT_DB = ROOT / "backend" / "app" / "database" / "dgs_phase1.sqlite3"
DEFAULT_OUT_DIR = ROOT / "backend" / "simulation_runs"
DEFAULT_JOB_FILE = DEFAULT_OUT_DIR / "current_3m_live_job.txt"
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


def answer_for_question(question: dict[str, Any], index: int) -> str:
    text = str(question.get("text") or question.get("question") or "").lower()
    qtype = str(question.get("type") or "").lower()
    choices = question.get("choices") or []

    if "horizon" in text or "time" in text or "months" in text:
        return "3"
    if qtype == "number" or "risk" in text:
        return "4"
    if "budget" in text or "constraint" in text or "limit" in text:
        return "$500 budget, weekends only, just myself, no prior ecommerce experience"
    if "platform" in text or "sell" in text or "channel" in text:
        return "maybe Etsy or Instagram, not sure yet"
    if "outcome" in text or "goal" in text:
        return "make extra money within 3 months without taking huge risk"
    if "experience" in text:
        return "no business experience, just starting"
    if choices:
        return str(choices[0])

    fallbacks = [
        "not sure yet, just starting",
        "no experience",
        "just weekends",
        "want to make some extra money",
        "no idea really",
        "maybe Etsy or Instagram",
        "just myself",
        "no",
    ]
    return fallbacks[index] if index < len(fallbacks) else "not sure"


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


def fetch_graph(client: httpx.Client, api_base: str, session_id: str) -> dict[str, Any]:
    return get_json(client, f"{api_base}/graph/{session_id}")


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

        print(f"\n[1] Clarifying live prompt: {args.prompt!r}")
        clarify = post_json(client, f"{api_base}/intake/clarify", {"prompt": args.prompt})
        questions = clarify.get("questions", [])
        print(f"Got {len(questions)} clarification question(s)")
        answers: dict[str, str] = {}
        for index, question in enumerate(questions):
            question_id = str(question.get("id") or question.get("question_id") or index)
            answer = answer_for_question(question, index)
            answers[question_id] = answer
            question_text = str(question.get("text") or question.get("question") or "")[:90]
            print(f"  {question_id}: {question_text} => {answer!r}")

        print("\n[2] Building live intent with Groq")
        intent = post_json(client, f"{api_base}/intake/build-intent", {"prompt": args.prompt, "answers": answers})
        intent_id = str(intent["id"])
        print(f"Intent ID: {intent_id}")
        print(f"Horizon: {intent.get('horizon_months')} months")
        print(f"Domain: {intent.get('domain')}")

        print("\n[3] Starting detailed 3-month simulation with live scraping enabled")
        sim_payload = {
            "user_intent_id": intent_id,
            "disable_scraping": False,
            "depth": args.depth,
            "branching_factor": args.branching_factor,
            "mode": "detailed",
        }
        job = post_json(client, f"{api_base}/simulate/start", sim_payload)
        job_id = str(job["job_id"])
        job_file.write_text(job_id, encoding="utf-8")
        print(f"Job ID: {job_id}")
        print(f"Job file: {job_file}")

        print("\n[4] Monitoring automatically")
        started = time.monotonic()
        last_line = ""
        latest_api_job: dict[str, Any] = {}
        latest_db_job: dict[str, Any] | None = None
        while True:
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
            if status in TERMINAL_STATUSES:
                break
            if elapsed > args.max_wait:
                raise TimeoutError(f"Simulation did not finish within {args.max_wait}s")
            time.sleep(args.poll_interval)

        full_dump: dict[str, Any] = {
            "scenario": args.prompt,
            "intent": intent,
            "clarification": {"questions": questions, "answers": answers},
            "simulation_request": sim_payload,
            "job_id": job_id,
            "job": latest_api_job,
            "job_db": latest_db_job,
            "graph": None,
        }

        if latest_api_job.get("status") != "completed":
            error = latest_api_job.get("error_message") or (latest_db_job or {}).get("error_message")
            failed_path = out_dir / f"run_output_3m_live_failed_{utc_stamp()}.json"
            failed_path.write_text(json.dumps(full_dump, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSimulation failed. Dumped failure payload: {failed_path}")
            print(f"Error: {error}")
            return 1

        result = latest_api_job.get("result") or (latest_db_job or {}).get("result") or {}
        session_id = str(result.get("session_id") or intent_id)
        print(f"\n[5] Fetching graph for session: {session_id}")
        graph = fetch_graph(client, api_base, session_id)
        full_dump["session_id"] = session_id
        full_dump["graph"] = graph

        stamp = utc_stamp()
        full_path = out_dir / f"run_output_3m_live_full_{stamp}.json"
        graph_path = out_dir / f"run_output_3m_live_graph_{stamp}.json"
        full_path.write_text(json.dumps(full_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nCompleted")
        print(f"Full dump: {full_path}")
        print(f"Graph dump: {graph_path}")
        print(f"Nodes: {len(graph.get('nodes', []))}")
        print(f"Edges: {len(graph.get('edges', []))}")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and monitor a full 3-month live DGS simulation.")
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
