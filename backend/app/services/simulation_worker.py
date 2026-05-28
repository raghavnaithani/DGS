from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from ..database.connection import get_connection
from ..database.jobs_store import SQLiteJobStore
from ..database.vector_store import LanceChunkStore
from ..engines.simulation import generate_branch_node, generate_initial_graph

logger = logging.getLogger(__name__)


class SimulationJobWorker:
    def __init__(self, *, job_store: SQLiteJobStore, vector_store: LanceChunkStore, poll_interval_seconds: float = 0.25):
        self.job_store = job_store
        self.vector_store = vector_store
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.set()
        self._thread = threading.Thread(target=self._run_loop, name="simulation-job-worker", daemon=True)
        self._thread.start()
        logger.info("Simulation worker started with db_path=%s", self.job_store.db_path)

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Simulation worker stopped")

    def kick(self) -> None:
        self._wake_event.set()

    def enqueue_start(self, *, user_intent_id: str, persona: str | None = None, webhook_url: str | None = None) -> Any:
        job = self.job_store.create_simulation_job(
            {
                "workflow": "start",
                "user_intent_id": user_intent_id,
                "persona": persona,
                "webhook_url": webhook_url,
            }
        )
        self.kick()
        return job

    def enqueue_branch(
        self,
        *,
        session_id: str,
        parent_node_id: str,
        action_description: str,
        persona: str | None = None,
        webhook_url: str | None = None,
    ) -> Any:
        job = self.job_store.create_simulation_job(
            {
                "workflow": "branch",
                "session_id": session_id,
                "parent_node_id": parent_node_id,
                "action_description": action_description,
                "persona": persona,
                "webhook_url": webhook_url,
            }
        )
        self.kick()
        return job

    def process_pending_jobs_once(self) -> int:
        processed = 0
        while True:
            job = self.job_store.claim_next_simulation_job()
            if job is None:
                break
            self._process_job(job.id, job.request)
            processed += 1
        return processed

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self.job_store.claim_next_simulation_job()
            if job is None:
                self._wake_event.wait(self.poll_interval_seconds)
                self._wake_event.clear()
                continue
            self._process_job(job.id, job.request)

    def _process_job(self, job_id: str, request: dict[str, Any]) -> None:
        try:
            self.job_store.update_job(job_id, status="running", progress=5, current_step="starting")
            workflow = str(request.get("workflow") or "start")
            if workflow == "start":
                result = self._run_start(request)
            elif workflow == "branch":
                result = self._run_branch(request)
            else:
                raise ValueError(f"Unsupported workflow: {workflow}")
            self.job_store.update_job(job_id, status="completed", progress=100, current_step="completed", result=result)
            self._dispatch_webhook(request, job_id=job_id, status="completed", result=result)
        except Exception as exc:
            logger.exception("Simulation job failed job_id=%s", job_id)
            self.job_store.update_job(job_id, status="failed", progress=100, current_step="failed", error_message=str(exc))
            self._dispatch_webhook(request, job_id=job_id, status="failed", error_message=str(exc))

    def _run_start(self, request: dict[str, Any]) -> dict[str, Any]:
        user_intent_id = str(request["user_intent_id"])
        user_intent = self._fetch_user_intent(user_intent_id)
        session_id = self._ensure_session(user_intent_id=user_intent_id, session_id=user_intent_id, title=user_intent["original_prompt"])
        graph = generate_initial_graph(user_intent, top_k=10)
        self._persist_graph(session_id=session_id, graph=graph)
        return {"session_id": session_id, "workflow": "start", **graph}

    def _run_branch(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        parent_node_id = str(request["parent_node_id"])
        action_description = str(request["action_description"])
        persona = request.get("persona")

        session = self._fetch_session(session_id)
        if session is None:
            raise KeyError(f"session:{session_id}")
        user_intent = self._fetch_user_intent(str(session["intent_id"]))
        parent_node = self._fetch_node(session_id=session_id, node_id=parent_node_id)
        if parent_node is None:
            raise KeyError(f"node:{parent_node_id}")

        branch = generate_branch_node(
            user_intent=user_intent,
            parent_node=parent_node,
            action_description=action_description,
            persona_prompt=str(persona) if persona else None,
            top_k=6,
        )
        node = branch["node"]
        self._persist_node(session_id=session_id, node=node)
        self._persist_edge(session_id=session_id, source_node_id=parent_node_id, target_node_id=str(node["id"]), action_description=action_description)
        return {
            "session_id": session_id,
            "workflow": "branch",
            "parent_node_id": parent_node_id,
            "node": node,
        }

    def _fetch_user_intent(self, intent_id: str) -> dict[str, Any]:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, original_prompt, domain, horizon_months, risk_tolerance,
                       constraints_json, personal_context, clarified_entities_json,
                       ambiguities_remaining_json, created_at
                FROM user_intents
                WHERE id = ?
                """,
                (intent_id,),
            ).fetchone()
        if row is None:
            raise KeyError(intent_id)
        return {
            "id": row["id"],
            "original_prompt": row["original_prompt"],
            "domain": row["domain"],
            "horizon_months": int(row["horizon_months"]),
            "risk_tolerance": int(row["risk_tolerance"]),
            "constraints": json.loads(row["constraints_json"] or "[]"),
            "personal_context": row["personal_context"],
            "clarified_entities": json.loads(row["clarified_entities_json"] or "[]"),
            "ambiguities_remaining": json.loads(row["ambiguities_remaining_json"] or "[]"),
            "created_at": row["created_at"],
        }

    def _ensure_session(self, *, user_intent_id: str, session_id: str, title: str) -> str:
        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions (id, intent_id, title)
                VALUES (?, ?, ?)
                """,
                (session_id, user_intent_id, title[:200]),
            )
            connection.commit()
        return session_id

    def _fetch_session(self, session_id: str) -> dict[str, Any] | None:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                "SELECT id, intent_id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "intent_id": row["intent_id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def _fetch_node(self, *, session_id: str, node_id: str) -> dict[str, Any] | None:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, session_id, title, summary, description, time_step, created_by_engine,
                       alternatives_json, risks_json, source_citations_json, confidence_score,
                       speculative, created_at
                FROM nodes
                WHERE session_id = ? AND id = ?
                """,
                (session_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "title": row["title"],
            "summary": row["summary"],
            "description": row["description"],
            "time_step": int(row["time_step"]),
            "created_by_engine": row["created_by_engine"],
            "alternatives": json.loads(row["alternatives_json"] or "[]"),
            "risks": json.loads(row["risks_json"] or "[]"),
            "source_citations": json.loads(row["source_citations_json"] or "[]"),
            "confidence_score": float(row["confidence_score"]),
            "speculative": bool(row["speculative"]),
            "created_at": row["created_at"],
        }

    def _persist_graph(self, *, session_id: str, graph: dict[str, Any]) -> None:
        for node in graph.get("nodes", []):
            self._persist_node(session_id=session_id, node=node)
        for edge in graph.get("edges", []):
            source_node_id = str(edge.get("source") or edge.get("from"))
            target_node_id = str(edge.get("target") or edge.get("to"))
            action_description = str(edge.get("action_description") or "initial transition")
            self._persist_edge(
                session_id=session_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                action_description=action_description,
            )

    def _persist_node(self, *, session_id: str, node: dict[str, Any]) -> None:
        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO nodes (
                    id, session_id, title, summary, description, time_step, created_by_engine,
                    alternatives_json, risks_json, source_citations_json, confidence_score,
                    speculative, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(node["id"]),
                    session_id,
                    str(node.get("title") or "Decision Node"),
                    str(node.get("summary") or ""),
                    str(node.get("description") or ""),
                    int(node.get("time_step", 0)),
                    str(node.get("created_by_engine") or "phase5.node_generator"),
                    json.dumps(node.get("alternatives", []), ensure_ascii=False),
                    json.dumps(node.get("risks", []), ensure_ascii=False),
                    json.dumps(node.get("source_citations", []), ensure_ascii=False),
                    float(node.get("confidence_score", 0.0)),
                    1 if bool(node.get("speculative")) else 0,
                    str(node.get("created_at") or ""),
                ),
            )
            connection.commit()

    def _persist_edge(self, *, session_id: str, source_node_id: str, target_node_id: str, action_description: str) -> None:
        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT INTO edges (session_id, source_node_id, target_node_id, action_description)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, source_node_id, target_node_id, action_description),
            )
            connection.commit()

    def _dispatch_webhook(self, request: dict[str, Any], *, job_id: str, status: str, result: dict[str, Any] | None = None, error_message: str | None = None) -> None:
        webhook_url = request.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "job_id": job_id,
            "status": status,
            "workflow": request.get("workflow", "start"),
            "result": result,
            "error_message": error_message,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(str(webhook_url), json=payload)
            logger.info("Webhook delivered job_id=%s status=%s code=%s", job_id, status, response.status_code)
        except Exception as exc:
            logger.warning("Webhook delivery failed job_id=%s status=%s error=%s", job_id, status, exc)
