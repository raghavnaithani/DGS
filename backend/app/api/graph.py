from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from ..database.connection import get_connection

router = APIRouter()
share_router = APIRouter()


def _load_graph(db_path: str, *, session_id: str) -> dict:
    with get_connection(db_path) as connection:
        session = connection.execute(
            "SELECT id, intent_id, title, created_at, updated_at FROM sessions WHERE id = ? AND status = 'active'",
            (session_id,),
        ).fetchone()
        if session is None:
            raise KeyError(session_id)

        nodes = connection.execute(
            """
            SELECT id, session_id, title, summary, description, time_step, created_by_engine,
                   alternatives_json, risks_json, source_citations_json, confidence_score,
                   speculative, watchpoints_json, created_at
            FROM nodes
            WHERE session_id = ?
            ORDER BY time_step ASC, created_at ASC
            """,
            (session_id,),
        ).fetchall()

        edges = connection.execute(
            """
            SELECT id, session_id, source_node_id, target_node_id, action_description, created_at
            FROM edges
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        share = connection.execute(
            "SELECT public_id FROM graph_shares WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    return {
        "session_id": session["id"],
        "title": session["title"],
        "intent_id": session["intent_id"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "public_id": share["public_id"] if share else None,
        "nodes": [
            {
                "id": row["id"],
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
                "watchpoints": json.loads(row["watchpoints_json"] or "[]"),
                "created_at": row["created_at"],
            }
            for row in nodes
        ],
        "edges": [
            {
                "id": str(row["id"]),
                "source": row["source_node_id"],
                "target": row["target_node_id"],
                "action_description": row["action_description"],
            }
            for row in edges
        ],
    }


@router.get("/{session_id}")
def get_graph(session_id: str, request: Request) -> dict:
    job_store = getattr(request.app.state, "job_store", None)
    if job_store is None:
        raise HTTPException(status_code=500, detail="Job store is unavailable")

    try:
        return _load_graph(str(job_store.db_path), session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/{session_id}/share")
def share_graph(session_id: str, request: Request) -> dict:
    job_store = getattr(request.app.state, "job_store", None)
    if job_store is None:
        raise HTTPException(status_code=500, detail="Job store is unavailable")

    db_path = str(job_store.db_path)
    with get_connection(db_path) as connection:
        session = connection.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        public_id = str(uuid4())
        connection.execute(
            "INSERT OR IGNORE INTO graph_shares (public_id, session_id) VALUES (?, ?)",
            (public_id, session_id),
        )
        existing = connection.execute(
            "SELECT public_id FROM graph_shares WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        connection.commit()

    resolved_public_id = str(existing["public_id"] if existing else public_id)
    return {"public_id": resolved_public_id, "public_url": f"/share/{resolved_public_id}"}


@share_router.get("/{public_id}")
def get_shared_graph(public_id: str, request: Request) -> dict:
    job_store = getattr(request.app.state, "job_store", None)
    if job_store is None:
        raise HTTPException(status_code=500, detail="Job store is unavailable")

    with get_connection(str(job_store.db_path)) as connection:
        row = connection.execute(
            "SELECT session_id FROM graph_shares WHERE public_id = ?",
            (public_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Shared graph not found")

    try:
        return _load_graph(str(job_store.db_path), session_id=str(row["session_id"]))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
