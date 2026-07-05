from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..auth.middleware import AuthenticatedUser, get_current_user
from ..database.connection import get_connection

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response / request models
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    """Lightweight session object returned in the history list — no node content."""
    id: str
    title: str
    domain: str
    horizon_months: int
    node_count: int
    created_at: str
    updated_at: str


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_path(request: Request) -> str:
    return str(request.app.state.job_store.db_path)


def _assert_owns_session(conn, session_id: str, user_id: str) -> None:
    """
    Verify the session exists, belongs to this user, and is not soft-deleted.
    Raises HTTP 404 on any mismatch (intentionally does not reveal whether
    the session exists for another user).
    """
    row = conn.execute(
        "SELECT id FROM sessions WHERE id = ? AND user_id = ? AND status = 'active'",
        (session_id, user_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/sessions",
    response_model=list[SessionSummary],
    summary="List the current user's decision graph sessions",
)
def list_sessions(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
    domain: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[SessionSummary]:
    """
    Returns paginated list of active sessions for the authenticated user,
    ordered by most recently updated first.

    Query params:
    - domain: optional filter (e.g. "career", "finance")
    - limit: default 20, max 100
    - offset: for pagination
    """
    limit = min(limit, 100)
    db_path = _db_path(request)

    with get_connection(db_path) as conn:
        query = """
            SELECT id, title, domain, horizon_months, node_count, created_at, updated_at
            FROM sessions
            WHERE user_id = ? AND status = 'active'
        """
        params: list = [user.user_id]

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

    return [
        SessionSummary(
            id=row["id"],
            title=row["title"] or "",
            domain=row["domain"] or "",
            horizon_months=int(row["horizon_months"] or 3),
            node_count=int(row["node_count"] or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


@router.delete(
    "/sessions/{session_id}",
    status_code=200,
    summary="Soft-delete a session (removes from history, breaks share links)",
)
def delete_session(
    session_id: str,
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """
    Soft-deletes the session by setting status = 'deleted'.
    The underlying nodes and edges are NOT deleted from SQLite — they are
    simply hidden from all list and graph endpoints.

    Returns 404 if the session doesn't exist or belongs to another user.
    """
    db_path = _db_path(request)
    with get_connection(db_path) as conn:
        _assert_owns_session(conn, session_id, user.user_id)
        conn.execute(
            "UPDATE sessions SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        logger.info("Soft-deleted session %s for user %s", session_id[:12], user.user_id[:8])
    return {"status": "deleted"}


@router.patch(
    "/sessions/{session_id}",
    response_model=dict,
    summary="Rename a session",
)
def rename_session(
    session_id: str,
    payload: RenameRequest,
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """
    Updates the session title.
    Returns 404 if the session doesn't exist or belongs to another user.
    """
    db_path = _db_path(request)
    with get_connection(db_path) as conn:
        _assert_owns_session(conn, session_id, user.user_id)
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (payload.title.strip(), session_id),
        )
        conn.commit()

    return {"id": session_id, "title": payload.title.strip()}
