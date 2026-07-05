from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth.middleware import AuthenticatedUser, get_current_user
from ..database.connection import get_connection
from ..models.user import UserProfile, UserProfileCreate, UserProfileUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(request: Request) -> str:
    return str(request.app.state.job_store.db_path)


def _load_profile(db_path: str, user_id: str) -> UserProfile:
    """
    Load a user profile from SQLite by user ID.
    Raises HTTP 404 if not found (before onboarding is complete).
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Please complete onboarding.",
        )

    return UserProfile(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        expertise_level=row["expertise_level"],
        risk_tolerance=int(row["risk_tolerance"]),
        values=json.loads(row["values_json"] or "[]"),
        life_situation=row["life_situation"] or "",
        decision_patterns=json.loads(row["decision_patterns_json"] or "{}"),
        onboarding_complete=bool(row["onboarding_complete"]),
        subscription_tier=row["subscription_tier"],
        stripe_customer_id=row["stripe_customer_id"],
        graphs_this_month=int(row["graphs_this_month"]),
    )


def _upsert_profile(
    db_path: str,
    user: AuthenticatedUser,
    data: dict,
) -> UserProfile:
    """
    INSERT OR REPLACE the profile row, then reload and return it.
    Always sets onboarding_complete = 1 (calling this endpoint means
    the wizard was submitted).
    """
    now = _now_iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_profiles
                (id, email, display_name, expertise_level, risk_tolerance,
                 values_json, life_situation, onboarding_complete, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name        = excluded.display_name,
                expertise_level     = excluded.expertise_level,
                risk_tolerance      = excluded.risk_tolerance,
                values_json         = excluded.values_json,
                life_situation      = excluded.life_situation,
                onboarding_complete = 1,
                updated_at          = excluded.updated_at
            """,
            (
                user.user_id,
                user.email,
                data.get("display_name"),
                data.get("expertise_level", "intermediate"),
                int(data.get("risk_tolerance", 5)),
                json.dumps(data.get("values", [])),
                data.get("life_situation", ""),
                now,
            ),
        )
        conn.commit()

    return _load_profile(db_path, user.user_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/profile",
    response_model=UserProfile,
    summary="Get the current user's profile",
)
def get_profile(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> UserProfile:
    """
    Returns the authenticated user's profile.
    Returns HTTP 404 if onboarding has not been completed yet.
    """
    return _load_profile(_db_path(request), user.user_id)


@router.post(
    "/profile",
    response_model=UserProfile,
    summary="Create or replace the current user's profile (onboarding submit)",
)
def create_or_update_profile(
    payload: UserProfileCreate,
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> UserProfile:
    """
    Submits the onboarding wizard data and persists the full profile.
    Sets onboarding_complete = true.
    Safe to call multiple times — subsequent calls update the profile.
    """
    return _upsert_profile(_db_path(request), user, payload.model_dump())


@router.patch(
    "/profile",
    response_model=UserProfile,
    summary="Partially update the current user's profile",
)
def patch_profile(
    payload: UserProfileUpdate,
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> UserProfile:
    """
    Updates only the provided fields.
    System-managed fields (subscription_tier, graphs_this_month, etc.)
    cannot be modified via this endpoint.
    """
    db_path = _db_path(request)
    current = _load_profile(db_path, user.user_id)

    # Merge patch fields onto current values
    merged = {
        "expertise_level": current.expertise_level,
        "risk_tolerance": current.risk_tolerance,
        "values": current.values,
        "life_situation": current.life_situation,
        "display_name": current.display_name,
    }
    for key, value in payload.model_dump(exclude_none=True).items():
        merged[key] = value

    return _upsert_profile(db_path, user, merged)
