from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth.middleware import AuthenticatedUser, get_current_user
from ..config import settings
from ..database.connection import get_connection

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_path(request: Request) -> str:
    return str(request.app.state.job_store.db_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/account/usage",
    summary="Get the current user's graph usage for this month",
)
def get_usage(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """
    Returns:
    - graphs_this_month: how many graphs the user has created this calendar month
    - graphs_limit: the maximum allowed (999999 for pro, free_tier_graph_limit for free)
    - subscription_tier: 'free' or 'pro'
    """
    db_path = _db_path(request)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT graphs_this_month, subscription_tier FROM user_profiles WHERE id = ?",
            (user.user_id,),
        ).fetchone()

    if row is None:
        # No profile yet (before onboarding) — return defaults
        return {
            "graphs_this_month": 0,
            "graphs_limit": settings.free_tier_graph_limit,
            "subscription_tier": "free",
        }

    tier = row["subscription_tier"]
    limit = settings.free_tier_graph_limit if tier == "free" else 999999
    return {
        "graphs_this_month": int(row["graphs_this_month"]),
        "graphs_limit": limit,
        "subscription_tier": tier,
    }


@router.post(
    "/account/checkout",
    summary="Create a Stripe checkout session for the Pro plan",
)
def create_checkout(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """
    Creates a Stripe Checkout Session and returns the URL to redirect the user to.
    Requires STRIPE_SECRET_KEY and STRIPE_PRO_PRICE_ID to be configured.
    """
    if not settings.stripe_secret_key or not settings.stripe_pro_price_id:
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured. Set STRIPE_SECRET_KEY and STRIPE_PRO_PRICE_ID.",
        )

    from ..services.stripe_service import create_checkout_session

    base = settings.frontend_base_url.rstrip("/")
    checkout_url = create_checkout_session(
        user_email=user.email,
        success_url=f"{base}/dashboard?upgraded=true",
        cancel_url=f"{base}/pricing",
    )
    return {"checkout_url": checkout_url}


@router.post(
    "/webhooks/stripe",
    summary="Stripe webhook receiver — upgrades user to Pro on subscription creation",
    include_in_schema=False,  # hide from OpenAPI docs (external endpoint)
)
async def stripe_webhook(request: Request) -> dict:
    """
    Verifies the Stripe webhook signature and processes subscription events.
    Must return HTTP 200 to prevent Stripe retrying.

    Handles:
    - customer.subscription.created → set subscription_tier = 'pro'
    - customer.subscription.deleted → set subscription_tier = 'free' (optional downgrade)
    """
    if not settings.stripe_webhook_secret:
        logger.warning("Stripe webhook received but STRIPE_WEBHOOK_SECRET not set — ignoring.")
        return {"status": "unconfigured"}

    from ..services.stripe_service import handle_webhook

    body = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event = handle_webhook(payload=body, sig_header=sig_header)

    if event is None:
        logger.warning("Stripe webhook signature verification failed.")
        return {"status": "ignored"}

    event_type: str = event.get("type", "")
    logger.info("Stripe webhook received: %s", event_type)

    db_path = _db_path(request)

    if event_type == "customer.subscription.created":
        customer_email = (
            event.get("data", {})
            .get("object", {})
            .get("customer_email", "")
        )
        if customer_email:
            with get_connection(db_path) as conn:
                conn.execute(
                    "UPDATE user_profiles SET subscription_tier = 'pro', updated_at = CURRENT_TIMESTAMP WHERE email = ?",
                    (customer_email,),
                )
                conn.commit()
            logger.info("Upgraded %s to Pro tier via Stripe webhook.", customer_email)

    elif event_type == "customer.subscription.deleted":
        customer_email = (
            event.get("data", {})
            .get("object", {})
            .get("customer_email", "")
        )
        if customer_email:
            with get_connection(db_path) as conn:
                conn.execute(
                    "UPDATE user_profiles SET subscription_tier = 'free', updated_at = CURRENT_TIMESTAMP WHERE email = ?",
                    (customer_email,),
                )
                conn.commit()
            logger.info("Downgraded %s to free tier via Stripe webhook.", customer_email)

    return {"status": "ok"}
