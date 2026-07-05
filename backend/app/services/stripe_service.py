from __future__ import annotations

import logging

from ..config import settings

logger = logging.getLogger(__name__)


def create_checkout_session(
    user_email: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Creates a Stripe Checkout Session for the Pro monthly subscription.
    Returns the Stripe-hosted checkout URL.
    """
    import stripe  # noqa: PLC0415 — lazy import keeps startup fast when Stripe not needed

    stripe.api_key = settings.stripe_secret_key

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="subscription",
        customer_email=user_email,
        line_items=[
            {
                "price": settings.stripe_pro_price_id,
                "quantity": 1,
            }
        ],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"source": "dgs_v0.2"},
    )
    return str(session.url)


def handle_webhook(payload: bytes, sig_header: str) -> dict | None:
    """
    Verifies the Stripe webhook HMAC signature and returns the event dict.
    Returns None if the signature is invalid.
    """
    import stripe  # noqa: PLC0415

    stripe.api_key = settings.stripe_secret_key

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=settings.stripe_webhook_secret,
        )
        return dict(event)
    except stripe.SignatureVerificationError as exc:
        logger.warning("Stripe signature verification failed: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Stripe webhook parsing error: %s", exc)
        return None
