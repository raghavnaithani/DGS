from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ExpertiseLevel = Literal["beginner", "intermediate", "expert"]
SubscriptionTier = Literal["free", "pro"]


# ---------------------------------------------------------------------------
# Full profile model (returned by GET /v1/profile)
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """
    Complete user profile as stored in the user_profiles table.
    Returned by GET and POST /v1/profile endpoints.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    email: str
    display_name: str | None = None
    expertise_level: ExpertiseLevel = "intermediate"
    risk_tolerance: int = Field(default=5, ge=1, le=10)
    values: list[str] = Field(default_factory=list)
    life_situation: str = ""
    decision_patterns: dict = Field(default_factory=dict)
    onboarding_complete: bool = False
    subscription_tier: SubscriptionTier = "free"
    stripe_customer_id: str | None = None
    graphs_this_month: int = 0


# ---------------------------------------------------------------------------
# Create model (onboarding wizard submit — POST /v1/profile)
# ---------------------------------------------------------------------------

class UserProfileCreate(BaseModel):
    """
    Payload for the onboarding wizard submit.
    All four fields are required — the wizard must complete all steps.
    """

    model_config = ConfigDict(extra="forbid")

    expertise_level: ExpertiseLevel
    risk_tolerance: int = Field(ge=1, le=10)
    values: list[str] = Field(default_factory=list, max_length=5)
    life_situation: str = Field(default="", max_length=1000)
    display_name: str | None = Field(default=None, max_length=100)


# ---------------------------------------------------------------------------
# Update model (PATCH /v1/profile — all fields optional)
# ---------------------------------------------------------------------------

class UserProfileUpdate(BaseModel):
    """
    Partial update payload. Only provided fields are changed.
    System-managed fields (id, email, subscription_tier, graphs_this_month,
    stripe_customer_id, decision_patterns, onboarding_complete) are excluded
    and cannot be set through this endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    expertise_level: ExpertiseLevel | None = None
    risk_tolerance: int | None = Field(default=None, ge=1, le=10)
    values: list[str] | None = Field(default=None, max_length=5)
    life_situation: str | None = Field(default=None, max_length=1000)
    display_name: str | None = Field(default=None, max_length=100)
