from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NON_MATERIAL_ACTION_TYPES = {"continue", "wait", "do nothing"}


class DGSBaseModel(BaseModel):
    # extra="ignore" lets LLM-generated fields (e.g. watchpoints) pass validation
    # without crashing; they are extracted manually before model_dump.
    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)


class Risk(DGSBaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["Low", "Medium", "High", "Critical"]
    likelihood: Literal["Low", "Medium", "High"]
    mitigation_strategy: str | None = None
    citation: str | None = None

    @field_validator("mitigation_strategy", "citation", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class Alternative(DGSBaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    expected_outcome_summary: str | None = None

    @field_validator("expected_outcome_summary", mode="before")
    @classmethod
    def _empty_summary_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class Watchpoint(DGSBaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DecisionNode(DGSBaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    description: str = Field(min_length=1)
    time_step: int = Field(ge=0)
    created_by_engine: str = Field(min_length=1)
    alternatives: list[Alternative] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    source_citations: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    speculative: bool
    created_at: datetime
    watchpoints: list[Watchpoint] = Field(default_factory=list)

    @field_validator("source_citations")
    @classmethod
    def _validate_source_citations(cls, value: list[str]) -> list[str]:
        if any(not citation for citation in value):
            raise ValueError("source_citations cannot contain empty values")
        if len(set(value)) != len(value):
            raise ValueError("source_citations must be unique")
        return value

    @model_validator(mode="after")
    def _validate_risk_rules(self) -> DecisionNode:
        if not self.risks:
            raise ValueError("DecisionNode must include at least one risk")

        material_actions = [
            alternative
            for alternative in self.alternatives
            if alternative.action_type.strip().lower() not in NON_MATERIAL_ACTION_TYPES
        ]
        has_material_action = bool(material_actions)
        has_high_risk = any(risk.severity in {"High", "Critical"} for risk in self.risks)

        if has_material_action and not has_high_risk:
            raise ValueError("Material action alternatives require at least one High or Critical risk")

        return self


class KnowledgeChunk(DGSBaseModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_title: str | None = None
    chunk_index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)
    created_at: datetime
    ttl_days: int = Field(default=30, ge=1)
    verification_status: Literal["verified", "unverified", "failed"]
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("source_title", mode="before")
    @classmethod
    def _empty_title_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class UserIntent(DGSBaseModel):
    id: str = Field(min_length=1)
    original_prompt: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    horizon_months: int = Field(ge=1)
    risk_tolerance: int = Field(ge=0, le=100)
    constraints: list[str] = Field(default_factory=list)
    personal_context: str = Field(min_length=1)
    clarified_entities: list[str] = Field(default_factory=list)
    ambiguities_remaining: list[str] = Field(default_factory=list)

