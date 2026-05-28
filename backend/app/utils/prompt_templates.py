from __future__ import annotations

from typing import Any
import json

from ..models.schemas import DecisionNode


def _decision_node_schema_str() -> str:
    try:
        schema = DecisionNode.model_json_schema()
        return json.dumps(schema, ensure_ascii=False, indent=2)
    except Exception:
        return "{\n  \"DecisionNode\": \"<schema unavailable>\"\n}"


def build_system_prompt(*, user_intent_json: str, evidence_chunks_json: str, parent_summary: str | None = None, persona_prompt: str | None = None, time_step: int = 0) -> str:
    schema_text = _decision_node_schema_str()
    parts = [
        "You are a strict, safety-first decision synthesis engine.",
        "Produce exactly one JSON object that validates against the DecisionNode schema below.",
        "Output must be JSON only, without any surrounding text or commentary.",
        "All factual claims must include a citation token in the form [Source: <cache id> | <url>].",
        "If a claim cannot be grounded to a source, set speculative: true and use [Source: speculative].",
        "Risks must be non-empty. If the node proposes material actions, include at least one risk with severity High or Critical.",
        "Do not include any fields outside the DecisionNode schema. Use ISO8601 for created_at.",
        "\nDecisionNode JSON schema:\n",
        schema_text,
        "\nInputs:\n",
        f"time_step: {time_step}",
        f"user_intent: {user_intent_json}",
        f"evidence_chunks: {evidence_chunks_json}",
    ]

    if parent_summary:
        parts.append(f"parent_summary: {parent_summary}")
    if persona_prompt:
        parts.append(f"persona: {persona_prompt}")

    parts.append("\nReturn a single JSON object that conforms exactly to the schema.")
    return "\n".join(parts)
SYSTEM_PROMPT = ""
