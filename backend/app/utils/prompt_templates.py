from __future__ import annotations

from typing import Any
import json


def build_system_prompt(*, user_intent_json: str, evidence_chunks_json: str, parent_summary: str | None = None, persona_prompt: str | None = None, time_step: int = 0, parent_choice_description: str | None = None, horizon_months: int = 3) -> str:
    parts = [
        "Return one compact DecisionNode JSON object only.",
        "Use provided evidence for factual claims; cite supported claims as [Source: <url>].",
        "If evidence is weak, keep claims practical/general and speculative=true.",
        "Strict limits: summary <= 15 words. description: 40-80 words, containing >=1 concrete action. alternatives: 1-2 max, description <= 15 words. risks: 1-2 max, mitigation <= 10 words.",
    ]

    parts.append("Include watchpoints: 2 concise items if asked, otherwise empty array.")

    parts.extend([
        "",
        "Schema:",
        '{"id": str, "title": str, "summary": str, "description": str, "time_step": int, "created_by_engine": str, "created_at": str,',
        '"alternatives": [{"id": str, "action_type": str, "description": str}],',
        '"risks": [{"id": str, "description": str, "severity": "Low|Medium|High|Critical", "likelihood": "Low|Medium|High", "mitigation_strategy": str}],',
        '"source_citations": [str], "confidence_score": float, "speculative": bool, "watchpoints": [{"id": str, "text": str}]}',
        "",
        f"Context: step={time_step}, intent={user_intent_json}, evidence={evidence_chunks_json}",
    ])

    if parent_choice_description and parent_choice_description != "initial transition":
        parts.append(f"Previously chose: {parent_choice_description}. Build on this, do not repeat.")

    if parent_summary:
        parts.append(f"parent: {parent_summary}")
        parts.append("child branch: explore a distinct action, tradeoff, or consequence.")
    if persona_prompt:
        parts.append(f"persona: {persona_prompt}")

    parts.append("\nReturn JSON only.")
    return "\n".join(parts)


def build_skeleton_system_prompt(*, target_nodes: int, horizon_months: int) -> str:
    return f"""Produce a compact decision graph skeleton JSON.
Schema: {{"nodes": [{{"id": str, "title": str, "summary": str, "time_step": int}}], "edges": [{{"source": str, "target": str, "action_description": str}}]}}
Rules:
- Target exactly {target_nodes} nodes total.
- Horizon: {horizon_months} months. time_steps: 0..{min(horizon_months, 4)}.
- Root is time_step 0.
Return JSON only."""
