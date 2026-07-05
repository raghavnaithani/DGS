from __future__ import annotations

from typing import Any
import json


def build_system_prompt(*, user_intent_json: str, evidence_chunks_json: str, parent_summary: str | None = None, persona_prompt: str | None = None, time_step: int = 0, parent_choice_description: str | None = None, horizon_months: int = 3, user_profile: dict | None = None) -> str:
    parts = [
        "Return one compact DecisionNode JSON object only.",
        "Use provided evidence for factual claims; cite supported claims as [Source: <url>].",
        "If evidence is weak, keep claims practical/general and speculative=true.",
        "Strict limits: summary <= 15 words. alternatives: 1-4 (ensure organic branching: some nodes should have 1, some 2, some 3-4), description <= 15 words. risks: 1-2 max, mitigation <= 10 words.",
        "description: Provide 3-4 concrete, numbered steps INSIDE THIS STRING. For each step, include the specific tool/platform/resource, estimated cost/time, and expected outcome.",
        "CRITICAL: Do NOT invent new JSON keys (e.g. do NOT create a 'steps' array). Output ONLY the keys defined in the schema.",
    ]
    
    if user_profile:
        expertise = user_profile.get("expertise_level", "intermediate")
        expertise_instruction = "use plain language, avoid jargon" if expertise == "beginner" else "use domain-specific terminology" if expertise == "expert" else "use balanced language"
        risk = int(user_profile.get("risk_tolerance") or 5)
        risk_instruction = "strongly emphasise mitigations, safety nets, conservative options" if risk <= 3 else "lead with upside, bold moves, accept calculated risk" if risk >= 8 else "balanced"
        values_list = user_profile.get("values", [])
        values_str = ", ".join(values_list)
        life_situation = str(user_profile.get("life_situation") or "")[:200]
        
        profile_block = (
            f"User Profile Context:\n"
            f"- Expertise: {expertise} ({expertise_instruction})\n"
            f"- Risk Tolerance: {risk}/10 ({risk_instruction})\n"
        )
        if values_str:
            profile_block += f"- Core Values: {values_str} (branch alternatives must reflect these values, e.g. 'Financial growth' means at least one alternative per node should have a financial/ROI angle)\n"
        if life_situation:
            profile_block += f"- Life Situation: {life_situation} (respect constraints implied by the situation)\n"
        parts.insert(0, profile_block)
    
    try:
        intent_dict = json.loads(user_intent_json)
        constraints = intent_dict.get("constraints_json", "None")
    except Exception:
        constraints = "None"
        
    parts.append(f"All suggested actions MUST respect the user's constraints: {constraints}. Do NOT propose steps that exceed their budget or availability.")

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
- Ensure organic branching: some nodes should have 1 outgoing edge, some 2, some 3-4.
Return JSON only."""
