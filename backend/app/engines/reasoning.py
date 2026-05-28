from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from ..config import settings
from ..models import DecisionNode, UserIntent
from ..utils.prompt_templates import build_system_prompt


class NodeGenerationError(Exception):
    pass


class NodeGenerator:
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def _chat_completion(self, system_prompt: str, user_message: str) -> str:
        api_key = settings.groq_api_key.strip()
        if not api_key:
            raise NodeGenerationError("GROQ_API_KEY is missing")
        payload = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": float(settings.simulation_temperature),
            "max_tokens": int(settings.simulation_max_tokens),
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.api_url, headers=headers, json=payload)
        except Exception as exc:
            raise NodeGenerationError(f"Groq API request failed: {exc}")
        if resp.status_code >= 400:
            raise NodeGenerationError(f"Groq API error: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise NodeGenerationError("Groq returned invalid response") from exc

    def _extract_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise NodeGenerationError("No JSON object found in model output")
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                raise NodeGenerationError("Failed to parse JSON from model output") from exc

    def _citation_audit(self, node_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        # Returns possibly modified node and boolean flagged_speculative
        flagged = False
        text_fields = []
        for key in ("summary", "description"):
            if key in node_dict and isinstance(node_dict[key], str):
                text_fields.append(node_dict[key])
        # gather alternatives and risks
        for alt in node_dict.get("alternatives", []):
            if isinstance(alt.get("description"), str):
                text_fields.append(alt["description"])
        for risk in node_dict.get("risks", []):
            if isinstance(risk.get("description"), str):
                text_fields.append(risk["description"])

        # simple heuristic: sentences containing numbers or proper nouns require citation
        missing = False
        for text in text_fields:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                if re.search(r"\d{1,4}|\b[A-Z][a-z]{2,}\b", sent):
                    if "[Source:" not in sent:
                        missing = True
                        # append speculative marker
                        text = text.replace(sent, sent + " [Source: speculative]")
                        flagged = True

        if flagged:
            node_dict["speculative"] = True
        return node_dict, flagged

    def _compute_confidence(self, evidence: list[dict[str, Any]], retries: int, citations_present: bool) -> float:
        # evidence: list of chunks with 'dense_similarity' and optional 'verification_status' or trust
        max_sim = 0.0
        trust_scores = []
        for ch in evidence:
            sim = float(ch.get("dense_similarity") or 0.0)
            if sim > max_sim:
                max_sim = sim
            ver = ch.get("verification_status")
            if ver == "verified":
                trust_scores.append(0.9)
            elif ver == "unverified":
                trust_scores.append(0.7)
            else:
                trust_scores.append(0.5)
        trust = (sum(trust_scores) / len(trust_scores)) if trust_scores else 0.8
        retries_penalty = float(retries) * float(settings.simulation_retry_penalty)
        citation_bonus = 0.05 if citations_present else -0.05
        raw = max_sim * trust - retries_penalty + citation_bonus
        score = max(0.0, min(1.0, raw))
        return score

    def generate_node(self, *, user_intent: UserIntent | dict[str, Any], evidence_chunks: list[dict[str, Any]], parent_summary: str | None = None, persona_prompt: str | None = None, time_step: int = 0, max_retries: int | None = None) -> tuple[dict[str, Any], str]:
        max_retries = int(max_retries or settings.simulation_max_retries)
        user_intent_json = json.dumps(user_intent if isinstance(user_intent, dict) else user_intent.model_dump(), ensure_ascii=False)
        evidence_json = json.dumps(evidence_chunks, ensure_ascii=False)
        system_prompt = build_system_prompt(user_intent_json=user_intent_json, evidence_chunks_json=evidence_json, parent_summary=parent_summary, persona_prompt=persona_prompt, time_step=time_step)
        user_message = "Generate one DecisionNode JSON object conforming to the schema."

        last_error = None
        for attempt in range(0, max_retries + 1):
            try:
                try:
                    content = self._chat_completion(system_prompt, user_message)
                except Exception as exc:
                    raise NodeGenerationError(str(exc))
                raw_completion = content
                node_dict = self._extract_json(content)
                # Citation auditor
                node_dict, flagged = self._citation_audit(node_dict)
                # Pydantic validation
                try:
                    validated = DecisionNode.model_validate(node_dict)
                except ValidationError as ve:
                    last_error = ve
                    if attempt < max_retries:
                        # re-prompt with error
                        system_prompt += "\nPrevious output failed validation: " + str(ve)
                        continue
                    else:
                        # final fallback
                        node_dict.setdefault("speculative", True)
                        node_dict.setdefault("source_citations", [])
                        node_dict.setdefault("confidence_score", 0.0)
                        return node_dict, raw_completion
                # compute confidence
                confidence = self._compute_confidence(evidence_chunks, retries=attempt, citations_present=not flagged)
                validated_dict = validated.model_dump()
                validated_dict["confidence_score"] = float(confidence)
                # ensure speculative set
                if flagged:
                    validated_dict["speculative"] = True
                return validated_dict, raw_completion
            except NodeGenerationError as exc:
                last_error = exc
                if attempt < max_retries:
                    continue
                raise NodeGenerationError(f"Node generation failed after retries: {exc}")
        raise NodeGenerationError(f"Node generation failed: {last_error}")
def generate_node(context):
    return {}
