from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from ..config import settings
from ..models import DecisionNode, UserIntent
from ..utils.prompt_templates import build_system_prompt


logger = logging.getLogger(__name__)


_GROUNDING_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "can",
    "could",
    "from",
    "have",
    "into",
    "more",
    "most",
    "must",
    "not",
    "only",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "when",
    "where",
    "which",
    "while",
    "with",
    "your",
}


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

    @staticmethod
    def _sentence_terms(sentence: str) -> set[str]:
        terms = set()
        for token in re.findall(r"[A-Za-z0-9]{4,}", sentence.lower()):
            if token not in _GROUNDING_STOPWORDS:
                terms.add(token)
        return terms

    def _evidence_support_score(self, sentence: str, evidence_chunks: list[dict[str, Any]]) -> float:
        if "[Source:" in sentence:
            return 1.0

        sentence_terms = self._sentence_terms(sentence)
        if not sentence_terms:
            return 0.0

        best_score = 0.0
        for chunk in evidence_chunks:
            content = str(chunk.get("content") or "")
            chunk_terms = self._sentence_terms(content)
            if not chunk_terms:
                continue
            overlap = len(sentence_terms & chunk_terms) / float(len(sentence_terms))
            similarity = float(chunk.get("dense_similarity") or chunk.get("bm25_score") or 0.0)
            score = max(overlap, similarity * 0.75 + overlap * 0.25)
            best_score = max(best_score, score)

        if (
            settings.simulation_use_nli_grounding
            and settings.groq_api_key.strip()
            and 0.35 <= best_score <= 0.75
            and evidence_chunks
        ):
            try:
                nli_score = self._nli_grounding_check(sentence, evidence_chunks[:3])
                best_score = max(best_score, nli_score)
            except Exception as exc:
                logger.debug("Grounding check fallback after NLI failure: %s", exc)

        return best_score

    def _nli_grounding_check(self, sentence: str, evidence_chunks: list[dict[str, Any]]) -> float:
        api_key = settings.groq_api_key.strip()
        if not api_key:
            return 0.0

        evidence_lines = []
        for chunk in evidence_chunks:
            evidence_lines.append(f"- {chunk.get('id', '')}: {str(chunk.get('content') or '')[:600]}")

        payload = {
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only Yes or No. Judge whether the claim is supported by the evidence.",
                },
                {
                    "role": "user",
                    "content": (
                        "Claim: "
                        + sentence
                        + "\nEvidence:\n"
                        + "\n".join(evidence_lines)
                        + "\nDoes the evidence support the claim?"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=20.0) as client:
            response = client.post(self.api_url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise NodeGenerationError(f"Grounding check failed: {response.text}")
        data = response.json()
        content = str(data["choices"][0]["message"]["content"]).strip().lower()
        return 1.0 if content.startswith("y") else 0.0

    def _citation_audit(self, node_dict: dict[str, Any], evidence_chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        flagged = False
        text_keys = ("summary", "description")

        for key in text_keys:
            text = node_dict.get(key)
            if not isinstance(text, str):
                continue
            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
            rewritten: list[str] = []
            for sentence in sentences:
                support_score = self._evidence_support_score(sentence, evidence_chunks)
                requires_source = bool(re.search(r"\d{1,4}|\b[A-Z][a-z]{2,}\b", sentence)) or support_score < 0.4
                if requires_source and "[Source:" not in sentence:
                    sentence = f"{sentence} [Source: speculative]"
                    flagged = True
                rewritten.append(sentence)
            node_dict[key] = " ".join(rewritten)

        for alt in node_dict.get("alternatives", []):
            description = alt.get("description")
            if not isinstance(description, str):
                continue
            support_score = self._evidence_support_score(description, evidence_chunks)
            if support_score < 0.4 and "[Source:" not in description:
                alt["description"] = f"{description} [Source: speculative]"
                flagged = True

        for risk in node_dict.get("risks", []):
            description = risk.get("description")
            if not isinstance(description, str):
                continue
            support_score = self._evidence_support_score(description, evidence_chunks)
            if support_score < 0.4 and "[Source:" not in description:
                risk["description"] = f"{description} [Source: speculative]"
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
                node_dict, flagged = self._citation_audit(node_dict, evidence_chunks)
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
                validated_dict = validated.model_dump(mode="json")
                validated_dict["confidence_score"] = float(confidence)
                # ensure speculative set
                if flagged:
                    validated_dict["speculative"] = True
                logger.info("Generated DecisionNode id=%s time_step=%s confidence=%.3f speculative=%s", validated_dict.get("id"), validated_dict.get("time_step"), confidence, validated_dict.get("speculative"))
                return validated_dict, raw_completion
            except NodeGenerationError as exc:
                last_error = exc
                if attempt < max_retries:
                    continue
                raise NodeGenerationError(f"Node generation failed after retries: {exc}")
        raise NodeGenerationError(f"Node generation failed: {last_error}")
def generate_node(context):
    return {}
