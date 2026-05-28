from __future__ import annotations

import json
import re
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..config import settings
from ..database.connection import get_connection
from ..models import UserIntent

router = APIRouter()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
MOCK_QUESTIONS = [
    {
        "id": "q1",
        "text": "What is the main outcome you want from this decision?",
        "type": "text",
        "hint": "Describe the result in one sentence.",
    },
    {
        "id": "q2",
        "text": "What time horizon should we optimize for?",
        "type": "choice",
        "choices": ["3", "6", "12"],
        "hint": "Pick 3, 6, or 12 months.",
    },
    {
        "id": "q3",
        "text": "How much risk are you comfortable taking?",
        "type": "number",
        "hint": "Use a scale from 1 to 10.",
    },
    {
        "id": "q4",
        "text": "What constraints must we respect?",
        "type": "text",
        "hint": "Budget, time, team, legal, or personal limits.",
    },
    {
        "id": "q5",
        "text": "Which entities or goals should we keep in mind?",
        "type": "text",
        "hint": "People, companies, projects, or priorities.",
    },
]


class ClarifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=2000)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    type: Literal["text", "choice", "number"]
    choices: list[str] | None = None
    hint: str | None = None


class ClarifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[Question]


class BuildIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=2000)
    answers: dict[str, str] = Field(default_factory=dict)


class MockClarifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[Question]


class MockBuildIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1)
    answers: dict[str, str] = Field(default_factory=dict)


def _require_groq_key() -> str:
    api_key = settings.groq_api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is missing")
    return api_key


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=30.0)


def _chat_completion(api_key: str, prompt: str) -> str:
    payload = {
        "model": settings.groq_model or DEFAULT_GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant that returns JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with _http_client() as client:
        response = client.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Groq API error: {response.text}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Groq response was missing content") from exc


def _parse_json_array(content: str) -> list[dict[str, object]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from exc
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as inner_exc:
            raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from inner_exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=502, detail="Groq returned a non-array response")
    return parsed


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from exc
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as inner_exc:
            raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from inner_exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Groq returned a non-object response")
    return parsed


def _normalize_questions(items: list[dict[str, object]]) -> list[Question]:
    normalized: list[Question] = []
    for index, item in enumerate(items[:7], start=1):
        question_data = {
            "id": str(item.get("id") or f"q{index}"),
            "text": str(item.get("text") or item.get("question") or ""),
            "type": item.get("type") or "text",
            "choices": item.get("choices"),
            "hint": item.get("hint"),
        }
        normalized.append(Question.model_validate(question_data))
    if len(normalized) < 5:
        raise HTTPException(status_code=502, detail="Groq returned fewer than 5 questions")
    return normalized


def _user_intent_db_path(request: Request | None) -> str | None:
    if request is None:
        return None
    job_store = getattr(request.app.state, "job_store", None)
    return str(job_store.db_path) if job_store is not None else None


def _persist_user_intent(intent: UserIntent, request: Request | None) -> UserIntent:
    connection = get_connection(_user_intent_db_path(request))
    with connection:
        connection.execute(
            """
            INSERT INTO user_intents (
                id,
                original_prompt,
                domain,
                horizon_months,
                risk_tolerance,
                constraints_json,
                personal_context,
                clarified_entities_json,
                ambiguities_remaining_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent.id,
                intent.original_prompt,
                intent.domain,
                intent.horizon_months,
                intent.risk_tolerance,
                json.dumps(intent.constraints, ensure_ascii=False),
                intent.personal_context,
                json.dumps(intent.clarified_entities, ensure_ascii=False),
                json.dumps(intent.ambiguities_remaining, ensure_ascii=False),
            ),
        )
        connection.commit()
    return intent


@router.post("/clarify", response_model=ClarifyResponse)
def clarify(payload: ClarifyRequest) -> ClarifyResponse:
    api_key = _require_groq_key()
    prompt = (
        "Given the following user prompt, generate 5-7 clarifying questions to resolve ambiguity and gather missing context. "
        'Return the questions as a JSON array of objects with fields: id (string), text (string), type ("text" | "choice" | "number"), '
        "choices (optional string array), hint (optional string).\n"
        f"User prompt: {payload.prompt}"
    )
    content = _chat_completion(api_key, prompt)
    return ClarifyResponse(questions=_normalize_questions(_parse_json_array(content)))


@router.post("/build-intent", response_model=UserIntent)
def build_intent(payload: BuildIntentRequest, request: Request) -> UserIntent:
    if not payload.answers:
        raise HTTPException(status_code=400, detail="answers are required")

    missing_answers = [key for key, value in payload.answers.items() if not str(value).strip()]
    if missing_answers:
        raise HTTPException(status_code=400, detail=f"Missing answers for: {', '.join(missing_answers)}")

    api_key = _require_groq_key()
    prompt = (
        "Based on the user's original prompt and their answers to clarifying questions, construct a UserIntent JSON object. "
        "The object must have the following fields: domain (string), horizon_months (integer 3|6|12), risk_tolerance (integer 1-10), "
        "constraints (array of strings), personal_context (string), clarified_entities (array of strings), ambiguities_remaining (array of strings). "
        "Be concise but thorough.\n"
        f"Original prompt: {payload.prompt}\n"
        f"Answers: {json.dumps(payload.answers, ensure_ascii=False)}"
    )
    content = _chat_completion(api_key, prompt)
    parsed = _parse_json_object(content)
    intent_data = {
        "id": str(uuid4()),
        "original_prompt": payload.prompt,
        **parsed,
    }
    validated_intent = UserIntent.model_validate(intent_data)
    if validated_intent.horizon_months not in {3, 6, 12}:
        raise HTTPException(status_code=502, detail="Groq returned an invalid horizon_months value")
    if not 1 <= validated_intent.risk_tolerance <= 10:
        raise HTTPException(status_code=502, detail="Groq returned an invalid risk_tolerance value")
    return _persist_user_intent(validated_intent, request)


@router.get("/mock-clarify", response_model=MockClarifyResponse)
def mock_clarify(prompt: str = "test") -> MockClarifyResponse:
    questions = [Question.model_validate(item) for item in MOCK_QUESTIONS]
    return MockClarifyResponse(questions=questions)


@router.post("/mock-build-intent", response_model=UserIntent)
def mock_build_intent(payload: MockBuildIntentRequest, request: Request) -> UserIntent:
    intent = {
        "id": str(uuid4()),
        "original_prompt": payload.prompt,
        "domain": "general decision making",
        "horizon_months": 6,
        "risk_tolerance": 5,
        "constraints": ["Uses mock intake flow"],
        "personal_context": "User is exploring the prompt with a development mock response.",
        "clarified_entities": ["goal", "timeline", "constraints"],
        "ambiguities_remaining": ["specific alternatives to compare"],
    }
    return _persist_user_intent(UserIntent.model_validate(intent), request)
