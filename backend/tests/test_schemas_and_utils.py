from __future__ import annotations

import pytest
from app.models import Alternative, DecisionNode, KnowledgeChunk, Risk, UserIntent
from datetime import datetime, timezone
from pydantic import ValidationError

def _valid_risk(severity: str = "Medium") -> Risk:
    return Risk(
        id="risk-1",
        description="Potential scheduling delay",
        severity=severity,
        likelihood="Medium",
        mitigation_strategy="Add buffer time",
        citation="https://example.com/risk",
    )

def _valid_node(**overrides):
    base = {
        "id": "node-1",
        "title": "Approve vendor change",
        "summary": "The system recommends evaluating a vendor switch.",
        "description": "1. Analyze current vendors.\n2. Evaluate a material vendor change.",
        "time_step": 1,
        "created_by_engine": "arbiter",
        "alternatives": [
            Alternative(
                id="alt-1",
                description="Switch to a new vendor",
                action_type="switch_vendor",
                expected_outcome_summary="Lower cost but higher transition risk.",
            )
        ],
        "risks": [_valid_risk("High")],
        "source_citations": ["https://example.com/source"],
        "confidence_score": 0.75,
        "speculative": False,
        "created_at": datetime(2026, 5, 24, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return DecisionNode(**base)

def test_pydantic_schemas():
    # Valid decision node passes validation
    node = _valid_node()
    assert node.id == "node-1"
    assert node.confidence_score == 0.75
    assert node.risks[0].severity == "High"

    # Empty risks list is rejected
    with pytest.raises(ValidationError):
        _valid_node(risks=[])

    # Confidence score outside bounds is rejected
    with pytest.raises(ValidationError):
        _valid_node(confidence_score=-0.1)
    with pytest.raises(ValidationError):
        _valid_node(confidence_score=1.1)

    # Material action without high risk is rejected
    with pytest.raises(ValidationError):
        _valid_node(
            alternatives=[
                Alternative(
                    id="alt-1",
                    description="Launch a new product line",
                    action_type="launch_product",
                )
            ],
            risks=[_valid_risk("Medium")],
        )

    # Knowledge chunk defaults and validation
    chunk = KnowledgeChunk(
        id="chunk-1",
        content="Important context from a source document.",
        source_url="https://example.com/doc",
        source_title="Example Doc",
        chunk_index=0,
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        verification_status="unverified",
    )
    assert chunk.ttl_days == 30
    assert chunk.embedding == [0.1, 0.2, 0.3]

    # User intent validation
    intent = UserIntent(
        id="intent-1",
        original_prompt="Plan a product launch",
        domain="product",
        horizon_months=6,
        risk_tolerance=40,
        constraints=["budget capped"],
        personal_context="Founder-led startup",
        clarified_entities=["launch plan"],
        ambiguities_remaining=["market timing"],
    )
    assert intent.risk_tolerance == 40
