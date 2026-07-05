import pytest
from app.utils.prompt_templates import build_system_prompt

def test_profile_in_system_prompt():
    profile = {
        "expertise_level": "expert",
        "risk_tolerance": 9,
        "values": ["Innovation"],
        "life_situation": "Single"
    }
    
    prompt = build_system_prompt(
        user_intent_json='{"original_prompt": "test"}',
        evidence_chunks_json="[]",
        user_profile=profile
    )
    
    assert "User Profile Context:" in prompt
    assert "Expertise: expert" in prompt
    assert "Risk Tolerance: 9/10" in prompt
    assert "Core Values: Innovation" in prompt
    assert "Life Situation: Single" in prompt

def test_beginner_prompt_contains_plain_language_instruction():
    profile = {
        "expertise_level": "beginner",
        "risk_tolerance": 5,
        "values": [],
        "life_situation": ""
    }
    
    prompt = build_system_prompt(
        user_intent_json='{"original_prompt": "test"}',
        evidence_chunks_json="[]",
        user_profile=profile
    )
    
    assert "use plain language, avoid jargon" in prompt
