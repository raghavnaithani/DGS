import asyncio
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app.engines.reasoning import NodeGenerator
from backend.app.utils.prompt_templates import build_system_prompt

def test_citations():
    evidence_chunks_json = json.dumps([
        {
            "content": "According to Reddit user u/startupguru, the best way to start an AI company is to build an MVP using Streamlit and deploy it on Heroku. Cost is $7/month.",
            "url": "https://www.reddit.com/r/Entrepreneur/comments/1234/how_to_start_ai/"
        },
        {
            "content": "Salesforce reports that 80% of successful AI startups focus on narrow B2B niches like legal document parsing rather than general chatbots.",
            "url": "https://www.salesforce.com/ai-startup-report"
        }
    ])
    
    intent_json = json.dumps({
        "prompt": "I want to launch a high-end tech startup in the AI space and secure venture capital.",
        "constraints_json": "Budget is $15k, want steady progress."
    })
    
    user_profile = {
        "expertise_level": "intermediate",
        "risk_tolerance": 5,
        "values": ["Sustainable growth", "Work-life balance"],
        "life_situation": "Testing Phase 1 with $15k budget, moderate risk-taker"
    }

    system_prompt = build_system_prompt(
        user_intent_json=intent_json,
        evidence_chunks_json=evidence_chunks_json,
        user_profile=user_profile,
        horizon_months=3,
        time_step=1
    )
    
    generator = NodeGenerator()
    print("Sending prompt to LLM to verify citations...\n")
    try:
        intent_dict = {
            "prompt": "I want to launch a high-end tech startup in the AI space and secure venture capital.",
            "constraints_json": "Budget is $15k, want steady progress."
        }
        result, _ = generator.generate_node(
            user_intent=intent_dict,
            evidence_chunks=[
                {"content": "According to Reddit user u/startupguru, the best way to start an AI company is to build an MVP using Streamlit and deploy it on Heroku. Cost is $7/month.", "url": "https://www.reddit.com/r/Entrepreneur/comments/1234/how_to_start_ai/"},
                {"content": "Salesforce reports that 80% of successful AI startups focus on narrow B2B niches like legal document parsing rather than general chatbots.", "url": "https://www.salesforce.com/ai-startup-report"}
            ],
            user_profile=user_profile,
            time_step=1,
            max_retries=1
        )
        print("--- Node Generated Successfully ---\n")
        print(f"Description:\n{result.get('description', '')}\n")
        print(f"Source Citations Array: {result.get('source_citations', [])}\n")
        print(f"Speculative: {result.get('speculative', False)}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_citations()
