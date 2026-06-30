from app.engines.reasoning import NodeGenerator
from app.models.schemas import UserIntent
import json
from datetime import datetime

OUT = "raw_logs/09_metrics_and_audits/2026-05-29Treasoning_smoke_run.log"

def main():
    gen = NodeGenerator()
    # minimal deterministic node output without citations
    raw_node = json.dumps({
        "id": "smoke-node",
        "title": "Smoke",
        "summary": "Projected AI job market growth in 2026.",
        "description": "Projected AI job market growth in 2026.",
        "time_step": 0,
        "created_by_engine": "smoke",
        "alternatives": [],
        "risks": [],
        "source_citations": [],
        "confidence_score": 0.5,
        "speculative": False,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })

    # monkeypatch _chat_completion to return our raw_node
    gen._chat_completion = lambda s, u: raw_node

    evidence = [{"id": "c1", "dense_similarity": 0.83, "content": "AI job market trends 2026 forecast", "source_url": "https://bls.gov/report"}]
    node, raw = gen.generate_node(user_intent={"id": "ix", "original_prompt": "smoke", "domain": "general", "horizon_months": 3, "risk_tolerance": 5, "constraints": [], "personal_context": "ctx", "clarified_entities": [], "ambiguities_remaining": []}, evidence_chunks=evidence, time_step=0, max_retries=0)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("RAW_COMPLETION:\n")
        fh.write(raw + "\n\n")
        fh.write("GENERATED_NODE:\n")
        fh.write(json.dumps(node, indent=2, ensure_ascii=False))

    print("Wrote smoke run to", OUT)

if __name__ == "__main__":
    main()
