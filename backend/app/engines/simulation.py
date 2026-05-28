from __future__ import annotations

from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from ..engines.retriever import assemble_evidence
from ..engines.reasoning import NodeGenerator
from ..models.schemas import DecisionNode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_initial_graph(user_intent: dict[str, Any] | Any, top_k: int = 10) -> dict[str, Any]:
    # assemble evidence
    query_text = user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt
    retrieval = assemble_evidence(query_text, top_k=top_k)
    # materialize evidence chunks as simple dicts
    evidence_chunks = []
    for item in retrieval.evidence:
        evidence_chunks.append({
            "id": item.id,
            "content": item.content,
            "source_url": item.source_url,
            "source_title": item.source_title,
            "chunk_index": item.chunk_index,
            "dense_similarity": item.dense_similarity or 0.0,
            "verification_status": "verified" if (item.source_title and item.source_title.strip()) else "unverified",
        })

    generator = NodeGenerator()
    nodes = []
    parent_summary = None
    for t in range(0, 3):
        node_id = str(uuid4())
        node_dict, raw = generator.generate_node(user_intent=user_intent, evidence_chunks=evidence_chunks, parent_summary=parent_summary, time_step=t)
        # ensure id and timestamps
        node_dict.setdefault("id", node_id)
        node_dict.setdefault("created_by_engine", "phase5.node_generator")
        node_dict.setdefault("created_at", _now_iso())
        node_dict.setdefault("time_step", t)
        nodes.append(node_dict)
        parent_summary = node_dict.get("summary")

    edges = []
    for i in range(len(nodes) - 1):
        edges.append({"from": nodes[i]["id"], "to": nodes[i + 1]["id"]})

    return {"nodes": nodes, "edges": edges}
def simulate():
    return {}
