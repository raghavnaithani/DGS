from __future__ import annotations

from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from ..engines.retriever import assemble_evidence
from ..engines.reasoning import NodeGenerator



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _materialize_evidence(retrieval) -> list[dict[str, Any]]:
    evidence_chunks = []
    for item in retrieval.evidence:
        evidence_chunks.append(
            {
                "id": item.id,
                "content": item.content,
                "source_url": item.source_url,
                "source_title": item.source_title,
                "chunk_index": item.chunk_index,
                "dense_similarity": item.dense_similarity or 0.0,
                "bm25_score": item.bm25_score,
                "verification_status": "verified" if (item.source_title and item.source_title.strip()) else "unverified",
            }
        )
    return evidence_chunks


def generate_initial_graph(user_intent: dict[str, Any] | Any, top_k: int = 10) -> dict[str, Any]:
    query_text = user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt
    retrieval = assemble_evidence(query_text, top_k=top_k)
    evidence_chunks = _materialize_evidence(retrieval)

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
        edges.append(
            {
                "source": nodes[i]["id"],
                "target": nodes[i + 1]["id"],
                "action_description": f"time_step_{nodes[i]['time_step']}_to_{nodes[i + 1]['time_step']}",
            }
        )

    return {"nodes": nodes, "edges": edges}


def generate_branch_node(
    *,
    user_intent: dict[str, Any] | Any,
    parent_node: dict[str, Any],
    action_description: str,
    persona_prompt: str | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    query_text = user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt
    branch_query = f"{query_text}\nAction: {action_description}"
    retrieval = assemble_evidence(branch_query, top_k=top_k)
    evidence_chunks = _materialize_evidence(retrieval)

    generator = NodeGenerator()
    parent_summary = " ".join(
        part
        for part in [
            f"Parent node {parent_node.get('id')}: {parent_node.get('summary', '')}",
            f"Chosen action: {action_description}",
        ]
        if part.strip()
    )
    node_dict, _raw = generator.generate_node(
        user_intent=user_intent,
        evidence_chunks=evidence_chunks,
        parent_summary=parent_summary,
        persona_prompt=persona_prompt,
        time_step=int(parent_node.get("time_step", 0)) + 1,
    )
    node_dict.setdefault("id", str(uuid4()))
    node_dict.setdefault("created_by_engine", "phase5.branch_generator")
    node_dict.setdefault("created_at", _now_iso())
    node_dict.setdefault("time_step", int(parent_node.get("time_step", 0)) + 1)
    return {"node": node_dict, "evidence_chunks": evidence_chunks}
