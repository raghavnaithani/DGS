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


def generate_llm_skeleton(
    *,
    user_intent: dict[str, Any] | Any,
    target_nodes: int = 6,
    horizon_months: int = 3,
) -> dict[str, Any]:
    """Call the LLM to produce a structured skeleton of nodes and edges."""
    import json as _json
    import re as _re

    prompt_text = user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt
    domain = (user_intent.get("domain") if isinstance(user_intent, dict) else getattr(user_intent, "domain", "")) or "general"

    system_prompt = (
        f"You are a strategic decision advisor. Generate a JSON decision graph skeleton for a {horizon_months}-month plan.\n"
        f"Output ONLY valid JSON with keys: nodes (array) and edges (array).\n"
        f"Generate exactly {target_nodes} nodes. Each node must have: id (string), title (string, <=10 words), "
        f"summary (string, <=15 words), time_step (0-based integer), alternatives (array of 1-2 objects with id+description <=15 words each), "
        f"risks (array of 1-2 objects with id, severity, likelihood, mitigation <=10 words).\n"
        f"Each edge must have: source (node id), target (node id), action_description (string).\n"
        f"time_step 0 = root node. Subsequent steps branch out organically. No markdown, no prose - JSON only."
    )
    user_message = (
        f"Scenario: {prompt_text}\n"
        f"Domain: {domain}\n"
        f"Horizon: {horizon_months} months\n"
        f"Target nodes: {target_nodes}\n"
        "Return the skeleton JSON now."
    )

    from ..engines.reasoning import NodeGenerator
    generator = NodeGenerator()
    raw = generator._chat_completion(system_prompt, user_message)

    match = _re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("LLM skeleton response contained no JSON object")
    skeleton = _json.loads(match.group(0))

    now = _now_iso()
    for node in skeleton.get("nodes", []):
        node.setdefault("id", str(uuid4()))
        node.setdefault("created_at", now)
        node.setdefault("created_by_engine", "phase5.skeleton_generator")
        node.setdefault("speculative", True)
        node.setdefault("confidence_score", 0.40)
        node.setdefault("source_citations", [])
        node.setdefault("alternatives", [])
        node.setdefault("risks", [])

    return skeleton


def generate_deterministic_skeleton(
    *,
    user_intent: dict[str, Any] | Any,
    horizon_months: int = 3,
) -> dict[str, Any]:
    """Build a simple deterministic skeleton tree as a guaranteed fallback."""
    import random as _random

    prompt_text = user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt

    if horizon_months <= 3:
        target_nodes = 6
    elif horizon_months <= 6:
        target_nodes = 12
    else:
        target_nodes = 14

    now = _now_iso()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    root_id = str(uuid4())
    nodes.append({
        "id": root_id,
        "title": f"Starting Point: {prompt_text[:60]}",
        "summary": f"Initial decision for {prompt_text[:40]}",
        "description": f"This is the starting point for: {prompt_text}",
        "time_step": 0,
        "created_by_engine": "phase5.deterministic_skeleton",
        "created_at": now,
        "speculative": True,
        "confidence_score": 0.40,
        "source_citations": [],
        "alternatives": [
            {"id": "alt_root_1", "action_type": "option", "description": "Research and plan before committing"},
            {"id": "alt_root_2", "action_type": "option", "description": "Start small with a low-risk experiment"},
        ],
        "risks": [
            {"id": "risk-root", "severity": "Medium", "likelihood": "Medium", "description": "Uncertainty in early stages", "mitigation_strategy": "Validate assumptions early"},
        ],
    })

    current_layer = [root_id]
    time_step = 1
    action_labels = [
        "Research and plan before committing",
        "Start small with a low-risk experiment",
        "Build incrementally and validate",
        "Seek expert guidance",
        "Combine multiple approaches",
        "Focus on quick wins",
        "Iterate and adapt based on feedback",
        "Scale what works",
        "Diversify and expand",
        "Optimize and refine processes",
    ]

    label_idx = 0
    while len(nodes) < target_nodes:
        next_layer: list[str] = []
        for parent_id in current_layer:
            if len(nodes) >= target_nodes:
                break
            n_children = _random.randint(1, 2)
            for _ in range(n_children):
                if len(nodes) >= target_nodes:
                    break
                label = action_labels[label_idx % len(action_labels)]
                label_idx += 1
                child_id = str(uuid4())
                nodes.append({
                    "id": child_id,
                    "title": label,
                    "summary": f"{label} for this scenario",
                    "description": f"{label} as part of the {horizon_months}-month plan for: {prompt_text[:60]}",
                    "time_step": time_step,
                    "created_by_engine": "phase5.deterministic_skeleton",
                    "created_at": now,
                    "speculative": True,
                    "confidence_score": 0.40,
                    "source_citations": [],
                    "alternatives": [],
                    "risks": [
                        {"id": f"risk-{child_id[:6]}", "severity": "Medium", "likelihood": "Low", "description": "Execution risk", "mitigation_strategy": "Monitor progress weekly"},
                    ],
                })
                edges.append({
                    "source": parent_id,
                    "target": child_id,
                    "action_description": label,
                })
                next_layer.append(child_id)
        if not next_layer:
            break
        current_layer = next_layer
        time_step += 1

    return {"nodes": nodes, "edges": edges}
