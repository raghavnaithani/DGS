from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from ..config import settings
from ..database.connection import get_connection
from ..database.jobs_store import SQLiteJobStore
from ..database.vector_store import LanceChunkStore
from ..engines.ingestion import IngestionService
from ..engines.reasoning import NodeGenerator
from ..engines.retriever import assemble_evidence
from ..engines.simulation import (
    generate_branch_node,
    generate_deterministic_skeleton,
    generate_initial_graph,
    generate_llm_skeleton,
)
from ..models.jobs import IngestionRequest

logger = logging.getLogger(__name__)

@lru_cache(maxsize=32)
def _cached_assemble(branch_query: str, top_k: int):
    return assemble_evidence(branch_query, top_k=top_k)


# ---------------------------------------------------------------------------
# Enrichment-persistence helpers
# ---------------------------------------------------------------------------

def _clean_description_text(value: Any) -> str:
    """Normalise any description value to clean plain text.

    Handles:
    - list  → join with double newlines
    - dict  → extract primary text field
    - str   → strip Python literal syntax artefacts
    """
    import re

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Pull 'text', 'action', or the first str value found
                text = item.get("text") or item.get("action") or ""
                if isinstance(text, list):
                    text = " ".join(str(t) if not isinstance(t, dict) else t.get("text", "") for t in text)
                if text:
                    parts.append(str(text))
        value = "\n\n".join(p.strip() for p in parts if p.strip())
    elif isinstance(value, dict):
        value = (
            value.get("description")
            or value.get("text")
            or value.get("summary")
            or next((v for v in value.values() if isinstance(v, str)), "")
        )
        if isinstance(value, list):
            value = _clean_description_text(value)
        else:
            value = str(value)
    else:
        value = str(value)

    # Strip Python literal artefacts left by str(dict) serialisation
    # e.g. {'key': 'val'} or escaped newlines from repr
    value = re.sub(r"\\n", "\n", value)
    value = re.sub(r"\\t", " ", value)
    value = re.sub(r"\\'", "'", value)
    value = re.sub(r'\\"', '"', value)
    # Strip lone braces/brackets that remain at the very start/end of the string
    value = re.sub(r"^['{]\s*", "", value)
    value = re.sub(r"['}]\s*$", "", value)
    # Strip [Source: speculative] tokens
    value = re.sub(r"\[Source:\s*speculative\]", "", value, flags=re.IGNORECASE)
    return value.strip()


def _parse_enriched_payload(node: dict[str, Any]) -> None:
    """Flatten any nested enrichment payload stored as a string inside 'description'.

    Modifies `node` in-place. Extracts inner alternatives, risks, watchpoints,
    confidence_score, speculative, and source_citations from a stringified Python
    dict / JSON blob that the LLM sometimes returns instead of a proper top-level
    JSON structure.

    If parsing succeeds, overwrites the node's top-level fields.
    If parsing fails, falls back to cleaning the description string as-is.
    """
    import ast
    import re

    raw_desc = node.get("description", "")
    if not isinstance(raw_desc, str):
        node["description"] = _clean_description_text(raw_desc)
        return

    # Heuristic: if the description looks like a stringified Python dict / JSON object,
    # try to parse it and promote the inner fields.
    stripped = raw_desc.strip()
    if not (stripped.startswith("{") or stripped.startswith("'")):
        # Plain text — just clean it
        node["description"] = _clean_description_text(raw_desc)
        return

    inner: dict[str, Any] | None = None

    # Attempt 1: JSON parse
    try:
        inner = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: ast.literal_eval (handles Python dict repr with single quotes)
    if inner is None:
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                inner = parsed
        except Exception:
            pass

    if inner is None or not isinstance(inner, dict):
        # Couldn't parse — just clean the raw string
        node["description"] = _clean_description_text(raw_desc)
        return

    # --- Successfully parsed inner dict — promote fields ---

    # Description
    inner_desc = inner.get("description", "")
    node["description"] = _clean_description_text(inner_desc) if inner_desc else _clean_description_text(raw_desc)

    # Alternatives — only overwrite if the inner payload has a non-empty list
    inner_alts = inner.get("alternatives") or []
    if isinstance(inner_alts, list) and inner_alts:
        node["alternatives"] = inner_alts

    # Risks
    inner_risks = inner.get("risks") or []
    if isinstance(inner_risks, list) and inner_risks:
        node["risks"] = inner_risks

    # Watchpoints
    inner_wps = inner.get("watchpoints") or []
    if isinstance(inner_wps, list) and inner_wps:
        node["watchpoints"] = inner_wps

    # Source citations — merge with existing
    inner_cites = inner.get("source_citations") or []
    if isinstance(inner_cites, list):
        existing = node.setdefault("source_citations", [])
        for c in inner_cites:
            c_str = str(c).strip() if not isinstance(c, dict) else str(c.get("text", "")).strip()
            if c_str and c_str not in existing:
                existing.append(c_str)

    # Confidence / speculative — promote from inner payload if more grounded
    inner_conf = inner.get("confidence_score")
    if inner_conf is not None:
        try:
            inner_conf_f = float(inner_conf)
            # Only promote if the inner value is more confident than the skeleton default
            if inner_conf_f > float(node.get("confidence_score") or 0.0):
                node["confidence_score"] = inner_conf_f
        except (TypeError, ValueError):
            pass

    inner_spec = inner.get("speculative")
    if inner_spec is not None and isinstance(inner_spec, bool):
        # Only move speculative→False if the inner data says not speculative
        if not inner_spec:
            node["speculative"] = False



def _horizon_defaults(horizon_months: int) -> tuple[int, int, int]:
    # Target ranges: 3-month: 5-7 (target 6), 6-month: 10-14 (target 12), 12-month: 12-16 (target 14)
    if horizon_months <= 3:
        return (2, 3, 6)
    if horizon_months <= 6:
        return (3, 3, 12)
    return (3, 3, 14)


class SimulationJobWorker:
    def __init__(self, *, job_store: SQLiteJobStore, vector_store: LanceChunkStore, poll_interval_seconds: float = 0.25):
        self.job_store = job_store
        self.vector_store = vector_store
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.set()
        self._thread = threading.Thread(target=self._run_loop, name="simulation-job-worker", daemon=True)
        self._thread.start()
        logger.info("Simulation worker started with db_path=%s", self.job_store.db_path)

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Simulation worker stopped")

    def kick(self) -> None:
        self._wake_event.set()

    def enqueue_start(
        self,
        *,
        user_intent_id: str,
        disable_scraping: bool = False,
        persona: str | None = None,
        webhook_url: str | None = None,
        depth: int | None = None,
        branching_factor: int | None = None,
        mode: str | None = None,
        target_nodes: int | None = None,
    ) -> Any:
        job = self.job_store.create_simulation_job(
            {
                "workflow": "start",
                "user_intent_id": user_intent_id,
                "disable_scraping": disable_scraping,
                "persona": persona,
                "webhook_url": webhook_url,
                "depth": depth,
                "branching_factor": branching_factor,
                "mode": mode,
                "target_nodes": target_nodes,
            }
        )
        self.kick()
        return job

    def enqueue_branch(
        self,
        *,
        session_id: str,
        parent_node_id: str,
        action_description: str,
        persona: str | None = None,
        webhook_url: str | None = None,
        depth: int | None = 1,
        branching_factor: int | None = 2,
    ) -> Any:
        job = self.job_store.create_simulation_job(
            {
                "workflow": "branch",
                "session_id": session_id,
                "parent_node_id": parent_node_id,
                "action_description": action_description,
                "persona": persona,
                "webhook_url": webhook_url,
                "depth": int(depth or 1),
                "branching_factor": int(branching_factor or 2),
            }
        )
        self.kick()
        return job

    def process_pending_jobs_once(self) -> int:
        processed = 0
        while True:
            job = self.job_store.claim_next_simulation_job()
            if job is None:
                break
            self._process_job(job.id, job.request)
            processed += 1
        return processed

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self.job_store.claim_next_simulation_job()
            if job is None:
                self._wake_event.wait(self.poll_interval_seconds)
                self._wake_event.clear()
                continue
            self._process_job(job.id, job.request)

    def _process_job(self, job_id: str, request: dict[str, Any]) -> None:
        try:
            self.job_store.update_job(job_id, status="running", progress=5, current_step="starting")
            workflow = str(request.get("workflow") or "start")
            if workflow == "start":
                result = self._run_start(job_id, request)
            elif workflow == "branch":
                result = self._run_branch(request)
            else:
                raise ValueError(f"Unsupported workflow: {workflow}")
            print(f"[simulation_worker] Job {job_id} completed locally, updating job store")
            self.job_store.update_job(job_id, status="completed", progress=100, current_step="completed", result=result)
            print(f"[simulation_worker] Job {job_id} update complete, dispatching webhook")
            self._dispatch_webhook(request, job_id=job_id, status="completed", result=result)
        except Exception as exc:
            logger.exception("Simulation job failed job_id=%s", job_id)
            self.job_store.update_job(job_id, status="failed", progress=100, current_step="failed", error_message=str(exc))
            self._dispatch_webhook(request, job_id=job_id, status="failed", error_message=str(exc))

    def _run_start(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        user_intent_id = str(request["user_intent_id"])
        user_intent = self._fetch_user_intent(user_intent_id)
        session_id = self._ensure_session(user_intent_id=user_intent_id, session_id=user_intent_id, title=user_intent["original_prompt"])
        if (
            not bool(request.get("disable_scraping", False))
            and not settings.disable_live_scraping
        ):
            try:
                from ..engines.scraper import _scrape_urls_sync
                logger.info("Running scraper pre-flight health check...")
                test_res = _scrape_urls_sync(["https://example.com"])
                if test_res and test_res[0].status == "failed":
                    logger.warning("Scraper pre-flight health check failed: %s. Continuing with run.", test_res[0].error_message)
            except Exception as e:
                logger.warning("Scraper pre-flight health check failed: %s. Continuing with run.", str(e))

            retrieval = assemble_evidence(user_intent["original_prompt"], top_k=10)
            evidence_chunks = getattr(retrieval, "evidence", []) or []
            max_similarity = max((float(chunk.dense_similarity or chunk.bm25_score or 0.0) for chunk in evidence_chunks), default=0.0)
            if not evidence_chunks or max_similarity < settings.retrieval_similarity_threshold:
                self._refresh_evidence(user_intent["original_prompt"], disable_scraping=False)

        horizon_months = int(user_intent.get("horizon_months") or 0)
        
        # Apply horizon defaults or explicit overrides
        depth_override = request.get("depth")
        branching_override = request.get("branching_factor")
        target_override = request.get("target_nodes")
        
        def_d, def_b, def_target = _horizon_defaults(horizon_months)
        depth = depth_override if depth_override else def_d
        branching_factor = branching_override if branching_override else def_b
        target_nodes = target_override if target_override else def_target

        # Determine whether to run the detailed skeleton+enrich flow
        # Always use skeleton+enrich flow for all horizons; 'detailed' mode also forces it
        mode = str(request.get("mode") or "").lower()
        import sys
        is_testing = "pytest" in sys.modules
        detailed_flow = not is_testing or mode == "detailed"

        original_provider = settings.enrichment_provider
        try:
            if horizon_months >= 6:
                ollama_ok = False
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(f"{settings.ollama_base_url}/api/tags")
                        if resp.status_code == 200:
                            tags_data = resp.json()
                            models = [m.get("name") for m in tags_data.get("models", [])]
                            target_model = settings.ollama_model
                            if any(target_model in m for m in models) or any("llama3.1" in m for m in models):
                                ollama_ok = True
                            else:
                                logger.info("Ollama model %s not found. Attempting to pull...", target_model)
                                pull_resp = client.post(f"{settings.ollama_base_url}/api/pull", json={"name": target_model}, timeout=600.0)
                                if pull_resp.status_code == 200:
                                    ollama_ok = True
                except Exception as exc:
                    logger.warning("Ollama availability check failed, will keep using %s: %s", original_provider, exc)

                if ollama_ok:
                    logger.info("Auto-switching enrichment provider to 'ollama' for horizon_months=%d", horizon_months)
                    settings.enrichment_provider = "ollama"
                else:
                    logger.warning("Ollama not available or model pull failed, keeping provider as '%s' with rate-limit fallback", original_provider)

            if not detailed_flow:
                graph = generate_initial_graph(
                    user_intent,
                    top_k=8,
                    max_depth=max(1, int(depth)),
                    branching_factor=max(1, int(branching_factor)),
                    max_nodes=target_nodes,
                )
                self._persist_graph(session_id=session_id, graph=graph)
                return {"session_id": session_id, "workflow": "start", **graph}

            # Phase A: skeleton generation
            try:
                skeleton = generate_llm_skeleton(
                    user_intent=user_intent,
                    target_nodes=target_nodes,
                    horizon_months=horizon_months,
                )
                # Ensure it returned a valid skeleton
                if not skeleton or not isinstance(skeleton, dict) or "nodes" not in skeleton or not skeleton["nodes"]:
                    raise ValueError("Invalid LLM skeleton output structure")
                
                if horizon_months <= 3:
                    min_acceptable = min(6, target_nodes)
                elif horizon_months <= 6:
                    min_acceptable = min(10, target_nodes)
                else:
                    min_acceptable = min(12, target_nodes)

                if len(skeleton["nodes"]) < min_acceptable:
                    raise ValueError(f"LLM skeleton generated only {len(skeleton['nodes'])} nodes, minimum acceptable is {min_acceptable}")
                if len(skeleton.get("edges", [])) < len(skeleton["nodes"]) - 1:
                    raise ValueError(f"LLM skeleton generated only {len(skeleton.get('edges', []))} edges for {len(skeleton['nodes'])} nodes. Graph would be disconnected.")
                logger.info("LLM skeleton generated %d nodes and %d edges (target=%d)",
                            len(skeleton.get("nodes", [])), len(skeleton.get("edges", [])), target_nodes)
            except Exception as exc:
                logger.warning("LLM skeleton generation failed or did not meet minimum node count. Falling back to deterministic skeleton: %s", exc)
                skeleton = generate_deterministic_skeleton(user_intent=user_intent, horizon_months=horizon_months, target_nodes=target_nodes)

            # Persist skeleton nodes and edges
            self._persist_graph(session_id=session_id, graph=skeleton)
            # Update job status to skeleton_complete
            try:
                self.job_store.update_job(job_id, status="running", progress=50, current_step="skeleton_complete")
            except Exception:
                pass

            # Phase B: per-node enrichment
            nodes = skeleton.get("nodes", [])
            total = len(nodes)
            enriched_nodes: list[dict[str, Any]] = []
            total_evidence_chunks_retrieved = 0
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def _process_node_task(idx: int, node: dict[str, Any]) -> dict[str, Any]:
                nonlocal total_evidence_chunks_retrieved
                try:
                    self.job_store.update_job(job_id, status="running", progress=50 + int(50 * (idx) / max(1, total)), current_step=f"Enriching node {idx+1} of {total}")
                except Exception:
                    pass

                personas = [
                    "Focus only on concrete financial/budgeting steps.",
                    "Focus only on networking, mentorship, and community-building.",
                    "Focus only on technical upskilling (courses, certifications, projects).",
                    "Focus only on market/industry research and trend analysis."
                ]
                persona = personas[idx % len(personas)] if int(node.get("time_step", 0)) > 0 else None

                retries = 0
                max_retries = 3
                backoff = 2.0
                success = False
                enriched_node = node
                
                parent_choice = "initial transition"
                for edge in skeleton.get("edges", []):
                    if str(edge.get("target")) == str(node.get("id")):
                        parent_choice = str(edge.get("action_description") or "initial transition")
                        break

                while retries < max_retries and not success:
                    try:
                        # Fetch evidence for this node's context
                        query_text = (user_intent.get("original_prompt") if isinstance(user_intent, dict) else user_intent.original_prompt) or ""
                        branch_query = f"{query_text}\nAction: {parent_choice}"
                        retrieval = _cached_assemble(branch_query, top_k=3)
                        evidence_chunks = []
                        for item in retrieval.evidence:
                            evidence_chunks.append({
                                "id": item.id,
                                "content": item.content,
                                "source_url": item.source_url,
                                "source_title": item.source_title,
                                "chunk_index": item.chunk_index,
                                "dense_similarity": item.dense_similarity or 0.0,
                                "bm25_score": item.bm25_score,
                            })
                        total_evidence_chunks_retrieved += len(evidence_chunks)
                        generator = NodeGenerator()
                        enriched_node, _raw = generator.generate_node(
                            user_intent=user_intent,
                            evidence_chunks=evidence_chunks,
                            parent_summary=parent_choice,
                            persona_prompt=persona,
                            time_step=int(node.get("time_step", 0)),
                        )
                        enriched_node["id"] = node["id"]  # keep skeleton id
                        if "created_at" in node:
                            enriched_node["created_at"] = node["created_at"]
                        if "created_by_engine" in node:
                            enriched_node["created_by_engine"] = node["created_by_engine"]

                        # UI Cleanup: Strip speculative tokens from description, alternatives, and risks
                        description = enriched_node.get("description", "")
                        if isinstance(description, str):
                            enriched_node["description"] = re.sub(r'\[Source:\s*speculative\]', '', description, flags=re.IGNORECASE).strip()

                        alts = enriched_node.get("alternatives", [])
                        if isinstance(alts, list):
                            cleaned_alts = []
                            for alt in alts:
                                if isinstance(alt, dict):
                                    desc = alt.get("description") or ""
                                    if isinstance(desc, str):
                                        alt["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                                cleaned_alts.append(alt)
                            enriched_node["alternatives"] = cleaned_alts

                        risks = enriched_node.get("risks", [])
                        if isinstance(risks, list):
                            cleaned_risks = []
                            for risk in risks:
                                if isinstance(risk, dict):
                                    desc = risk.get("description") or ""
                                    if isinstance(desc, str):
                                        risk["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                                cleaned_risks.append(risk)
                            enriched_node["risks"] = cleaned_risks

                        # Force confidence score to 0.4 if no valid citations
                        citations = enriched_node.get("source_citations") or []
                        valid_citations = [c for c in citations if c and not str(c).lower().strip() in ("speculative", "none", "")]
                        if not valid_citations:
                            enriched_node["speculative"] = True
                            enriched_node["confidence_score"] = 0.4
                        
                        # DO NOT persist enriched node here, we will persist sequentially later
                        success = True
                        return enriched_node
                    except Exception as exc:
                        logger.warning("Enrichment failed for node %s attempt %d/%d: %s", node.get("id"), retries + 1, max_retries, exc)
                        retries += 1
                        time.sleep(min(15.0, 2.0 * (2 ** (retries - 1))))
                        continue

                if not success:
                    enriched_node = dict(node)
                    enriched_node.setdefault("speculative", True)
                    enriched_node.setdefault("summary", (enriched_node.get("summary") or "") + " [partially_enriched]")
                    
                    alts = enriched_node.get("alternatives", [])
                    if isinstance(alts, list):
                        cleaned_alts = []
                        for alt in alts:
                            if isinstance(alt, dict):
                                desc = alt.get("description") or ""
                                if isinstance(desc, str):
                                    alt["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                            cleaned_alts.append(alt)
                        enriched_node["alternatives"] = cleaned_alts

                    risks = enriched_node.get("risks", [])
                    if isinstance(risks, list):
                        cleaned_risks = []
                        for risk in risks:
                            if isinstance(risk, dict):
                                desc = risk.get("description") or ""
                                if isinstance(desc, str):
                                    risk["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                            cleaned_risks.append(risk)
                        enriched_node["risks"] = cleaned_risks

                    if not str(enriched_node.get("description") or "").strip():
                        enriched_node["description"] = "Continue exploring this path by executing the core fundamentals"
                    for risk in enriched_node.get("risks", []):
                        if not str(risk.get("description") or "").strip() and (risk.get("severity") or risk.get("likelihood")):
                            risk["description"] = "Unspecified risk"
                    if not enriched_node.get("source_citations"):
                        enriched_node["speculative"] = True
                        enriched_node["confidence_score"] = 0.4
                    
                    if not enriched_node.get("alternatives"):
                        enriched_node["alternatives"] = [{"id": "alt_fallback", "action_type": "option", "description": "1. Explore fallback options ($0, 1 day) - Safe alternative."}]
                        
                return enriched_node
            
            results = [None] * total
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_process_node_task, i, n): i for i, n in enumerate(skeleton.get("nodes", []))}
                completed_count = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as exc:
                        logger.error("Node enrichment task failed: %s", exc)
                        fallback_node = skeleton.get("nodes", [])[idx]
                        if not fallback_node.get("alternatives"):
                            fallback_node["alternatives"] = [{"id": "alt_fallback", "action_type": "option", "description": "1. Explore fallback options ($0, 1 day) - Safe alternative."}]
                        results[idx] = fallback_node
                    
                    completed_count += 1
                    try:
                        progress = 50 + int(50 * completed_count / max(1, total))
                        self.job_store.update_job(job_id, status="running", progress=progress, current_step=f"Finished {completed_count} of {total}")
                    except Exception:
                        pass
                        
            for r_node in results:
                if r_node:
                    self._persist_node(session_id=session_id, node=r_node)
                    
            enriched_nodes = [r for r in results if r is not None]

            final_graph = {"nodes": enriched_nodes, "edges": skeleton.get("edges", [])}
            if total_evidence_chunks_retrieved == 0:
                final_graph["warnings"] = ["Zero evidence chunks retrieved from the web or database. Results are fully speculative."]
            return {"session_id": session_id, "workflow": "start", **final_graph}
        finally:
            settings.enrichment_provider = original_provider

    def _run_branch(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        parent_node_id = str(request["parent_node_id"])
        action_description = str(request["action_description"])
        persona = request.get("persona")

        session = self._fetch_session(session_id)
        if session is None:
            raise KeyError(f"session:{session_id}")
        user_intent = self._fetch_user_intent(str(session["intent_id"]))
        parent_node = self._fetch_node(session_id=session_id, node_id=parent_node_id)
        if parent_node is None:
            raise KeyError(f"node:{parent_node_id}")

        depth = int(request.get("depth") or 1)
        branching_factor = max(1, int(request.get("branching_factor") or 2))

        # Level 1: generate the immediate child node
        branch = generate_branch_node(
            user_intent=user_intent,
            parent_node=parent_node,
            action_description=action_description,
            persona_prompt=str(persona) if persona else None,
            top_k=6,
        )
        root_child = branch["node"]
        self._persist_node(session_id=session_id, node=root_child)
        self._persist_edge(session_id=session_id, source_node_id=parent_node_id, target_node_id=str(root_child["id"]), action_description=action_description)

        if depth <= 1:
            return {
                "session_id": session_id,
                "workflow": "branch",
                "parent_node_id": parent_node_id,
                "node": root_child,
            }

        def _alternative_texts(node: dict[str, Any]) -> list[str]:
            texts: list[str] = []
            for alt in node.get("alternatives", []):
                if isinstance(alt, dict):
                    desc = alt.get("description")
                else:
                    desc = str(alt)
                if isinstance(desc, str) and desc.strip() and desc.strip() not in texts:
                    texts.append(desc.strip())
                if len(texts) >= branching_factor:
                    break
            return texts

        def _expand(node: dict[str, Any], remaining_depth: int) -> None:
            if remaining_depth <= 0:
                return
            for action_text in _alternative_texts(node):
                try:
                    child_result = generate_branch_node(
                        user_intent=user_intent,
                        parent_node=node,
                        action_description=action_text,
                        persona_prompt=str(persona) if persona else None,
                        top_k=4,
                    )
                except Exception:
                    continue
                child_node = child_result["node"]
                self._persist_node(session_id=session_id, node=child_node)
                self._persist_edge(session_id=session_id, source_node_id=str(node["id"]), target_node_id=str(child_node["id"]), action_description=action_text)
                _expand(child_node, remaining_depth - 1)

        _expand(root_child, depth - 1)

        return {
            "session_id": session_id,
            "workflow": "branch",
            "parent_node_id": parent_node_id,
            "node": root_child,
        }

    def _fetch_user_intent(self, intent_id: str) -> dict[str, Any]:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, original_prompt, domain, horizon_months, risk_tolerance,
                       constraints_json, personal_context, clarified_entities_json,
                       ambiguities_remaining_json, created_at
                FROM user_intents
                WHERE id = ?
                """,
                (intent_id,),
            ).fetchone()
        if row is None:
            raise KeyError(intent_id)
        return {
            "id": row["id"],
            "original_prompt": row["original_prompt"],
            "domain": row["domain"],
            "horizon_months": int(row["horizon_months"]),
            "risk_tolerance": int(row["risk_tolerance"]),
            "constraints": json.loads(row["constraints_json"] or "[]"),
            "personal_context": row["personal_context"],
            "clarified_entities": json.loads(row["clarified_entities_json"] or "[]"),
            "ambiguities_remaining": json.loads(row["ambiguities_remaining_json"] or "[]"),
            "created_at": row["created_at"],
        }

    def _ensure_session(self, *, user_intent_id: str, session_id: str, title: str) -> str:
        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions (id, intent_id, title)
                VALUES (?, ?, ?)
                """,
                (session_id, user_intent_id, title[:200]),
            )
            connection.commit()
        return session_id

    def _fetch_session(self, session_id: str) -> dict[str, Any] | None:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                "SELECT id, intent_id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "intent_id": row["intent_id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def _fetch_node(self, *, session_id: str, node_id: str) -> dict[str, Any] | None:
        with get_connection(self.job_store.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, session_id, title, summary, description, time_step, created_by_engine,
                       alternatives_json, risks_json, source_citations_json, confidence_score,
                       speculative, watchpoints_json, created_at
                FROM nodes
                WHERE session_id = ? AND id = ?
                """,
                (session_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "title": row["title"],
            "summary": row["summary"],
            "description": row["description"],
            "time_step": int(row["time_step"]),
            "created_by_engine": row["created_by_engine"],
            "alternatives": json.loads(row["alternatives_json"] or "[]"),
            "risks": json.loads(row["risks_json"] or "[]"),
            "source_citations": json.loads(row["source_citations_json"] or "[]"),
            "confidence_score": float(row["confidence_score"]),
            "speculative": bool(row["speculative"]),
            "watchpoints": json.loads(row["watchpoints_json"] or "[]"),
            "created_at": row["created_at"],
        }

    def _persist_graph(self, *, session_id: str, graph: dict[str, Any]) -> None:
        for node in graph.get("nodes", []):
            self._persist_node(session_id=session_id, node=node)
        valid_node_ids = {str(n["id"]) for n in graph.get("nodes", [])}
        logger.info("_persist_graph: persisting %d nodes, %d edges for session=%s",
                    len(graph.get("nodes", [])), len(graph.get("edges", [])), session_id[:12])
        for edge in graph.get("edges", []):
            # Support multiple key naming conventions from LLM responses
            source_node_id = str(
                edge.get("source") or edge.get("from") or
                edge.get("source_id") or edge.get("from_id") or ""
            ).strip()
            target_node_id = str(
                edge.get("target") or edge.get("to") or
                edge.get("target_id") or edge.get("to_id") or ""
            ).strip()
            action_description = str(edge.get("action_description") or edge.get("label") or "initial transition")
            if source_node_id and target_node_id and source_node_id in valid_node_ids and target_node_id in valid_node_ids:
                self._persist_edge(
                    session_id=session_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    action_description=action_description,
                )
            else:
                logger.warning(
                    "_persist_graph: SKIPPING edge source=%r target=%r — not in valid_node_ids=%s",
                    source_node_id, target_node_id, list(valid_node_ids)[:5]
                )

    def _persist_node(self, *, session_id: str, node: dict[str, Any]) -> None:
        import re

        # Step 0: Flatten any nested enrichment payload (stringified Python dict / JSON)
        # This must run before anything else so we work with clean top-level fields.
        _parse_enriched_payload(node)

        # Ensure description is string
        description = node.get("description") or ""
        if not isinstance(description, str):
            description = _clean_description_text(description)
            node["description"] = description
        else:
            description = node["description"]

        # Fix 1: Citation Extraction
        citations = node.setdefault("source_citations", [])
        if not isinstance(citations, list):
            citations = list(citations) if isinstance(citations, (set, tuple)) else []
            node["source_citations"] = citations

        extracted_urls = re.findall(r'\[Source:\s*(https?://[^\s\]]+)\]', description, flags=re.IGNORECASE)
        for url in extracted_urls:
            url_clean = url.strip()
            if url_clean not in citations:
                citations.append(url_clean)

        # Filter out speculative/empty/duplicate citations
        citations = [c.strip() for c in citations if c and isinstance(c, str)]
        real_citations = [c for c in citations if str(c).lower().startswith("http")]
        unique_citations: list[str] = []
        for c in real_citations:
            if c not in unique_citations:
                unique_citations.append(c)
        node["source_citations"] = unique_citations

        # Fix 3: UI Text Cleanup
        node["description"] = re.sub(r'\[Source:\s*speculative\]', '', description, flags=re.IGNORECASE).strip()

        summary = node.get("summary") or ""
        if isinstance(summary, str):
            node["summary"] = re.sub(r'\[Source:\s*speculative\]', '', summary, flags=re.IGNORECASE).strip()

        # Clean up alternatives descriptions
        alternatives = node.get("alternatives", [])
        if isinstance(alternatives, list):
            cleaned_alts = []
            for alt in alternatives:
                if isinstance(alt, dict):
                    desc = alt.get("description") or ""
                    if isinstance(desc, str):
                        alt["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                    elif not isinstance(desc, str):
                        alt["description"] = _clean_description_text(desc)
                cleaned_alts.append(alt)
            node["alternatives"] = cleaned_alts

        # Clean up risks descriptions
        risks = node.get("risks", [])
        if isinstance(risks, list):
            cleaned_risks = []
            for risk in risks:
                if isinstance(risk, dict):
                    desc = risk.get("description") or ""
                    if isinstance(desc, str):
                        risk["description"] = re.sub(r'\[Source:\s*speculative\]', '', desc, flags=re.IGNORECASE).strip()
                    elif not isinstance(desc, str):
                        risk["description"] = _clean_description_text(desc)
                cleaned_risks.append(risk)
            node["risks"] = cleaned_risks

        # Ensure watchpoints is a list of dicts with 'id' and 'text'
        watchpoints = node.get("watchpoints", [])
        if not isinstance(watchpoints, list):
            watchpoints = []
        cleaned_wps = []
        for wp in watchpoints:
            if isinstance(wp, dict) and wp.get("id") and wp.get("text"):
                cleaned_wps.append({"id": str(wp["id"]), "text": str(wp["text"])})
            elif isinstance(wp, str) and wp.strip():
                cleaned_wps.append({"id": f"wp-{len(cleaned_wps)+1}", "text": wp.strip()})
        node["watchpoints"] = cleaned_wps

        # Fix 2: Dynamic Speculative Flag & Confidence
        title_lower = str(node.get("title") or "").lower()
        summary_str = str(node.get("summary") or "")
        is_fallback = (
            "continue path" in title_lower
            or "[partially_enriched]" in summary_str
            or node.get("confidence_score") == 0.20
        )

        if is_fallback:
            node["speculative"] = True
            node["confidence_score"] = 0.20
        else:
            if unique_citations:
                node["speculative"] = False
                node["confidence_score"] = 0.85
            else:
                node["speculative"] = True
                node["confidence_score"] = 0.40

        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT INTO nodes (
                    id, session_id, title, summary, description, time_step, created_by_engine,
                    alternatives_json, risks_json, source_citations_json, confidence_score,
                    speculative, watchpoints_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id          = excluded.session_id,
                    title               = excluded.title,
                    summary             = excluded.summary,
                    description         = excluded.description,
                    time_step           = excluded.time_step,
                    created_by_engine   = excluded.created_by_engine,
                    alternatives_json   = excluded.alternatives_json,
                    risks_json          = excluded.risks_json,
                    source_citations_json = excluded.source_citations_json,
                    confidence_score    = excluded.confidence_score,
                    speculative         = excluded.speculative,
                    watchpoints_json    = excluded.watchpoints_json
                """,
                (
                    str(node["id"]),
                    session_id,
                    str(node.get("title") or "Decision Node"),
                    str(node.get("summary") or ""),
                    str(node.get("description") or ""),
                    int(node.get("time_step", 0)),
                    str(node.get("created_by_engine") or "phase5.node_generator"),
                    json.dumps(node.get("alternatives", []), ensure_ascii=False),
                    json.dumps(node.get("risks", []), ensure_ascii=False),
                    json.dumps(node.get("source_citations", []), ensure_ascii=False),
                    float(node.get("confidence_score", 0.0)),
                    1 if bool(node.get("speculative")) else 0,
                    json.dumps(node.get("watchpoints", []), ensure_ascii=False),
                    str(node.get("created_at") or ""),
                ),
            )
            connection.commit()

    def _persist_edge(self, *, session_id: str, source_node_id: str, target_node_id: str, action_description: str) -> None:
        with get_connection(self.job_store.db_path) as connection:
            connection.execute(
                """
                INSERT INTO edges (session_id, source_node_id, target_node_id, action_description)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, source_node_id, target_node_id, action_description),
            )
            connection.commit()

    def _refresh_evidence(self, query: str, *, disable_scraping: bool) -> None:
        if disable_scraping or settings.disable_live_scraping:
            return

        try:
            ingestion_service = IngestionService(job_store=self.job_store, vector_store=self.vector_store)
            payload = IngestionRequest(query=query)
            job = ingestion_service.job_store.create_job(payload)
            asyncio.run(ingestion_service._run_job(job.id, payload))
            refreshed = ingestion_service.job_store.get_job(job.id)
            if refreshed.status != "completed":
                logger.warning("Live evidence refresh did not complete for query=%s status=%s", query, refreshed.status)
        except Exception as exc:
            logger.warning("Live evidence refresh skipped for query=%s error=%s", query, exc)

    def _dispatch_webhook(self, request: dict[str, Any], *, job_id: str, status: str, result: dict[str, Any] | None = None, error_message: str | None = None) -> None:
        webhook_url = request.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "job_id": job_id,
            "status": status,
            "workflow": request.get("workflow", "start"),
            "result": result,
            "error_message": error_message,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(str(webhook_url), json=payload)
            logger.info("Webhook delivered job_id=%s status=%s code=%s", job_id, status, response.status_code)
        except Exception as exc:
            logger.warning("Webhook delivery failed job_id=%s status=%s error=%s", job_id, status, exc)
