# [F-memory/R2/C4] Memory tree orchestration — all pipeline wiring
# R2 — Application. Depends: R0 config, R1 domain, R4 storage. No contract imports.
# v0.5 — predicate table as vocabulary source, predicate-aware decay/compression,
#         bootstrap prior from table, cold branch via physics gate,
#         summary injection, consensus preserves predicate structure,
#         classify_predicate with axis-based gap/synonym detection

from __future__ import annotations
import uuid
import json
import os
from typing import Optional, Callable

from src.domain.nodes import (
    Node, is_duplicate, is_compression_eligible,
    record_access, current_relevance,
    derive_predicate_props,
)
from src.domain.retrieval import (
    compute_tfidf, compute_tf, tokenize, update_corpus_idf,
    conditional_injection, format_injection, build_query_vector,
    activate_relational, compute_distribution, get_threshold,
    compute_signal_strength, compute_bootstrap_prior, is_cold_branch,
)
from src.domain.compression import (
    should_compress_branch, compress_branch,
    compress_cold_branch, compute_mean_relevance,
)
from src.exec.storage import Storage


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG = _load_config()


# ── MemoryTree ────────────────────────────────────────────────────────────────

class MemoryTree:
    """
    Orchestrates the three-dimensional memory space.

      D1 — Permanence axis:   backbone (permanent) → active → cold → archive
      D2 — Domain manifolds:  dynamic branches, created at runtime
      D3 — Relevance space:   cosine similarity × predicate-aware decay buoyancy

    All writes go through Storage (R4) atomically.
    All domain logic lives in R1.
    This layer wires them together, manages system state, and holds the
    predicate props cache — the single runtime lookup table for predicate physics.
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self.storage         = Storage(db_path)
        self._corpus_idf:    dict[str, float]        = {}
        self._predicate_map: dict[str, dict]         = {}  # predicate → props dict
        self._bootstrap_prior: tuple[float, float]   = None
        self._init()

    def _init(self) -> None:
        """Load predicate table, compute bootstrap prior, rebuild IDF."""
        self._reload_predicate_map()
        self._init_bootstrap_prior()
        self._rebuild_idf()

    def _reload_predicate_map(self) -> None:
        """Load all predicates from table into memory cache."""
        rows = self.storage.get_all_predicates()
        self._predicate_map = {r["predicate"]: r for r in rows}

    def _init_bootstrap_prior(self) -> None:
        """
        Load bootstrap prior from meta if already computed.
        Compute from predicate table if not yet stored.
        Store in meta for future sessions.
        """
        stored_mean = self.storage.get_meta("bootstrap_mean")
        stored_std  = self.storage.get_meta("bootstrap_std")

        if stored_mean and stored_std:
            self._bootstrap_prior = (float(stored_mean), float(stored_std))
            return

        predicates = list(self._predicate_map.values())
        mean, std  = compute_bootstrap_prior(predicates)
        self.storage.set_meta("bootstrap_mean", str(mean))
        self.storage.set_meta("bootstrap_std",  str(std))
        self._bootstrap_prior = (mean, std)

    def _rebuild_idf(self) -> None:
        raw_nodes        = self.storage.get_all_active_nodes()
        nodes            = [self._node_from_dict(n) for n in raw_nodes]
        self._corpus_idf = update_corpus_idf(nodes)

    def _system_accesses(self) -> int:
        return self.storage.get_system_accesses()

    def _node_from_dict(self, d: dict) -> Node:
        """Deserialize node dict and inject predicate props from cache."""
        node = Node.from_dict(d)
        node._predicate_props = self._predicate_map.get(node.predicate, {})
        return node

    def _get_props(self, predicate: str) -> dict:
        """Return predicate physics props from cache. Empty dict if unknown."""
        return self._predicate_map.get(predicate, {})


    # ── Branch management ─────────────────────────────────────────────────────

    def add_branch(self, branch_name: str) -> None:
        self.storage.upsert_branch(branch_name, is_backbone=False)

    def ensure_backbone(self) -> None:
        self.storage.upsert_branch("backbone", is_backbone=True)


    # ── Predicate classification ──────────────────────────────────────────────

    def classify_predicate(self, predicate: str,
                            axes: dict = None) -> dict:
        """
        Classify an unknown predicate against the predicate table.

        If axes provided: compare axis signature against all known predicates.
            Exact match on all 5 axes → synonym, map to existing.
            Unique signature → gap confirmed, derive props, insert new predicate.
            Equidistant (ties across 2+ predicates) → ambiguous, quarantine.

        If axes not provided: quarantine. Caller must supply axes for
        auto-classification. v0.6 will add embedding-based axis inference.

        Returns:
            {"action": "known",     "predicate": str}
            {"action": "synonym",   "mapped_to": str, "original": str}
            {"action": "inserted",  "predicate": str, "props": dict}
            {"action": "quarantine","predicate": str, "reason": str}
        """
        # Already known
        if predicate in self._predicate_map:
            return {"action": "known", "predicate": predicate}

        # Check synonym table
        mapped = self.storage.get_synonym_mapping(predicate)
        if mapped:
            return {"action": "synonym", "mapped_to": mapped, "original": predicate}

        # No axes provided — quarantine
        if not axes:
            return {
                "action": "quarantine",
                "predicate": predicate,
                "reason": "no axis signature provided — supply polarity, temporality, "
                          "directionality, certainty, agency to classify"
            }

        AXES = ["polarity", "temporality", "directionality", "certainty", "agency"]

        # Compute axis agreement score against each known predicate
        matches: list[tuple[int, str]] = []
        for known_pred, known_props in self._predicate_map.items():
            score = sum(1 for ax in AXES if axes.get(ax) == known_props.get(ax))
            matches.append((score, known_pred))

        matches.sort(key=lambda x: x[0], reverse=True)
        top_score    = matches[0][0] if matches else 0
        top_matches  = [m for m in matches if m[0] == top_score]

        # Exact match on all 5 axes — synonym
        if top_score == 5:
            mapped_to = top_matches[0][1]
            sig = json.dumps({ax: axes[ax] for ax in AXES if ax in axes})
            self.storage.log_synonym(predicate, mapped_to, sig)
            return {"action": "synonym", "mapped_to": mapped_to, "original": predicate}

        # Ambiguous — tied across 2+ predicates at high agreement
        if len(top_matches) > 1 and top_score >= 4:
            return {
                "action": "quarantine",
                "predicate": predicate,
                "reason": f"ambiguous — tied between {[m[1] for m in top_matches]}"
            }

        # Gap confirmed — derive props and insert
        props = derive_predicate_props(
            axes.get("polarity", "neutral"),
            axes.get("temporality", "transient"),
            axes.get("directionality", "self"),
            axes.get("certainty", "belief"),
            axes.get("agency", "active"),
        )
        props["predicate"] = predicate
        props["source"]    = "discovered"
        props["version"]   = _CFG["predicates"]["version"]

        self.storage.insert_predicate(props)
        self._reload_predicate_map()

        return {"action": "inserted", "predicate": predicate, "props": props}

    def _resolve_predicate(self, predicate: Optional[str],
                            axes: dict = None) -> tuple[str, dict]:
        """
        Resolve predicate to a known term and its physics props.

        Returns (resolved_predicate, props_dict).

        None → default 'knows' (free-text fallback).
        Known → pass through with props.
        Unknown with axes → classify (synonym/gap/quarantine).
        Unknown without axes → quarantine, fall back to 'knows'.
        """
        if predicate is None:
            props = self._get_props("knows")
            return "knows", props

        if predicate in self._predicate_map:
            return predicate, self._predicate_map[predicate]

        # Check synonym table first
        mapped = self.storage.get_synonym_mapping(predicate)
        if mapped:
            return mapped, self._predicate_map.get(mapped, {})

        # Attempt classification
        result = self.classify_predicate(predicate, axes)

        if result["action"] == "synonym":
            mapped_to = result["mapped_to"]
            return mapped_to, self._predicate_map.get(mapped_to, {})

        if result["action"] == "inserted":
            return predicate, result["props"]

        # quarantine — fall back to 'knows', preserve memory
        return "knows", self._get_props("knows")


    # ── Write ─────────────────────────────────────────────────────────────────

    def add_node(self,
                 content:      Optional[str]      = None,
                 branch_name:  str                = "backbone",
                 tags:         Optional[list]     = None,
                 normalize_fn: Optional[Callable] = None,
                 agent_id:     Optional[str]      = None,
                 subject:      Optional[str]      = None,
                 predicate:    Optional[str]      = None,
                 object_text:  Optional[str]      = None,
                 embedding_fn: Optional[Callable] = None,
                 axes:         Optional[dict]     = None) -> Node:
        """
        Full write pipeline (v0.5).

        Structured write:  subject + predicate + object_text
        Free-text fallback: content only, subject=None, predicate='knows'

        1. Resolve predicate — validate against table or classify via axes
        2. Build object_text and content
        3. Vectorize — obj_vector from object_text, content_vector from content
        4. Dedup check — subject + predicate + obj_vector cosine
        5. Insert or update
        6. Rebuild IDF
        7. φ-triggered OLS check
        """
        tags = tags or []

        # ── Resolve predicate ─────────────────────────────────────────────────
        resolved_predicate, pred_props = self._resolve_predicate(predicate, axes)

        # ── Structured vs free-text ───────────────────────────────────────────
        if object_text is not None:
            resolved_subject = subject
            resolved_object  = normalize_fn(object_text) if normalize_fn else object_text
            resolved_content = (
                f"{resolved_subject} {resolved_predicate} {resolved_object}"
                if resolved_subject is not None
                else resolved_object
            )
        else:
            resolved_subject = None
            resolved_content = normalize_fn(content or "") if normalize_fn else (content or "")
            resolved_object  = resolved_content

        self.storage.upsert_branch(branch_name, is_backbone=(branch_name == "backbone"))

        # ── Vectorize ─────────────────────────────────────────────────────────
        obj_vec = (
            embedding_fn(resolved_object)
            if embedding_fn
            else compute_tf(tokenize(resolved_object))
        )
        content_vector = compute_tf(tokenize(resolved_content))

        sys_accesses = self._system_accesses()

        # ── Load branch nodes for dedup ───────────────────────────────────────
        raw_branch   = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes = [self._node_from_dict(n) for n in raw_branch]

        # ── Compute live dedup threshold ──────────────────────────────────────
        all_raw   = self.storage.get_all_active_nodes()
        all_nodes = [self._node_from_dict(n) for n in all_raw]
        mean, std = compute_distribution(all_nodes, self._bootstrap_prior)
        dedup_thresh = get_threshold(
            _CFG["physics"]["beta"]["dedup"]["value"], mean, std
        )

        # ── Dedup check ───────────────────────────────────────────────────────
        existing = is_duplicate(
            obj_vec, resolved_subject, resolved_predicate,
            branch_nodes, dedup_thresh
        )

        if existing:
            existing.content        = resolved_content
            existing.object_text    = resolved_object
            existing.obj_vector     = obj_vec
            existing.content_vector = content_vector
            existing.access_count  += 1
            existing.system_access_snapshot = sys_accesses
            for tag in tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
            self.storage.update_node(existing.to_dict())
            self._rebuild_idf()
            return existing

        # ── New node ──────────────────────────────────────────────────────────
        is_backbone = (branch_name == "backbone")
        node = Node(
            content                 = resolved_content,
            branch_name             = branch_name,
            is_backbone             = is_backbone,
            base_score              = 1.0,
            access_count            = 0,
            system_access_snapshot  = sys_accesses,
            tags                    = tags,
            content_vector          = content_vector,
            agent_id                = agent_id,
            subject                 = resolved_subject,
            predicate               = resolved_predicate,
            object_text             = resolved_object,
            obj_vector              = obj_vec,
        )
        node._predicate_props = pred_props
        self.storage.insert_node(node.to_dict())
        self._rebuild_idf()

        if not is_backbone:
            self._check_and_compress(branch_name)

        return node

    def add_backbone_node(self, content: str,
                          tags:         Optional[list]     = None,
                          normalize_fn: Optional[Callable] = None,
                          agent_id:     Optional[str]      = None,
                          subject:      Optional[str]      = None,
                          predicate:    Optional[str]      = None,
                          object_text:  Optional[str]      = None,
                          embedding_fn: Optional[Callable] = None,
                          axes:         Optional[dict]     = None) -> Node:
        self.ensure_backbone()
        return self.add_node(
            content=content, branch_name="backbone", tags=tags,
            normalize_fn=normalize_fn, agent_id=agent_id,
            subject=subject, predicate=predicate,
            object_text=object_text, embedding_fn=embedding_fn, axes=axes,
        )


    # ── OLS compression pipeline ──────────────────────────────────────────────

    def _check_and_compress(self, branch_name: str) -> bool:
        """
        φ-triggered branch-level OLS.
        Fires automatically after every non-backbone write.
        Returns True if compression occurred.
        """
        branch_sizes = self.storage.get_branch_sizes()
        all_sizes    = list(branch_sizes.values())
        raw_nodes    = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes = [self._node_from_dict(n) for n in raw_nodes]

        if not should_compress_branch(branch_nodes, all_sizes):
            return False

        sys_accesses = self._system_accesses()
        all_raw      = self.storage.get_all_active_nodes()
        all_nodes    = [self._node_from_dict(n) for n in all_raw]

        try:
            compressed, consumed_ids = compress_branch(
                branch_nodes, branch_name, sys_accesses,
                all_nodes, self._predicate_map
            )
        except ValueError:
            return False

        log_entry = {
            "log_id":          str(uuid.uuid4()),
            "compressed_node": compressed.node_id,
            "source_ids":      consumed_ids,
            "branch_name":     branch_name,
            "level":           "branch",
        }
        self.storage.compress_nodes_atomic(consumed_ids, compressed.to_dict(), log_entry)
        self._rebuild_idf()
        return True

    def run_root_ols(self) -> list[str]:
        """
        Root-level OLS — compress cold branches into archive nodes.
        Cold = mean node relevance below retrieval threshold (physics gate).
        Call at session boundaries.
        Returns list of branch names compressed.
        """
        sys_accesses = self._system_accesses()
        all_raw      = self.storage.get_all_active_nodes()
        all_nodes    = [self._node_from_dict(n) for n in all_raw]

        # Compute retrieval threshold for cold gate
        mean, std = compute_distribution(all_nodes, self._bootstrap_prior)
        retrieval_threshold = get_threshold(
            _CFG["physics"]["beta"]["retrieval"]["value"], mean, std
        )

        branch_names = [
            b["branch_name"] for b in self.storage.get_all_branches()
            if not b["is_backbone"]
        ]

        compressed_branches = []

        for branch_name in branch_names:
            raw_nodes    = self.storage.get_nodes_for_branch(branch_name)
            if not raw_nodes:
                continue
            branch_nodes = [self._node_from_dict(n) for n in raw_nodes]

            if not is_cold_branch(
                branch_nodes, sys_accesses,
                retrieval_threshold, self._predicate_map
            ):
                continue

            try:
                compressed, consumed_ids = compress_cold_branch(
                    branch_name, branch_nodes, sys_accesses,
                    all_nodes, self._predicate_map
                )
            except ValueError:
                continue

            log_entry = {
                "log_id":          str(uuid.uuid4()),
                "compressed_node": compressed.node_id,
                "source_ids":      consumed_ids,
                "branch_name":     branch_name,
                "level":           "root",
            }
            self.storage.compress_nodes_atomic(consumed_ids, compressed.to_dict(), log_entry)
            compressed_branches.append(branch_name)

        if compressed_branches:
            self._rebuild_idf()

        return compressed_branches


    # ── Query / retrieval ─────────────────────────────────────────────────────

    def query(self, prompt: str, formatted: bool = True,
              agent_id: Optional[str] = None) -> "str | list[Node]":
        """
        Conditional injection — D3 slice.
        Scores all active nodes against prompt using predicate-aware decay.
        Returns summary-grouped formatted string or raw node list.
        """
        sys_accesses = self._system_accesses()
        raw_nodes    = self.storage.get_all_active_nodes()
        all_nodes    = [self._node_from_dict(n) for n in raw_nodes]

        mean, std = compute_distribution(all_nodes, self._bootstrap_prior)
        retrieval_thresh = get_threshold(
            _CFG["physics"]["beta"]["retrieval"]["value"], mean, std
        )

        injected = conditional_injection(
            prompt, all_nodes, self._corpus_idf, sys_accesses,
            retrieval_threshold=retrieval_thresh,
            predicate_props_map=self._predicate_map,
            bootstrap_prior=self._bootstrap_prior,
        )

        # Relational activation — Hebbian co-access partners
        co_access_partners = {}
        for node in injected:
            partners = self.storage.get_co_access_partners(
                node.node_id, agent_id=agent_id
            )
            if partners:
                co_access_partners[node.node_id] = partners

        chars_injected   = sum(len(n.content) for n in injected)
        budget_remaining = _CFG["injection"]["budget_chars"] - chars_injected
        activated = activate_relational(
            injected, all_nodes, co_access_partners, budget_remaining
        )
        injected = injected + activated

        # Record accesses
        new_total = self.storage.increment_system_accesses(by=len(injected))
        for node in injected:
            record_access(node, new_total)
            self.storage.update_node_access(
                node.node_id, node.access_count, node.system_access_snapshot
            )
            self.storage.increment_branch_access(node.branch_name)

        if len(injected) > 1:
            self.storage.log_co_access(
                [n.node_id for n in injected], agent_id=agent_id
            )

        if formatted:
            return format_injection(injected)
        return injected


    # ── Decay scan ────────────────────────────────────────────────────────────

    def decay_scan(self) -> dict[str, float]:
        """Current predicate-aware relevance scores for all active non-backbone nodes."""
        sys_accesses = self._system_accesses()
        raw_nodes    = self.storage.get_all_active_nodes()
        scores       = {}
        for raw in raw_nodes:
            node = self._node_from_dict(raw)
            if not node.is_backbone:
                scores[node.node_id] = current_relevance(
                    node, sys_accesses, self._predicate_map.get(node.predicate)
                )
        return scores


    # ── Multi-agent consensus pipeline ────────────────────────────────────────

    def add_agent_node(self, content: str,
                       branch_name:  str,
                       agent_id:     str,
                       concept_key:  str,
                       tags:         Optional[list]     = None,
                       normalize_fn: Optional[Callable] = None,
                       stakes:       str                = "standard",
                       subject:      Optional[str]      = None,
                       predicate:    Optional[str]      = None,
                       object_text:  Optional[str]      = None,
                       embedding_fn: Optional[Callable] = None,
                       axes:         Optional[dict]     = None) -> dict:
        """
        Multi-agent write pipeline. Routes backbone content through consensus.
        Non-backbone writes directly with agent_id provenance.
        """
        tags = tags or []
        if normalize_fn:
            content = normalize_fn(content)

        if branch_name != "backbone":
            node = self.add_node(
                content=content, branch_name=branch_name, tags=tags,
                agent_id=agent_id, subject=subject, predicate=predicate,
                object_text=object_text, embedding_fn=embedding_fn, axes=axes,
            )
            return {"status": "written", "node_id": node.node_id}

        # Backbone — resolve predicate and route through consensus
        resolved_predicate, pred_props = self._resolve_predicate(predicate, axes)
        resolved_subject = subject
        resolved_object  = object_text if object_text is not None else content
        resolved_content = (
            f"{resolved_subject} {resolved_predicate} {resolved_object}"
            if resolved_subject is not None else resolved_object
        )

        content_vector = compute_tf(tokenize(resolved_content))
        obj_vec = (
            embedding_fn(resolved_object)
            if embedding_fn
            else compute_tf(tokenize(resolved_object))
        )
        sys_acc = self._system_accesses()

        pending = {
            "pending_id":      f"{concept_key}:{agent_id}",
            "concept_key":     concept_key,
            "agent_id":        agent_id,
            "content":         resolved_content,
            "branch_name":     branch_name,
            "content_vector":  content_vector,
            "subject":         resolved_subject,
            "predicate":       resolved_predicate,
            "object_text":     resolved_object,
            "obj_vector":      obj_vec,
            "tags":            tags,
        }
        self.storage.write_pending(pending)

        return self._check_consensus(concept_key, branch_name, sys_acc, stakes)

    def _check_consensus(self, concept_key: str, branch_name: str,
                         sys_accesses: int, stakes: str = "standard") -> dict:
        """
        Evaluate signal strength across pending agent versions.

        Signal = mean pairwise cosine similarity across content_vectors.
        Threshold = SDT-grounded, emergent from live tree distribution.

        On commit: merged backbone node inherits subject+predicate from agents.
        Predicate divergence across agents → structural conflict regardless of signal.
        """
        pending_raw = self.storage.get_pending_by_concept(concept_key)

        if len(pending_raw) < 2:
            return {
                "status": "pending", "signal": 0.0,
                "threshold": None, "agents": len(pending_raw)
            }

        # ── Predicate divergence check — structural conflict ──────────────────
        predicates = {p.get("predicate", "knows") for p in pending_raw}
        subjects   = {p.get("subject") for p in pending_raw}

        if len(predicates) > 1:
            # Agents disagree on predicate — structural conflict, skip signal check
            conflict = {
                "conflict_id": str(uuid.uuid4()),
                "concept_key": concept_key,
                "branch_name": branch_name,
                "agent_ids":   [p["agent_id"] for p in pending_raw],
                "signal":      0.0,
                "threshold":   0.0,
            }
            self.storage.flag_conflict(conflict)
            return {
                "status": "conflict",
                "reason": "predicate_divergence",
                "predicates": list(predicates),
                "agents": conflict["agent_ids"],
            }

        # ── Signal evaluation ─────────────────────────────────────────────────
        pending_nodes = [
            Node(
                content        = p["content"],
                content_vector = p.get("content_vector", {}),
                agent_id       = p["agent_id"],
                branch_name    = branch_name,
                subject        = p.get("subject"),
                predicate      = p.get("predicate", "knows"),
                object_text    = p.get("object_text", ""),
                obj_vector     = p.get("obj_vector", {}),
            )
            for p in pending_raw
        ]

        signal = compute_signal_strength(pending_nodes)

        all_raw   = self.storage.get_all_active_nodes()
        all_nodes = [self._node_from_dict(n) for n in all_raw]
        mean, std = compute_distribution(all_nodes, self._bootstrap_prior)

        stakes_mult = _CFG["consensus"]["stakes_multiplier"].get(stakes, 1.0)
        threshold   = get_threshold(
            _CFG["physics"]["beta"]["consensus"]["value"],
            mean, std, stakes_multiplier=stakes_mult
        )

        if signal >= threshold:
            # ── Commit — inherit subject+predicate from agents ────────────────
            from src.domain.compression import _merge_content, _merge_vectors, _merge_obj_vectors
            merged_content    = _merge_content(pending_nodes)
            merged_cv         = _merge_vectors(pending_nodes)
            merged_obj_vector = _merge_obj_vectors(pending_nodes)

            # All agents agreed on predicate — safe to inherit
            consensus_predicate = predicates.pop()
            consensus_subject   = subjects.pop() if len(subjects) == 1 else None

            object_text = (
                merged_content if consensus_subject is None
                else " | ".join(p["object_text"] for p in pending_raw if p.get("object_text"))
            )

            node_dict = {
                "node_id":                str(uuid.uuid4()),
                "content":                merged_content,
                "branch_name":            "backbone",
                "is_backbone":            True,
                "base_score":             1.0,
                "access_count":           0,
                "system_access_snapshot": sys_accesses,
                "tags":                   list({t for p in pending_raw for t in p.get("tags", [])}),
                "compressed_from":        [p["pending_id"] for p in pending_raw],
                "content_vector":         merged_cv,
                "agent_id":               None,
                "subject":                consensus_subject,
                "predicate":              consensus_predicate,
                "object_text":            object_text,
                "obj_vector":             merged_obj_vector,
            }

            self.storage.upsert_branch("backbone", is_backbone=True)
            self.storage.commit_pending_to_backbone(concept_key, node_dict)
            self._rebuild_idf()

            return {
                "status":    "committed",
                "signal":    signal,
                "threshold": threshold,
                "agents":    [p["agent_id"] for p in pending_raw],
                "predicate": consensus_predicate,
                "subject":   consensus_subject,
            }

        else:
            conflict = {
                "conflict_id": str(uuid.uuid4()),
                "concept_key": concept_key,
                "branch_name": branch_name,
                "agent_ids":   [p["agent_id"] for p in pending_raw],
                "signal":      signal,
                "threshold":   threshold,
            }
            self.storage.flag_conflict(conflict)
            return {
                "status":    "conflict",
                "signal":    signal,
                "threshold": threshold,
                "agents":    conflict["agent_ids"],
            }

    def pending_conflicts(self) -> list[dict]:
        return self.storage.get_pending_conflicts()

    def resolve_conflict(self, conflict_id: str,
                         resolved_content: str,
                         concept_key: str,
                         subject:     Optional[str] = None,
                         predicate:   Optional[str] = None,
                         object_text: Optional[str] = None,
                         axes:        Optional[dict] = None) -> None:
        """
        Write externally resolved content to backbone. Atomic.
        Preserves triple structure — caller supplies subject+predicate if known.
        """
        sys_acc = self._system_accesses()

        resolved_predicate, pred_props = self._resolve_predicate(predicate, axes)
        resolved_subject = subject
        resolved_object  = object_text if object_text is not None else resolved_content
        final_content    = (
            f"{resolved_subject} {resolved_predicate} {resolved_object}"
            if resolved_subject is not None else resolved_content
        )

        content_vector = compute_tf(tokenize(final_content))
        obj_vec        = compute_tf(tokenize(resolved_object))

        node_dict = {
            "node_id":                str(uuid.uuid4()),
            "content":                final_content,
            "branch_name":            "backbone",
            "is_backbone":            True,
            "base_score":             1.0,
            "access_count":           0,
            "system_access_snapshot": sys_acc,
            "tags":                   ["resolved"],
            "compressed_from":        [],
            "content_vector":         content_vector,
            "agent_id":               "resolved",
            "subject":                resolved_subject,
            "predicate":              resolved_predicate,
            "object_text":            resolved_object,
            "obj_vector":             obj_vec,
        }

        self.storage.upsert_branch("backbone", is_backbone=True)
        self.storage.commit_pending_to_backbone(concept_key, node_dict)
        self.storage.resolve_conflict(conflict_id)
        self._rebuild_idf()


    # ── Predicate table ───────────────────────────────────────────────────────

    def predicates(self) -> list[dict]:
        """Return full predicate table — vocabulary + axis properties."""
        return self.storage.get_all_predicates()

    def predicate_synonyms(self) -> list[dict]:
        """Return all logged synonym mappings."""
        cur = self.storage._conn.execute("SELECT * FROM predicate_synonyms")
        return [dict(r) for r in cur.fetchall()]


    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str = "living_memory_export.json") -> str:
        return self.storage.export_json(path)

    def close(self) -> None:
        self.storage.close()
