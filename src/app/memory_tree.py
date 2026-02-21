# [F-memory/R2/C4] Memory tree orchestration — backbone, branches, OLS triggers
# R2 — Application. Depends: R0 config, R1 domain, R4 storage. No contract imports.
# v0.4 — predicate-aware writes, unknown predicate routing, embedding_fn hook

from __future__ import annotations
import uuid
import json
import os
from typing import Optional, Callable

from src.domain.nodes import (
    Node, compute_decay, is_duplicate, is_compression_eligible,
    is_known_predicate, DEFAULT_PREDICATE,
    record_access, current_relevance
)
from src.domain.retrieval import (
    compute_tfidf, compute_tf, tokenize, update_corpus_idf,
    conditional_injection, format_injection, build_query_vector,
    activate_relational, compute_distribution, get_threshold,
    compute_signal_strength
)
from src.domain.compression import (
    should_compress_branch, compress_branch,
    compress_cold_branch, is_cold_branch
)
from src.exec.storage import Storage


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG = _load_config()
_PROMOTION_THRESHOLD = _CFG["unknown_predicates"]["promotion_threshold"]


# ── MemoryTree ────────────────────────────────────────────────────────────────

class MemoryTree:
    """
    Orchestrates the three-dimensional memory space:

      D1 — Permanence axis:   backbone (permanent) → active → cold → archive
      D2 — Domain manifolds:  dynamic branches, created at runtime
      D3 — Relevance space:   cosine similarity × decay buoyancy

    All writes go through Storage (R4) atomically.
    All domain logic lives in R1.
    This layer wires them together and manages system state.
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self.storage      = Storage(db_path)
        self._corpus_idf: dict[str, float] = {}
        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        raw_nodes  = self.storage.get_all_active_nodes()
        nodes      = [Node.from_dict(n) for n in raw_nodes]
        self._corpus_idf = update_corpus_idf(nodes)

    def _system_accesses(self) -> int:
        return self.storage.get_system_accesses()


    # ── Branch management ─────────────────────────────────────────────────────

    def add_branch(self, branch_name: str) -> None:
        self.storage.upsert_branch(branch_name, is_backbone=False)

    def ensure_backbone(self) -> None:
        self.storage.upsert_branch("backbone", is_backbone=True)


    # ── Predicate routing ─────────────────────────────────────────────────────

    def _resolve_predicate(self, predicate: Optional[str]) -> str:
        """
        Validate predicate against locked vocabulary.
        Known predicate → pass through.
        Unknown predicate → log to unknown_predicates, return DEFAULT_PREDICATE.
        None → return DEFAULT_PREDICATE (free-text fallback).
        """
        if predicate is None:
            return DEFAULT_PREDICATE
        if is_known_predicate(predicate):
            return predicate
        # Unknown — log and fall back
        count = self.storage.log_unknown_predicate(predicate)
        if count >= _PROMOTION_THRESHOLD:
            # Surface to caller via tag on the node — external review required
            pass  # promotion surfacing is handled at API layer
        return DEFAULT_PREDICATE

    def _check_promotion_candidates(self) -> list[dict]:
        """Return unknown predicates that have reached promotion threshold."""
        return self.storage.get_promotion_candidates(_PROMOTION_THRESHOLD)


    # ── Write ─────────────────────────────────────────────────────────────────

    def add_node(self,
                 content:      Optional[str]   = None,
                 branch_name:  str             = "backbone",
                 tags:         Optional[list]  = None,
                 normalize_fn: Optional[Callable] = None,
                 agent_id:     Optional[str]   = None,
                 subject:      Optional[str]   = None,
                 predicate:    Optional[str]   = None,
                 object_text:  Optional[str]   = None,
                 embedding_fn: Optional[Callable] = None) -> Node:
        """
        Add a node to a branch. Full write pipeline (v0.4):

        Structured write:  subject + predicate + object_text provided
        Free-text fallback: content provided, subject=None, predicate=knows

        1. Resolve predicate (validate or log-and-fallback)
        2. Build object_text and content
        3. Vectorize: obj_vector from object_text, tfidf_vector from content
        4. Dedup check — subject+predicate+obj_vector cosine
        5. Insert or update
        6. Rebuild IDF
        7. OLS check
        """
        tags = tags or []

        # ── Structured vs free-text resolution ───────────────────────────────
        if object_text is not None:
            # Structured write
            resolved_predicate = self._resolve_predicate(predicate)
            resolved_subject   = subject
            resolved_object    = object_text
            if normalize_fn is not None:
                resolved_object = normalize_fn(resolved_object)
            resolved_content = (
                f"{resolved_subject} {resolved_predicate} {resolved_object}"
                if resolved_subject is not None
                else resolved_object
            )
        else:
            # Free-text fallback — full content string becomes object_text
            resolved_predicate = DEFAULT_PREDICATE
            resolved_subject   = None
            resolved_content   = content or ""
            if normalize_fn is not None:
                resolved_content = normalize_fn(resolved_content)
            resolved_object = resolved_content

        self.storage.upsert_branch(branch_name, is_backbone=(branch_name == "backbone"))

        # ── Vectorize ─────────────────────────────────────────────────────────
        # obj_vector: object only — TF default, pluggable via embedding_fn
        if embedding_fn is not None:
            obj_vec = embedding_fn(resolved_object)
        else:
            obj_vec = compute_tf(tokenize(resolved_object))

        # tfidf_vector: full content — corpus-independent TF for retrieval
        tf_vector    = compute_tf(tokenize(resolved_content))
        tfidf_vector = compute_tfidf(resolved_content, self._corpus_idf)

        sys_accesses = self._system_accesses()

        # ── Load branch nodes for dedup ───────────────────────────────────────
        raw_branch   = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes = [Node.from_dict(n) for n in raw_branch]

        # ── Compute live dedup threshold ──────────────────────────────────────
        all_raw   = self.storage.get_all_active_nodes()
        all_nodes = [Node.from_dict(n) for n in all_raw]
        mean, std = compute_distribution(all_nodes)
        dedup_thresh = get_threshold(_CFG["thresholds"]["dedup"]["beta"], mean, std)

        # ── Dedup check ───────────────────────────────────────────────────────
        existing = is_duplicate(
            obj_vec, resolved_subject, resolved_predicate,
            branch_nodes, dedup_thresh
        )

        if existing:
            existing.content      = resolved_content
            existing.object_text  = resolved_object
            existing.obj_vector   = obj_vec
            existing.tfidf_vector = tf_vector
            existing.access_count += 1
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
            tfidf_vector            = tf_vector,
            agent_id                = agent_id,
            subject                 = resolved_subject,
            predicate               = resolved_predicate,
            object_text             = resolved_object,
            obj_vector              = obj_vec,
        )
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
                          embedding_fn: Optional[Callable] = None) -> Node:
        self.ensure_backbone()
        return self.add_node(
            content=content, branch_name="backbone", tags=tags,
            normalize_fn=normalize_fn, agent_id=agent_id,
            subject=subject, predicate=predicate,
            object_text=object_text, embedding_fn=embedding_fn
        )


    # ── OLS compression pipeline ──────────────────────────────────────────────

    def _check_and_compress(self, branch_name: str) -> bool:
        """
        φ-triggered branch-level OLS.
        Fires automatically after every write to a non-backbone branch.
        Neutral buoyancy gate — nodes sinking below system mean are eligible.
        Returns True if compression occurred.
        """
        branch_sizes = self.storage.get_branch_sizes()
        all_sizes    = list(branch_sizes.values())
        raw_nodes    = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes = [Node.from_dict(n) for n in raw_nodes]

        if not should_compress_branch(branch_nodes, all_sizes):
            return False

        sys_accesses  = self._system_accesses()
        all_raw       = self.storage.get_all_active_nodes()
        all_nodes     = [Node.from_dict(n) for n in all_raw]

        try:
            compressed, consumed_ids = compress_branch(
                branch_nodes, branch_name, sys_accesses, all_nodes
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
        Call at session boundaries.
        Returns list of branch names compressed.
        """
        branch_accesses = self.storage.get_branch_access_counts()
        all_counts      = [v for k, v in branch_accesses.items() if k != "backbone"]
        sys_accesses    = self._system_accesses()
        all_raw         = self.storage.get_all_active_nodes()
        all_nodes       = [Node.from_dict(n) for n in all_raw]

        compressed_branches = []

        for branch_name, access_count in branch_accesses.items():
            if branch_name == "backbone":
                continue
            if not is_cold_branch(access_count, all_counts):
                continue

            raw_nodes = self.storage.get_nodes_for_branch(branch_name)
            if not raw_nodes:
                continue

            branch_nodes = [Node.from_dict(n) for n in raw_nodes]

            try:
                compressed, consumed_ids = compress_cold_branch(
                    branch_name, branch_nodes, sys_accesses, all_nodes
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
              agent_id: Optional[str] = None) -> str | list[Node]:
        """
        Conditional injection — D3 slice.
        Scores all active nodes against prompt. Returns top-N within budget.
        """
        sys_accesses = self._system_accesses()
        raw_nodes    = self.storage.get_all_active_nodes()
        all_nodes    = [Node.from_dict(n) for n in raw_nodes]

        mean, std        = compute_distribution(all_nodes)
        retrieval_thresh = get_threshold(
            _CFG["thresholds"]["retrieval"]["beta"], mean, std
        )

        injected = conditional_injection(
            prompt, all_nodes, self._corpus_idf, sys_accesses,
            retrieval_threshold=retrieval_thresh
        )

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

        new_total = self.storage.increment_system_accesses(by=len(injected))
        for node in injected:
            record_access(node, new_total)
            self.storage.update_node_access(
                node.node_id, node.access_count, node.system_access_snapshot
            )
            self.storage.increment_branch_access(node.branch_name)

        if len(injected) > 1:
            self.storage.log_co_access([n.node_id for n in injected], agent_id=agent_id)

        if formatted:
            return format_injection(injected)
        return injected


    # ── Decay scan ────────────────────────────────────────────────────────────

    def decay_scan(self) -> dict[str, float]:
        sys_accesses = self._system_accesses()
        raw_nodes    = self.storage.get_all_active_nodes()
        scores       = {}

        for raw in raw_nodes:
            node = Node.from_dict(raw)
            if not node.is_backbone:
                scores[node.node_id] = current_relevance(node, sys_accesses)

        return scores


    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str = "living_memory_export.json") -> str:
        return self.storage.export_json(path)

    def close(self) -> None:
        self.storage.close()


    # ── Multi-agent consensus pipeline ────────────────────────────────────────

    def add_agent_node(self, content: str, branch_name: str,
                       agent_id:     str,
                       concept_key:  str,
                       tags:         Optional[list]     = None,
                       normalize_fn: Optional[Callable] = None,
                       stakes:       str                = "standard",
                       subject:      Optional[str]      = None,
                       predicate:    Optional[str]      = None,
                       object_text:  Optional[str]      = None,
                       embedding_fn: Optional[Callable] = None) -> dict:
        """
        Multi-agent write pipeline. Routes backbone content through consensus.
        Non-backbone writes directly with agent_id provenance.
        """
        tags = tags or []

        if normalize_fn is not None:
            content = normalize_fn(content)

        if branch_name != "backbone":
            node = self.add_node(
                content=content, branch_name=branch_name, tags=tags,
                agent_id=agent_id, subject=subject, predicate=predicate,
                object_text=object_text, embedding_fn=embedding_fn
            )
            return {"status": "written", "node_id": node.node_id}

        # Backbone — resolve predicate and route through consensus
        resolved_predicate = self._resolve_predicate(predicate)
        resolved_subject   = subject
        resolved_object    = object_text if object_text is not None else content
        resolved_content   = (
            f"{resolved_subject} {resolved_predicate} {resolved_object}"
            if resolved_subject is not None else resolved_object
        )

        tf_vector = compute_tf(tokenize(resolved_content))
        sys_acc   = self._system_accesses()

        pending = {
            "pending_id":   f"{concept_key}:{agent_id}",
            "concept_key":  concept_key,
            "agent_id":     agent_id,
            "content":      resolved_content,
            "branch_name":  branch_name,
            "tfidf_vector": tf_vector,
            "tags":         tags,
        }
        self.storage.write_pending(pending)

        return self._check_consensus(concept_key, branch_name, sys_acc, stakes)

    def _check_consensus(self, concept_key: str, branch_name: str,
                         sys_accesses: int, stakes: str = "standard") -> dict:
        pending_raw = self.storage.get_pending_by_concept(concept_key)

        if len(pending_raw) < 2:
            return {"status": "pending", "signal": 0.0, "threshold": None,
                    "agents": len(pending_raw)}

        pending_nodes = [
            Node(
                content      = p["content"],
                tfidf_vector = p["tfidf_vector"],
                agent_id     = p["agent_id"],
                branch_name  = branch_name,
            )
            for p in pending_raw
        ]

        signal = compute_signal_strength(pending_nodes)

        all_raw   = self.storage.get_all_active_nodes()
        all_nodes = [Node.from_dict(n) for n in all_raw]
        mean, std = compute_distribution(all_nodes)

        stakes_mult = _CFG["consensus"]["stakes_multiplier"].get(stakes, 1.0)
        threshold   = get_threshold(
            _CFG["thresholds"]["consensus"]["beta"],
            mean, std, stakes_multiplier=stakes_mult
        )

        if signal >= threshold:
            from src.domain.compression import _merge_content, _merge_vectors
            merged_content = _merge_content(pending_nodes)
            merged_vector  = _merge_vectors(pending_nodes)
            agent_ids      = [p["agent_id"] for p in pending_raw]

            node_dict = {
                "node_id":                str(uuid.uuid4()),
                "content":                merged_content,
                "branch_name":            "backbone",
                "is_backbone":            True,
                "base_score":             1.0,
                "access_count":           0,
                "system_access_snapshot": sys_accesses,
                "tags":                   list({t for p in pending_raw for t in p["tags"]}),
                "compressed_from":        [p["pending_id"] for p in pending_raw],
                "tfidf_vector":           merged_vector,
                "agent_id":               None,
                "subject":                None,
                "predicate":              DEFAULT_PREDICATE,
                "object_text":            merged_content,
                "obj_vector":             merged_vector,
            }

            self.storage.upsert_branch("backbone", is_backbone=True)
            self.storage.commit_pending_to_backbone(concept_key, node_dict)
            self._rebuild_idf()

            return {"status": "committed", "signal": signal,
                    "threshold": threshold, "agents": agent_ids}

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

            return {"status": "conflict", "signal": signal,
                    "threshold": threshold,
                    "agents": conflict["agent_ids"]}

    def pending_conflicts(self) -> list[dict]:
        return self.storage.get_pending_conflicts()

    def resolve_conflict(self, conflict_id: str,
                         resolved_content: str,
                         concept_key: str,
                         branch_name: str = "backbone") -> None:
        sys_acc   = self._system_accesses()
        tf_vector = compute_tf(tokenize(resolved_content))

        node_dict = {
            "node_id":                str(uuid.uuid4()),
            "content":                resolved_content,
            "branch_name":            "backbone",
            "is_backbone":            True,
            "base_score":             1.0,
            "access_count":           0,
            "system_access_snapshot": sys_acc,
            "tags":                   ["resolved"],
            "compressed_from":        [],
            "tfidf_vector":           tf_vector,
            "agent_id":               "resolved",
            "subject":                None,
            "predicate":              DEFAULT_PREDICATE,
            "object_text":            resolved_content,
            "obj_vector":             tf_vector,
        }

        self.storage.upsert_branch("backbone", is_backbone=True)

        with self.storage._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
            """, (
                node_dict["node_id"], node_dict["content"],
                "backbone", 1, 1.0, 0, sys_acc,
                json.dumps(["resolved"]),
                json.dumps([]),
                json.dumps(tf_vector),
                "resolved",
                None,
                DEFAULT_PREDICATE,
                resolved_content,
                json.dumps(tf_vector),
            ))
            cur.execute(
                "UPDATE conflicts SET resolved=1 WHERE conflict_id=?",
                (conflict_id,)
            )
            cur.execute(
                "DELETE FROM pending_consensus WHERE concept_key=?",
                (concept_key,)
            )

        self._rebuild_idf()

    def promotion_candidates(self) -> list[dict]:
        """Return unknown predicates at or above promotion threshold."""
        return self._check_promotion_candidates()
