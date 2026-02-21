# [F-memory/R2/C2] Memory tree orchestration — backbone, branches, OLS triggers
# R2 — Application. Depends: R0 config, R1 domain, R4 storage. No contract imports.

from __future__ import annotations
import uuid
import json
import os
from typing import Optional

from src.domain.nodes import (
    Node, compute_decay, is_duplicate, record_access, current_relevance
)
from src.domain.retrieval import (
    compute_tfidf, compute_tf, tokenize, update_corpus_idf,
    conditional_injection, format_injection, build_query_vector
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
        """Recompute corpus IDF from all active nodes. Called after mutations."""
        raw_nodes  = self.storage.get_all_active_nodes()
        nodes      = [Node.from_dict(n) for n in raw_nodes]
        self._corpus_idf = update_corpus_idf(nodes)

    def _system_accesses(self) -> int:
        return self.storage.get_system_accesses()


    # ── Branch management ─────────────────────────────────────────────────────

    def add_branch(self, branch_name: str) -> None:
        """
        Dynamically create a new memory branch (D2 manifold).
        Branches are runtime objects — never hardcoded.
        """
        self.storage.upsert_branch(branch_name, is_backbone=False)

    def ensure_backbone(self) -> None:
        """Backbone is permanent — created once, never removed."""
        self.storage.upsert_branch("backbone", is_backbone=True)


    # ── Write ─────────────────────────────────────────────────────────────────

    def add_node(self, content: str, branch_name: str,
                 tags: Optional[list[str]] = None) -> Node:
        """
        Add a node to a branch. Full write pipeline:

        1. Ensure branch exists
        2. Vectorize content (TF-IDF against current corpus)
        3. Dedup check — if near-duplicate exists (≥0.85), update it
        4. Otherwise insert new node
        5. Rebuild IDF corpus
        6. Trigger OLS check — compress if φ threshold crossed
        7. Return the node (new or updated)
        """
        tags = tags or []
        self.storage.upsert_branch(branch_name, is_backbone=(branch_name == "backbone"))

        # Raw TF vector — corpus-independent, used for dedup comparison only.
        # TF is stable across IDF rebuilds. Using TF-IDF for dedup breaks when
        # corpus is small (IDF collapses to 0 for all terms in a 1-doc corpus).
        tf_vector    = compute_tf(tokenize(content))
        # TF-IDF vector — corpus-dependent, used for retrieval ranking and storage.
        tfidf_vector = compute_tfidf(content, self._corpus_idf)
        sys_accesses = self._system_accesses()

        # Load existing branch nodes for dedup check
        raw_branch   = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes = [Node.from_dict(n) for n in raw_branch]

        # Dedup — compare incoming TF against stored TF vectors.
        # Stored nodes carry tf_vector in a parallel field populated at insert.
        # Fall back to tfidf_vector if tf_vector not yet stored (legacy nodes).
        def _tf_vec(node: Node) -> dict:
            return node.tfidf_vector  # stored as TF on first insert (see below)

        existing = is_duplicate(tf_vector, branch_nodes)
        if existing:
            existing.content      = content
            # Keep stored tfidf_vector as TF (identity vector) — do NOT overwrite
            # with the new TF-IDF which is corpus-dependent and unstable for dedup.
            # Update it to the new content's TF so future dedup checks stay accurate.
            existing.tfidf_vector = tf_vector
            existing.access_count += 1
            existing.system_access_snapshot = sys_accesses
            for tag in tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
            self.storage.update_node(existing.to_dict())
            self._rebuild_idf()
            return existing

        # New node — store TF vector (not TF-IDF) so future dedup comparisons
        # are corpus-independent and stable. tfidf_vector is updated on _rebuild_idf.
        is_backbone = (branch_name == "backbone")
        node = Node(
            content                 = content,
            branch_name             = branch_name,
            is_backbone             = is_backbone,
            base_score              = 1.0,
            access_count            = 0,
            system_access_snapshot  = sys_accesses,
            tags                    = tags,
            tfidf_vector            = tf_vector,  # store TF — stable identity vector
        )
        self.storage.insert_node(node.to_dict())
        self._rebuild_idf()

        # OLS trigger — only on non-backbone branches
        if not is_backbone:
            self._check_and_compress(branch_name)

        return node

    def add_backbone_node(self, content: str,
                          tags: Optional[list[str]] = None) -> Node:
        """Convenience — add permanent node to backbone. Never decays."""
        self.ensure_backbone()
        return self.add_node(content, "backbone", tags=tags)


    # ── OLS compression pipeline ──────────────────────────────────────────────

    def _check_and_compress(self, branch_name: str) -> bool:
        """
        φ-triggered branch-level OLS.
        Fires automatically after every write to a non-backbone branch.
        Returns True if compression occurred.
        """
        branch_sizes  = self.storage.get_branch_sizes()
        all_sizes     = list(branch_sizes.values())
        raw_nodes     = self.storage.get_nodes_for_branch(branch_name)
        branch_nodes  = [Node.from_dict(n) for n in raw_nodes]

        if not should_compress_branch(branch_nodes, all_sizes):
            return False

        sys_accesses = self._system_accesses()
        compressed, consumed_ids = compress_branch(branch_nodes, branch_name, sys_accesses)

        log_entry = {
            "log_id":          str(uuid.uuid4()),
            "compressed_node": compressed.node_id,
            "source_ids":      consumed_ids,
            "branch_name":     branch_name,
            "level":           "branch",
        }

        # Atomic: insert compressed, archive sources, log — all or nothing
        self.storage.compress_nodes_atomic(consumed_ids, compressed.to_dict(), log_entry)
        self._rebuild_idf()
        return True

    def run_root_ols(self) -> list[str]:
        """
        Root-level OLS — compress cold branches into archive nodes.
        Call periodically (e.g. on agent session start/end).
        Returns list of branch names that were root-compressed.
        """
        branch_accesses = self.storage.get_branch_access_counts()
        all_counts      = [v for k, v in branch_accesses.items() if k != "backbone"]
        sys_accesses    = self._system_accesses()
        compressed_branches = []

        for branch_name, access_count in branch_accesses.items():
            if branch_name == "backbone":
                continue
            if not is_cold_branch(access_count, all_counts):
                continue

            raw_nodes = self.storage.get_nodes_for_branch(branch_name)
            if len(raw_nodes) < 3:  # below min_nodes_to_compress — nothing to merge
                continue

            branch_nodes = [Node.from_dict(n) for n in raw_nodes]
            compressed, consumed_ids = compress_cold_branch(
                branch_name, branch_nodes, sys_accesses
            )

            log_entry = {
                "log_id":          str(uuid.uuid4()),
                "compressed_node": compressed.node_id,
                "source_ids":      consumed_ids,
                "branch_name":     branch_name,
                "level":           "root",
            }

            # Atomic: insert archive node, archive all branch nodes, log
            self.storage.compress_nodes_atomic(consumed_ids, compressed.to_dict(), log_entry)
            compressed_branches.append(branch_name)

        if compressed_branches:
            self._rebuild_idf()

        return compressed_branches


    # ── Query / retrieval ─────────────────────────────────────────────────────

    def query(self, prompt: str, formatted: bool = True) -> str | list[Node]:
        """
        Conditional injection — D3 slice.

        Scores all active nodes against prompt context.
        Returns top-N within budget cap.
        Bumps access counts atomically on retrieved nodes.

        Args:
            prompt:    current agent query / context
            formatted: if True returns injection string, else raw Node list
        """
        sys_accesses = self._system_accesses()
        raw_nodes    = self.storage.get_all_active_nodes()
        all_nodes    = [Node.from_dict(n) for n in raw_nodes]

        injected = conditional_injection(
            prompt, all_nodes, self._corpus_idf, sys_accesses
        )

        # Atomic access bumps — each node retrieved gets buoyancy restored
        new_total = self.storage.increment_system_accesses(by=len(injected))
        for node in injected:
            record_access(node, new_total)
            self.storage.update_node_access(
                node.node_id, node.access_count, node.system_access_snapshot
            )
            self.storage.increment_branch_access(node.branch_name)

        if formatted:
            return format_injection(injected)
        return injected


    # ── Decay scan ────────────────────────────────────────────────────────────

    def decay_scan(self) -> dict[str, float]:
        """
        Compute current relevance for all active non-backbone nodes.
        Returns dict of node_id → current_relevance score.
        Read-only — decay is computed on demand, not stored.
        Stored relevance would require constant writes; computed relevance is free.
        """
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
