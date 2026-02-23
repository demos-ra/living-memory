# [F-memory/R3/C3] Public contract — all external access to LM goes through here
# R3 — Contract. Depends: R0 config, R2 app only. No R4 imports.
# v0.5 — classify_predicate surface, predicate table access, axes parameter,
#         resolve_conflict with triple structure, promotion_candidates removed

from __future__ import annotations
from typing import Optional, Callable

from src.app.memory_tree import MemoryTree


class LivingMemory:
    """
    Living Memory — persistent, context-aware agent memory.

    Every memory is a triple: subject + predicate + object_text.
    Predicates are physics — they determine how memory decays, compresses,
    and resolves conflict. The predicate table is self-managing: unknown
    predicates are classified by axis signature, synonyms are mapped,
    gaps are inserted automatically.

    Free-text fallback is always available — no migration required from v0.3/v0.4.

    Single-agent:
        lm = LivingMemory(db_path="agent.db")
        lm.remember("goals", subject="user", predicate="targets",
                    object="ship LM v1")
        context = lm.recall("current goals")
        lm.compress()
        lm.close()

    Multi-agent:
        r = lm.contribute("backbone", subject="design", predicate="requires",
                          object="self-organization",
                          agent_id="grok", concept_key="design:requires")
        # r["status"]: "committed" | "pending" | "conflict" | "written"

    Free-text (v0.3/v0.4 compat — unchanged):
        lm.remember("threads", "Open: benchmark embeddings vs TF")
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self._tree = MemoryTree(db_path)
        self._tree.ensure_backbone()


    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(self, branch: str,
                 content:      Optional[str]       = None,
                 tags:         Optional[list[str]]  = None,
                 normalize_fn: Optional[Callable]   = None,
                 agent_id:     Optional[str]        = None,
                 subject:      Optional[str]        = None,
                 predicate:    Optional[str]        = None,
                 object:       Optional[str]        = None,
                 embedding_fn: Optional[Callable]   = None,
                 axes:         Optional[dict]       = None) -> dict:
        """
        Store a memory.

        Structured triple:
            lm.remember("goals", subject="user", predicate="targets",
                        object="ship LM v1")

        Free-text fallback (v0.3/v0.4 compat):
            lm.remember("goals", "Ship LM v1")

        branch="backbone" — permanent, never decays, never compresses.

        axes — optional dict for unknown predicate classification:
            {"polarity": "positive", "temporality": "transient",
             "directionality": "self", "certainty": "belief", "agency": "active"}
        Unknown predicates without axes fall back to "knows" — memory is never lost.

        embedding_fn — optional callable(str) -> dict[str,float] for object vectorization.
        normalize_fn — optional callable(str) -> str for semantic canonicalization.

        Returns serialized node dict.
        """
        if branch == "backbone":
            node = self._tree.add_backbone_node(
                content=content, tags=tags, normalize_fn=normalize_fn,
                agent_id=agent_id, subject=subject, predicate=predicate,
                object_text=object, embedding_fn=embedding_fn, axes=axes,
            )
        else:
            node = self._tree.add_node(
                content=content, branch_name=branch, tags=tags,
                normalize_fn=normalize_fn, agent_id=agent_id,
                subject=subject, predicate=predicate,
                object_text=object, embedding_fn=embedding_fn, axes=axes,
            )
        return node.to_dict()


    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(self, query: str,
               agent_id: Optional[str] = None) -> str:
        """
        Retrieve memory relevant to query.

        Returns summary-grouped context string within character budget.
        Format: "[branch] subject predicate: object1 | object2"
        Empty string if nothing meets retrieval threshold.
        """
        return self._tree.query(query, formatted=True, agent_id=agent_id)

    def recall_nodes(self, query: str,
                     agent_id: Optional[str] = None) -> list[dict]:
        """Same as recall() but returns raw node dicts for programmatic use."""
        nodes = self._tree.query(query, formatted=False, agent_id=agent_id)
        return [n.to_dict() for n in nodes]


    # ── Multi-agent ───────────────────────────────────────────────────────────

    def contribute(self, branch: str,
                   content:      Optional[str]      = None,
                   agent_id:     str                = None,
                   concept_key:  str                = None,
                   tags:         Optional[list[str]] = None,
                   normalize_fn: Optional[Callable]  = None,
                   stakes:       str                 = "standard",
                   subject:      Optional[str]       = None,
                   predicate:    Optional[str]       = None,
                   object:       Optional[str]       = None,
                   embedding_fn: Optional[Callable]  = None,
                   axes:         Optional[dict]      = None) -> dict:
        """
        Multi-agent write. Routes backbone content through consensus pipeline.

        concept_key — groups agent versions of the same concept for signal comparison.
                      Convention for structured writes: "subject:predicate".
        stakes      — "high" or "standard". Scales consensus threshold.
                      High stakes = harder to commit, safer for critical backbone facts.

        Returns status dict:
            {"status": "committed", "signal": float, "predicate": str, ...}
            {"status": "pending",   "signal": 0.0, "agents": int}
            {"status": "conflict",  "signal": float, "reason": str, ...}
            {"status": "written",   "node_id": str}  ← non-backbone branches
        """
        return self._tree.add_agent_node(
            content=content, branch_name=branch,
            agent_id=agent_id, concept_key=concept_key,
            tags=tags, normalize_fn=normalize_fn, stakes=stakes,
            subject=subject, predicate=predicate,
            object_text=object, embedding_fn=embedding_fn, axes=axes,
        )

    def pending_conflicts(self) -> list[dict]:
        """Return all unresolved conflicts awaiting external resolution."""
        return self._tree.pending_conflicts()

    def resolve_conflict(self, conflict_id: str,
                         resolved_content: str,
                         concept_key:      str,
                         subject:          Optional[str] = None,
                         predicate:        Optional[str] = None,
                         object:           Optional[str] = None,
                         axes:             Optional[dict] = None) -> None:
        """
        Write externally resolved content to backbone. Atomic.

        Provide subject+predicate+object to preserve triple structure.
        If omitted, resolved_content written as free-text — structure lost.
        """
        self._tree.resolve_conflict(
            conflict_id=conflict_id,
            resolved_content=resolved_content,
            concept_key=concept_key,
            subject=subject,
            predicate=predicate,
            object_text=object,
            axes=axes,
        )


    # ── Predicate vocabulary ──────────────────────────────────────────────────

    def classify_predicate(self, predicate: str,
                            axes: Optional[dict] = None) -> dict:
        """
        Classify a predicate against the vocabulary table.

        Returns:
            {"action": "known",     "predicate": str}
            {"action": "synonym",   "mapped_to": str, "original": str}
            {"action": "inserted",  "predicate": str, "props": dict}
            {"action": "quarantine","predicate": str, "reason": str}

        axes — five axis values for classification:
            polarity:       positive | negative | neutral
            temporality:    permanent | semi-permanent | transient
            directionality: self | relational
            certainty:      fact | belief
            agency:         active | passive
        """
        return self._tree.classify_predicate(predicate, axes)

    def predicates(self) -> list[dict]:
        """
        Return full predicate table.
        Includes seed predicates and any discovered predicates.
        Each row includes all axis properties and derived physics.
        """
        return self._tree.predicates()

    def predicate_synonyms(self) -> list[dict]:
        """Return all logged synonym mappings — unknown predicate → mapped term."""
        return self._tree.predicate_synonyms()


    # ── Maintenance ───────────────────────────────────────────────────────────

    def compress(self) -> list[str]:
        """
        Run root-level OLS on cold branches.
        Cold = mean node relevance below retrieval threshold.
        Call at session boundaries.
        Returns list of branch names compressed.
        """
        return self._tree.run_root_ols()

    def decay_scores(self) -> dict[str, float]:
        """
        Current predicate-aware relevance scores for all active non-backbone nodes.
        Score = base_score × e^(−λ_base × decay_multiplier × inactivity_ratio)
        """
        return self._tree.decay_scan()

    def status(self) -> dict:
        """Snapshot of current memory state."""
        branch_sizes = self._tree.storage.get_branch_sizes()
        all_branches = self._tree.storage.get_all_branches()
        sys_accesses = self._tree.storage.get_system_accesses()
        active_nodes = self._tree.storage.get_all_active_nodes()

        return {
            "system_total_accesses": sys_accesses,
            "active_node_count":     len(active_nodes),
            "predicate_count":       len(self._tree._predicate_map),
            "bootstrap_prior":       self._tree._bootstrap_prior,
            "branches": {
                b["branch_name"]: {
                    "node_count":   branch_sizes.get(b["branch_name"], 0),
                    "access_count": b["access_count"],
                    "is_backbone":  bool(b["is_backbone"]),
                }
                for b in all_branches
            },
        }


    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, path: str = "living_memory_export.json") -> str:
        """Export full memory tree to JSON. Returns path of written file."""
        return self._tree.export_json(path)


    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connection. Call at agent session end."""
        self._tree.close()

    def __enter__(self) -> "LivingMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()
