# [F-memory/R3/C4] Public contract — all external access to LM goes through here
# R3 — Contract. Depends: R0 config, R1 domain, R2 app. No R4 imports.
# v0.4 — structured triple write surface, embedding_fn hook, promotion_candidates

from __future__ import annotations
from typing import Optional, Callable

from src.app.memory_tree import MemoryTree


class LivingMemory:
    """
    Public API for Living Memory.

    All agent interaction goes through this class.
    Instantiate once per agent session. Pass db_path for persistence.

    Single-agent free-text (v0.3 compat — unchanged):
        lm = LivingMemory(db_path="agent.db")
        lm.remember("backbone", "User is Demos", tags=["identity"])
        lm.remember("goals", "Ship LIV by Q1")
        context = lm.recall("what are the current goals")
        lm.close()

    Structured triple (v0.4):
        lm.remember("goals", subject="user", predicate="targets",
                    object="ship LM v1 as open source library")
        lm.remember("prefs", subject="user", predicate="prefers",
                    object="dark mode", embedding_fn=my_embed_fn)

    Unknown predicates are logged, not rejected. Check candidates:
        lm.promotion_candidates()
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self._tree = MemoryTree(db_path)
        self._tree.ensure_backbone()


    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(self, branch: str,
                 content:      Optional[str]      = None,
                 tags:         Optional[list[str]] = None,
                 normalize_fn: Optional[Callable]  = None,
                 agent_id:     Optional[str]       = None,
                 subject:      Optional[str]       = None,
                 predicate:    Optional[str]       = None,
                 object:       Optional[str]       = None,
                 embedding_fn: Optional[Callable]  = None) -> dict:
        """
        Store a memory in the given branch.

        Free-text (v0.3 compat — always works):
            lm.remember("goals", "Ship LM v1")

        Structured triple (v0.4):
            lm.remember("goals", subject="user", predicate="targets",
                        object="ship LM v1")

        branch="backbone" — permanent, never decays, never compresses.
        Unknown predicates are logged not rejected — memory is never lost.
        embedding_fn — optional callable(str) -> dict[str, float] for object vectorization.

        Returns serialized node dict.
        """
        if branch == "backbone":
            node = self._tree.add_backbone_node(
                content=content, tags=tags, normalize_fn=normalize_fn,
                agent_id=agent_id, subject=subject, predicate=predicate,
                object_text=object, embedding_fn=embedding_fn
            )
        else:
            node = self._tree.add_node(
                content=content, branch_name=branch, tags=tags,
                normalize_fn=normalize_fn, agent_id=agent_id,
                subject=subject, predicate=predicate,
                object_text=object, embedding_fn=embedding_fn
            )
        return node.to_dict()


    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(self, query: str, agent_id: Optional[str] = None) -> str:
        """
        Retrieve memory relevant to query.
        Returns top-N within character budget as formatted string.
        Empty string if nothing meets retrieval threshold.
        """
        return self._tree.query(query, formatted=True, agent_id=agent_id)

    def recall_nodes(self, query: str,
                     agent_id: Optional[str] = None) -> list[dict]:
        """Same as recall() but returns raw node dicts."""
        nodes = self._tree.query(query, formatted=False, agent_id=agent_id)
        return [n.to_dict() for n in nodes]


    # ── Multi-agent ───────────────────────────────────────────────────────────

    def contribute(self, branch: str, content: str,
                   agent_id:     str,
                   concept_key:  str,
                   tags:         Optional[list[str]] = None,
                   normalize_fn: Optional[Callable]  = None,
                   stakes:       str                 = "standard",
                   subject:      Optional[str]       = None,
                   predicate:    Optional[str]       = None,
                   object:       Optional[str]       = None,
                   embedding_fn: Optional[Callable]  = None) -> dict:
        """
        Multi-agent write. Routes backbone content through consensus pipeline.

        concept_key — groups agent versions of same concept for comparison.
                      For structured writes, convention is "subject:predicate".
        stakes      — 'high' or 'standard'. Scales consensus threshold.

        Returns status dict: committed | pending | conflict | written
        """
        return self._tree.add_agent_node(
            content=content, branch_name=branch,
            agent_id=agent_id, concept_key=concept_key,
            tags=tags, normalize_fn=normalize_fn, stakes=stakes,
            subject=subject, predicate=predicate,
            object_text=object, embedding_fn=embedding_fn
        )

    def pending_conflicts(self) -> list[dict]:
        """Return all unresolved conflicts for external resolution."""
        return self._tree.pending_conflicts()

    def resolve_conflict(self, conflict_id: str,
                         resolved_content: str,
                         concept_key: str) -> None:
        """Write externally resolved content to backbone. Atomic."""
        self._tree.resolve_conflict(conflict_id, resolved_content, concept_key)


    # ── Predicate vocabulary ──────────────────────────────────────────────────

    def promotion_candidates(self) -> list[dict]:
        """
        Return unknown predicates that have reached promotion threshold.
        Each entry: {predicate, count, first_seen, last_seen}.
        Review and promote manually via vocabulary version increment in config.
        """
        return self._tree.promotion_candidates()

    def unknown_predicates(self) -> list[dict]:
        """Return all logged unknown predicates ordered by count."""
        return self._tree.storage.get_unknown_predicates()


    # ── Maintenance ───────────────────────────────────────────────────────────

    def compress(self) -> list[str]:
        """Run root-level OLS on cold branches. Call at session boundaries."""
        return self._tree.run_root_ols()

    def decay_scores(self) -> dict[str, float]:
        """Current relevance scores for all active non-backbone nodes."""
        return self._tree.decay_scan()

    def status(self) -> dict:
        """Snapshot of current memory state."""
        branch_sizes  = self._tree.storage.get_branch_sizes()
        all_branches  = self._tree.storage.get_all_branches()
        sys_accesses  = self._tree.storage.get_system_accesses()
        active_nodes  = self._tree.storage.get_all_active_nodes()

        return {
            "system_total_accesses": sys_accesses,
            "active_node_count":     len(active_nodes),
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
