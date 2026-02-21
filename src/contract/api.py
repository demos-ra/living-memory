# [F-memory/R3/C3] Public contract — all external access to LM goes through here
# R3 — Contract. Depends: R0 config, R1 domain, R2 app. No R4 imports.

from __future__ import annotations
from typing import Optional

from src.app.memory_tree import MemoryTree


class LivingMemory:
    """
    Public API for Living Memory.

    All agent interaction goes through this class.
    Instantiate once per agent session. Pass db_path for persistence.

    Example:
        lm = LivingMemory(db_path="agent.db")
        lm.remember("backbone", "User is Demos", tags=["identity"])
        lm.remember("goals", "Ship LIV by Q1")
        context = lm.recall("what are the current goals")
        lm.close()
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self._tree = MemoryTree(db_path)
        self._tree.ensure_backbone()

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(self, branch: str, content: str,
                 tags: Optional[list[str]] = None) -> dict:
        """
        Store a memory in the given branch.

        branch="backbone" — permanent, never decays, never compresses.
        Any other branch — dynamic, subject to OLS compression and decay.
        Branch is created automatically if it does not exist.

        Returns serialized node dict.
        """
        if branch == "backbone":
            node = self._tree.add_backbone_node(content, tags=tags)
        else:
            node = self._tree.add_node(content, branch, tags=tags)
        return node.to_dict()

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(self, query: str) -> str:
        """
        Retrieve memory relevant to query.

        Scores all active nodes by cosine similarity × decay buoyancy.
        Returns top-N within character budget as formatted string.
        Empty string if nothing meets retrieval threshold.
        """
        return self._tree.query(query, formatted=True)

    def recall_nodes(self, query: str) -> list[dict]:
        """
        Same as recall() but returns raw node dicts instead of formatted string.
        Useful for programmatic access to retrieved memory.
        """
        nodes = self._tree.query(query, formatted=False)
        return [n.to_dict() for n in nodes]

    # ── Maintenance ───────────────────────────────────────────────────────────

    def compress(self) -> list[str]:
        """
        Run root-level OLS on cold branches.
        Call at session boundaries (start or end).
        Returns list of branch names that were compressed.
        """
        return self._tree.run_root_ols()

    def decay_scores(self) -> dict[str, float]:
        """
        Current relevance scores for all active non-backbone nodes.
        Returns dict of node_id → relevance float in [0, 1].
        Read-only. Useful for inspection and debugging.
        """
        return self._tree.decay_scan()

    def status(self) -> dict:
        """
        Snapshot of current memory state.
        Returns branch sizes, system access count, active node count.
        """
        branch_sizes  = self._tree.storage.get_branch_sizes()
        all_branches  = self._tree.storage.get_all_branches()
        sys_accesses  = self._tree.storage.get_system_accesses()
        active_nodes  = self._tree.storage.get_all_active_nodes()

        return {
            "system_total_accesses": sys_accesses,
            "active_node_count":     len(active_nodes),
            "branches":              {
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
        """
        Export full memory tree to JSON.
        Returns path of written file.
        """
        return self._tree.export_json(path)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connection. Call at agent session end."""
        self._tree.close()

    def __enter__(self) -> "LivingMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()
