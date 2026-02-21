# [F-storage/R4/C2] SQLite persistence, atomic transactions, JSON export, archival
# R4 — Execution. Depends: R0 config only. No domain imports. Pure I/O.

from __future__ import annotations
import sqlite3
import json
import os
from contextlib import contextmanager
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG = _load_config()


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches (
    branch_name    TEXT PRIMARY KEY,
    is_backbone    INTEGER NOT NULL DEFAULT 0,
    access_count   INTEGER NOT NULL DEFAULT 0,
    created_at     INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id                TEXT PRIMARY KEY,
    content                TEXT NOT NULL,
    branch_name            TEXT NOT NULL,
    is_backbone            INTEGER NOT NULL DEFAULT 0,
    base_score             REAL NOT NULL DEFAULT 1.0,
    access_count           INTEGER NOT NULL DEFAULT 0,
    system_access_snapshot INTEGER NOT NULL DEFAULT 0,
    tags                   TEXT NOT NULL DEFAULT '[]',
    compressed_from        TEXT NOT NULL DEFAULT '[]',
    tfidf_vector           TEXT NOT NULL DEFAULT '{}',
    archived               INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (branch_name) REFERENCES branches(branch_name)
);

CREATE TABLE IF NOT EXISTS compression_log (
    log_id          TEXT PRIMARY KEY,
    compressed_node TEXT NOT NULL,
    source_ids      TEXT NOT NULL,
    branch_name     TEXT NOT NULL,
    level           TEXT NOT NULL,
    timestamp       INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_nodes_branch   ON nodes(branch_name);
CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived);
CREATE INDEX IF NOT EXISTS idx_nodes_backbone ON nodes(is_backbone);
"""


# ── Connection ────────────────────────────────────────────────────────────────

class Storage:
    """
    All SQLite I/O lives here. Every write operation is atomic.
    R4 — pure I/O. Receives and returns plain dicts, not Node objects.
    Node objects live in R1. Storage works with serialized dicts only.
    """

    def __init__(self, db_path: str = "living_memory.db"):
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._transaction() as cur:
            cur.executescript(SCHEMA)
            # Init system access counter if not present
            cur.execute(
                "INSERT OR IGNORE INTO meta VALUES ('system_total_accesses', '0')"
            )

    @contextmanager
    def _transaction(self):
        """
        Atomic transaction context manager.
        All writes inside this block commit together or rollback entirely.
        This is the atomicity guarantee for every write operation.
        """
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


    # ── Meta ──────────────────────────────────────────────────────────────────

    def get_system_accesses(self) -> int:
        cur = self._conn.execute("SELECT value FROM meta WHERE key='system_total_accesses'")
        row = cur.fetchone()
        return int(row["value"]) if row else 0

    def increment_system_accesses(self, by: int = 1) -> int:
        """Atomic increment — read-modify-write in single transaction."""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) "
                "WHERE key = 'system_total_accesses'",
                (by,)
            )
        return self.get_system_accesses()


    # ── Branches ──────────────────────────────────────────────────────────────

    def upsert_branch(self, branch_name: str, is_backbone: bool = False) -> None:
        with self._transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO branches (branch_name, is_backbone) VALUES (?, ?)",
                (branch_name, int(is_backbone))
            )

    def increment_branch_access(self, branch_name: str) -> None:
        with self._transaction() as cur:
            cur.execute(
                "UPDATE branches SET access_count = access_count + 1 WHERE branch_name = ?",
                (branch_name,)
            )

    def get_all_branches(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM branches")
        return [dict(row) for row in cur.fetchall()]

    def get_branch_access_counts(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT branch_name, access_count FROM branches")
        return {row["branch_name"]: row["access_count"] for row in cur.fetchall()}


    # ── Nodes — reads ─────────────────────────────────────────────────────────

    def get_nodes_for_branch(self, branch_name: str,
                             include_archived: bool = False) -> list[dict]:
        query = "SELECT * FROM nodes WHERE branch_name = ?"
        if not include_archived:
            query += " AND archived = 0"
        cur = self._conn.execute(query, (branch_name,))
        return [self._deserialize_node(dict(row)) for row in cur.fetchall()]

    def get_all_active_nodes(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM nodes WHERE archived = 0")
        return [self._deserialize_node(dict(row)) for row in cur.fetchall()]

    def get_backbone_nodes(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM nodes WHERE is_backbone = 1 AND archived = 0")
        return [self._deserialize_node(dict(row)) for row in cur.fetchall()]

    def get_branch_sizes(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT branch_name, COUNT(*) as n FROM nodes "
            "WHERE archived = 0 AND is_backbone = 0 GROUP BY branch_name"
        )
        return {row["branch_name"]: row["n"] for row in cur.fetchall()}


    # ── Nodes — writes (all atomic) ───────────────────────────────────────────

    def insert_node(self, node_dict: dict) -> None:
        """Insert a new node. Atomic."""
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived)
                VALUES (?,?,?,?,?,?,?,?,?,?,0)
            """, (
                node_dict["node_id"],
                node_dict["content"],
                node_dict["branch_name"],
                int(node_dict["is_backbone"]),
                node_dict["base_score"],
                node_dict["access_count"],
                node_dict["system_access_snapshot"],
                json.dumps(node_dict["tags"]),
                json.dumps(node_dict["compressed_from"]),
                json.dumps(node_dict["tfidf_vector"]),
            ))

    def update_node(self, node_dict: dict) -> None:
        """Update existing node fields. Atomic."""
        with self._transaction() as cur:
            cur.execute("""
                UPDATE nodes SET
                    content                = ?,
                    base_score             = ?,
                    access_count           = ?,
                    system_access_snapshot = ?,
                    tags                   = ?,
                    tfidf_vector           = ?
                WHERE node_id = ?
            """, (
                node_dict["content"],
                node_dict["base_score"],
                node_dict["access_count"],
                node_dict["system_access_snapshot"],
                json.dumps(node_dict["tags"]),
                json.dumps(node_dict["tfidf_vector"]),
                node_dict["node_id"],
            ))

    def update_node_access(self, node_id: str,
                           access_count: int,
                           system_snapshot: int) -> None:
        """Lightweight access bump — called on every retrieval. Atomic."""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE nodes SET access_count=?, system_access_snapshot=? WHERE node_id=?",
                (access_count, system_snapshot, node_id)
            )

    def compress_nodes_atomic(self,
                               consumed_ids: list[str],
                               compressed_node_dict: dict,
                               log_entry: dict) -> None:
        """
        OLS compression is a single atomic transaction:
          1. Insert compressed node
          2. Mark source nodes archived (never delete — audit trail)
          3. Write compression log entry
        All three succeed or all three rollback.
        """
        with self._transaction() as cur:
            # 1. Insert compressed node
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived)
                VALUES (?,?,?,?,?,?,?,?,?,?,0)
            """, (
                compressed_node_dict["node_id"],
                compressed_node_dict["content"],
                compressed_node_dict["branch_name"],
                int(compressed_node_dict["is_backbone"]),
                compressed_node_dict["base_score"],
                compressed_node_dict["access_count"],
                compressed_node_dict["system_access_snapshot"],
                json.dumps(compressed_node_dict["tags"]),
                json.dumps(compressed_node_dict["compressed_from"]),
                json.dumps(compressed_node_dict["tfidf_vector"]),
            ))

            # 2. Archive source nodes — never hard delete
            cur.execute(
                f"UPDATE nodes SET archived=1 WHERE node_id IN "
                f"({','.join('?'*len(consumed_ids))})",
                consumed_ids
            )

            # 3. Log compression event
            cur.execute("""
                INSERT INTO compression_log
                    (log_id, compressed_node, source_ids, branch_name, level)
                VALUES (?,?,?,?,?)
            """, (
                log_entry["log_id"],
                log_entry["compressed_node"],
                json.dumps(log_entry["source_ids"]),
                log_entry["branch_name"],
                log_entry["level"],
            ))

    def archive_branch_nodes(self, branch_name: str) -> None:
        """Mark all nodes in a branch as archived. Atomic."""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE nodes SET archived=1 WHERE branch_name=? AND archived=0",
                (branch_name,)
            )


    # ── Serialization helpers ─────────────────────────────────────────────────

    @staticmethod
    def _deserialize_node(row: dict) -> dict:
        """JSON fields come back as strings from SQLite — parse them."""
        for field in ("tags", "compressed_from", "tfidf_vector"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
        row["is_backbone"] = bool(row["is_backbone"])
        return row


    # ── JSON export ───────────────────────────────────────────────────────────

    def export_json(self, path: str = "living_memory_export.json") -> str:
        """
        Full tree snapshot as human-readable JSON.
        Point-in-time read — not transactional with writes,
        but consistent within the read (single connection, WAL mode).
        """
        branches   = self.get_all_branches()
        all_nodes  = self.get_all_active_nodes()
        cur        = self._conn.execute("SELECT * FROM compression_log ORDER BY timestamp")
        comp_log   = [dict(r) for r in cur.fetchall()]
        meta_cur   = self._conn.execute("SELECT * FROM meta")
        meta       = {r["key"]: r["value"] for r in meta_cur.fetchall()}

        # Group nodes by branch
        by_branch: dict[str, list] = {}
        for node in all_nodes:
            bn = node["branch_name"]
            by_branch.setdefault(bn, []).append(node)

        export = {
            "meta":             meta,
            "branches":         branches,
            "nodes_by_branch":  by_branch,
            "compression_log":  comp_log,
        }

        with open(path, "w") as f:
            json.dump(export, f, indent=2)

        return path

    def close(self) -> None:
        self._conn.close()
