# [F-storage/R4/C4] SQLite persistence, atomic transactions, JSON export, archival
# R4 — Execution. Depends: R0 config only. No domain imports. Pure I/O.
# v0.4 — triple fields on nodes, unknown_predicates table

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
    agent_id               TEXT,
    subject                TEXT,
    predicate              TEXT NOT NULL DEFAULT 'knows',
    object_text            TEXT NOT NULL DEFAULT '',
    obj_vector             TEXT NOT NULL DEFAULT '{}',
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

CREATE INDEX IF NOT EXISTS idx_nodes_branch     ON nodes(branch_name);
CREATE INDEX IF NOT EXISTS idx_nodes_archived   ON nodes(archived);
CREATE INDEX IF NOT EXISTS idx_nodes_backbone   ON nodes(is_backbone);

CREATE TABLE IF NOT EXISTS co_access_log (
    node_a    TEXT NOT NULL,
    node_b    TEXT NOT NULL,
    agent_id  TEXT,
    count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (node_a, node_b, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_co_access_a     ON co_access_log(node_a);
CREATE INDEX IF NOT EXISTS idx_co_access_b     ON co_access_log(node_b);
CREATE INDEX IF NOT EXISTS idx_co_access_agent ON co_access_log(agent_id);

CREATE TABLE IF NOT EXISTS pending_consensus (
    pending_id   TEXT PRIMARY KEY,
    concept_key  TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    content      TEXT NOT NULL,
    branch_name  TEXT NOT NULL,
    tfidf_vector TEXT NOT NULL DEFAULT '{}',
    tags         TEXT NOT NULL DEFAULT '[]',
    created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_pending_concept ON pending_consensus(concept_key);
CREATE INDEX IF NOT EXISTS idx_pending_agent   ON pending_consensus(agent_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id  TEXT PRIMARY KEY,
    concept_key  TEXT NOT NULL,
    branch_name  TEXT NOT NULL,
    agent_ids    TEXT NOT NULL,
    signal       REAL NOT NULL,
    threshold    REAL NOT NULL,
    resolved     INTEGER NOT NULL DEFAULT 0,
    created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(resolved);

CREATE TABLE IF NOT EXISTS unknown_predicates (
    predicate  TEXT PRIMARY KEY,
    count      INTEGER NOT NULL DEFAULT 1,
    first_seen INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    last_seen  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
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
            cur.execute(
                "INSERT OR IGNORE INTO meta VALUES ('system_total_accesses', '0')"
            )
        # Migrate existing databases — add new columns if absent
        self._migrate()

    def _migrate(self) -> None:
        """
        Safe migration for existing v0.3 databases.
        Adds new columns if they don't exist. Idempotent.
        """
        cur = self._conn.execute("PRAGMA table_info(nodes)")
        existing_columns = {row["name"] for row in cur.fetchall()}

        migrations = []
        if "subject"     not in existing_columns:
            migrations.append("ALTER TABLE nodes ADD COLUMN subject TEXT")
        if "predicate"   not in existing_columns:
            migrations.append("ALTER TABLE nodes ADD COLUMN predicate TEXT NOT NULL DEFAULT 'knows'")
        if "object_text" not in existing_columns:
            migrations.append("ALTER TABLE nodes ADD COLUMN object_text TEXT NOT NULL DEFAULT ''")
        if "obj_vector"  not in existing_columns:
            migrations.append("ALTER TABLE nodes ADD COLUMN obj_vector TEXT NOT NULL DEFAULT '{}'")

        if migrations:
            with self._transaction() as cur:
                for sql in migrations:
                    cur.execute(sql)

        # Ensure indexes on new columns exist — safe to run after migration
        with self._transaction() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_predicate ON nodes(predicate)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_subject   ON nodes(subject)")

    @contextmanager
    def _transaction(self):
        """
        Atomic transaction context manager.
        All writes inside this block commit together or rollback entirely.
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
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
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
                node_dict.get("agent_id"),
                node_dict.get("subject"),
                node_dict.get("predicate", "knows"),
                node_dict.get("object_text", ""),
                json.dumps(node_dict.get("obj_vector", {})),
            ))

    def update_node(self, node_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                UPDATE nodes SET
                    content                = ?,
                    base_score             = ?,
                    access_count           = ?,
                    system_access_snapshot = ?,
                    tags                   = ?,
                    tfidf_vector           = ?,
                    object_text            = ?,
                    obj_vector             = ?
                WHERE node_id = ?
            """, (
                node_dict["content"],
                node_dict["base_score"],
                node_dict["access_count"],
                node_dict["system_access_snapshot"],
                json.dumps(node_dict["tags"]),
                json.dumps(node_dict["tfidf_vector"]),
                node_dict.get("object_text", ""),
                json.dumps(node_dict.get("obj_vector", {})),
                node_dict["node_id"],
            ))

    def update_node_access(self, node_id: str,
                           access_count: int,
                           system_snapshot: int) -> None:
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
        OLS compression — single atomic transaction:
          1. Insert compressed node
          2. Archive source nodes (never delete — audit trail)
          3. Write compression log entry
        """
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
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
                compressed_node_dict.get("agent_id"),
                compressed_node_dict.get("subject"),
                compressed_node_dict.get("predicate", "knows"),
                compressed_node_dict.get("object_text", ""),
                json.dumps(compressed_node_dict.get("obj_vector", {})),
            ))

            cur.execute(
                f"UPDATE nodes SET archived=1 WHERE node_id IN "
                f"({','.join('?'*len(consumed_ids))})",
                consumed_ids
            )

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
        with self._transaction() as cur:
            cur.execute(
                "UPDATE nodes SET archived=1 WHERE branch_name=? AND archived=0",
                (branch_name,)
            )


    # ── Serialization helpers ─────────────────────────────────────────────────

    @staticmethod
    def _deserialize_node(row: dict) -> dict:
        """JSON fields come back as strings from SQLite — parse them."""
        for field in ("tags", "compressed_from", "tfidf_vector", "obj_vector"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
            elif row.get(field) is None:
                row[field] = {}
        row["is_backbone"] = bool(row["is_backbone"])
        row.setdefault("agent_id",    None)
        row.setdefault("subject",     None)
        row.setdefault("predicate",   "knows")
        row.setdefault("object_text", "")
        row.setdefault("obj_vector",  {})
        return row


    # ── Unknown predicates ────────────────────────────────────────────────────

    def log_unknown_predicate(self, predicate: str) -> int:
        """
        Log an unrecognized predicate attempt. Upsert — atomic.
        Returns current count after increment.
        """
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO unknown_predicates (predicate, count)
                VALUES (?, 1)
                ON CONFLICT(predicate) DO UPDATE SET
                    count     = count + 1,
                    last_seen = strftime('%s','now')
            """, (predicate,))
        cur2 = self._conn.execute(
            "SELECT count FROM unknown_predicates WHERE predicate = ?", (predicate,)
        )
        row = cur2.fetchone()
        return row["count"] if row else 1

    def get_unknown_predicates(self) -> list[dict]:
        """Return all logged unknown predicates ordered by count descending."""
        cur = self._conn.execute(
            "SELECT * FROM unknown_predicates ORDER BY count DESC"
        )
        return [dict(row) for row in cur.fetchall()]

    def get_promotion_candidates(self, threshold: int) -> list[dict]:
        """Return unknown predicates that have reached the promotion threshold."""
        cur = self._conn.execute(
            "SELECT * FROM unknown_predicates WHERE count >= ? ORDER BY count DESC",
            (threshold,)
        )
        return [dict(row) for row in cur.fetchall()]


    # ── JSON export ───────────────────────────────────────────────────────────

    def export_json(self, path: str = "living_memory_export.json") -> str:
        branches   = self.get_all_branches()
        all_nodes  = self.get_all_active_nodes()
        cur        = self._conn.execute("SELECT * FROM compression_log ORDER BY timestamp")
        comp_log   = [dict(r) for r in cur.fetchall()]
        meta_cur   = self._conn.execute("SELECT * FROM meta")
        meta       = {r["key"]: r["value"] for r in meta_cur.fetchall()}
        unk_cur    = self._conn.execute("SELECT * FROM unknown_predicates ORDER BY count DESC")
        unknown    = [dict(r) for r in unk_cur.fetchall()]

        by_branch: dict[str, list] = {}
        for node in all_nodes:
            bn = node["branch_name"]
            by_branch.setdefault(bn, []).append(node)

        export = {
            "meta":                meta,
            "branches":            branches,
            "nodes_by_branch":     by_branch,
            "compression_log":     comp_log,
            "unknown_predicates":  unknown,
        }

        with open(path, "w") as f:
            json.dump(export, f, indent=2)

        return path

    def close(self) -> None:
        self._conn.close()


    # ── Co-access (relational activation) ────────────────────────────────────

    def log_co_access(self, node_ids: list[str], agent_id: str = None) -> None:
        if len(node_ids) < 2:
            return
        with self._transaction() as cur:
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    a, b = sorted([node_ids[i], node_ids[j]])
                    cur.execute("""
                        INSERT INTO co_access_log (node_a, node_b, agent_id, count)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(node_a, node_b, agent_id)
                        DO UPDATE SET count = count + 1
                    """, (a, b, agent_id))

    def get_co_access_partners(self, node_id: str,
                               agent_id: str = None,
                               top_n: int = 5) -> list[str]:
        if agent_id is not None:
            cur = self._conn.execute("""
                SELECT CASE WHEN node_a = ? THEN node_b ELSE node_a END AS partner,
                       SUM(count) as total
                FROM co_access_log
                WHERE (node_a = ? OR node_b = ?) AND agent_id = ?
                GROUP BY partner
                ORDER BY total DESC
                LIMIT ?
            """, (node_id, node_id, node_id, agent_id, top_n))
        else:
            cur = self._conn.execute("""
                SELECT CASE WHEN node_a = ? THEN node_b ELSE node_a END AS partner,
                       SUM(count) as total
                FROM co_access_log
                WHERE node_a = ? OR node_b = ?
                GROUP BY partner
                ORDER BY total DESC
                LIMIT ?
            """, (node_id, node_id, node_id, top_n))
        return [row["partner"] for row in cur.fetchall()]


    # ── Pending consensus ─────────────────────────────────────────────────────

    def write_pending(self, pending_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO pending_consensus
                    (pending_id, concept_key, agent_id, content,
                     branch_name, tfidf_vector, tags)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(pending_id) DO UPDATE SET
                    content      = excluded.content,
                    tfidf_vector = excluded.tfidf_vector,
                    tags         = excluded.tags
            """, (
                pending_dict["pending_id"],
                pending_dict["concept_key"],
                pending_dict["agent_id"],
                pending_dict["content"],
                pending_dict["branch_name"],
                json.dumps(pending_dict.get("tfidf_vector", {})),
                json.dumps(pending_dict.get("tags", [])),
            ))

    def get_pending_by_concept(self, concept_key: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM pending_consensus WHERE concept_key = ?",
            (concept_key,)
        )
        rows = []
        for row in cur.fetchall():
            r = dict(row)
            r["tfidf_vector"] = json.loads(r["tfidf_vector"])
            r["tags"]         = json.loads(r["tags"])
            rows.append(r)
        return rows

    def commit_pending_to_backbone(self, concept_key: str,
                                   node_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, tfidf_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
            """, (
                node_dict["node_id"],
                node_dict["content"],
                node_dict["branch_name"],
                1,
                node_dict.get("base_score", 1.0),
                node_dict.get("access_count", 0),
                node_dict.get("system_access_snapshot", 0),
                json.dumps(node_dict.get("tags", [])),
                json.dumps(node_dict.get("compressed_from", [])),
                json.dumps(node_dict.get("tfidf_vector", {})),
                node_dict.get("agent_id"),
                node_dict.get("subject"),
                node_dict.get("predicate", "knows"),
                node_dict.get("object_text", ""),
                json.dumps(node_dict.get("obj_vector", {})),
            ))
            cur.execute(
                "DELETE FROM pending_consensus WHERE concept_key = ?",
                (concept_key,)
            )

    def flag_conflict(self, conflict_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                INSERT OR IGNORE INTO conflicts
                    (conflict_id, concept_key, branch_name,
                     agent_ids, signal, threshold, resolved)
                VALUES (?,?,?,?,?,?,0)
            """, (
                conflict_dict["conflict_id"],
                conflict_dict["concept_key"],
                conflict_dict["branch_name"],
                json.dumps(conflict_dict["agent_ids"]),
                conflict_dict["signal"],
                conflict_dict["threshold"],
            ))

    def get_pending_conflicts(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM conflicts WHERE resolved = 0 ORDER BY created_at"
        )
        rows = []
        for row in cur.fetchall():
            r = dict(row)
            r["agent_ids"] = json.loads(r["agent_ids"])
            rows.append(r)
        return rows

    def resolve_conflict(self, conflict_id: str) -> None:
        with self._transaction() as cur:
            cur.execute(
                "UPDATE conflicts SET resolved = 1 WHERE conflict_id = ?",
                (conflict_id,)
            )
