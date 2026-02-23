# [F-storage/R4/C4] SQLite persistence, atomic transactions, JSON export
# R4 — Execution. Depends: R0 config only. No domain imports. Pure I/O.
# v0.5 — predicates table, predicate_synonyms, content_vector rename,
#         bootstrap prior in meta, cold branch via physics gate

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
    content_vector         TEXT NOT NULL DEFAULT '{}',
    archived               INTEGER NOT NULL DEFAULT 0,
    agent_id               TEXT,
    subject                TEXT,
    predicate              TEXT NOT NULL DEFAULT 'knows',
    object_text            TEXT NOT NULL DEFAULT '',
    obj_vector             TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (branch_name) REFERENCES branches(branch_name)
);

CREATE INDEX IF NOT EXISTS idx_nodes_branch     ON nodes(branch_name);
CREATE INDEX IF NOT EXISTS idx_nodes_archived   ON nodes(archived);
CREATE INDEX IF NOT EXISTS idx_nodes_backbone   ON nodes(is_backbone);
CREATE INDEX IF NOT EXISTS idx_nodes_predicate  ON nodes(predicate);
CREATE INDEX IF NOT EXISTS idx_nodes_subject    ON nodes(subject);

CREATE TABLE IF NOT EXISTS compression_log (
    log_id          TEXT PRIMARY KEY,
    compressed_node TEXT NOT NULL,
    source_ids      TEXT NOT NULL,
    branch_name     TEXT NOT NULL,
    level           TEXT NOT NULL,
    timestamp       INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

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
    pending_id      TEXT PRIMARY KEY,
    concept_key     TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    content         TEXT NOT NULL,
    branch_name     TEXT NOT NULL,
    content_vector  TEXT NOT NULL DEFAULT '{}',
    subject         TEXT,
    predicate       TEXT NOT NULL DEFAULT 'knows',
    object_text     TEXT NOT NULL DEFAULT '',
    obj_vector      TEXT NOT NULL DEFAULT '{}',
    tags            TEXT NOT NULL DEFAULT '[]',
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s','now'))
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

CREATE TABLE IF NOT EXISTS predicates (
    predicate            TEXT PRIMARY KEY,
    polarity             TEXT NOT NULL,
    temporality          TEXT NOT NULL,
    directionality       TEXT NOT NULL,
    certainty            TEXT NOT NULL,
    agency               TEXT NOT NULL,
    decay_multiplier     REAL NOT NULL DEFAULT 1.0,
    compression          TEXT NOT NULL DEFAULT 'eligible',
    conflict_sensitivity TEXT NOT NULL DEFAULT 'medium',
    version              INTEGER NOT NULL DEFAULT 1,
    source               TEXT NOT NULL DEFAULT 'seed'
);

CREATE TABLE IF NOT EXISTS predicate_synonyms (
    unknown_predicate TEXT PRIMARY KEY,
    mapped_to         TEXT NOT NULL,
    axis_signature    TEXT NOT NULL,
    created_at        INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"""


# ── Storage ───────────────────────────────────────────────────────────────────

class Storage:
    """
    All SQLite I/O. Every write is atomic.
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
        self._migrate()
        self._seed_predicates()

    def _migrate(self) -> None:
        """
        Idempotent migration. Handles v0.3 → v0.5 column additions.
        content_vector replaces tfidf_vector. Both kept during transition.
        """
        cur = self._conn.execute("PRAGMA table_info(nodes)")
        existing = {row["name"] for row in cur.fetchall()}

        migrations = []

        # v0.4 fields
        if "subject"     not in existing:
            migrations.append("ALTER TABLE nodes ADD COLUMN subject TEXT")
        if "predicate"   not in existing:
            migrations.append("ALTER TABLE nodes ADD COLUMN predicate TEXT NOT NULL DEFAULT 'knows'")
        if "object_text" not in existing:
            migrations.append("ALTER TABLE nodes ADD COLUMN object_text TEXT NOT NULL DEFAULT ''")
        if "obj_vector"  not in existing:
            migrations.append("ALTER TABLE nodes ADD COLUMN obj_vector TEXT NOT NULL DEFAULT '{}'")

        # v0.5 — content_vector (was tfidf_vector)
        # SQLite doesn't support RENAME COLUMN before 3.25 — add new column, copy data
        if "content_vector" not in existing and "tfidf_vector" in existing:
            migrations.append(
                "ALTER TABLE nodes ADD COLUMN content_vector TEXT NOT NULL DEFAULT '{}'"
            )
            migrations.append(
                "UPDATE nodes SET content_vector = tfidf_vector"
            )
        elif "content_vector" not in existing:
            migrations.append(
                "ALTER TABLE nodes ADD COLUMN content_vector TEXT NOT NULL DEFAULT '{}'"
            )

        # pending_consensus: add triple fields if absent
        cur2 = self._conn.execute("PRAGMA table_info(pending_consensus)")
        pc_existing = {row["name"] for row in cur2.fetchall()}
        if "subject"    not in pc_existing:
            migrations.append("ALTER TABLE pending_consensus ADD COLUMN subject TEXT")
        if "predicate"  not in pc_existing:
            migrations.append("ALTER TABLE pending_consensus ADD COLUMN predicate TEXT NOT NULL DEFAULT 'knows'")
        if "object_text" not in pc_existing:
            migrations.append("ALTER TABLE pending_consensus ADD COLUMN object_text TEXT NOT NULL DEFAULT ''")
        if "obj_vector" not in pc_existing:
            migrations.append("ALTER TABLE pending_consensus ADD COLUMN obj_vector TEXT NOT NULL DEFAULT '{}'")
        if "content_vector" not in pc_existing:
            migrations.append("ALTER TABLE pending_consensus ADD COLUMN content_vector TEXT NOT NULL DEFAULT '{}'")

        if migrations:
            with self._transaction() as cur:
                for sql in migrations:
                    cur.execute(sql)

    def _seed_predicates(self) -> None:
        """
        Load seed predicates from config into predicates table on first init.
        Idempotent — INSERT OR IGNORE, never overwrites discovered predicates.
        """
        seed = _CFG["predicates"]["seed"]
        with self._transaction() as cur:
            for p in seed:
                cur.execute("""
                    INSERT OR IGNORE INTO predicates
                        (predicate, polarity, temporality, directionality,
                         certainty, agency, decay_multiplier, compression,
                         conflict_sensitivity, version, source)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    p["predicate"], p["polarity"], p["temporality"],
                    p["directionality"], p["certainty"], p["agency"],
                    p["decay_multiplier"], p["compression"],
                    p["conflict_sensitivity"],
                    _CFG["predicates"]["version"], "seed"
                ))

    @contextmanager
    def _transaction(self):
        """Atomic transaction — all writes commit together or rollback entirely."""
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

    def get_meta(self, key: str) -> Optional[str]:
        cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (key, value)
            )


    # ── Predicates ────────────────────────────────────────────────────────────

    def get_predicate(self, predicate: str) -> Optional[dict]:
        """Return predicate row or None if not in vocabulary."""
        cur = self._conn.execute(
            "SELECT * FROM predicates WHERE predicate = ?", (predicate,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_predicates(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM predicates ORDER BY predicate")
        return [dict(row) for row in cur.fetchall()]

    def insert_predicate(self, predicate_dict: dict) -> None:
        """Insert a newly discovered predicate. Atomic."""
        with self._transaction() as cur:
            cur.execute("""
                INSERT OR IGNORE INTO predicates
                    (predicate, polarity, temporality, directionality,
                     certainty, agency, decay_multiplier, compression,
                     conflict_sensitivity, version, source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                predicate_dict["predicate"],
                predicate_dict["polarity"],
                predicate_dict["temporality"],
                predicate_dict["directionality"],
                predicate_dict["certainty"],
                predicate_dict["agency"],
                predicate_dict["decay_multiplier"],
                predicate_dict["compression"],
                predicate_dict["conflict_sensitivity"],
                predicate_dict.get("version", 1),
                predicate_dict.get("source", "discovered"),
            ))

    def log_synonym(self, unknown: str, mapped_to: str,
                    axis_signature: str) -> None:
        """Log a predicate that was classified as a synonym of an existing one."""
        with self._transaction() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO predicate_synonyms
                    (unknown_predicate, mapped_to, axis_signature)
                VALUES (?,?,?)
            """, (unknown, mapped_to, axis_signature))

    def get_synonym_mapping(self, predicate: str) -> Optional[str]:
        """Return what an unknown predicate was mapped to, or None."""
        cur = self._conn.execute(
            "SELECT mapped_to FROM predicate_synonyms WHERE unknown_predicate=?",
            (predicate,)
        )
        row = cur.fetchone()
        return row["mapped_to"] if row else None


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

    def _node_insert_params(self, nd: dict) -> tuple:
        return (
            nd["node_id"], nd["content"], nd["branch_name"],
            int(nd["is_backbone"]), nd["base_score"],
            nd["access_count"], nd["system_access_snapshot"],
            json.dumps(nd["tags"]),
            json.dumps(nd["compressed_from"]),
            json.dumps(nd.get("content_vector", {})),
            nd.get("agent_id"),
            nd.get("subject"),
            nd.get("predicate", "knows"),
            nd.get("object_text", ""),
            json.dumps(nd.get("obj_vector", {})),
        )

    def insert_node(self, node_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, content_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
            """, self._node_insert_params(node_dict))

    def update_node(self, node_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                UPDATE nodes SET
                    content                = ?,
                    base_score             = ?,
                    access_count           = ?,
                    system_access_snapshot = ?,
                    tags                   = ?,
                    content_vector         = ?,
                    object_text            = ?,
                    obj_vector             = ?
                WHERE node_id = ?
            """, (
                node_dict["content"],
                node_dict["base_score"],
                node_dict["access_count"],
                node_dict["system_access_snapshot"],
                json.dumps(node_dict["tags"]),
                json.dumps(node_dict.get("content_vector", {})),
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

    def compress_nodes_atomic(self, consumed_ids: list[str],
                               compressed_node_dict: dict,
                               log_entry: dict) -> None:
        """
        OLS compression — single atomic transaction:
          1. Insert compressed node
          2. Archive source nodes (never delete — full audit trail)
          3. Write compression log entry
        """
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, content_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
            """, self._node_insert_params(compressed_node_dict))

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


    # ── Serialization ─────────────────────────────────────────────────────────

    @staticmethod
    def _deserialize_node(row: dict) -> dict:
        """Parse JSON fields. Normalize nullable fields. Handle legacy tfidf_vector."""
        for field in ("tags", "compressed_from", "obj_vector"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
            elif row.get(field) is None:
                row[field] = {}

        # content_vector — handle both new column and legacy tfidf_vector
        if "content_vector" in row and row["content_vector"]:
            if isinstance(row["content_vector"], str):
                row["content_vector"] = json.loads(row["content_vector"])
        elif "tfidf_vector" in row and row["tfidf_vector"]:
            cv = row["tfidf_vector"]
            row["content_vector"] = json.loads(cv) if isinstance(cv, str) else cv
        else:
            row["content_vector"] = {}

        row["is_backbone"] = bool(row["is_backbone"])
        row.setdefault("agent_id",    None)
        row.setdefault("subject",     None)
        row.setdefault("predicate",   "knows")
        row.setdefault("object_text", "")
        row.setdefault("obj_vector",  {})
        return row


    # ── Co-access (Hebbian activation) ────────────────────────────────────────

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
                GROUP BY partner ORDER BY total DESC LIMIT ?
            """, (node_id, node_id, node_id, agent_id, top_n))
        else:
            cur = self._conn.execute("""
                SELECT CASE WHEN node_a = ? THEN node_b ELSE node_a END AS partner,
                       SUM(count) as total
                FROM co_access_log
                WHERE node_a = ? OR node_b = ?
                GROUP BY partner ORDER BY total DESC LIMIT ?
            """, (node_id, node_id, node_id, top_n))
        return [row["partner"] for row in cur.fetchall()]


    # ── Pending consensus ─────────────────────────────────────────────────────

    def write_pending(self, pending_dict: dict) -> None:
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO pending_consensus
                    (pending_id, concept_key, agent_id, content, branch_name,
                     content_vector, subject, predicate, object_text, obj_vector, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pending_id) DO UPDATE SET
                    content        = excluded.content,
                    content_vector = excluded.content_vector,
                    subject        = excluded.subject,
                    predicate      = excluded.predicate,
                    object_text    = excluded.object_text,
                    obj_vector     = excluded.obj_vector,
                    tags           = excluded.tags
            """, (
                pending_dict["pending_id"],
                pending_dict["concept_key"],
                pending_dict["agent_id"],
                pending_dict["content"],
                pending_dict["branch_name"],
                json.dumps(pending_dict.get("content_vector", {})),
                pending_dict.get("subject"),
                pending_dict.get("predicate", "knows"),
                pending_dict.get("object_text", ""),
                json.dumps(pending_dict.get("obj_vector", {})),
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
            for field in ("content_vector", "obj_vector", "tags"):
                if isinstance(r.get(field), str):
                    r[field] = json.loads(r[field])
            rows.append(r)
        return rows

    def commit_pending_to_backbone(self, concept_key: str,
                                   node_dict: dict) -> None:
        """Atomic: insert backbone node + clear pending. All or nothing."""
        with self._transaction() as cur:
            cur.execute("""
                INSERT INTO nodes
                    (node_id, content, branch_name, is_backbone, base_score,
                     access_count, system_access_snapshot, tags,
                     compressed_from, content_vector, archived, agent_id,
                     subject, predicate, object_text, obj_vector)
                VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
            """, self._node_insert_params({**node_dict, "is_backbone": True}))
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


    # ── JSON export ───────────────────────────────────────────────────────────

    def export_json(self, path: str = "living_memory_export.json") -> str:
        branches  = self.get_all_branches()
        all_nodes = self.get_all_active_nodes()
        comp_log  = [dict(r) for r in self._conn.execute(
            "SELECT * FROM compression_log ORDER BY timestamp"
        ).fetchall()]
        meta      = {r["key"]: r["value"] for r in self._conn.execute(
            "SELECT * FROM meta"
        ).fetchall()}
        predicates = self.get_all_predicates()
        synonyms   = [dict(r) for r in self._conn.execute(
            "SELECT * FROM predicate_synonyms"
        ).fetchall()]

        by_branch: dict[str, list] = {}
        for node in all_nodes:
            by_branch.setdefault(node["branch_name"], []).append(node)

        export = {
            "meta":              meta,
            "branches":          branches,
            "nodes_by_branch":   by_branch,
            "compression_log":   comp_log,
            "predicates":        predicates,
            "predicate_synonyms": synonyms,
        }

        with open(path, "w") as f:
            json.dump(export, f, indent=2)

        return path

    def close(self) -> None:
        self._conn.close()
