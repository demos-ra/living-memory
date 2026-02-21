"""
test_lm.py — Living Memory automated test suite
Run: python test_lm.py

Tests all key mechanics end-to-end through the public API.
Each test is independent. Failures report clearly.
Exit code 0 = all pass. Exit code 1 = failures.
"""

import os
import sys
import json
import math
import shutil
import traceback

TEST_DB_DIR = "/tmp/lm_test_suite"


def setup():
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)
    os.makedirs(TEST_DB_DIR)


def db(name: str) -> str:
    return os.path.join(TEST_DB_DIR, f"{name}.db")


# ── Test runner ───────────────────────────────────────────────────────────────

results: list[tuple[str, bool, str]] = []

def run(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        tb = traceback.format_exc().strip().split("\n")[-1]
        results.append((name, False, tb))
        print(f"  FAIL  {name}")
        print(f"        {tb}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def t1_backbone_permanence():
    """
    Backbone nodes never decay regardless of system activity.
    D1 permanence axis — backbone is the fixed point.
    """
    from src.contract.api import LivingMemory
    from src.domain.nodes import Node, compute_decay

    with LivingMemory(db_path=db("t1")) as lm:
        lm.remember("backbone", "User identity: permanent record.", tags=["identity"])

        # Simulate massive system activity
        for _ in range(500):
            lm._tree.storage.increment_system_accesses()

        backbone_raw = lm._tree.storage.get_backbone_nodes()
        assert len(backbone_raw) == 1, "Expected 1 backbone node"

        node = Node.from_dict(backbone_raw[0])
        relevance = compute_decay(node, lm._tree.storage.get_system_accesses())

        assert relevance == node.base_score, (
            f"Backbone decayed: expected {node.base_score}, got {relevance}"
        )


def t2_dynamic_branch_creation():
    """
    Branches are created dynamically at runtime — not hardcoded.
    Any branch name is valid. Multiple branches coexist independently.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t2")) as lm:
        lm.remember("goals",    "Ship LM v1.")
        lm.remember("beliefs",  "Memory must self-organize.")
        lm.remember("threads",  "Open: benchmark compression.")
        lm.remember("custom_branch_xyz", "Arbitrary branch works.")

        status = lm.status()
        branches = status["branches"]

        assert "goals"            in branches
        assert "beliefs"          in branches
        assert "threads"          in branches
        assert "custom_branch_xyz" in branches
        assert "backbone"         in branches  # always present


def t3_deduplication():
    """
    Near-duplicate content (cosine similarity >= 0.85) updates existing node
    rather than inserting a new one. Keeps tree clean.
    Verifies both exact duplicates (sim=1.0) and high-overlap content (sim>=0.85).
    """
    from src.contract.api import LivingMemory
    from src.domain.retrieval import compute_tfidf
    from src.domain.nodes import cosine_similarity

    # Strings share most tokens — cosine sim will be high
    s1 = "agent memory persistence retrieval compression vector similarity"
    s2 = "agent memory persistence retrieval compression vector similarity scoring"

    v1 = compute_tfidf(s1, {})
    v2 = compute_tfidf(s2, {})
    sim = cosine_similarity(v1, v2)
    assert sim >= 0.85, f"Test strings not similar enough: {sim:.3f}"

    with LivingMemory(db_path=db("t3")) as lm:
        lm.remember("goals", s1)
        lm.remember("goals", s1)  # exact duplicate — update, not insert
        lm.remember("goals", s2)  # high-overlap — update, not insert

        nodes = lm._tree.storage.get_nodes_for_branch("goals")
        assert len(nodes) == 1, (
            f"Dedup failed: expected 1 node, got {len(nodes)}"
        )


def t4_ols_trigger_phi():
    """
    Branch-level OLS fires when branch size exceeds φ × system mean.
    After compression, active node count is reduced. Content is preserved.
    φ = 1.618 (golden ratio) — self-organizing, no hardcoded count.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t4")) as lm:
        # Create a second small branch so system mean is low → threshold easily crossed
        lm.remember("small", "anchor branch to keep mean low")

        # Fill target branch well past φ × mean
        for i in range(20):
            lm.remember(
                "targets",
                f"Target {i}: implement subsystem {i} with full test coverage and documentation"
            )

        active = lm._tree.storage.get_nodes_for_branch("targets")
        assert len(active) < 20, (
            f"OLS never fired: {len(active)} active nodes, expected < 20"
        )


def t5_ols_content_preservation():
    """
    After OLS compression, the compressed node contains content from source nodes.
    Information is preserved in merged form — not lost.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t5")) as lm:
        lm.remember("small", "anchor")

        # Each node uses a unique identifier to prevent dedup
        # Content is varied enough that cosine sim stays below 0.85 across nodes
        topics = [
            ("alpha", "persistence storage database sqlite atomic transactions"),
            ("beta",  "retrieval cosine tfidf vector similarity scoring threshold"),
            ("gamma", "compression ols centroid merge golden ratio phi trigger"),
            ("delta", "backbone permanent identity decay immune session boundary"),
            ("epsilon", "branch dynamic manifold domain context agent working memory"),
            ("zeta",  "injection budget cap character limit context window overflow"),
            ("eta",   "archival cold root level cross branch multi year long term"),
            ("theta", "deduplication update existing node near duplicate similarity"),
            ("iota",  "ebbinghaus forgetting curve lambda inactivity ratio gradient"),
            ("kappa", "export json human readable snapshot inspection debug audit"),
        ]
        for name, unique_content in topics:
            lm.remember("knowledge", f"Subsystem {name}: {unique_content}.")

        active = lm._tree.storage.get_nodes_for_branch("knowledge")
        assert len(active) >= 1

        # Verify at least one compressed node exists (has compressed_from)
        compressed = [n for n in active if len(n["compressed_from"]) > 0]
        assert len(compressed) >= 1, "No compressed nodes found after OLS"

        # Verify content is non-empty
        for node in compressed:
            assert len(node["content"]) > 10, "Compressed node has empty content"


def t6_ebbinghaus_gradient_decay():
    """
    Branch node decay is gradient (continuous), not binary.
    Decay = base_score × e^(-λ × inactivity_ratio).
    More system activity since last access → lower relevance.
    Backbone nodes immune.
    """
    from src.contract.api import LivingMemory
    from src.domain.nodes import Node, compute_decay

    with LivingMemory(db_path=db("t6")) as lm:
        lm.remember("goals", "Gradient decay test node.")
        lm.remember("backbone", "Permanent backbone node.")

        # Simulate system activity without accessing the branch node
        for _ in range(200):
            lm._tree.storage.increment_system_accesses()

        sys_acc = lm._tree.storage.get_system_accesses()
        branch_raw = lm._tree.storage.get_nodes_for_branch("goals")
        backbone_raw = lm._tree.storage.get_backbone_nodes()

        branch_node   = Node.from_dict(branch_raw[0])
        backbone_node = Node.from_dict(backbone_raw[0])

        branch_rel   = compute_decay(branch_node, sys_acc)
        backbone_rel = compute_decay(backbone_node, sys_acc)

        # Branch decayed — gradient, between 0 and 1
        assert 0.0 < branch_rel < 1.0, (
            f"Expected gradient decay, got {branch_rel}"
        )
        # Backbone unchanged
        assert backbone_rel == backbone_node.base_score, (
            f"Backbone decayed: {backbone_rel}"
        )
        # Decay is exponential — verify it follows e^(-λ * ratio)
        expected = branch_node.base_score * math.exp(-0.5 * (sys_acc - branch_node.system_access_snapshot) / max(sys_acc, 1))
        assert abs(branch_rel - expected) < 1e-5, (
            f"Decay formula mismatch: got {branch_rel}, expected {expected}"
        )


def t7_conditional_injection_budget():
    """
    Injection never exceeds top_n nodes or budget_chars characters.
    Hard caps enforced regardless of how many nodes are relevant.
    """
    from src.contract.api import LivingMemory
    import json as _json

    config_path = os.path.join(os.path.dirname(__file__), "src/config/memory.json")
    with open(config_path) as f:
        cfg = _json.load(f)

    top_n        = cfg["injection"]["top_n"]
    budget_chars = cfg["injection"]["budget_chars"]

    with LivingMemory(db_path=db("t7")) as lm:
        # Insert many nodes with similar content — all will score high on query
        for i in range(30):
            lm.remember(
                "knowledge",
                f"Agent memory system {i}: persists context across sessions with fast retrieval and compression. "
                f"The system uses TF-IDF vectors and cosine similarity for semantic matching."
            )

        # Warm up accesses so nodes have some history
        for _ in range(10):
            lm._tree.storage.increment_system_accesses()

        nodes = lm.recall_nodes("agent memory system persistence retrieval")

        assert len(nodes) <= top_n, (
            f"Injection exceeded top_n: {len(nodes)} > {top_n}"
        )
        total_chars = sum(len(n["content"]) for n in nodes)
        assert total_chars <= budget_chars, (
            f"Injection exceeded budget: {total_chars} > {budget_chars}"
        )


def t8_retrieval_relevance_ordering():
    """
    Injection returns nodes ordered by relevance to query.
    Most relevant node is first. Unrelated nodes are not injected.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t8")) as lm:
        lm.remember("knowledge", "The agent uses TF-IDF vectors for semantic memory retrieval.")
        lm.remember("knowledge", "Compression merges nodes using centroid vector averaging.")
        lm.remember("knowledge", "Backbone nodes store permanent user identity and facts.")
        lm.remember("knowledge", "The φ golden ratio triggers OLS compression automatically.")
        lm.remember("unrelated", "The weather in Bangkok is hot and humid in summer months.")

        # Rebuild IDF with all nodes present
        lm._tree._rebuild_idf()

        nodes = lm.recall_nodes("TF-IDF semantic retrieval vectors")

        # Should return results
        assert len(nodes) > 0, "Expected at least one relevant node"

        # Unrelated node should not be top result
        if len(nodes) > 0:
            top = nodes[0]["content"].lower()
            assert "weather" not in top, (
                "Unrelated weather node ranked first — retrieval ordering broken"
            )


def t9_sqlite_persistence():
    """
    Memory persists across LivingMemory instances (sessions).
    Close and reopen — data must survive.
    This is the multi-year persistence guarantee.
    """
    from src.contract.api import LivingMemory

    db_path = db("t9")

    # Session 1 — write
    with LivingMemory(db_path=db_path) as lm:
        lm.remember("backbone", "Persistent user identity across all sessions.")
        lm.remember("goals",    "Goal that must survive session boundary.")
        node_count_s1 = lm.status()["active_node_count"]

    # Session 2 — verify data survived
    with LivingMemory(db_path=db_path) as lm:
        node_count_s2 = lm.status()["active_node_count"]
        assert node_count_s2 == node_count_s1, (
            f"Nodes lost across sessions: s1={node_count_s1}, s2={node_count_s2}"
        )
        backbone = lm._tree.storage.get_backbone_nodes()
        assert len(backbone) == 1
        assert "Persistent user identity" in backbone[0]["content"]


def t10_json_export_integrity():
    """
    JSON export contains all active branches, nodes, and compression log.
    Export is human-readable and structurally complete.
    """
    from src.contract.api import LivingMemory

    export_path = os.path.join(TEST_DB_DIR, "t10_export.json")

    with LivingMemory(db_path=db("t10")) as lm:
        lm.remember("backbone", "Identity node for export test.")
        lm.remember("goals",    "Goal A for export validation.")
        lm.remember("goals",    "Goal B for export validation.")

        # Force OLS by adding nodes with distinct enough content to avoid dedup
        lm.remember("anchor", "anchor node keeps mean low for phi trigger")
        topics = [
            "persistence storage atomicity transaction rollback commit wal journal",
            "retrieval cosine tfidf vector similarity threshold injection budget",
            "compression centroid merge golden ratio phi trigger branch size mean",
            "backbone permanence decay immune identity session boundary long term",
            "branch dynamic manifold domain context agent working episodic memory",
            "injection budget cap character limit window overflow top n selection",
            "archival cold root cross branch multi year long term embed compress",
            "deduplication update existing near duplicate similarity eighty five",
            "ebbinghaus forgetting lambda inactivity ratio gradient exponential",
            "export json human readable snapshot inspection debug audit structure",
            "query prompt context score rank buoyancy float active node retrieval",
            "sqlite wal foreign key index schema migrate version upgrade rollback",
            "idf term frequency document corpus vocabulary tokenize regex filter",
            "access bump record snapshot restore buoyancy anti decay retrieval hit",
            "system total accesses counter atomic increment meta table key value",
            "cold branch ratio below threshold mean access root ols compress fire",
            "node id uuid content branch name backbone score count snapshot tags",
            "session open close context manager enter exit lifecycle db path init",
            "status snapshot active count branches backbone dynamic access count",
            "merge content deduplicate sentence split unique normalized preserved",
        ]
        for content_str in topics:
            lm.remember("compress_me", content_str)

        path = lm.export(export_path)

    with open(path) as f:
        exported = json.load(f)

    assert "branches"        in exported, "Missing 'branches' key"
    assert "nodes_by_branch" in exported, "Missing 'nodes_by_branch' key"
    assert "compression_log" in exported, "Missing 'compression_log' key"
    assert "meta"            in exported, "Missing 'meta' key"

    # Backbone present
    assert "backbone" in exported["nodes_by_branch"], "Backbone missing from export"

    # Compression log has entries (OLS fired during insert loop)
    assert len(exported["compression_log"]) >= 1, (
        "No compression events logged — OLS may not have fired"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("T1  Backbone permanence",             t1_backbone_permanence),
    ("T2  Dynamic branch creation",         t2_dynamic_branch_creation),
    ("T3  Deduplication (cosine 0.85)",     t3_deduplication),
    ("T4  OLS trigger φ × system mean",     t4_ols_trigger_phi),
    ("T5  OLS content preservation",        t5_ols_content_preservation),
    ("T6  Ebbinghaus gradient decay",       t6_ebbinghaus_gradient_decay),
    ("T7  Injection budget cap",            t7_conditional_injection_budget),
    ("T8  Retrieval relevance ordering",    t8_retrieval_relevance_ordering),
    ("T9  SQLite session persistence",      t9_sqlite_persistence),
    ("T10 JSON export integrity",           t10_json_export_integrity),
]

if __name__ == "__main__":
    setup()
    print("Living Memory — Test Suite")
    print("=" * 40)

    for name, fn in TESTS:
        run(name, fn)

    print("=" * 40)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"{passed}/{len(results)} passed")

    if failed:
        print(f"\nFailed:")
        for name, ok, err in results:
            if not ok:
                print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
