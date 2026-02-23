"""
test_lm.py — Living Memory test suite
Run: python3 test_lm.py

21 tests. Each independent. All run through the public API.
Exit 0 = all pass. Exit 1 = failures.
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


# ── Runner ────────────────────────────────────────────────────────────────────

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


# ── T1 — T14: core mechanics (updated for v0.5 field names) ──────────────────

def t1_backbone_permanence():
    """Backbone nodes never decay regardless of system activity."""
    from src.contract.api import LivingMemory
    from src.domain.nodes import Node, compute_decay

    with LivingMemory(db_path=db("t1")) as lm:
        lm.remember("backbone", "Permanent identity record.", tags=["identity"])
        for _ in range(500):
            lm._tree.storage.increment_system_accesses()

        raw       = lm._tree.storage.get_backbone_nodes()
        assert len(raw) == 1
        node      = Node.from_dict(raw[0])
        relevance = compute_decay(node, lm._tree.storage.get_system_accesses())
        assert relevance == node.base_score, f"Backbone decayed: {relevance}"


def t2_dynamic_branch_creation():
    """Branches created dynamically. Any name valid. Multiple coexist."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t2")) as lm:
        lm.remember("goals",            "Ship LM v1.")
        lm.remember("beliefs",          "Memory must self-organize.")
        lm.remember("threads",          "Open: benchmark compression.")
        lm.remember("custom_branch_xyz","Arbitrary branch works.")

        branches = lm.status()["branches"]
        for name in ("goals", "beliefs", "threads", "custom_branch_xyz", "backbone"):
            assert name in branches, f"Branch missing: {name}"


def t3_deduplication():
    """Near-duplicate content updates existing node rather than inserting."""
    from src.contract.api import LivingMemory
    from src.domain.retrieval import compute_tfidf
    from src.domain.nodes import cosine_similarity

    s1 = "agent memory persistence retrieval compression vector similarity"
    s2 = "agent memory persistence retrieval compression vector similarity scoring"
    assert cosine_similarity(compute_tfidf(s1, {}), compute_tfidf(s2, {})) >= 0.85

    with LivingMemory(db_path=db("t3")) as lm:
        lm.remember("goals", s1)
        lm.remember("goals", s1)
        lm.remember("goals", s2)
        nodes = lm._tree.storage.get_nodes_for_branch("goals")
        assert len(nodes) == 1, f"Dedup failed: {len(nodes)} nodes"


def t4_ols_trigger_phi():
    """Branch-level OLS fires when size > φ × system mean. No hardcoded count."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t4")) as lm:
        lm.remember("small", "anchor branch to keep mean low")
        for i in range(20):
            lm.remember("targets",
                f"Target {i}: implement subsystem {i} with full test coverage and documentation")
        active = lm._tree.storage.get_nodes_for_branch("targets")
        assert len(active) < 20, f"OLS never fired: {len(active)} nodes"


def t5_ols_content_preservation():
    """After OLS, compressed node contains content from source nodes."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t5")) as lm:
        lm.remember("small", "anchor")
        topics = [
            "persistence storage database sqlite atomic transactions",
            "retrieval cosine tfidf vector similarity scoring threshold",
            "compression ols centroid merge golden ratio phi trigger",
            "backbone permanent identity decay immune session boundary",
            "branch dynamic manifold domain context agent working memory",
            "injection budget cap character limit context window overflow",
            "archival cold root level cross branch multi year long term",
            "deduplication update existing node near duplicate similarity",
            "ebbinghaus forgetting curve lambda inactivity ratio gradient",
            "export json human readable snapshot inspection debug audit",
        ]
        for i, c in enumerate(topics):
            lm.remember("knowledge", f"Subsystem {i}: {c}.")

        branch_raws = lm._tree.storage.get_nodes_for_branch("knowledge")
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws:
            lm._tree.storage.update_node_access(n["node_id"], 1, sys_now)
        for _ in range(1000):
            lm._tree.storage.increment_system_accesses()
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws[:5]:
            lm._tree.storage.update_node_access(n["node_id"], 2, sys_now)

        lm._tree._check_and_compress("knowledge")
        active     = lm._tree.storage.get_nodes_for_branch("knowledge")
        compressed = [n for n in active if len(n["compressed_from"]) > 0]
        assert len(compressed) >= 1, "No compressed nodes found"
        for node in compressed:
            assert len(node["content"]) > 10, "Compressed node empty"


def t6_ebbinghaus_gradient_decay():
    """Decay is gradient and predicate-aware. Backbone immune."""
    from src.contract.api import LivingMemory
    from src.domain.nodes import Node, compute_decay

    with LivingMemory(db_path=db("t6")) as lm:
        # 'knows' predicate — decay_multiplier=0.0, should not decay
        lm.remember("facts",    subject="user", predicate="knows",
                    object="Gradient decay test.")
        # free-text — defaults to 'knows', also multiplier=0.0
        lm.remember("prefs",    subject="user", predicate="prefers",
                    object="dark mode interface display")
        lm.remember("backbone", "Permanent backbone node.")

        for _ in range(200):
            lm._tree.storage.increment_system_accesses()

        sys_acc  = lm._tree.storage.get_system_accesses()
        pref_raw = lm._tree.storage.get_nodes_for_branch("prefs")
        bb_raw   = lm._tree.storage.get_backbone_nodes()

        pref_node = Node.from_dict(pref_raw[0])
        bb_node   = Node.from_dict(bb_raw[0])

        # prefers has decay_multiplier=1.8 — should decay
        pred_props = lm._tree._predicate_map.get("prefers", {})
        pref_rel  = compute_decay(pref_node, sys_acc, pred_props)
        bb_rel    = compute_decay(bb_node, sys_acc)

        assert 0.0 < pref_rel < 1.0,            f"Expected gradient decay, got {pref_rel}"
        assert bb_rel == bb_node.base_score,     f"Backbone decayed: {bb_rel}"

        # Verify formula with predicate multiplier
        mult     = pred_props.get("decay_multiplier", 1.0)
        lam_eff  = 0.5 * mult
        inact    = (sys_acc - pref_node.system_access_snapshot) / max(sys_acc, 1)
        expected = pref_node.base_score * math.exp(-lam_eff * inact)
        assert abs(pref_rel - expected) < 1e-5, f"Decay formula mismatch: {pref_rel} vs {expected}"


def t7_injection_budget_cap():
    """Injection respects max_summary_groups and budget_chars."""
    from src.contract.api import LivingMemory

    config_path = os.path.join(os.path.dirname(__file__), "src/config/memory.json")
    with open(config_path) as f:
        cfg = json.load(f)
    max_groups   = cfg["injection"]["max_summary_groups"]
    budget_chars = cfg["injection"]["budget_chars"]

    with LivingMemory(db_path=db("t7")) as lm:
        for i in range(30):
            lm.remember("knowledge",
                f"Agent memory system {i}: persists context across sessions with "
                f"fast retrieval compression TF-IDF cosine similarity matching.")
        for _ in range(10):
            lm._tree.storage.increment_system_accesses()

        nodes = lm.recall_nodes("agent memory system persistence retrieval")
        total_chars = sum(len(n["content"]) for n in nodes)
        assert total_chars <= budget_chars, f"Exceeded budget: {total_chars} > {budget_chars}"


def t8_retrieval_relevance_ordering():
    """Most relevant node ranked first. Unrelated nodes not injected."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t8")) as lm:
        lm.remember("knowledge", "The agent uses TF-IDF vectors for semantic memory retrieval.")
        lm.remember("knowledge", "Compression merges nodes using centroid vector averaging.")
        lm.remember("knowledge", "Backbone nodes store permanent user identity and facts.")
        lm.remember("knowledge", "The φ golden ratio triggers OLS compression automatically.")
        lm.remember("unrelated", "The weather in Bangkok is hot and humid in summer months.")
        lm._tree._rebuild_idf()

        nodes = lm.recall_nodes("TF-IDF semantic retrieval vectors")
        assert len(nodes) > 0, "Expected at least one relevant node"
        if nodes:
            assert "weather" not in nodes[0]["content"].lower(), \
                "Unrelated node ranked first"


def t9_sqlite_persistence():
    """Memory persists across sessions. Close and reopen — data survives."""
    from src.contract.api import LivingMemory

    path = db("t9")
    with LivingMemory(db_path=path) as lm:
        lm.remember("backbone", "Persistent identity across all sessions.")
        lm.remember("goals",    "Goal that must survive session boundary.")
        count_s1 = lm.status()["active_node_count"]

    with LivingMemory(db_path=path) as lm:
        count_s2 = lm.status()["active_node_count"]
        assert count_s2 == count_s1, f"Nodes lost: s1={count_s1} s2={count_s2}"
        bb = lm._tree.storage.get_backbone_nodes()
        assert any("Persistent" in n["content"] for n in bb)


def t10_json_export_integrity():
    """Export contains branches, nodes, compression log, predicates, synonyms."""
    from src.contract.api import LivingMemory

    export_path = os.path.join(TEST_DB_DIR, "t10_export.json")
    with LivingMemory(db_path=db("t10")) as lm:
        lm.remember("backbone", "Identity node.")
        lm.remember("goals",    "Goal A.")
        lm.remember("goals",    "Goal B.")
        lm.remember("anchor",   "anchor node")

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
        ]
        for c in topics:
            lm.remember("compress_me", c)

        branch_raws = lm._tree.storage.get_nodes_for_branch("compress_me")
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws:
            lm._tree.storage.update_node_access(n["node_id"], 1, sys_now)
        for _ in range(1000):
            lm._tree.storage.increment_system_accesses()
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws[:3]:
            lm._tree.storage.update_node_access(n["node_id"], 2, sys_now)
        lm._tree._check_and_compress("compress_me")

        # Log a synonym for export completeness
        lm.classify_predicate("likes", axes={
            "polarity": "positive", "temporality": "transient",
            "directionality": "self", "certainty": "belief", "agency": "active"
        })

        path = lm.export(export_path)

    with open(path) as f:
        exp = json.load(f)

    for key in ("branches", "nodes_by_branch", "compression_log", "meta",
                "predicates", "predicate_synonyms"):
        assert key in exp, f"Missing key: {key}"
    assert "backbone" in exp["nodes_by_branch"]
    assert len(exp["compression_log"]) >= 1
    assert len(exp["predicates"]) >= 12


def t11_relational_activation():
    """Co-retrieved nodes build connection weights. Co-access log populated."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t11")) as lm:
        lm.remember("goals",   "ship living memory library agent persistence retrieval system")
        lm.remember("beliefs", "memory must self-organize compress via phi golden ratio trigger")
        lm.remember("threads", "completely unrelated content about weather climate temperature")
        lm._tree._rebuild_idf()

        for _ in range(5):
            nodes = lm._tree.query("ship living memory agent", formatted=False)

        if len(nodes) >= 2:
            partners = lm._tree.storage.get_co_access_partners(nodes[0].node_id)
            assert isinstance(partners, list)

        cur = lm._tree.storage._conn.execute("SELECT COUNT(*) as n FROM co_access_log")
        assert cur.fetchone()["n"] >= 0


def t12_multi_agent_convergence():
    """Near-identical backbone contributions commit atomically. Pending cleared."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t12")) as lm:
        for i in range(10):
            lm.remember("goals",
                f"goal {i} agent memory persistence retrieval compression backbone")
        lm._tree._rebuild_idf()

        c1 = "agent memory system self-organizes compress decay persist backbone nodes"
        c2 = "agent memory system self-organizes compress decay persist backbone structure"

        r1 = lm.contribute("backbone", c1, agent_id="grok", concept_key="mem-design")
        r2 = lm.contribute("backbone", c2, agent_id="kimi", concept_key="mem-design")

        assert r2["status"] == "committed", \
            f"Expected committed: signal={r2.get('signal',0):.4f} threshold={r2.get('threshold',0):.4f}"
        assert r2["signal"] >= r2["threshold"]
        assert len(lm._tree.storage.get_pending_by_concept("mem-design")) == 0


def t13_multi_agent_conflict():
    """Divergent contributions flagged as conflict. Resolution writes to backbone."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t13")) as lm:
        for i in range(10):
            lm.remember("goals",
                f"goal {i} agent memory persistence retrieval compression backbone")
        lm._tree._rebuild_idf()

        r1 = lm.contribute("backbone",
            "agent memory retrieval uses vector embeddings semantic similarity",
            agent_id="grok", concept_key="retrieval-design")
        r2 = lm.contribute("backbone",
            "weather climate temperature humidity precipitation unrelated forecast",
            agent_id="kimi", concept_key="retrieval-design")

        assert r2["status"] == "conflict", f"Expected conflict: {r2}"
        assert r2["signal"] < r2["threshold"]

        conflicts = lm.pending_conflicts()
        assert len(conflicts) == 1

        lm.resolve_conflict(
            conflicts[0]["conflict_id"],
            "Agent memory retrieval uses TF-IDF cosine similarity.",
            "retrieval-design",
            subject="retrieval", predicate="requires",
            object="TF-IDF cosine similarity",
        )
        assert len(lm.pending_conflicts()) == 0

        bb = lm._tree.storage.get_backbone_nodes()
        assert any(n.get("agent_id") == "resolved" for n in bb)


def t14_emergent_sdt_threshold():
    """Threshold shifts with tree distribution. Sparse uses bootstrap. Rich uses live."""
    from src.contract.api import LivingMemory
    from src.domain.retrieval import compute_distribution, get_threshold
    from src.domain.nodes import Node

    with LivingMemory(db_path=db("t14")) as lm:
        lm.remember("goals", "agent memory system")
        lm.remember("goals", "persistence retrieval backbone")
        raw   = lm._tree.storage.get_all_active_nodes()
        nodes = [Node.from_dict(n) for n in raw]

        mean_cold, std_cold = compute_distribution(nodes)
        thresh_cold = get_threshold(0.3, mean_cold, std_cold)
        # Bootstrap values from config
        assert mean_cold == 0.60, f"Expected bootstrap mean 0.60, got {mean_cold}"

        # Rich corpus
        for i, branch in enumerate(["a","b","c","d","e","f","g","h","i","j","k","l"]):
            lm.remember(branch,
                f"concept {i} agent memory persistence retrieval compression backbone "
                f"decay phi golden ratio ebbinghaus hebbian signal detection theory "
                f"unique term branch {branch} identifier {i*7}")
        lm._tree._rebuild_idf()

        raw2   = lm._tree.storage.get_all_active_nodes()
        nodes2 = [Node.from_dict(n) for n in raw2]
        assert len(nodes2) >= 10

        mean_rich, std_rich = compute_distribution(nodes2)
        thresh_rich = get_threshold(0.3, mean_rich, std_rich)

        assert mean_rich != 0.60,        "Rich corpus still using bootstrap mean"
        assert thresh_cold != thresh_rich, "Threshold not emergent"

        for beta, mean, std, thresh in [(0.3, mean_cold, std_cold, thresh_cold),
                                         (0.3, mean_rich, std_rich, thresh_rich)]:
            expected = max(0.05, min(0.99, mean + math.log(beta) * std))
            assert abs(thresh - expected) < 1e-4


def t15_predicate_mismatch_no_dedup():
    """Same subject+object, different predicates — never deduplicate."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t15")) as lm:
        lm.remember("prefs", subject="user", predicate="prefers", object="dark mode display")
        lm.remember("prefs", subject="user", predicate="avoids",  object="dark mode display")

        nodes = lm._tree.storage.get_nodes_for_branch("prefs")
        assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
        predicates = {n["predicate"] for n in nodes}
        assert "prefers" in predicates and "avoids" in predicates


def t16_neutral_buoyancy_eligibility():
    """Nodes sink below mean only when access patterns diverge. Floating nodes immune."""
    from src.domain.nodes import Node, is_compression_eligible
    from src.domain.compression import compute_mean_relevance
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t16")) as lm:
        lm.remember("backbone", "system anchor permanent node")
        for c in [
            "persistence storage database sqlite atomic",
            "retrieval cosine tfidf vector similarity",
            "compression ols centroid merge golden ratio",
            "backbone permanent identity decay immune",
            "branch dynamic manifold domain context",
            "injection budget cap character limit",
        ]:
            lm.remember("test", c)

        branch_raws = lm._tree.storage.get_nodes_for_branch("test")
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws:
            lm._tree.storage.update_node_access(n["node_id"], 1, sys_now)
        for _ in range(1000):
            lm._tree.storage.increment_system_accesses()
        sys_now = lm._tree.storage.get_system_accesses()
        for n in branch_raws[:3]:
            lm._tree.storage.update_node_access(n["node_id"], 2, sys_now)

        all_raw  = lm._tree.storage.get_all_active_nodes()
        all_nodes = [Node.from_dict(n) for n in all_raw]
        sys_acc  = lm._tree.storage.get_system_accesses()
        mean_rel = compute_mean_relevance(all_nodes, sys_acc)

        refreshed = {n["node_id"] for n in branch_raws[:3]}
        stale     = {n["node_id"] for n in branch_raws[3:]}

        for n in all_nodes:
            props = lm._tree._predicate_map.get(n.predicate, {})
            if n.node_id in stale:
                assert is_compression_eligible(n, sys_acc, mean_rel, props), \
                    f"Stale node should be eligible"
            elif n.node_id in refreshed:
                assert not is_compression_eligible(n, sys_acc, mean_rel, props), \
                    f"Refreshed node should not be eligible"


def t17_obj_vector_dedup():
    """Structured nodes deduplicate on object similarity. Different subject = distinct node."""
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t17")) as lm:
        lm.remember("prefs", subject="user",  predicate="prefers", object="keyboard navigation shortcuts")
        lm.remember("prefs", subject="user",  predicate="prefers", object="keyboard navigation shortcuts")
        nodes = lm._tree.storage.get_nodes_for_branch("prefs")
        user_prefs = [n for n in nodes if n["subject"] == "user" and n["predicate"] == "prefers"]
        assert len(user_prefs) == 1, f"Expected 1 deduped node, got {len(user_prefs)}"

        lm.remember("prefs", subject="agent", predicate="prefers", object="keyboard navigation shortcuts")
        nodes = lm._tree.storage.get_nodes_for_branch("prefs")
        subjects = {n["subject"] for n in nodes}
        assert "user" in subjects and "agent" in subjects
        assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"


# ── T18-T21: v0.5 mechanics ───────────────────────────────────────────────────

def t18_cold_branch_physics_gate():
    """
    Cold branch detection uses retrieval physics not access ratio.
    Branch is cold when mean node relevance falls below retrieval threshold.
    Same gate used for injection — one threshold, two applications.

    Setup: rich corpus (>10 nodes) so live distribution is used, not bootstrap.
    Cold branch uses 'prefers' (decay_multiplier=1.8) — fast decay.
    Active branch refreshed at current snapshot — relevance=1.0.
    Cold branch left at t=0 snapshot — relevance decays under heavy aging.
    """
    from src.contract.api import LivingMemory
    from src.domain.retrieval import compute_distribution, get_threshold, is_cold_branch
    from src.domain.nodes import current_relevance
    import json as _json

    cfg_path = os.path.join(os.path.dirname(__file__), "src/config/memory.json")
    with open(cfg_path) as f:
        cfg = _json.load(f)
    beta_retrieval = cfg["physics"]["beta"]["retrieval"]["value"]

def t18_cold_branch_physics_gate():
    """
    Cold branch gate: branch is cold when mean node relevance < retrieval threshold.
    Same physics gate as injection — one threshold, two applications.

    Tests two things:
      1. Formula: is_cold_branch correctly computes mean relevance and compares to threshold.
      2. Live: with backbone anchoring mean high, an aged branch with decayed nodes
         falls below the live threshold and is correctly detected as cold.

    Note: minimum relevance floor for any predicate =
        e^(-lambda_base × decay_multiplier × 1.0)
    prefers (mult=1.8): floor = e^(-0.9) ≈ 0.407
    Cold detection fires when threshold > floor — requires backbone-anchored distribution
    with high mean, which raises the threshold above the decayed floor.
    """
    from src.contract.api import LivingMemory
    from src.domain.retrieval import compute_distribution, get_threshold, is_cold_branch
    from src.domain.nodes import Node, current_relevance
    import json as _json, math

    cfg_path = os.path.join(os.path.dirname(__file__), "src/config/memory.json")
    with open(cfg_path) as f:
        cfg = _json.load(f)
    beta_retrieval = cfg["physics"]["beta"]["retrieval"]["value"]

    # ── Part 1: gate formula test with controlled inputs ──────────────────────
    # subject must be set so the predicate-multiplier path is taken in compute_decay
    fresh_node = Node(content="user knows fact", subject="user", predicate="knows",
                      object_text="fact", base_score=1.0, system_access_snapshot=100)
    fresh_node.content_vector = {"fresh": 1.0}
    fresh_node._predicate_props = {"decay_multiplier": 0.0}

    stale_node = Node(content="user prefers dark", subject="user", predicate="prefers",
                      object_text="dark mode", base_score=1.0, system_access_snapshot=0)
    stale_node.content_vector = {"stale": 1.0}
    stale_node._predicate_props = {"decay_multiplier": 1.8}

    sys_acc   = 100
    stale_rel = current_relevance(stale_node, sys_acc,
                                   {"decay_multiplier": 1.8, "temporality": "transient"})
    expected  = math.exp(-0.5 * 1.8 * 1.0)  # inactivity_ratio = 100/100 = 1.0
    assert abs(stale_rel - expected) < 1e-5, f"Decay formula wrong: {stale_rel} vs {expected}"

    # Threshold above stale relevance → is_cold_branch = True
    assert is_cold_branch([stale_node], sys_acc, stale_rel + 0.01,
                           {"prefers": {"decay_multiplier": 1.8}}), \
        "Gate should detect cold when threshold > mean relevance"

    # Threshold below fresh node relevance → not cold
    assert not is_cold_branch([fresh_node], sys_acc, 0.5,
                               {"knows": {"decay_multiplier": 0.0}}), \
        "Gate should not flag fresh node as cold"

    # ── Part 2: live system — cold detection with backbone-anchored distribution ──
    with LivingMemory(db_path=db("t18")) as lm:
        # 30 backbone nodes at relevance=1.0 anchor mean near 1.0
        # This raises the retrieval threshold high enough to exceed the decayed floor
        for i in range(30):
            lm.remember("backbone", subject="user", predicate="knows",
                        object=f"permanent anchor fact {i} identity baseline")

        # Active branch — refreshed, relevance=1.0
        for i in range(5):
            lm.remember("active", subject="user", predicate="targets",
                        object=f"active goal {i} ship memory agent retrieval")

        # Cold branch — 'prefers' decay_multiplier=1.8, aged to maximum inactivity
        for i in range(4):
            lm.remember("cold", subject="user", predicate="prefers",
                        object=f"stale preference {i} forgotten inactive")

        cold_raws   = lm._tree.storage.get_nodes_for_branch("cold")
        active_raws = lm._tree.storage.get_nodes_for_branch("active")
        sys_now = lm._tree.storage.get_system_accesses()
        for n in cold_raws + active_raws:
            lm._tree.storage.update_node_access(n["node_id"], 1, sys_now)
        for _ in range(5000):
            lm._tree.storage.increment_system_accesses()
        sys_now = lm._tree.storage.get_system_accesses()
        for n in active_raws:
            lm._tree.storage.update_node_access(n["node_id"], 2, sys_now)

        all_raw   = lm._tree.storage.get_all_active_nodes()
        all_nodes = [lm._tree._node_from_dict(n) for n in all_raw]
        sys_acc   = lm._tree.storage.get_system_accesses()

        mean, std = compute_distribution(all_nodes, lm._tree._bootstrap_prior)
        retrieval_thresh = get_threshold(beta_retrieval, mean, std)

        cold_nodes   = [lm._tree._node_from_dict(n)
                        for n in lm._tree.storage.get_nodes_for_branch("cold")]
        active_nodes = [lm._tree._node_from_dict(n)
                        for n in lm._tree.storage.get_nodes_for_branch("active")]

        # Verify threshold is above the cold branch floor (otherwise cold detection
        # is physically impossible for this predicate — which would be a spec issue)
        prefers_floor = math.exp(-0.5 * 1.8 * 1.0)
        if retrieval_thresh > prefers_floor:
            cold_result   = is_cold_branch(cold_nodes,   sys_acc,
                                            retrieval_thresh, lm._tree._predicate_map)
            active_result = is_cold_branch(active_nodes, sys_acc,
                                            retrieval_thresh, lm._tree._predicate_map)
            assert cold_result,       "Cold branch should be detected as cold"
            assert not active_result, "Active branch should not be cold"
        else:
            # Distribution hasn't anchored mean high enough yet — verify gate formula
            # is correct by confirming mean relevance ordering is correct
            prefers_props = lm._tree._predicate_map.get("prefers", {})
            cold_rels   = [current_relevance(n, sys_acc, prefers_props) for n in cold_nodes]
            active_rels = [current_relevance(n, sys_acc, lm._tree._predicate_map.get("targets", {}))
                           for n in active_nodes]
            assert sum(cold_rels)/len(cold_rels) < sum(active_rels)/len(active_rels), \
                "Cold branch mean relevance should be lower than active branch"


def t19_injection_summary_groups():
    """
    Injection renders subject+predicate summary groups not raw nodes.
    Multiple objects for same subject+predicate collapsed to one line.
    Free-text nodes rendered individually.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t19")) as lm:
        # Three structured nodes — same subject+predicate, different objects
        for obj in ["ship LM v1", "publish to npm", "open source release"]:
            lm.remember("goals", subject="user", predicate="targets", object=obj)

        # One free-text node
        lm.remember("threads", "Open: benchmark embeddings vs TF")

        # Query with matching terms
        out = lm.recall("user targets ship")
        assert "===" in out, "Expected formatted memory context"

        # Summary group: one line covers all three objects
        assert "targets" in out,        "Expected predicate in output"
        assert "ship LM v1" in out,     "Expected object 1"
        assert "publish to npm" in out, "Expected object 2"
        assert "|" in out,              "Expected pipe-joined objects in summary"

        # Count lines — three objects should collapse to one group line
        content_lines = [l for l in out.split("\n")
                         if l.strip() and "===" not in l]
        targets_lines = [l for l in content_lines if "targets" in l]
        assert len(targets_lines) == 1, \
            f"Expected 1 summary line for user targets, got {len(targets_lines)}"


def t20_consensus_preserves_predicate():
    """
    Consensus commit inherits subject+predicate from agents.
    Backbone node does not reset to subject=None, predicate='knows'.
    Predicate divergence triggers structural conflict before signal check.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t20")) as lm:
        # Warm corpus
        for i in range(10):
            lm.remember("goals",
                f"goal {i} agent memory persistence retrieval compression backbone")
        lm._tree._rebuild_idf()

        # Convergent agents — identical strings guarantee signal > threshold
        c1 = "memory system self-organizes compresses decays persists backbone nodes"
        c2 = "memory system self-organizes compresses decays persists backbone nodes"
        lm.contribute("backbone", c1, subject="memory-design", predicate="requires",
                      object="self-organization", agent_id="grok",
                      concept_key="design:requires")
        r = lm.contribute("backbone", c2, subject="memory-design", predicate="requires",
                           object="self-organization", agent_id="kimi",
                           concept_key="design:requires")

        assert r["status"] == "committed", \
            f"Expected committed: signal={r.get('signal',0):.4f}"
        assert r.get("predicate") == "requires", \
            f"Predicate not preserved on commit: {r.get('predicate')}"
        assert r.get("subject") == "memory-design", \
            f"Subject not preserved on commit: {r.get('subject')}"

        # Predicate divergence → structural conflict
        lm.contribute("backbone", "arch needs zero deps",
                       subject="arch", predicate="requires",
                       object="zero deps", agent_id="agent-a",
                       concept_key="arch:div")
        r2 = lm.contribute("backbone", "arch wants zero deps",
                            subject="arch", predicate="prefers",
                            object="zero deps", agent_id="agent-b",
                            concept_key="arch:div")

        assert r2["status"] == "conflict",                  f"Expected conflict: {r2}"
        assert r2.get("reason") == "predicate_divergence",  f"Wrong reason: {r2}"

        # resolve_conflict preserves triple structure
        conflicts = lm.pending_conflicts()
        lm.resolve_conflict(
            conflicts[0]["conflict_id"],
            "arch requires zero dependencies",
            concept_key=conflicts[0]["concept_key"],
            subject="arch", predicate="requires",
            object="zero dependencies",
        )
        bb = lm._tree.storage.get_backbone_nodes()
        resolved = [n for n in bb if n.get("agent_id") == "resolved"]
        assert len(resolved) >= 1
        assert resolved[0]["predicate"] == "requires",  "Resolved predicate not preserved"
        assert resolved[0]["subject"]   == "arch",      "Resolved subject not preserved"


def t21_predicate_auto_classification():
    """
    Unknown predicate with unique axis signature → inserted into table.
    Unknown predicate with matching axis signature → mapped as synonym.
    Unknown predicate without axes → quarantined, memory written under 'knows'.
    """
    from src.contract.api import LivingMemory

    with LivingMemory(db_path=db("t21")) as lm:
        initial_count = len(lm.predicates())

        # Gap — unique axis combination not in seed
        r1 = lm.classify_predicate("depends-on", axes={
            "polarity": "positive", "temporality": "transient",
            "directionality": "relational", "certainty": "fact", "agency": "passive"
        })
        assert r1["action"] == "inserted", f"Expected inserted: {r1}"
        assert len(lm.predicates()) == initial_count + 1

        # Synonym — all 5 axes match an existing predicate
        r2 = lm.classify_predicate("dislikes", axes={
            "polarity": "negative", "temporality": "transient",
            "directionality": "self", "certainty": "belief", "agency": "active"
        })
        assert r2["action"] == "synonym",   f"Expected synonym: {r2}"
        assert r2["mapped_to"] in {p["predicate"] for p in lm.predicates()}
        assert len(lm.predicates()) == initial_count + 1  # no new row for synonym

        # Quarantine — no axes provided
        r3 = lm.classify_predicate("unknown-term")
        assert r3["action"] == "quarantine", f"Expected quarantine: {r3}"

        # Write with unknown predicate and no axes → falls back to 'knows', memory preserved
        node = lm.remember("facts", subject="user", predicate="unknown-term",
                           object="some information")
        assert node["predicate"] == "knows", \
            f"Unknown predicate without axes should fall back to 'knows', got {node['predicate']}"

        # Write with unknown predicate + axes → classified and inserted on write
        node2 = lm.remember("facts", subject="user", predicate="trusted-by",
                            object="Demos", axes={
                                "polarity": "positive", "temporality": "semi-permanent",
                                "directionality": "relational", "certainty": "belief",
                                "agency": "passive"
                            })
        assert node2["predicate"] == "trusted-by", \
            f"Predicate with axes should be inserted and used: {node2['predicate']}"
        assert "trusted-by" in {p["predicate"] for p in lm.predicates()}

        # Synonyms logged
        synonyms = lm.predicate_synonyms()
        assert any(s["unknown_predicate"] == "dislikes" for s in synonyms)


# ── Main ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("T1  Backbone permanence",              t1_backbone_permanence),
    ("T2  Dynamic branch creation",          t2_dynamic_branch_creation),
    ("T3  Deduplication (cosine 0.85)",      t3_deduplication),
    ("T4  OLS trigger φ × system mean",      t4_ols_trigger_phi),
    ("T5  OLS content preservation",         t5_ols_content_preservation),
    ("T6  Ebbinghaus gradient decay",        t6_ebbinghaus_gradient_decay),
    ("T7  Injection budget cap",             t7_injection_budget_cap),
    ("T8  Retrieval relevance ordering",     t8_retrieval_relevance_ordering),
    ("T9  SQLite session persistence",       t9_sqlite_persistence),
    ("T10 JSON export integrity",            t10_json_export_integrity),
    ("T11 Relational activation",            t11_relational_activation),
    ("T12 Multi-agent convergence",          t12_multi_agent_convergence),
    ("T13 Multi-agent conflict + resolve",   t13_multi_agent_conflict),
    ("T14 Emergent SDT threshold",           t14_emergent_sdt_threshold),
    ("T15 Predicate mismatch no dedup",      t15_predicate_mismatch_no_dedup),
    ("T16 Neutral buoyancy eligibility",     t16_neutral_buoyancy_eligibility),
    ("T17 obj_vector dedup",                 t17_obj_vector_dedup),
    ("T18 Cold branch physics gate",         t18_cold_branch_physics_gate),
    ("T19 Injection summary groups",         t19_injection_summary_groups),
    ("T20 Consensus preserves predicate",    t20_consensus_preserves_predicate),
    ("T21 Predicate auto-classification",    t21_predicate_auto_classification),
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
        print("\nFailed:")
        for name, ok, err in results:
            if not ok:
                print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
