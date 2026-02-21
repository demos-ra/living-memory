# [F-retrieval/R1/C2] TF-IDF vectors, SDT-grounded thresholds, conditional injection
# R1 — Domain. Depends: R0 config, nodes.py. No I/O. Pure logic.

from __future__ import annotations
import math
import json
import os
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nodes import Node

from .nodes import score_against_query, record_access


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG        = _load_config()
TOP_N       = _CFG["injection"]["top_n"]
BUDGET_CHARS = _CFG["injection"]["budget_chars"]
_T          = _CFG["thresholds"]
_BOOTSTRAP  = _T["bootstrap"]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total  = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_tfidf(text: str, corpus_idf: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(text)
    tf     = compute_tf(tokens)
    if not corpus_idf:
        return tf
    tfidf = {}
    for term, tf_val in tf.items():
        idf = corpus_idf.get(term, math.log(2))
        tfidf[term] = round(tf_val * idf, 6)
    return tfidf


def update_corpus_idf(all_nodes: list["Node"]) -> dict[str, float]:
    n = len(all_nodes)
    if n == 0:
        return {}
    df: dict[str, int] = Counter()
    for node in all_nodes:
        for term in node.tfidf_vector:
            df[term] += 1
    return {term: round(math.log(n / count), 6) for term, count in df.items()}


def build_query_vector(query: str, corpus_idf: dict[str, float]) -> dict[str, float]:
    return compute_tfidf(query, corpus_idf)


# ── SDT threshold engine ──────────────────────────────────────────────────────

def compute_distribution(nodes: list["Node"]) -> tuple[float, float]:
    """
    Compute mean and std of pairwise cosine similarities across all active nodes.
    Returns (mean, std). Used as the base distribution for all SDT thresholds.

    Falls back to bootstrap values when corpus is below min_corpus_size.
    Bootstrap prevents threshold collapse on cold start.
    """
    from .nodes import cosine_similarity

    if len(nodes) < _BOOTSTRAP["min_corpus_size"]:
        return _BOOTSTRAP["bootstrap_mean"], _BOOTSTRAP["bootstrap_std"]

    similarities = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_similarity(nodes[i].tfidf_vector, nodes[j].tfidf_vector)
            similarities.append(sim)

    if not similarities:
        return _BOOTSTRAP["bootstrap_mean"], _BOOTSTRAP["bootstrap_std"]

    mean = sum(similarities) / len(similarities)
    variance = sum((s - mean) ** 2 for s in similarities) / len(similarities)
    std  = math.sqrt(variance)

    return round(mean, 6), round(std, 6)


def get_threshold(beta: float, mean: float, std: float,
                  stakes_multiplier: float = 1.0) -> float:
    """
    SDT-grounded threshold.

    threshold = (mean + log(β) × std) × stakes_multiplier

    β = cost(false_positive) / cost(false_negative)
    log(β) is the mathematically correct criterion position in SDT.

    β > 1 → log(β) > 0 → threshold above mean (conservative, avoid false alarms)
    β < 1 → log(β) < 0 → threshold below mean (generous, avoid misses)
    β = 1 → log(β) = 0 → threshold at mean (symmetric costs)
    """
    criterion = math.log(beta) * std
    threshold = (mean + criterion) * stakes_multiplier
    # Clamp to [0.05, 0.99] — prevent degenerate thresholds
    return round(max(0.05, min(0.99, threshold)), 6)


# ── Signal strength (consensus) ───────────────────────────────────────────────

def compute_signal_strength(agent_versions: list["Node"]) -> float:
    """
    Mean pairwise cosine similarity across agent versions of same concept.
    High signal = agents have converged. Low signal = agents diverge.
    Used to determine whether pending content commits to backbone.
    """
    from .nodes import cosine_similarity

    if len(agent_versions) < 2:
        return 0.0

    similarities = []
    for i in range(len(agent_versions)):
        for j in range(i + 1, len(agent_versions)):
            sim = cosine_similarity(
                agent_versions[i].tfidf_vector,
                agent_versions[j].tfidf_vector
            )
            similarities.append(sim)

    return round(sum(similarities) / len(similarities), 6)


# ── Injection ─────────────────────────────────────────────────────────────────

def conditional_injection(
    query: str,
    all_nodes: list["Node"],
    corpus_idf: dict[str, float],
    system_total_accesses: int,
    retrieval_threshold: float = None,
) -> list["Node"]:
    """
    Core injection mechanic — D3 slice.

    Threshold is SDT-grounded and emergent from tree distribution.
    Passed in from memory_tree which computes it once per query.
    Falls back to bootstrap if not provided.
    """
    if retrieval_threshold is None:
        # Fallback — should always be provided by memory_tree
        mean, std = compute_distribution(all_nodes)
        retrieval_threshold = get_threshold(_T["retrieval"]["beta"], mean, std)

    query_tf     = compute_tf(tokenize(query))
    scored: list[tuple[float, "Node"]] = []

    for node in all_nodes:
        s = score_against_query(node, query_tf, system_total_accesses)
        if s >= retrieval_threshold:
            scored.append((s, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    injected: list["Node"] = []
    chars_used = 0

    for score, node in scored[:TOP_N]:
        node_len = len(node.content)
        if chars_used + node_len > BUDGET_CHARS:
            break
        injected.append(node)
        chars_used += node_len
        record_access(node, system_total_accesses)

    return injected


def format_injection(nodes: list["Node"]) -> str:
    if not nodes:
        return ""
    lines = ["=== MEMORY CONTEXT ==="]
    for node in nodes:
        tag_str    = f"[{', '.join(node.tags)}]" if node.tags else ""
        agent_str  = f"({node.agent_id})" if node.agent_id else ""
        lines.append(f"[{node.branch_name}]{agent_str} {tag_str} {node.content}".strip())
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


# ── Relational activation ─────────────────────────────────────────────────────

def activate_relational(
    injected: list["Node"],
    candidate_nodes: list["Node"],
    co_access_partners: dict[str, list[str]],
    budget_remaining: int,
) -> list["Node"]:
    """
    Hebbian relational activation — nodes that fire together wire together.
    Pulls co-access partners of injected nodes into context within budget.
    Edges are implicit and emergent from activity history.
    """
    if not co_access_partners:
        return []

    injected_ids = {n.node_id for n in injected}
    node_by_id   = {n.node_id: n for n in candidate_nodes}
    activated    = []
    chars_used   = 0
    seen: set[str] = set()

    for node in injected:
        for partner_id in co_access_partners.get(node.node_id, []):
            if partner_id in injected_ids or partner_id in seen:
                continue
            seen.add(partner_id)
            partner = node_by_id.get(partner_id)
            if partner is None:
                continue
            if chars_used + len(partner.content) > budget_remaining:
                break
            activated.append(partner)
            chars_used += len(partner.content)

    return activated
