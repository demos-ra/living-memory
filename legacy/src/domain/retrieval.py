# [F-retrieval/R1/C4] TF-IDF, SDT thresholds, injection, summary grouping
# R1 — Domain. Depends: R0 config, nodes.py. No I/O. Pure logic.
# v0.5 — content_vector rename, group_and_summarize, bootstrap from predicate table,
#         cold branch via retrieval physics gate

from __future__ import annotations
import math
import json
import os
from collections import Counter
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .nodes import Node

from .nodes import score_against_query, record_access, cosine_similarity


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG       = _load_config()
_PHYSICS   = _CFG["physics"]
MAX_GROUPS = _CFG["injection"]["max_summary_groups"]
BUDGET     = _CFG["injection"]["budget_chars"]
_BOOTSTRAP = _PHYSICS["bootstrap"]


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
        for term in node.content_vector:
            df[term] += 1
    return {term: round(math.log(n / count), 6) for term, count in df.items()}


def build_query_vector(query: str, corpus_idf: dict[str, float]) -> dict[str, float]:
    return compute_tfidf(query, corpus_idf)


# ── SDT threshold engine ──────────────────────────────────────────────────────

def compute_distribution(nodes: list["Node"],
                          bootstrap_prior: tuple[float, float] = None) -> tuple[float, float]:
    """
    Mean and std of pairwise cosine similarities across all active nodes.
    Bootstrap prior used when corpus below min_corpus_size.

    bootstrap_prior: (mean, std) derived from predicate table at init.
    Falls back to emergency values from config if prior not yet computed.
    """
    if len(nodes) < _BOOTSTRAP["min_corpus_size"]:
        if bootstrap_prior:
            return bootstrap_prior
        return _BOOTSTRAP["emergency_mean"], _BOOTSTRAP["emergency_std"]

    similarities = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_similarity(nodes[i].content_vector, nodes[j].content_vector)
            similarities.append(sim)

    if not similarities:
        if bootstrap_prior:
            return bootstrap_prior
        return _BOOTSTRAP["emergency_mean"], _BOOTSTRAP["emergency_std"]

    mean = sum(similarities) / len(similarities)
    variance = sum((s - mean) ** 2 for s in similarities) / len(similarities)
    std  = math.sqrt(variance)
    return round(mean, 6), round(std, 6)


def get_threshold(beta: float, mean: float, std: float,
                  stakes_multiplier: float = 1.0) -> float:
    """
    SDT threshold: (mean + log(β) × std) × stakes_multiplier
    Clamped to [0.05, 0.99].
    """
    criterion = math.log(beta) * std
    threshold = (mean + criterion) * stakes_multiplier
    return round(max(0.05, min(0.99, threshold)), 6)


def compute_bootstrap_prior(predicates: list[dict]) -> tuple[float, float]:
    """
    Derive cold-start prior from predicate table axis variance.

    Computes expected similarity between predicate pairs based on axis agreement.
    Same-axis-signature pairs → high similarity.
    Cross-axis pairs → lower similarity.
    Returns (mean, std) as initialization prior stored in meta.
    """
    AXES = ["polarity", "temporality", "directionality", "certainty", "agency"]

    if len(predicates) < 2:
        return _BOOTSTRAP["emergency_mean"], _BOOTSTRAP["emergency_std"]

    similarities = []
    for i in range(len(predicates)):
        for j in range(i + 1, len(predicates)):
            pa, pb = predicates[i], predicates[j]
            # Axis agreement ratio — fraction of axes that match
            matches = sum(1 for ax in AXES if pa.get(ax) == pb.get(ax))
            sim = matches / len(AXES)
            similarities.append(sim)

    mean = sum(similarities) / len(similarities)
    variance = sum((s - mean) ** 2 for s in similarities) / len(similarities)
    std  = math.sqrt(variance)
    return round(mean, 4), round(std, 4)


# ── Signal strength (consensus) ───────────────────────────────────────────────

def compute_signal_strength(agent_versions: list["Node"]) -> float:
    if len(agent_versions) < 2:
        return 0.0
    similarities = []
    for i in range(len(agent_versions)):
        for j in range(i + 1, len(agent_versions)):
            sim = cosine_similarity(
                agent_versions[i].content_vector,
                agent_versions[j].content_vector
            )
            similarities.append(sim)
    return round(sum(similarities) / len(similarities), 6)


# ── Cold branch physics gate ──────────────────────────────────────────────────

def is_cold_branch(branch_nodes: list["Node"],
                   system_total_accesses: int,
                   retrieval_threshold: float,
                   predicate_props_map: dict[str, dict] = None) -> bool:
    """
    Branch is cold when mean node relevance falls below retrieval threshold.
    Same physics gate used for injection — one threshold, two applications.

    predicate_props_map: {predicate: props_dict} for decay multiplier lookup.
    """
    if not branch_nodes:
        return True
    props_map = predicate_props_map or {}
    from .nodes import current_relevance
    relevances = [
        current_relevance(n, system_total_accesses, props_map.get(n.predicate))
        for n in branch_nodes
    ]
    mean_rel = sum(relevances) / len(relevances)
    return mean_rel < retrieval_threshold


# ── Summary grouping ──────────────────────────────────────────────────────────

def group_and_summarize(nodes: list["Node"]) -> list[dict]:
    """
    Group injected nodes by (subject, predicate) before rendering.
    Each group becomes one summary line — reduces injection slots,
    preserves all object information.

    Free-text nodes (subject=None) each form their own group — no collapsing.

    Returns list of summary dicts ordered by group size descending:
        {
            "subject":     str | None,
            "predicate":   str,
            "objects":     [str],
            "branch_name": str,
            "tags":        [str],
            "is_freetext": bool
        }
    """
    groups: dict[tuple, dict] = {}

    for node in nodes:
        if node.subject is None:
            # Free-text — each node is its own group, keyed by node_id
            key = ("__freetext__", node.node_id)
            groups[key] = {
                "subject":     None,
                "predicate":   node.predicate,
                "objects":     [node.object_text or node.content],
                "branch_name": node.branch_name,
                "tags":        node.tags,
                "is_freetext": True,
            }
        else:
            key = (node.subject, node.predicate)
            if key not in groups:
                groups[key] = {
                    "subject":     node.subject,
                    "predicate":   node.predicate,
                    "objects":     [],
                    "branch_name": node.branch_name,
                    "tags":        [],
                    "is_freetext": False,
                }
            if node.object_text and node.object_text not in groups[key]["objects"]:
                groups[key]["objects"].append(node.object_text)
            for tag in node.tags:
                if tag not in groups[key]["tags"]:
                    groups[key]["tags"].append(tag)

    # Sort: structured groups first (by object count desc), then free-text
    structured = [g for g in groups.values() if not g["is_freetext"]]
    freetext   = [g for g in groups.values() if g["is_freetext"]]
    structured.sort(key=lambda g: len(g["objects"]), reverse=True)

    return structured + freetext


# ── Injection ─────────────────────────────────────────────────────────────────

def conditional_injection(query: str,
                           all_nodes: list["Node"],
                           corpus_idf: dict[str, float],
                           system_total_accesses: int,
                           retrieval_threshold: float = None,
                           predicate_props_map: dict[str, dict] = None,
                           bootstrap_prior: tuple[float, float] = None) -> list["Node"]:
    """
    D3 injection — score all nodes, return those above retrieval threshold.

    Threshold is SDT-grounded and emergent from tree distribution.
    Budget cap enforced after threshold filter.
    """
    if retrieval_threshold is None:
        mean, std = compute_distribution(all_nodes, bootstrap_prior)
        retrieval_threshold = get_threshold(_CFG["physics"]["beta"]["retrieval"]["value"],
                                            mean, std)

    props_map  = predicate_props_map or {}
    query_tf   = compute_tf(tokenize(query))
    scored: list[tuple[float, "Node"]] = []

    for node in all_nodes:
        props = props_map.get(node.predicate)
        s = score_against_query(node, query_tf, system_total_accesses, props)
        if s >= retrieval_threshold:
            scored.append((s, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    injected: list["Node"] = []
    chars_used = 0

    for score, node in scored:
        node_len = len(node.content)
        if chars_used + node_len > BUDGET:
            break
        injected.append(node)
        chars_used += node_len
        props = props_map.get(node.predicate)
        record_access(node, system_total_accesses)

    return injected


def format_injection(nodes: list["Node"]) -> str:
    """
    Format injected nodes as summary groups.
    Structured nodes grouped by subject+predicate — one line per group.
    Free-text nodes rendered individually.
    """
    if not nodes:
        return ""

    groups = group_and_summarize(nodes)
    # Respect max_summary_groups cap
    groups = groups[:MAX_GROUPS]

    lines = ["=== MEMORY CONTEXT ==="]
    for g in groups:
        if g["is_freetext"]:
            branch = g["branch_name"]
            content = g["objects"][0] if g["objects"] else ""
            lines.append(f"[{branch}] {content}")
        else:
            branch    = g["branch_name"]
            subject   = g["subject"]
            predicate = g["predicate"]
            objects   = " | ".join(g["objects"])
            lines.append(f"[{branch}] {subject} {predicate}: {objects}")
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


# ── Relational activation ─────────────────────────────────────────────────────

def activate_relational(injected: list["Node"],
                         candidate_nodes: list["Node"],
                         co_access_partners: dict[str, list[str]],
                         budget_remaining: int) -> list["Node"]:
    """
    Hebbian activation — nodes that fire together wire together.
    Pulls co-access partners of injected nodes into context within budget.
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
