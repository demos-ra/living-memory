# [F-retrieval/R1/C2] TF-IDF vectors, cosine retrieval, conditional injection
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

_CFG            = _load_config()
TOP_N           = _CFG["injection"]["top_n"]
BUDGET_CHARS    = _CFG["injection"]["budget_chars"]
RETRIEVAL_THRESH = _CFG["similarity"]["retrieval_threshold"]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Bare-bones tokenizer. Lowercase, split on non-alphanumeric.
    No external deps. Sufficient for agent memory content.
    """
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Remove single-char tokens — low signal
    return [t for t in tokens if len(t) > 1]


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency — normalized by document length."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total  = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_tfidf(text: str, corpus_idf: dict[str, float]) -> dict[str, float]:
    """
    TF-IDF vector for a piece of text given a corpus IDF table.
    If corpus_idf is empty (first node), falls back to pure TF.
    This is a sparse vector — only non-zero terms stored.
    """
    tokens = tokenize(text)
    tf     = compute_tf(tokens)

    if not corpus_idf:
        return tf  # bootstrap: no corpus yet, use raw TF

    tfidf = {}
    for term, tf_val in tf.items():
        idf = corpus_idf.get(term, math.log(2))  # unseen term gets log(2) ≈ 0.693
        tfidf[term] = round(tf_val * idf, 6)

    return tfidf


def update_corpus_idf(all_nodes: list["Node"]) -> dict[str, float]:
    """
    Recompute IDF table from all active nodes.
    Called after insert/compress operations.
    IDF(term) = log(N / df(term)) where df = document frequency.
    """
    n = len(all_nodes)
    if n == 0:
        return {}

    df: dict[str, int] = Counter()
    for node in all_nodes:
        for term in node.tfidf_vector:
            df[term] += 1

    idf = {}
    for term, count in df.items():
        idf[term] = round(math.log(n / count), 6)

    return idf


# ── Injection ─────────────────────────────────────────────────────────────────

def build_query_vector(query: str, corpus_idf: dict[str, float]) -> dict[str, float]:
    """Vectorize the incoming query against current corpus IDF."""
    return compute_tfidf(query, corpus_idf)


def conditional_injection(
    query: str,
    all_nodes: list["Node"],
    corpus_idf: dict[str, float],
    system_total_accesses: int,
) -> list["Node"]:
    """
    Core injection mechanic — D3 slice.

    1. Vectorize query
    2. Score every node: cosine_similarity * current_relevance (buoyancy)
    3. Filter below retrieval threshold
    4. Sort descending
    5. Take top-N within budget_chars hard cap

    Returns ordered list of nodes to inject into agent context.
    Mutates access counts on returned nodes (buoyancy restoration).
    """
    query_vector = build_query_vector(query, corpus_idf)

    scored: list[tuple[float, "Node"]] = []
    for node in all_nodes:
        s = score_against_query(node, query_vector, system_total_accesses)
        if s >= RETRIEVAL_THRESH:
            scored.append((s, node))

    # Sort by score descending — highest buoyancy first
    scored.sort(key=lambda x: x[0], reverse=True)

    # Apply budget cap — never blow context window
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
    """
    Format injected nodes as a clean string for agent context.
    Each node prefixed with its branch and tags for agent orientation.
    """
    if not nodes:
        return ""

    lines = ["=== MEMORY CONTEXT ==="]
    for node in nodes:
        tag_str = f"[{', '.join(node.tags)}]" if node.tags else ""
        lines.append(f"[{node.branch_name}] {tag_str} {node.content}".strip())
    lines.append("=== END MEMORY ===")

    return "\n".join(lines)
