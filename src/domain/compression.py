# [F-compression/R1/C2] OLS branch-level and root-level compression
# R1 — Domain. Depends: R0 config, nodes.py, retrieval.py. No I/O. Pure logic.

from __future__ import annotations
import json
import os
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .nodes import Node, cosine_similarity
from .retrieval import tokenize, compute_tf, update_corpus_idf


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG            = _load_config()
PHI             = _CFG["compression"]["ols_multiplier"]   # 1.618 — golden ratio
MIN_TO_COMPRESS = _CFG["compression"]["min_nodes_to_compress"]
COLD_RATIO      = _CFG["archival"]["cold_branch_access_ratio"]


# ── OLS trigger ───────────────────────────────────────────────────────────────

def should_compress_branch(branch_nodes: list[Node],
                            all_branch_sizes: list[int]) -> bool:
    """
    φ-based self-organizing trigger.

    Compress when: len(branch) > φ × mean(all_branch_sizes)

    No hardcoded node count. The threshold emerges from current system state.
    As the agent uses more memory and branches grow, threshold rises naturally.
    """
    if len(branch_nodes) < MIN_TO_COMPRESS:
        return False

    if not all_branch_sizes:
        return False

    system_mean = sum(all_branch_sizes) / len(all_branch_sizes)
    threshold   = PHI * system_mean

    return len(branch_nodes) > threshold


def is_cold_branch(branch_access_count: int,
                   all_branch_access_counts: list[int]) -> bool:
    """
    A branch is cold when its access ratio falls below COLD_RATIO × system mean.
    Cold branches are candidates for root-level OLS compression.
    """
    if not all_branch_access_counts:
        return False

    system_mean = sum(all_branch_access_counts) / len(all_branch_access_counts)
    threshold   = COLD_RATIO * system_mean

    return branch_access_count < threshold


# ── OLS compression core ──────────────────────────────────────────────────────

def _merge_content(nodes: list[Node]) -> str:
    """
    Merge node contents into a single compressed representation.
    Deduplicates sentences, preserves unique information.
    Simple but effective for agent memory content.
    """
    seen:     set[str]  = set()
    segments: list[str] = []

    for node in nodes:
        # Split on sentence boundaries
        parts = [p.strip() for p in node.content.replace("\n", ". ").split(".") if p.strip()]
        for part in parts:
            # Deduplicate by normalized form
            normalized = " ".join(part.lower().split())
            if normalized not in seen and len(normalized) > 5:
                seen.add(normalized)
                segments.append(part)

    return ". ".join(segments) + ("." if segments else "")


def _merge_vectors(nodes: list[Node]) -> dict[str, float]:
    """
    OLS equilibrium vector — centroid of all node TF-IDF vectors.
    This is the 'line of best fit' representation of the node cluster.
    Minimizes squared distance from all source vectors.
    """
    if not nodes:
        return {}

    all_terms: set[str] = set()
    for node in nodes:
        all_terms.update(node.tfidf_vector)

    centroid: dict[str, float] = {}
    n = len(nodes)

    for term in all_terms:
        centroid[term] = round(sum(node.tfidf_vector.get(term, 0.0) for node in nodes) / n, 6)

    # Normalize centroid vector
    magnitude = math.sqrt(sum(v * v for v in centroid.values()))
    if magnitude > 0:
        centroid = {t: round(v / magnitude, 6) for t, v in centroid.items()}

    return centroid


# ── Branch-level OLS ──────────────────────────────────────────────────────────

def compress_branch(nodes: list[Node], branch_name: str,
                    system_total_accesses: int) -> tuple[Node, list[str]]:
    """
    Branch-level OLS compression.

    Takes N nodes from a branch, produces 1 compressed node.
    Returns: (compressed_node, list_of_consumed_node_ids)

    The compressed node:
    - Inherits the highest base_score of its sources (preserves peak relevance)
    - Carries compressed_from list for audit trail
    - Gets centroid TF-IDF vector (OLS equilibrium)
    - Merged content deduplicates and preserves unique information
    """
    if len(nodes) < MIN_TO_COMPRESS:
        raise ValueError(f"Need at least {MIN_TO_COMPRESS} nodes to compress, got {len(nodes)}")

    merged_content = _merge_content(nodes)
    merged_vector  = _merge_vectors(nodes)
    consumed_ids   = [n.node_id for n in nodes]

    # Inherit peak score — preserve the highest buoyancy from sources
    peak_score = max(n.base_score for n in nodes)
    # Inherit most recent access snapshot
    peak_snapshot = max(n.system_access_snapshot for n in nodes)
    # Aggregate tags
    all_tags = list({tag for n in nodes for tag in n.tags})

    compressed = Node(
        content                 = merged_content,
        branch_name             = branch_name,
        is_backbone             = False,
        base_score              = peak_score,
        access_count            = sum(n.access_count for n in nodes),
        system_access_snapshot  = peak_snapshot,
        tags                    = all_tags,
        compressed_from         = consumed_ids,
        tfidf_vector            = merged_vector,
    )

    return compressed, consumed_ids


# ── Root-level OLS ────────────────────────────────────────────────────────────

def compress_cold_branch(branch_name: str, nodes: list[Node],
                         system_total_accesses: int) -> tuple[Node, list[str]]:
    """
    Root-level OLS — compresses an entire cold branch into one archive node.

    This is the cross-branch compression that enables multi-year memory.
    Cold branches sink through D1 and get compressed at the root boundary
    before archival. The archive node retains semantic content but takes
    minimal space.
    """
    if not nodes:
        raise ValueError("Cannot compress empty branch")

    # For root-level, we compress everything in the branch
    compressed, consumed_ids = compress_branch(nodes, branch_name, system_total_accesses)
    # Mark clearly as an archive-level compressed node
    compressed.tags.append("archived")
    compressed.tags.append(f"was:{branch_name}")

    return compressed, consumed_ids
