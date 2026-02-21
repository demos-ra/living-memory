# [F-compression/R1/C4] OLS branch-level and root-level compression
# R1 — Domain. Depends: R0 config, nodes.py, retrieval.py. No I/O. Pure logic.
# v0.4 — predicate-aware compression gates, physics-derived eligibility

from __future__ import annotations
import json
import os
import math
from typing import TYPE_CHECKING

from .nodes import Node, cosine_similarity, is_compression_eligible, current_relevance
from .retrieval import tokenize, compute_tf, update_corpus_idf


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG   = _load_config()
PHI    = _CFG["compression"]["ols_multiplier"]   # 1.618 — golden ratio
COLD_RATIO = _CFG["archival"]["cold_branch_access_ratio"]


# ── OLS trigger ───────────────────────────────────────────────────────────────

def should_compress_branch(branch_nodes: list[Node],
                            all_branch_sizes: list[int]) -> bool:
    """
    φ-based self-organizing trigger.

    Compress when: len(branch) > φ × mean(all_branch_sizes)

    Threshold emerges from current system state — no hardcoded count.
    """
    if not branch_nodes or not all_branch_sizes:
        return False

    system_mean = sum(all_branch_sizes) / len(all_branch_sizes)
    threshold   = PHI * system_mean

    return len(branch_nodes) > threshold


def is_cold_branch(branch_access_count: int,
                   all_branch_access_counts: list[int]) -> bool:
    """
    Branch is cold when its access ratio falls below COLD_RATIO × system mean.
    Cold branches are candidates for root-level OLS compression.
    """
    if not all_branch_access_counts:
        return False

    system_mean = sum(all_branch_access_counts) / len(all_branch_access_counts)
    threshold   = COLD_RATIO * system_mean

    return branch_access_count < threshold


# ── Compression eligibility ───────────────────────────────────────────────────

def _has_active_conflict(node: Node, branch_nodes: list[Node]) -> bool:
    """
    Activity-based conflict detection (v0.4).

    A subject+predicate pair is in active conflict if a node with the same
    subject+predicate exists that has never been accessed since it was written
    (access_count == 0). Zero accesses means the system hasn't resolved which
    claim is correct — compressing now would erase an unresolved distinction.

    Returns True if conflict is active (do not compress).
    """
    if node.subject is None:
        # Free-text fallback nodes — no structured conflict possible
        return False

    for other in branch_nodes:
        if other.node_id == node.node_id:
            continue
        if other.subject != node.subject:
            continue
        if other.predicate != node.predicate:
            continue
        # Same subject+predicate — check if unresolved (never accessed)
        if other.access_count == 0:
            return True

    return False


def compute_mean_relevance(nodes: list[Node], system_total_accesses: int) -> float:
    """
    Mean current relevance across all active nodes — the fluid density.

    Backbone nodes included: they hold relevance=1.0 permanently and
    act as the anchor that raises the mean, making the bar for sinking
    appropriately higher. Part-to-whole reference point.
    """
    if not nodes:
        return 1.0  # empty system — nothing is below mean
    scores = [current_relevance(n, system_total_accesses) for n in nodes]
    return sum(scores) / len(scores)


def get_compressible_nodes(branch_nodes: list[Node],
                            all_active_nodes: list[Node],
                            system_total_accesses: int) -> list[Node]:
    """
    Filter branch nodes to those eligible for compression.

    Four gates — all required (v0.4):
      1. Not backbone
      2. current_relevance < mean_relevance(all_active_nodes)  (neutral buoyancy)
      3. Has been accessed at least once                        (evidence gate)
      4. No active conflict on subject+predicate               (recency gate)

    mean_relevance computed from full active pool — part evaluated in
    context of whole. Backbone nodes anchor the mean at 1.0.
    """
    mean_rel = compute_mean_relevance(all_active_nodes, system_total_accesses)
    eligible = []
    for node in branch_nodes:
        if node.is_backbone:
            continue
        if not is_compression_eligible(node, system_total_accesses, mean_rel):
            continue
        if node.access_count == 0:
            continue
        if _has_active_conflict(node, branch_nodes):
            continue
        eligible.append(node)
    return eligible


# ── OLS compression core ──────────────────────────────────────────────────────

def _merge_content(nodes: list[Node]) -> str:
    """
    Merge node contents into a single compressed representation.
    Deduplicates sentences, preserves unique information.
    """
    seen:     set[str]  = set()
    segments: list[str] = []

    for node in nodes:
        parts = [p.strip() for p in node.content.replace("\n", ". ").split(".") if p.strip()]
        for part in parts:
            normalized = " ".join(part.lower().split())
            if normalized not in seen and len(normalized) > 5:
                seen.add(normalized)
                segments.append(part)

    return ". ".join(segments) + ("." if segments else "")


def _merge_vectors(nodes: list[Node]) -> dict[str, float]:
    """
    OLS equilibrium vector — centroid of all node tfidf_vectors.
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
        centroid[term] = round(
            sum(node.tfidf_vector.get(term, 0.0) for node in nodes) / n, 6
        )

    magnitude = math.sqrt(sum(v * v for v in centroid.values()))
    if magnitude > 0:
        centroid = {t: round(v / magnitude, 6) for t, v in centroid.items()}

    return centroid


def _merge_obj_vectors(nodes: list[Node]) -> dict[str, float]:
    """
    Centroid of obj_vectors for compressed node.
    Falls back to empty dict for free-text fallback nodes (no obj_vector).
    """
    vectors = [n.obj_vector for n in nodes if n.obj_vector]
    if not vectors:
        return {}

    all_terms: set[str] = set()
    for v in vectors:
        all_terms.update(v)

    n        = len(vectors)
    centroid = {t: round(sum(v.get(t, 0.0) for v in vectors) / n, 6) for t in all_terms}

    magnitude = math.sqrt(sum(v * v for v in centroid.values()))
    if magnitude > 0:
        centroid = {t: round(v / magnitude, 6) for t, v in centroid.items()}

    return centroid


# ── Branch-level OLS ──────────────────────────────────────────────────────────

def compress_branch(nodes: list[Node], branch_name: str,
                    system_total_accesses: int,
                    all_active_nodes: list[Node]) -> tuple[Node, list[str]]:
    """
    Branch-level OLS compression (v0.4).

    Groups eligible nodes by subject+predicate before merging.
    Only nodes sharing the same subject+predicate are merged together.
    Cross-predicate merges are structurally impossible.

    Returns: (compressed_node, list_of_consumed_node_ids)

    If multiple subject+predicate groups exist, merges the largest eligible
    group. Caller loops for full branch compression.
    """
    eligible = get_compressible_nodes(nodes, all_active_nodes, system_total_accesses)

    if not eligible:
        raise ValueError("No eligible nodes to compress in branch")

    # Group by (subject, predicate) — hard structural boundary
    groups: dict[tuple, list[Node]] = {}
    for node in eligible:
        key = (node.subject, node.predicate)
        groups.setdefault(key, []).append(node)

    # Pick largest group — most compression value
    group_key   = max(groups, key=lambda k: len(groups[k]))
    to_compress = groups[group_key]

    if len(to_compress) < 2:
        raise ValueError(
            f"Largest eligible group ({group_key}) has only {len(to_compress)} node — need at least 2"
        )

    merged_content    = _merge_content(to_compress)
    merged_vector     = _merge_vectors(to_compress)
    merged_obj_vector = _merge_obj_vectors(to_compress)
    consumed_ids      = [n.node_id for n in to_compress]

    subject, predicate = group_key

    object_text = merged_content if subject is None else " | ".join(
        n.object_text for n in to_compress if n.object_text
    )

    peak_score    = max(n.base_score for n in to_compress)
    peak_snapshot = max(n.system_access_snapshot for n in to_compress)
    all_tags      = list({tag for n in to_compress for tag in n.tags})

    compressed = Node(
        content                 = merged_content,
        branch_name             = branch_name,
        is_backbone             = False,
        base_score              = peak_score,
        access_count            = sum(n.access_count for n in to_compress),
        system_access_snapshot  = peak_snapshot,
        tags                    = all_tags,
        compressed_from         = consumed_ids,
        tfidf_vector            = merged_vector,
        subject                 = subject,
        predicate               = predicate,
        object_text             = object_text,
        obj_vector              = merged_obj_vector,
    )

    return compressed, consumed_ids


# ── Root-level OLS ────────────────────────────────────────────────────────────

def compress_cold_branch(branch_name: str, nodes: list[Node],
                         system_total_accesses: int,
                         all_active_nodes: list[Node]) -> tuple[Node, list[str]]:
    """
    Root-level OLS — compresses an entire cold branch into one archive node.
    Passes all_active_nodes through to compress_branch for neutral buoyancy gating.
    """
    if not nodes:
        raise ValueError("Cannot compress empty branch")

    compressed, consumed_ids = compress_branch(
        nodes, branch_name, system_total_accesses, all_active_nodes
    )
    compressed.tags.append("archived")
    compressed.tags.append(f"was:{branch_name}")

    return compressed, consumed_ids
