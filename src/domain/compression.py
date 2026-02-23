# [F-compression/R1/C4] OLS branch-level and root-level compression
# R1 — Domain. Depends: R0 config, nodes.py, retrieval.py. No I/O. Pure logic.
# v0.5 — content_vector rename, predicate compression property gates,
#         cold branch replaced by retrieval physics gate in retrieval.py

from __future__ import annotations
import math
import json
import os
from typing import Optional

from .nodes import (
    Node, cosine_similarity, is_compression_eligible, current_relevance
)
from .retrieval import tokenize, compute_tf, update_corpus_idf


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG = _load_config()
PHI  = _CFG["physics"]["phi"]["value"]


# ── OLS trigger ───────────────────────────────────────────────────────────────

def should_compress_branch(branch_nodes: list[Node],
                            all_branch_sizes: list[int]) -> bool:
    """
    φ-based self-organizing trigger.

    Compress when: len(branch) > φ × mean(all_branch_sizes)

    Threshold emerges from current system state — no hardcoded count.
    φ is the structural packing ratio at which branching tension reaches
    critical density — the same ratio that governs crystal formation and
    phyllotaxis. When a branch exceeds this ratio relative to the system mean,
    it has accumulated beyond its natural carrying capacity.
    """
    if not branch_nodes or not all_branch_sizes:
        return False
    system_mean = sum(all_branch_sizes) / len(all_branch_sizes)
    return len(branch_nodes) > PHI * system_mean


# ── Compression eligibility ───────────────────────────────────────────────────

def _has_active_conflict(node: Node, branch_nodes: list[Node]) -> bool:
    """
    Activity-based conflict detection.

    A subject+predicate pair is in active conflict when a sibling node at
    the same subject+predicate has never been accessed since it was written
    (access_count == 0). Zero accesses = the system has not resolved which
    claim is correct. Compressing now would erase an unresolved distinction.

    Free-text nodes (subject=None) have no structured conflict — skip.
    """
    if node.subject is None:
        return False
    for other in branch_nodes:
        if other.node_id == node.node_id:
            continue
        if other.subject != node.subject:
            continue
        if other.predicate != node.predicate:
            continue
        if other.access_count == 0:
            return True
    return False


def compute_mean_relevance(nodes: list[Node],
                            system_total_accesses: int,
                            predicate_props_map: dict[str, dict] = None) -> float:
    """
    Mean current relevance across all active nodes — the fluid density.

    Backbone nodes included: they hold relevance=1.0 permanently and anchor
    the mean upward. A node only sinks below mean when it is genuinely less
    active than its peers including the permanent backbone.

    predicate_props_map: {predicate: props_dict} for decay multiplier lookup.
    """
    if not nodes:
        return 1.0
    props_map = predicate_props_map or {}
    scores = [
        current_relevance(n, system_total_accesses, props_map.get(n.predicate))
        for n in nodes
    ]
    return sum(scores) / len(scores)


def get_compressible_nodes(branch_nodes: list[Node],
                            all_active_nodes: list[Node],
                            system_total_accesses: int,
                            predicate_props_map: dict[str, dict] = None) -> list[Node]:
    """
    Filter branch nodes to those eligible for compression.

    Four gates — all required:
      1. Not backbone
      2. compression != 'never'            (predicate physics gate)
      3. current_relevance < mean_relevance (neutral buoyancy — part in context of whole)
      4. access_count > 0                  (evidence gate — never compress unaccessed)
      5. No active conflict on subject+predicate (recency gate)

    mean_relevance computed from full active pool including backbone.
    Backbone anchors the mean at 1.0 — raises the bar appropriately.
    """
    props_map = predicate_props_map or {}
    mean_rel  = compute_mean_relevance(all_active_nodes, system_total_accesses, props_map)
    eligible  = []

    for node in branch_nodes:
        if node.is_backbone:
            continue

        props = props_map.get(node.predicate, {})

        # Gate 1 — predicate physics: never-compress predicates are structurally prohibited
        if props.get("compression") == "never":
            continue

        # Gate 2 — neutral buoyancy: must have sunk below system mean
        if not is_compression_eligible(node, system_total_accesses, mean_rel, props):
            continue

        # Gate 3 — evidence: must have been accessed at least once
        if node.access_count == 0:
            continue

        # Gate 4 — no active conflict on this subject+predicate pair
        if _has_active_conflict(node, branch_nodes):
            continue

        eligible.append(node)

    return eligible


# ── OLS core ─────────────────────────────────────────────────────────────────

def _merge_content(nodes: list[Node]) -> str:
    """
    Merge node contents — deduplicate sentences, preserve unique information.
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
    OLS centroid vector — minimizes squared distance from all source content_vectors.
    Normalized to unit magnitude.
    """
    if not nodes:
        return {}
    all_terms: set[str] = set()
    for node in nodes:
        all_terms.update(node.content_vector)
    n        = len(nodes)
    centroid = {
        term: round(sum(node.content_vector.get(term, 0.0) for node in nodes) / n, 6)
        for term in all_terms
    }
    magnitude = math.sqrt(sum(v * v for v in centroid.values()))
    if magnitude > 0:
        centroid = {t: round(v / magnitude, 6) for t, v in centroid.items()}
    return centroid


def _merge_obj_vectors(nodes: list[Node]) -> dict[str, float]:
    """
    Centroid of obj_vectors for compressed node.
    Falls back to empty dict for free-text nodes (no obj_vector).
    """
    vectors = [n.obj_vector for n in nodes if n.obj_vector]
    if not vectors:
        return {}
    all_terms: set[str] = set()
    for v in vectors:
        all_terms.update(v)
    n        = len(vectors)
    centroid = {
        t: round(sum(v.get(t, 0.0) for v in vectors) / n, 6)
        for t in all_terms
    }
    magnitude = math.sqrt(sum(v * v for v in centroid.values()))
    if magnitude > 0:
        centroid = {t: round(v / magnitude, 6) for t, v in centroid.items()}
    return centroid


# ── Branch-level OLS ──────────────────────────────────────────────────────────

def compress_branch(nodes: list[Node],
                    branch_name: str,
                    system_total_accesses: int,
                    all_active_nodes: list[Node],
                    predicate_props_map: dict[str, dict] = None) -> tuple[Node, list[str]]:
    """
    Branch-level OLS compression.

    Groups eligible nodes by (subject, predicate) — hard structural boundary.
    Cross-predicate merges are structurally impossible.
    Selects largest eligible group for maximum compression value.
    Caller loops for full branch compression.

    Returns: (compressed_node, list_of_consumed_node_ids)

    Compressed node inherits:
      - subject, predicate from group
      - peak base_score and system_access_snapshot
      - summed access_count
      - merged content_vector (OLS centroid)
      - merged obj_vector (centroid of object vectors)
      - object_text as pipe-joined objects for structured nodes
    """
    eligible = get_compressible_nodes(
        nodes, all_active_nodes, system_total_accesses, predicate_props_map
    )

    if not eligible:
        raise ValueError("No eligible nodes to compress in branch")

    # Group by (subject, predicate) — hard structural boundary
    groups: dict[tuple, list[Node]] = {}
    for node in eligible:
        key = (node.subject, node.predicate)
        groups.setdefault(key, []).append(node)

    # Pick largest group — most compression value per operation
    group_key   = max(groups, key=lambda k: len(groups[k]))
    to_compress = groups[group_key]

    if len(to_compress) < 2:
        raise ValueError(
            f"Largest eligible group {group_key} has only "
            f"{len(to_compress)} node — need at least 2"
        )

    subject, predicate = group_key

    merged_content    = _merge_content(to_compress)
    merged_cv         = _merge_vectors(to_compress)
    merged_obj_vector = _merge_obj_vectors(to_compress)
    consumed_ids      = [n.node_id for n in to_compress]

    # object_text: pipe-joined for structured, merged content for free-text
    object_text = (
        merged_content if subject is None
        else " | ".join(n.object_text for n in to_compress if n.object_text)
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
        content_vector          = merged_cv,
        subject                 = subject,
        predicate               = predicate,
        object_text             = object_text,
        obj_vector              = merged_obj_vector,
    )

    return compressed, consumed_ids


# ── Root-level OLS ────────────────────────────────────────────────────────────

def compress_cold_branch(branch_name: str,
                          nodes: list[Node],
                          system_total_accesses: int,
                          all_active_nodes: list[Node],
                          predicate_props_map: dict[str, dict] = None) -> tuple[Node, list[str]]:
    """
    Root-level OLS — compresses an entire cold branch into one archive node.
    Cold branch detection is handled by caller via retrieval physics gate.
    Passes predicate_props_map through to compress_branch for full physics gating.
    """
    if not nodes:
        raise ValueError("Cannot compress empty branch")
    compressed, consumed_ids = compress_branch(
        nodes, branch_name, system_total_accesses, all_active_nodes, predicate_props_map
    )
    compressed.tags.append("archived")
    compressed.tags.append(f"was:{branch_name}")
    return compressed, consumed_ids
