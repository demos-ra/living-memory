# [F-memory/R1/C2] Node structure, similarity scoring, activity-based decay
# R1 — Domain. Depends: R0 config only. No I/O. Pure logic.

from __future__ import annotations
import math
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ── Config loader (R0 → R1) ──────────────────────────────────────────────────

def _load_config() -> dict:
    import os
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG = _load_config()
LAMBDA         = _CFG["decay"]["lambda"]
DEDUP_THRESH   = _CFG["similarity"]["dedup_threshold"]
RETRIEVAL_THRESH = _CFG["similarity"]["retrieval_threshold"]


# ── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    Single unit of memory. Lives inside a branch or the backbone.

    Dimensions:
      D1 — permanence:  backbone nodes never decay; branch nodes do
      D2 — domain:      branch_name places node in conceptual manifold
      D3 — relevance:   base_score * decay = current buoyancy in relevance-space
    """
    node_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:      str            = ""
    branch_name:  str            = "backbone"
    is_backbone:  bool           = False
    base_score:   float          = 1.0    # initial relevance weight
    access_count: int            = 0      # how many times retrieved
    system_access_snapshot: int  = 0      # system total accesses at last touch
    tags:         list[str]      = field(default_factory=list)
    compressed_from: list[str]   = field(default_factory=list)  # node_ids merged into this

    # TF-IDF vector — populated by retrieval.py at insert time
    tfidf_vector: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id":               self.node_id,
            "content":               self.content,
            "branch_name":           self.branch_name,
            "is_backbone":           self.is_backbone,
            "base_score":            self.base_score,
            "access_count":          self.access_count,
            "system_access_snapshot":self.system_access_snapshot,
            "tags":                  self.tags,
            "compressed_from":       self.compressed_from,
            "tfidf_vector":          self.tfidf_vector,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        # Strip storage-layer fields that don't belong on the domain object
        _STORAGE_FIELDS = {"archived"}
        clean = {k: v for k, v in d.items() if k not in _STORAGE_FIELDS}
        return cls(**clean)


# ── Decay ─────────────────────────────────────────────────────────────────────

def compute_decay(node: Node, system_total_accesses: int) -> float:
    """
    Ebbinghaus exponential decay — activity-relative, not clock-relative.

    Inactivity = how much the system has been accessed since this node was
    last touched. Backbone nodes return 1.0 always — they never decay.

    relevance = base_score * e^(-λ * inactivity_ratio)

    inactivity_ratio = (system_total - node_snapshot) / max(system_total, 1)
    Range [0, 1]. 0 = just accessed. 1 = never accessed relative to system.
    """
    if node.is_backbone:
        return node.base_score  # permanent — no decay

    accesses_since = system_total_accesses - node.system_access_snapshot
    inactivity_ratio = accesses_since / max(system_total_accesses, 1)
    decay = node.base_score * math.exp(-LAMBDA * inactivity_ratio)
    return round(decay, 6)


def current_relevance(node: Node, system_total_accesses: int) -> float:
    """Public interface for current buoyancy of a node in relevance-space (D3)."""
    return compute_decay(node, system_total_accesses)


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Cosine similarity between two TF-IDF sparse vectors.
    Returns float in [0, 1]. 1.0 = identical direction.
    """
    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a) & set(vec_b)
    if not shared:
        return 0.0

    dot    = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a  = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return round(dot / (mag_a * mag_b), 6)


def is_duplicate(new_vector: dict[str, float], existing_nodes: list[Node]) -> Optional[Node]:
    """
    Check if new content is a near-duplicate of any existing node.
    Returns the most similar node if above DEDUP_THRESH, else None.
    Standard threshold: 0.85 (Manning & Schütze, IR canonical).
    """
    best_score = 0.0
    best_node  = None

    for node in existing_nodes:
        score = cosine_similarity(new_vector, node.tfidf_vector)
        if score > best_score:
            best_score = score
            best_node  = node

    if best_score >= DEDUP_THRESH:
        return best_node
    return None


def score_against_query(node: Node, query_vector: dict[str, float],
                        system_total_accesses: int) -> float:
    """
    Combined relevance score for injection ranking.
    = cosine_similarity(node, query) * current_relevance(node)

    Nodes that are semantically similar BUT have decayed heavily rank lower.
    Nodes that are active AND relevant rank highest.
    This is buoyancy in D3 — relevance-space.
    """
    semantic = cosine_similarity(node.tfidf_vector, query_vector)
    buoyancy = current_relevance(node, system_total_accesses)
    return round(semantic * buoyancy, 6)


# ── Access bump ───────────────────────────────────────────────────────────────

def record_access(node: Node, system_total_accesses: int) -> Node:
    """
    When a node is retrieved, bump its access count and reset its snapshot.
    This resets the inactivity_ratio to ~0, restoring buoyancy (anti-decay).
    """
    node.access_count += 1
    node.system_access_snapshot = system_total_accesses
    return node
