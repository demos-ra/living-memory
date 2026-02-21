# [F-memory/R1/C4] Node structure, similarity scoring, activity-based decay
# R1 — Domain. Depends: R0 config only. No I/O. Pure logic.
# v0.4 — triple memory unit (subject, predicate, object_text, obj_vector)

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

_CFG             = _load_config()
LAMBDA           = _CFG["decay"]["lambda"]
# Thresholds are SDT-grounded and computed dynamically in retrieval.py
# Bootstrap values kept as cold-start fallback only
_BOOTSTRAP       = _CFG["thresholds"]["bootstrap"]
DEDUP_THRESH     = _BOOTSTRAP["bootstrap_mean"] + 0.25   # fallback only
RETRIEVAL_THRESH = _BOOTSTRAP["bootstrap_mean"] - 0.25   # fallback only

# Predicate vocabulary — locked v1, loaded from R0
_PRED_CFG        = _CFG["predicates"]
PREDICATE_VOCAB  = set(_PRED_CFG["vocabulary"])
DEFAULT_PREDICATE = _PRED_CFG["default_unstructured"]  # "knows"


# ── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    Single unit of memory. Lives inside a branch or the backbone.

    Dimensions:
      D1 — permanence:  backbone nodes never decay; branch nodes do
      D2 — domain:      branch_name places node in conceptual manifold
      D3 — relevance:   base_score * decay = current buoyancy in relevance-space

    Triple structure (v0.4):
      subject    — who or what the memory is about. None = free-text fallback.
      predicate  — semantic role from locked vocabulary. Default: "knows".
      object_text — the actual claim. Free text.
      obj_vector  — vector of object_text only. TF default, pluggable via embedding_fn.

    content is always subject + predicate + object_text joined for backward compat.
    For free-text fallback, content = object_text = full original string.
    """
    node_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:      str            = ""
    branch_name:  str            = "backbone"
    is_backbone:  bool           = False
    base_score:   float          = 1.0
    access_count: int            = 0
    system_access_snapshot: int  = 0
    tags:         list[str]      = field(default_factory=list)
    compressed_from: list[str]   = field(default_factory=list)

    # Full-content vector — corpus-independent TF, used for retrieval ranking
    tfidf_vector: dict[str, float] = field(default_factory=dict)

    # Agent provenance — None = single-agent mode
    agent_id: Optional[str] = None

    # Triple fields (v0.4)
    # subject=None signals free-text fallback — predicate defaults to "knows"
    subject:      Optional[str]            = None
    predicate:    str                      = DEFAULT_PREDICATE
    object_text:  str                      = ""
    obj_vector:   dict[str, float]         = field(default_factory=dict)

    def __post_init__(self):
        """
        Ensure content is always populated.
        Structured triple: content = "subject predicate object_text"
        Free-text fallback (subject=None): content = object_text
        If content was provided directly (legacy or compressed nodes), leave it.
        """
        if not self.content:
            if self.subject is not None:
                self.content = f"{self.subject} {self.predicate} {self.object_text}"
            else:
                self.content = self.object_text

    def to_dict(self) -> dict:
        return {
            "node_id":                self.node_id,
            "content":                self.content,
            "branch_name":            self.branch_name,
            "is_backbone":            self.is_backbone,
            "base_score":             self.base_score,
            "access_count":           self.access_count,
            "system_access_snapshot": self.system_access_snapshot,
            "tags":                   self.tags,
            "compressed_from":        self.compressed_from,
            "tfidf_vector":           self.tfidf_vector,
            "agent_id":               self.agent_id,
            "subject":                self.subject,
            "predicate":              self.predicate,
            "object_text":            self.object_text,
            "obj_vector":             self.obj_vector,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        # Strip storage-layer fields that don't belong on the domain object
        _STORAGE_FIELDS = {"archived"}
        clean = {k: v for k, v in d.items() if k not in _STORAGE_FIELDS}
        return cls(**clean)

    def matches_agent(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return True
        return self.agent_id == agent_id


# ── Predicate validation ──────────────────────────────────────────────────────

def is_known_predicate(predicate: str) -> bool:
    """True if predicate is in the locked v1 vocabulary."""
    return predicate in PREDICATE_VOCAB


# ── Decay ─────────────────────────────────────────────────────────────────────

def compute_decay(node: Node, system_total_accesses: int) -> float:
    """
    Ebbinghaus exponential decay — activity-relative, not clock-relative.

    relevance = base_score * e^(-λ * inactivity_ratio)

    inactivity_ratio = (system_total - node_snapshot) / max(system_total, 1)
    Range [0, 1]. 0 = just accessed. 1 = never accessed relative to system.
    Backbone nodes return base_score always — permanent, no decay.
    """
    if node.is_backbone:
        return node.base_score

    accesses_since   = system_total_accesses - node.system_access_snapshot
    inactivity_ratio = accesses_since / max(system_total_accesses, 1)
    decay            = node.base_score * math.exp(-LAMBDA * inactivity_ratio)
    return round(decay, 6)


def current_relevance(node: Node, system_total_accesses: int) -> float:
    """Public interface for current buoyancy of a node in relevance-space (D3)."""
    return compute_decay(node, system_total_accesses)


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Cosine similarity between two sparse vectors.
    Returns float in [0, 1]. 1.0 = identical direction.
    """
    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a) & set(vec_b)
    if not shared:
        return 0.0

    dot   = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return round(dot / (mag_a * mag_b), 6)


def is_duplicate(
    new_obj_vector: dict[str, float],
    new_subject: Optional[str],
    new_predicate: str,
    existing_nodes: list[Node],
    dedup_threshold: float,
) -> Optional[Node]:
    """
    Predicate-aware deduplication (v0.4).

    All three conditions required for a duplicate match:
      1. subject match  — None matches None only
      2. predicate match — hard gate, different predicate = never duplicate
      3. cosine(obj_vector_a, obj_vector_b) >= dedup_threshold

    Returns the most similar matching node, or None.
    """
    best_score = 0.0
    best_node  = None

    for node in existing_nodes:
        # Gate 1 — subject must match
        if node.subject != new_subject:
            continue
        # Gate 2 — predicate must match (hard — different predicate = different claim)
        if node.predicate != new_predicate:
            continue
        # Gate 3 — object vector similarity
        score = cosine_similarity(new_obj_vector, node.obj_vector)
        if score > best_score:
            best_score = score
            best_node  = node

    if best_score >= dedup_threshold:
        return best_node
    return None


def is_compression_eligible(node: Node, system_total_accesses: int,
                             mean_relevance: float) -> bool:
    """
    Neutral buoyancy compression eligibility (v0.4).

    A node is eligible when its current relevance falls below the mean
    relevance of all active nodes — it is denser than the surrounding fluid.

    Part-to-whole: the gate is relative, not absolute. A node only sinks
    when it is genuinely less active than its peers. If everything has aged
    equally, nothing is below mean — nothing compresses. Compression only
    fires when access patterns have diverged and created real density differences.

    Backbone nodes hold relevance=1.0 always — permanently above any mean.
    They are the anchor density of the fluid. Never eligible.
    """
    if node.is_backbone:
        return False
    return current_relevance(node, system_total_accesses) < mean_relevance


def score_against_query(node: Node, query_vector: dict[str, float],
                        system_total_accesses: int) -> float:
    """
    Combined relevance score for injection ranking.
    = cosine_similarity(tfidf_vector, query) * current_relevance

    Semantically similar + recently active = highest rank.
    """
    semantic = cosine_similarity(node.tfidf_vector, query_vector)
    buoyancy = current_relevance(node, system_total_accesses)
    return round(semantic * buoyancy, 6)


# ── Access bump ───────────────────────────────────────────────────────────────

def record_access(node: Node, system_total_accesses: int) -> Node:
    """
    Bump access count and reset snapshot on retrieval.
    Resets inactivity_ratio to ~0, restoring buoyancy (anti-decay).
    """
    node.access_count += 1
    node.system_access_snapshot = system_total_accesses
    return node
