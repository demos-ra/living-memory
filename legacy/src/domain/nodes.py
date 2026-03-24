# [F-memory/R1/C4] Node structure, similarity, predicate-aware decay
# R1 — Domain. Depends: R0 config only. No I/O. Pure logic.
# v0.5 — content_vector rename, predicate decay multiplier from physics.lambda,
#         predicate properties loaded at runtime from predicates table via injected dict

from __future__ import annotations
import math
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    import os
    config_path = os.path.join(os.path.dirname(__file__), "../config/memory.json")
    with open(os.path.normpath(config_path)) as f:
        return json.load(f)

_CFG             = _load_config()
_PHYSICS         = _CFG["physics"]
LAMBDA_BASE      = _PHYSICS["lambda"]["base"]
LAMBDA_MULT      = _PHYSICS["lambda"]["temporality_multiplier"]
PHI              = _PHYSICS["phi"]["value"]

# Fallback predicate properties when table lookup unavailable
_DEFAULT_PREDICATE_PROPS = {
    "decay_multiplier":     1.0,
    "compression":          "eligible",
    "conflict_sensitivity": "medium",
    "temporality":          "transient",
}


# ── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    Single unit of memory in the three-dimensional memory space.

    D1 — Permanence:  backbone (permanent) → active → cold → archive
    D2 — Domain:      branch_name places node in conceptual manifold
    D3 — Relevance:   cosine similarity × decay buoyancy

    Triple structure (v0.4+):
        subject     — who or what the memory is about. None = free-text fallback.
        predicate   — semantic role from predicates table.
        object_text — the actual claim. Free text.
        obj_vector  — vector of object_text only. TF default, pluggable.

    content_vector (v0.5, was tfidf_vector):
        Full-content vector for retrieval scoring. Corpus-independent TF.

    content = "{subject} {predicate} {object_text}" for structured nodes.
    content = object_text for free-text fallback (subject=None).
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

    # Content vector — corpus-independent TF of full content, used for retrieval
    content_vector: dict[str, float] = field(default_factory=dict)

    # Agent provenance — None = single-agent mode
    agent_id: Optional[str] = None

    # Triple fields
    subject:      Optional[str]        = None
    predicate:    str                  = "knows"
    object_text:  str                  = ""
    obj_vector:   dict[str, float]     = field(default_factory=dict)

    # Predicate physics — injected from predicates table at write/load time
    # Not persisted — reloaded from table on every instantiation
    _predicate_props: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
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
            "content_vector":         self.content_vector,
            "agent_id":               self.agent_id,
            "subject":                self.subject,
            "predicate":              self.predicate,
            "object_text":            self.object_text,
            "obj_vector":             self.obj_vector,
        }

    @classmethod
    def from_dict(cls, d: dict, predicate_props: dict = None) -> "Node":
        _STORAGE_FIELDS = {"archived", "tfidf_vector"}
        clean = {k: v for k, v in d.items() if k not in _STORAGE_FIELDS}
        # Handle legacy tfidf_vector → content_vector
        if "tfidf_vector" in d and "content_vector" not in d:
            clean["content_vector"] = d["tfidf_vector"]
        node = cls(**{k: v for k, v in clean.items() if k != "_predicate_props"})
        if predicate_props:
            node._predicate_props = predicate_props
        return node

    def matches_agent(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return True
        return self.agent_id == agent_id

    @property
    def predicate_temporality(self) -> str:
        return self._predicate_props.get("temporality", "transient")

    @property
    def compression_eligibility(self) -> str:
        return self._predicate_props.get("compression", "eligible")

    @property
    def conflict_sensitivity(self) -> str:
        return self._predicate_props.get("conflict_sensitivity", "medium")


# ── Predicate validation ──────────────────────────────────────────────────────

def classify_predicate_axis(predicate: str,
                             known_predicates: list[dict]) -> dict:
    """
    Classify an unknown predicate by axis signature comparison.

    Compares against all known predicates on five axes:
        polarity, temporality, directionality, certainty, agency

    Returns:
        {"action": "synonym", "mapped_to": str}   — exact axis match found
        {"action": "gap",     "props": dict}       — unique signature, gap confirmed
        {"action": "ambiguous"}                    — equidistant between 2+ predicates

    Caller handles each case:
        synonym  → map to existing, log in predicate_synonyms
        gap      → insert new predicate with computed props
        ambiguous → quarantine node, do not insert predicate
    """
    AXES = ["polarity", "temporality", "directionality", "certainty", "agency"]

    # Build axis signature for the unknown predicate — requires caller to provide axes
    # This function operates on already-provided axis values, not inference
    # Inference is a v0.6 concern (embedding-based axis projection)
    # For v0.5: caller provides axes explicitly on contribute/remember
    return {"action": "gap", "props": {}}


def derive_predicate_props(polarity: str, temporality: str,
                            directionality: str, certainty: str,
                            agency: str) -> dict:
    """
    Derive matrix properties from axis positions.
    No human judgment — all properties computed from axes.

    decay_multiplier: from temporality (permanent=0, semi-permanent=0.3, transient=1.8)
    compression:      from certainty (fact=conservative, belief=eligible) + temporality
                      (permanent=never for facts, never for completed-style)
    conflict_sensitivity: from polarity (negative=high) + certainty (fact=high)
    """
    # decay_multiplier from temporality axis
    decay_mult = LAMBDA_MULT.get(temporality, 1.0)

    # compression eligibility
    if temporality == "permanent" and certainty == "fact":
        compression = "never"    # permanent facts never compress (completed, owns, knows)
    elif certainty == "fact":
        compression = "conservative"
    else:
        compression = "eligible"

    # conflict sensitivity
    if polarity == "negative" and certainty == "fact":
        conflict_sens = "high"
    elif polarity == "negative" or certainty == "fact":
        conflict_sens = "medium"
    else:
        conflict_sens = "low"

    return {
        "polarity":             polarity,
        "temporality":          temporality,
        "directionality":       directionality,
        "certainty":            certainty,
        "agency":               agency,
        "decay_multiplier":     decay_mult,
        "compression":          compression,
        "conflict_sensitivity": conflict_sens,
    }


# ── Decay ─────────────────────────────────────────────────────────────────────

def compute_decay(node: Node, system_total_accesses: int,
                  predicate_props: dict = None) -> float:
    """
    Ebbinghaus exponential decay — activity-relative, not clock-relative.

    relevance = base_score × e^(−λ_effective × inactivity_ratio)

    λ_effective = lambda_base × predicate_decay_multiplier
    Backbone nodes return base_score always — permanent, no decay.
    completed and permanent-fact predicates have multiplier=0 — effectively backbone.
    """
    if node.is_backbone:
        return node.base_score

    # Free-text fallback (subject=None) uses base lambda directly.
    # Predicate physics apply to structured triples only.
    if node.subject is None:
        lambda_effective = LAMBDA_BASE
    else:
        props = predicate_props or node._predicate_props or _DEFAULT_PREDICATE_PROPS
        decay_multiplier = props.get("decay_multiplier", 1.0)
        lambda_effective = LAMBDA_BASE * decay_multiplier

    accesses_since   = system_total_accesses - node.system_access_snapshot
    inactivity_ratio = accesses_since / max(system_total_accesses, 1)
    decay            = node.base_score * math.exp(-lambda_effective * inactivity_ratio)
    return round(decay, 6)


def current_relevance(node: Node, system_total_accesses: int,
                      predicate_props: dict = None) -> float:
    return compute_decay(node, system_total_accesses, predicate_props)


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
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


def is_duplicate(new_obj_vector: dict[str, float],
                 new_subject: Optional[str],
                 new_predicate: str,
                 existing_nodes: list[Node],
                 dedup_threshold: float) -> Optional[Node]:
    """
    Predicate-aware deduplication.

    All three conditions required:
      1. subject match
      2. predicate match — hard gate, different predicate = different claim
      3. cosine(obj_vector_a, obj_vector_b) >= dedup_threshold
    """
    best_score = 0.0
    best_node  = None
    for node in existing_nodes:
        if node.subject != new_subject:
            continue
        if node.predicate != new_predicate:
            continue
        score = cosine_similarity(new_obj_vector, node.obj_vector)
        if score > best_score:
            best_score = score
            best_node  = node
    if best_score >= dedup_threshold:
        return best_node
    return None


def is_compression_eligible(node: Node, system_total_accesses: int,
                             mean_relevance: float,
                             predicate_props: dict = None) -> bool:
    """
    Neutral buoyancy gate — node sinks below system mean relevance.
    Predicate compression property adds additional gates:
        'never'        — structurally prohibited (completed, conflicts-with)
        'conservative' — requires access_count > 0 (evidence gate)
        'eligible'     — standard neutral buoyancy gate only
    """
    if node.is_backbone:
        return False

    props = predicate_props or node._predicate_props or _DEFAULT_PREDICATE_PROPS
    compression = props.get("compression", "eligible")

    if compression == "never":
        return False

    rel = current_relevance(node, system_total_accesses, props)
    if rel >= mean_relevance:
        return False

    return True


def score_against_query(node: Node, query_vector: dict[str, float],
                        system_total_accesses: int,
                        predicate_props: dict = None) -> float:
    semantic = cosine_similarity(node.content_vector, query_vector)
    buoyancy = current_relevance(node, system_total_accesses, predicate_props)
    return round(semantic * buoyancy, 6)


def record_access(node: Node, system_total_accesses: int) -> Node:
    node.access_count += 1
    node.system_access_snapshot = system_total_accesses
    return node
