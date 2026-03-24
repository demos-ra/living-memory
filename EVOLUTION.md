# Evolution

**Living Memory v1.0**

---

## Overview

Living Memory developed across three generations. Each retained the core thesis — structure determines relevance — and shed imposed structure that the previous generation mistook for essential. The trajectory is one of successive reduction: from a prototype with correct intuitions and incorrect mechanisms, to a formally specified system with frozen axioms.

This document records what each generation contributed, what it got wrong, and what the transition to the next generation revealed about the distinction between imposed and essential structure.

---

## Gen 0 — Living Memory (Python)

**Repository:** [legacy/](legacy/)
**Language:** Python
**Architecture:** Self-organizing tree on Living Architecture v2.0 (R0–R4 layers)
**Dependencies:** None

Gen 0 was a prototype. It represented memory as a tree with a backbone of core concepts and branches of associated detail. Retrieval used TF-IDF cosine similarity. Consolidation used OLS compression weighted by φ (the golden ratio). Decay followed an Ebbinghaus curve calibrated to activity rather than clock time. Relations between nodes strengthened through Hebbian co-access: nodes activated together developed stronger connections.

### What Gen 0 got right

**Activity-relative decay.** Decay was not a function of wall-clock time but of traversal activity. A node accessed frequently decayed slowly regardless of calendar duration. This was the first recognition that persistence should be structural, not temporal — though the mechanism was still an explicit decay function rather than a conservation consequence.

**Emergent relations.** Hebbian co-access produced relational structure that was not declared at creation time. Nodes that were repeatedly activated together developed strong edges. The system discovered associations rather than having them assigned.

**Identity persistence.** Backbone concepts resisted decay and anchored the tree. The insight — that some nodes are more central than others and should be harder to displace — survived into all subsequent generations, though the mechanism changed from a hardcoded list to emergent attractor dynamics.

### What Gen 0 got wrong

**Hardcoded backbone.** Core concepts were declared, not computed. The system could not discover that a topic had become central to identity — it had to be told. This violated what would later become MA5 (Emergent Identity).

**Tree, not graph.** A tree enforces a single parent per node. Memory is not hierarchical in this way. A node can belong to multiple contexts simultaneously, which requires a graph.

**Explicit decay function.** Ebbinghaus decay, even activity-relative, is an imposed correction. It assumes that forgetting is a process that must be actively managed. Under conservation (MA1), forgetting is a consequence of redistribution — not a separate mechanism.

**Separate read and write.** Storage and retrieval were distinct operations with distinct code paths. This created the possibility that what was stored and what was retrieved could diverge.

**Lexical matching.** TF-IDF cosine similarity operates on surface tokens. Two statements with identical meaning but different vocabulary would not match. Embedding-based binding eliminated this limitation.

---

## Gen 1 — F-memory v0.6 (TypeScript)

**Location:** Proprietary, inside LIV (livai.dev)
**Language:** TypeScript
**Architecture:** Weighted directed graph with energy conservation

Gen 1 moved from tree to graph. It introduced energy conservation (Σ activations = 1), lateral inhibition, embedding-based similarity, and a vital signs system for monitoring graph health. Memory was stored in Postgres. Nodes had embeddings and edges had weights. The system was substantially more capable than Gen 0.

### What Gen 1 got right

**Graph topology.** The move from tree to graph was the most important structural change across all three generations. It enabled multi-membership, lateral connections, and cycle participation — all prerequisites for the layer transition mechanisms of v1.0.

**Energy conservation.** Total activation summed to one. This was the first implementation of what became MA1. It eliminated runaway accumulation and made competition between nodes intrinsic to the system rather than externally managed.

**Lateral inhibition.** Nodes competed for finite activation. This prevented hub monopoly — a single highly connected node could not dominate the graph indefinitely.

**Vital signs.** Observable metrics (activation entropy, degree distribution, layer distribution) provided a diagnostic surface without altering the graph's dynamics. Observation was separated from structure.

**Embeddings.** Semantic similarity replaced lexical matching. Two statements with different words but the same meaning produced similar binding scores.

### What Gen 1 got wrong

**Concept/episode type split.** Nodes were tagged as either `concept` or `episode`. This imposed a categorical distinction that the graph's own topology should determine. A node's role — whether it functions as a stable pattern or a transient event — is a consequence of its connectivity, not a label assigned at creation. This violated what would later become MA5.

**Hardcoded backbone concepts.** As in Gen 0, identity nodes were declared (`BACKBONE_CONCEPTS`). The system still could not discover its own center. MA5 again.

**Separate read and write.** `remember()` and `recall()` were distinct functions. The same divergence risk as Gen 0, now in a more complex system. MA2.

**Explicit structural decay.** A constant (`STRUCTURAL_DECAY = 0.05`) was applied per session. Decay was still an imposed correction rather than a conservation consequence. MA1.

**Predicate-based persistence tiers.** Nodes were assigned persistence levels (`HIGH`, `MED`, `LOW`) by a predicate function. Persistence was a property of the node rather than a property of the node's position in the graph. MA3.

**Concept extraction.** An `extractConcepts()` function tokenized input and applied statistical weighting. This imposed a decomposition that the graph should solve through binding (§7.2 of SPEC.md). MA2.

---

## Gen 2 — Living Memory v1.0

**Specification:** [SPEC.md](SPEC.md)
**Axioms:** [AXIOMS.md](AXIOMS.md)
**Language:** Implementation-independent (specification only in this repository)

Gen 2 is the current version. It eliminates every imposed mechanism identified in Gen 0 and Gen 1 and replaces them with emergent equivalents under six frozen axioms.

### What changed

| Imposed (Gen 0/Gen 1) | Emergent (Gen 2) | Governing Axiom |
|------------------------|-------------------|-----------------|
| Hardcoded backbone / `BACKBONE_CONCEPTS` | Attractor basin computed by random walk convergence | MA5 |
| Concept/episode type tags | Layer computed from connectivity | MA5 |
| Ebbinghaus decay / `STRUCTURAL_DECAY = 0.05` | Conservation. Redistribution is the only dynamic. | MA1 |
| `remember()` / `recall()` | Single traversal operation | MA2 |
| `extractConcepts()` / tokenization | Embedding projection + graph-solved decomposition | MA2 |
| `PERSISTENCE_HIGH/MED/LOW` | Connectivity-based survival | MA3 |
| `computeUtility()` | M1 purpose bias applied during propagation | MA2 |
| `derivePersistence()` predicate | Structural closure and stability thresholds | MA3 |

### What Gen 2 introduced

**Five M-layers.** A crystallization gradient from raw input (M4) to identity (M0), with each transition adding a qualitatively new invariant. Layers are computed, not assigned.

**One traversal operation.** Read, write, and learning collapsed into a single pass. No separate code paths. No synchronization hazards.

**Propagation as decomposition.** Binding projects input onto the graph's embedding space. The graph decomposes compound input into weighted activation across multiple subgraphs. No external extraction required.

**Arms.** Community-detected overlapping subgraphs that span layers. Arm depth indicates identity integration. Arm merging is identity evolution.

**Present State Layer.** A derived computation that reads the graph and produces structured statements about the user's current state. Confidence tiers, source text availability, and arm categorization — all derived from topology, not from extraction.

**Source text lifecycle.** Original input text is retained at M3 and pruned at the M3 → M2 transition. The episodic wrapper drops; structural position carries meaning.

**Graph-relative thresholds.** Every threshold is derived from the graph's own statistical properties. The graph defines its own noise floor and its own promotion criteria.

**Frozen axioms.** Six axioms (MA1–MA6) that define the irreducible constraints. Axiom text is immutable. This is the first generation to draw a formal boundary between what can change (rules, parameters, implementation) and what cannot (axioms).

---

## The Trajectory

Each generation preserved one insight and discarded the scaffolding around it.

Gen 0 discovered that structure determines relevance — but imposed the structure (tree, backbone, decay function) rather than letting it emerge.

Gen 1 moved to graph topology and conservation — but retained imposed categories (concept/episode), imposed identity (backbone concepts), and imposed persistence (predicate tiers).

Gen 2 eliminated all imposed structure. Layer is computed. Identity is emergent. Persistence is connectivity. Decay is conservation. Read and write are one operation. The six axioms define the boundary: everything inside them is frozen; everything outside them is implementation.

The pattern across generations: what looked like a necessary mechanism was, each time, an imposed approximation of a property that the graph could compute for itself.

---

## References

- [AXIOMS.md](AXIOMS.md) — Six frozen axioms
- [SPEC.md](SPEC.md) — Full architectural specification
- [README.md](README.md) — Overview
- [legacy/](legacy/) — Gen 0 Python source

---

*Living Memory v1.0 · Implementation: [livai.dev](https://livai.dev)*
