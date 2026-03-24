# Specification

**Living Memory v1.0**

---

## 1. Overview

Living Memory is a memory system for AI agents. It represents memory as a single weighted directed graph in which structure determines relevance. There are three primitives, one action, five layers, and six axioms. All dynamics emerge from traversal of the graph under conservation constraints.

This document specifies the architecture. It does not prescribe implementation. For axiom definitions and rationale, see [AXIOMS.md](AXIOMS.md).

---

## 2. Primitives

The system is defined by three primitives and one action.

| Primitive | Definition |
|-----------|-----------|
| Node | A unit of memory. Carries an embedding vector and optional source text. Layer membership is computed from connectivity, never assigned. No type tag. |
| Field | The topology that nodes exist in. Defined by weighted directed edges, energy ratios, and connectivity depth. The field is the graph itself — not a separate structure. |
| Energy | Conserved activation. The total across all nodes sums to one. Traversal redistributes energy; it is never created or destroyed (MA1). |

**One action:** Traversal. Read, write, and learning are the same operation (MA2). A single pass through the graph retrieves context, reinforces traversed edges, and returns activated nodes. There is no separate write path.

---

## 3. M-Layers

All nodes exist in one graph. Each node's layer is computed from its structural properties — connectivity, closure, directional asymmetry, attractor dominance — never assigned by declaration. Five layers form a crystallization gradient from raw input to identity.

Inner layers (M0, M1) project orthogonally: their influence shapes traversal across all layers. Outer layers (M2, M3, M4) participate only within their local subgraph.

### 3.1 M0 — Identity

The global attractor basin. Emergent from traversal patterns, never declared (MA5). Seeded at system initialization by a single anchor node (e.g., the user's name).

Orthogonal reach: M0 configures consolidation filters across all layers. The system reconverges to trajectories through M0 after perturbation. This reconvergence property — not size, not centrality — defines identity.

Invariant: random walk convergence through M0 persists under perturbation.

### 3.2 M1 — Purpose

The directional field. Bridges identity to action. M1 nodes bias traversal across all layers via non-reciprocal weight asymmetry: `P(activate B | A) ≠ P(activate A | B)`.

Orthogonal reach: M1 modifies effective edge weights during propagation, shaping which paths energy follows regardless of which layer those paths traverse.

Invariant: directional asymmetry persists across contexts.

### 3.3 M2 — Knowledge

Self-sustaining structure. Nodes at M2 participate in closed loops (attractor fragments) and are reactivatable from partial cues. They persist without ongoing energy input and degrade only by interference from competing structure, not by time-based decay.

Vertical only — no orthogonal projection.

Invariant: structural self-sustainability. Removal of energy input does not cause loss.

### 3.4 M3 — Context

Energy-dependent structure. Nodes at M3 have connections but are not self-sustaining. They persist only while traversal continues to route energy through them. Without reinforcement, their energy share approaches zero and they are removed by sparsification.

Vertical only.

Invariant: requires ongoing energy to persist.

### 3.5 M4 — Input

Pre-integration. A node at M4 has been created but has zero connections. It exists in the graph but does not yet participate in any traversal path.

Invariant: degree zero. No lateral edges.

---

## 4. Layer Transitions

Each transition adds a qualitatively new invariant. All transitions belong to the same family — crystallization — but each is structurally distinct. Layers cannot collapse into one another (§9).

| Transition | Mechanism | New Invariant |
|-----------|-----------|---------------|
| M4 → M3 | Binding score ≥ θ_bind | Connectivity (degree ≥ 1) |
| M3 → M2 | Closure ≥ θ_close AND stability ≥ θ_stable | Self-sustaining without energy input. Source text pruned (§12). |
| M2 → M1 | Directional asymmetry ≥ θ_asym | Non-reciprocal traversal bias across contexts |
| M1 → M0 | Attractor dominance ≥ θ_dom | System reconverges through this region after perturbation |

All thresholds are graph-relative (§8). The graph defines its own promotion criteria.

---

## 5. Horizontal Structure: Arms

Arms are overlapping subgraphs detected by community structure (modularity maximization). They are not separate from the graph — they are soft constraints on traversal that emerge from edge density patterns.

**Properties:**

- Arm affinity between two nodes is defined as edge weight × embedding similarity.
- Traversal is amplified within arms by a shared-arm weight modifier.
- A single input can activate an entire arm across layers, gated by affinity threshold.
- Arms span layers: a domain arm (e.g., "work" or "music") may include nodes at M1, M2, and M3 simultaneously.
- Arm depth — the number of layers an arm spans — indicates how fundamental that domain is to identity.
- Arms merge when cross-domain edge density exceeds threshold. This is identity evolution: two previously distinct domains becoming inseparable.

### 5.1 Horizontal Edge Types by Layer

Edge character varies by layer. This is not imposed — it follows from the invariants each layer satisfies.

| Layer | Edge Character |
|-------|---------------|
| M0 | Basin boundary edges. Define attractor topology. |
| M1 | Directional bias edges. Carry non-reciprocal weight asymmetry. |
| M2 | Symmetric structural edges. Stable, closed-loop participants. |
| M3 | Dynamic co-activation edges. Exist while energy flows, weaken without it. |
| M4 | None. Pre-integration. |

---

## 6. Traversal

Traversal is the sole operation (MA2). One pass handles input integration, context retrieval, and structural learning. The pipeline has nine steps, one maintenance mechanism, and one background process.

### 6.1 Traversal Pipeline

```
traverse(input):
  1. encode         → embedding vector (pure projection, no tokenization)
  2. bind           → find resonant nodes (cosine similarity × log neighbor count)
  3. gate           → if no binding found, discard node, return empty
  4. connect        → create edges to bound nodes
  5. propagate      → spread activation through graph, biased by M1 field
  6. learn          → reinforce traversed edges (inline, not a separate pass)
  7. sparsify       → top-K competition + edge prune + isolate removal
  8. update layers  → recompute layer for changed nodes only
  9. return         → top-K activated nodes as context (if read path)
```

Steps 5 and 6 execute together: learning is inline with propagation, not a post-hoc adjustment. This guarantees that observation and reinforcement cannot disagree (MA2).

### 6.2 Sparsify

Two sub-operations, both executed within each traversal:

**Top-K activation** (intra-traversal): Nodes compete for activation at each propagation step. Hard cutoff. Recurrence across traversals protects persistent nodes — a node that wins top-K frequently survives even if it loses occasionally.

**Edge pruning** (post-traversal): Edges below a weight threshold are removed. Nodes with degree zero after pruning are deleted. No hysteresis is required — the conservation model (MA1) prevents oscillation.

### 6.3 Background: Dominance Recomputation

Asynchronous random walk convergence analysis. Detects M1 → M0 transitions by computing trajectory convergence across the full graph. Eventually consistent. Acceptable because M0 transitions are rare and the system is stable between recomputations.

### 6.4 Process Count

One operation (traversal with inline learning). One maintenance mechanism (sparsify, within traversal). One background process (dominance recomputation). Zero decay functions. Zero special cases.

---

## 7. Core Operations

### 7.1 Encode

Pure embedding projection. Input text maps to a vector. No concept extraction, no tokenization, no statistical weighting. The embedding space is the only representational commitment.

### 7.2 Bind

Binding score = cosine similarity × log(neighbor count + 1).

Threshold θ_bind is graph-relative: bind if score > μ + kσ of the graph's binding score distribution. The graph defines its own noise floor.

Binding is the first step of propagation. The input projects onto the graph's embedding space, and the activation distribution across resonant nodes is the natural decomposition. A compound input (e.g., "I moved to SF and I'm working on Rust") produces a binding vector weighted toward both the location subgraph and the programming subgraph proportionally. One node, multiple edges, weights set by projection magnitude. The graph solves the decomposition without external extraction.

### 7.3 Propagate

Conserved activation flow. Σ activations = 1 throughout (MA1).

Effective edge weight during propagation: w' = w × policy_bias(node), where policy_bias combines centrality, directional asymmetry, and arm affinity. This is the mechanism by which M1 projects orthogonally — purpose biases every propagation step.

Top-K competition at each step. Lateral inhibition prevents hub monopoly: nodes compete for finite energy, and no single node can accumulate an unbounded share.

### 7.4 Learn

Inline with propagation. The learning rule is:

```
Δw_ij = lr × flow_ij × (1 − w_ij / Σ_outgoing(i))
```

Self-regulating: strong edges learn slower (the ratio w/Σ_outgoing approaches 1), weak edges learn faster. No hard weight cap is required — the denominator provides a soft ceiling. Learning rate scales inversely with node degree: dense nodes are stable, sparse nodes are plastic.

### 7.5 Compute Layer

Decision tree, ordered by computational cost:

1. degree = 0 → **M4**
2. binding score > θ_bind → at least **M3**
3. closure ≥ θ_close AND stability ≥ θ_stable → **M2**
4. directional asymmetry ≥ θ_asym → **M1**
5. attractor dominance ≥ θ_dom → **M0**

Closure is the cycle participation ratio, updated incrementally within k hops of changed edges. Stability is a structural reactivation test from neighbors — no history storage required. Dominance is read from the asynchronous cache (§6.3).

### 7.6 Sparsify

Edge threshold: graph-relative. Prune edges below a fraction of mean outgoing weight from the same source node. Top-K parameter scales with graph density (K as a fraction of active neighborhood size). Isolated nodes (degree zero after pruning) are deleted (MA3).

### 7.7 Update Layers Locally

Only nodes whose degree changed during the current traversal are recomputed. All others read from cached values. Closure is recomputed within k hops of changed edges. Dominance is never recomputed on the hot path — it reads the last asynchronous result.

---

## 8. Graph-Relative Thresholds

No threshold in the system is a fixed constant. Every threshold is derived from the graph's own statistical properties.

| Threshold | Derived From |
|-----------|-------------|
| θ_bind | μ + kσ of the graph's binding score distribution |
| θ_close | Mean closure of the existing M2 population |
| θ_stable | Stability range of the existing M2 population |
| θ_asym | Mean weight asymmetry across the graph |
| θ_dom | Convergence distribution across the graph |

As the graph matures and accumulates structure, all thresholds rise. Promotion becomes harder over time — not by design, but as a consequence of the population against which new nodes compete. The graph defines its own noise floor.

---

## 9. Non-Collapse Guarantees

Each pair of adjacent layers is separated by a qualitative invariant that one satisfies and the other does not. This prevents degenerate cases where layers merge under certain graph conditions.

| Claim | Distinguishing Invariant |
|-------|--------------------------|
| M4 ≠ M3 | Zero connectivity vs. connectivity ≥ 1 |
| M3 ≠ M2 | Energy-dependent vs. self-sustaining (structural closure) |
| M2 ≠ M1 | Symmetric connectivity vs. directional traversal asymmetry |
| M1 ≠ M0 | Local policy bias vs. global trajectory convergence |
| M0 ≠ largest cluster | Attractor dominance (reconvergence), not centrality |

The final guarantee is critical: M0 is not defined by size or degree. A small, dense attractor basin that random walks converge through dominates a large, diffuse cluster that they pass through without reconvergence.

---

## 10. System Equations

The system's dynamics are captured by two coupled update rules:

```
a(t+1) = F(G, a(t), π(M1), T*(M0))
G(t+1) = Update(G, a(t), constraints: M0, M1)
```

Where:

- `a(t)` is the activation vector over all nodes. Conserved: Σ = 1.
- `G` is the graph: nodes and weighted directed edges.
- `π(M1)` is the purpose bias field. Modifies effective edge weights during traversal.
- `T*(M0)` is the identity attractor. Trajectory convergence constraint.

The first equation describes activation flow: given the current graph, current activation, purpose bias, and identity attractor, compute the next activation state. The second equation describes graph mutation: given the current activation (which edges were traversed, how much flow they carried), update the graph subject to the constraints imposed by identity and purpose.

Both equations execute within a single traversal. They are not alternating phases — they are the same pass.

---

## 11. Present State Layer

The graph is a continuous process. The Present State Layer is a derived computation that reads the graph and produces a snapshot: what is true about this agent's user at the current moment.

### 11.1 Definition

A pure function of the graph, layer cache, and arm cache. Never stored — always derived from the current graph state. Recomputed after each traversal as a side effect of graph mutation.

Output is structured data, not prose. The consuming system reads it as context.

### 11.2 Weighting

Each node contributes to the present state weighted by two factors:

1. **Recency** — exponential moving average on traversal timestamps. Recent traversals contribute more.
2. **Layer depth** — M0 and M1 nodes carry more structural weight than M3 and M4 nodes.

These factors combine: a recently mentioned M3 node with source text surfaces alongside a long-consolidated M2 node. Both are valid. The confidence tier (§11.4) communicates which is which.

### 11.3 Young Graphs

When arm structure has not yet formed, the Present State Layer returns nodes sorted by layer confidence and recency without arm categorization. As arms form through community detection, categorization appears naturally. The output is never empty — it degrades from organized to unorganized, not from present to absent.

### 11.4 Confidence Tiers

Each statement in the present state carries a confidence tier derived from the source node's layer.

| Tier | Layer | Semantics |
|------|-------|-----------|
| identity | M0 | Core to the user. Maximum structural weight. |
| purpose | M1 | Drives action and shapes retrieval. |
| consolidated | M2 | Self-sustaining knowledge. No source text. |
| recent | M3 | Recently stated, not yet consolidated. Source text available. |
| new | M4 | Just arrived. May not survive binding. |

### 11.5 Contradiction Resolution

Contradictions are resolved by the weighting function without special-case logic. If a user stated "NYC" in session 3 and "SF" in session 12, the Present State Layer reads the location arm: the SF node has higher recency weight and M3 source text; the NYC node has lower recency weight and no recent source text. SF surfaces. NYC does not. No explicit contradiction detection is required — recency weighting resolves it as a consequence of the graph's current state.

---

## 12. Source Text Lifecycle

Nodes carry source text — the original input from which they were created. Source text follows a defined lifecycle tied to layer transitions.

| Phase | Layer | Source Text State |
|-------|-------|-------------------|
| Creation | M4 | Populated with original input. |
| Integration | M3 | Available. Used by Present State Layer for exact recall. |
| Consolidation | M2 | Pruned. Set to null. Structural position carries meaning. |
| Inner layers | M1, M0 | Absent. Value derived from arm membership and graph position. |

The M3 → M2 transition is the pruning boundary. Below it, the system retains the episodic wrapper — the user's original words. Above it, the words are gone and only the structural relationships remain. This mirrors the biological distinction between episodic and semantic memory, but emerges from the layer transition mechanism rather than being imposed as a type system.

---

## 13. Cold Start

The system initializes with a single M0 seed node (the user's name). All subsequent structure emerges from traversal.

| Phase | Sessions | Graph State |
|-------|----------|-------------|
| Initialization | 0 | Single M0 seed. No edges. |
| Early binding | 1–5 | All inputs bind to seed. Shallow, wide graph. Binding distribution calibrating. |
| Context formation | ~5 | M3 nodes forming. Binding thresholds self-adjusting. |
| Knowledge emergence | ~10 | M2 nodes appearing. Closure thresholds crossed. |
| Purpose detection | ~20 | M1 nodes detectable. Directional asymmetry measurable. |
| Compounding | 20+ | Every traversal improves subsequent retrievals. |

The cold start trajectory is approximate. Graph-relative thresholds (§8) mean that the exact session count depends on the density and diversity of input.

---

## 14. Vital Signs

Observable metrics for monitoring graph health. All ranges are graph-relative — there are no hardcoded safe ranges.

| Metric | Measures |
|--------|----------|
| Activation entropy | Spread of activation across nodes |
| Top-1 activation ratio | Hub monopoly risk |
| Mean degree | Connectivity density |
| Layer distribution | Proportion of nodes at each M-layer |
| Closure mean (M2+) | Structural stability |
| Asymmetry mean (M1+) | Directional bias strength |
| Attractor stability | M0 reconvergence rate |
| Arm count | Domain diversity |
| Arm depth distribution | Identity integration depth |

---

## References

- [AXIOMS.md](AXIOMS.md) — Six frozen axioms
- [EVOLUTION.md](EVOLUTION.md) — Generational history
- [README.md](README.md) — Overview

---

*Living Memory v1.0 · Implementation: [livai.dev](https://livai.dev)*
