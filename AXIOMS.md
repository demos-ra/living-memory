# Axioms

**Living Memory v1.0**
**Status:** Frozen

---

## Preamble

Six axioms define the irreducible constraints of Living Memory. Each axiom is independent: no axiom is derivable from the others. Together they are sufficient: any system satisfying all six exhibits the dynamics described in the specification.

Axiom text is immutable. Rule changes require versioning. Axiom changes require a new major version.

---

## MA1 — Conservation

**Total energy is constant. Traversal redistributes, never creates or destroys.**

Remove conservation and activation becomes unbounded. Frequently visited regions accumulate energy without limit, producing runaway attractors that dominate the graph regardless of structural relevance. Decay functions are then required to counteract the unbounded growth — but decay is an external correction imposed on a system that should not have diverged. Conservation eliminates the need for decay by construction: redistribution is the only possible outcome of traversal, and what loses energy does so because something else gained it. The constraint is thermodynamic, not algorithmic.

## MA2 — One Action

**Traversal is the only primitive operation. Read, write, and learning are one pass.**

Remove this axiom and read and write become separate code paths with separate semantics. The system must then synchronize two operations that modify the same structure, introducing ordering dependencies, stale-read hazards, and divergence between what was retrieved and what was reinforced. Every retrieval system that separates storage from query inherits this problem. A single traversal that reads the graph, reinforces traversed edges, and returns activated nodes in one pass eliminates the class of bugs entirely. The reduction is not an optimization — it is a guarantee that observation and modification cannot disagree.

## MA3 — Connectivity Is Persistence

**A node survives if and only if its connectivity sustains its energy share under rebalancing. Isolation is the only death.**

Remove this axiom and node survival must be governed by an external policy: time-to-live counters, importance scores, or predicate-based persistence tiers. Each policy encodes assumptions about what should persist that may not hold as the graph evolves. Connectivity-based persistence requires no external judgment. A node that participates in traversal paths retains edges; edges carry energy; energy sustains the node's share of the conserved total. A node that no longer participates loses edges, loses energy share, and is removed by sparsification. The graph determines what matters. Nothing else does.

## MA4 — Directional Dependency

**Lower layers never import from higher. Energy enters at M4, sinks toward M0.**

Remove directionality and identity (M0) can be overwritten by transient input (M4). A single novel stimulus could propagate upward and reshape the attractor basin, producing an unstable system where identity fluctuates with every interaction. Directional dependency ensures that consolidation is irreversible under normal flow: energy enters at the periphery and must survive successive rounds of competition, binding, and structural closure before reaching inner layers.

## MA5 — Emergent Identity

**M0 is computed, never declared. The highest-density attractor basin.**

Remove emergence and identity must be assigned — a seed list, a configuration file, a human labeling which nodes are "core." Any declared identity is frozen at the moment of declaration and diverges from the actual structure of the graph as it evolves. Emergent identity tracks the graph's true center of mass: the region that random walks converge toward, that perturbations reconverge through, that accumulates the highest energy density through traversal. If the person changes, the attractor basin shifts. No declaration needs to be updated because no declaration was made.

## MA6 — Deletability

**Any node can be removed and the system re-equilibrates. No node is structurally required.**

Remove deletability and some nodes become load-bearing — their removal would partition the graph or violate an invariant. Load-bearing nodes cannot be pruned, cannot be challenged by competing structure, and accumulate influence indefinitely. The system ossifies around them. Deletability guarantees that every node earns its position through ongoing connectivity, including nodes at M0. If the highest-density attractor is removed, the graph re-equilibrates to a new attractor basin. This is not a failure mode — it is the mechanism by which identity evolves rather than calcifies.

---

## References

- [SPEC.md](SPEC.md) — Full architectural specification
- [EVOLUTION.md](EVOLUTION.md) — Generational history
- [README.md](README.md) — Overview

---

*Living Memory v1.0 · Implementation: [livai.dev](https://livai.dev)*
