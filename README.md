# Living Memory

A memory system that mirrors human memory — same architecture, higher fidelity. Not a retrieval engine. A living topology where structure is relevance.

---

## The System

Three primitives: **node**, **field**, **energy**. One action: **traversal** — read, write, and learning are the same operation. Five layers (**M0** Identity, **M1** Purpose, **M2** Knowledge, **M3** Context, **M4** Input) computed from connectivity, never assigned. Six frozen axioms that define the irreducible constraints.

Energy is conserved. Traversal redistributes it across the graph, biased by purpose and attracted toward identity. Nodes that sustain connectivity persist. Nodes that lose connectivity fade. There is no decay function — conservation plus redistribution is decay. There are no fixed thresholds — the graph defines its own noise floor from its own statistical properties.

The model is gravitational. M0 is the supermassive center: emergent, never declared, computed from random walk convergence. Energy enters at M4 and sinks inward through successive crystallization transitions, each adding a qualitatively new invariant. What cannot connect does not survive. What connects deeply becomes self-sustaining. What biases traversal across contexts becomes purpose. What the system reconverges through becomes identity.

---

## Key Concepts

**Present State Layer.** A derived computation that reads the graph and produces structured statements about the user's current state — with confidence tiers, arm categorization, and source text where available. The bridge between living process and immediate usability. Specified in §11 of SPEC.md.

**Arms.** Community-detected overlapping subgraphs that span layers. A domain's arm depth indicates how fundamental it is to identity. Arms merge when cross-domain edge density makes them indistinguishable — identity evolution. Specified in §5 of SPEC.md.

**Source text lifecycle.** Original input is retained at M3 and pruned at the M3 → M2 transition. Below the boundary, exact recall. Above it, structural position carries meaning. Specified in §12 of SPEC.md.

---

## Documents

| Document | Contents |
|----------|----------|
| [AXIOMS.md](AXIOMS.md) | Six frozen axioms (MA1–MA6) with rationale |
| [SPEC.md](SPEC.md) | Full architectural specification |
| [EVOLUTION.md](EVOLUTION.md) | Three generations: Gen 0 (Python) → Gen 1 (F-memory v0.6) → Gen 2 (v1.0) |
| [legacy/](legacy/) | Gen 0 Python source |

---

## Implementation

[livai.dev](https://livai.dev)

---

## License

MIT
