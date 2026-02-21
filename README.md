# Living Memory (LM)

Persistent, context-aware agent memory. Built on [Living Architecture](https://github.com/demos-ra/living-architecture).

---

## What it does

Stores agent memory as a self-organizing tree. Memory compresses automatically, retrieves by semantic relevance, activates relationally by co-access history, reaches consensus across multiple agents, and persists across sessions indefinitely.

No time-based expiry. No manual management. No context window bloat.

---

## Unified framework

Every mechanic is an instance of energy minimization under uncertainty — the system finding its lowest energy state given noisy, incomplete information. Operating at five timescales:

```
Instant        — threshold (SDT β): every read/write decision
Query          — Hebbian: co-access weights update on every retrieval
Session        — decay: relevance shifts with activity patterns
Multi-session  — compression: branch structure reorganizes via φ
Multi-agent    — consensus: shared state emerges from signal convergence
```

Two free parameters: β (cost asymmetry, Signal Detection Theory), λ (forgetting rate, Ebbinghaus). Everything else emergent from system state.

---

## Core mechanics

**Backbone**
Permanent nodes. Never decay. Never compress. Single-agent writes directly. Multi-agent writes require consensus. Store identity, long-term facts, validated beliefs.

**Dynamic branches**
Created on first write. Any name. Subject to OLS compression and Ebbinghaus decay.

**OLS compression (branch-level)**
Fires when branch size exceeds φ (1.618) × mean branch size across all active branches. Merges N nodes into 1 centroid node. Source nodes archived, not deleted. Self-organizing — no hardcoded node counts.

**OLS compression (root-level)**
Compresses cold branches (access ratio below 10% of system mean) into single archive nodes. Call at session boundaries.

**Ebbinghaus decay**
`relevance = base_score × e^(−λ × inactivity_ratio)`
Inactivity is activity-relative, not clock-relative. λ = 0.5 (configurable).
Backbone nodes immune.

**SDT-grounded emergent thresholds**
All thresholds computed from live tree similarity distribution:
`threshold = mean_sim + log(β) × std_sim`
β is cost asymmetry: cost(false_positive) / cost(false_negative).
Bootstrap fallback (mean=0.60, std=0.10) when corpus below min_corpus_size.
No hardcoded similarity values anywhere.

```
dedup β=3.0     — false alarm (losing distinct info) 3x worse than miss
retrieval β=0.3 — miss (lost context) 3x worse than false alarm
consensus β=5.0 — false alarm (bad permanent memory) 5x worse than miss
```

**Deduplication**
Raw TF cosine similarity checked against existing branch nodes on every write. Above dedup threshold: update existing node, do not insert. Pass `normalize_fn` for semantic canonicalization before dedup.

**Conditional injection**
All active nodes scored by cosine similarity × decay buoyancy per query. Top-N returned within character budget cap. Threshold emergent from live distribution.

**Relational activation**
Nodes retrieved together accumulate co-access weight. After primary injection, top co-access partners pulled into context within remaining budget. Partitioned by agent_id — each agent's Hebbian patterns isolated. Edges emergent from activity, not defined upfront.

**Write provenance**
Every node records agent_id. Retrieval filterable by agent. Co-access weights partitioned per agent.

**Pending consensus**
Backbone-bound multi-agent content stages in pending_consensus. Signal strength computed as mean pairwise cosine similarity across agent versions. Commits when signal crosses emergent consensus threshold. Stakes multiplier scales threshold for high-risk decisions.

**Conflict flagging**
When agents diverge and signal stays below threshold, written to conflicts table. Resolved externally — CEO, specialist, or higher-order agent. Resolution writes directly to backbone. All operations atomic.

---

## File structure

```
living-memory/
├── src/
│   ├── config/          R0 — memory.json (β, λ, φ, bootstrap, stakes)
│   ├── domain/          R1 — nodes.py, retrieval.py, compression.py
│   ├── app/             R2 — memory_tree.py
│   ├── contract/        R3 — api.py
│   └── exec/            R4 — storage.py
├── main.py
└── test_lm.py
```

Built under Living Architecture v2.0 R0–R4 layer constraints and F-tag feature manifolds.

---

## Usage

**Single agent**
```python
from src.contract.api import LivingMemory

with LivingMemory(db_path="agent.db") as lm:
    lm.remember("backbone", "User is Demos.", tags=["identity"])
    lm.remember("goals", "Ship LM v1.", agent_id="grok")

    context = lm.recall("current goals", agent_id="grok")
    lm.compress()
    lm.export("snapshot.json")
```

**Multi-agent**
```python
with LivingMemory(db_path="shared.db") as lm:
    # Each agent contributes its version under a concept key
    r1 = lm.contribute("backbone", "Memory must self-organize.",
                        agent_id="grok", concept_key="memory-design")
    r2 = lm.contribute("backbone", "Memory must self-organize via phi.",
                        agent_id="kimi", concept_key="memory-design")
    # r2["status"]: "committed" | "pending" | "conflict"

    # Check and resolve conflicts
    for conflict in lm.pending_conflicts():
        lm.resolve_conflict(
            conflict["conflict_id"],
            "Memory self-organizes via phi-triggered OLS compression.",
            conflict["concept_key"]
        )
```

---

## Configuration

`src/config/memory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `compression.ols_multiplier` | 1.618 | φ — branch compression trigger |
| `compression.min_nodes_to_compress` | 3 | Minimum nodes for OLS to fire |
| `decay.lambda` | 0.5 | Ebbinghaus decay rate |
| `thresholds.dedup.beta` | 3.0 | SDT β for deduplication |
| `thresholds.retrieval.beta` | 0.3 | SDT β for injection |
| `thresholds.consensus.beta` | 5.0 | SDT β for backbone consensus |
| `thresholds.bootstrap.min_corpus_size` | 10 | Nodes required for live distribution |
| `thresholds.bootstrap.bootstrap_mean` | 0.60 | Cold-start distribution mean |
| `thresholds.bootstrap.bootstrap_std` | 0.10 | Cold-start distribution std |
| `consensus.stakes_multiplier.high` | 1.25 | Threshold scale for high-stakes decisions |
| `injection.top_n` | 10 | Maximum nodes injected per query |
| `injection.budget_chars` | 4000 | Hard character cap on total injection |
| `archival.cold_branch_access_ratio` | 0.1 | Cold branch detection ratio |

---

## Tests

```bash
python3 test_lm.py
```

14 tests covering: backbone permanence, dynamic branch creation, deduplication, OLS trigger (φ), OLS content preservation, Ebbinghaus gradient decay, injection budget cap, retrieval relevance ordering, SQLite session persistence, JSON export integrity, relational activation, multi-agent convergence, multi-agent conflict + resolution, emergent SDT threshold.

---

## Run demo

```bash
python3 main.py
```

---

## Dependencies

None. Python 3.10+ standard library only. SQLite built-in.

---

## Known limitations

- Deduplication is lexical (TF cosine). Use `normalize_fn` with an LLM call for semantic canonicalization before insert.
- Relational activation and consensus signal both build from zero on first session — strengthen with use.

---

## Living Architecture layers

| Layer | Directory | Role |
|-------|-----------|------|
| R0 | `src/config` | β, λ, φ, bootstrap, stakes — all system parameters |
| R1 | `src/domain` | Pure logic — nodes, TF-IDF, OLS, decay, SDT thresholds, signal strength |
| R2 | `src/app` | Orchestration — consensus pipeline, query, compression wiring |
| R3 | `src/contract` | Public API — remember, recall, contribute, pending_conflicts, resolve_conflict |
| R4 | `src/exec` | I/O — SQLite, pending_consensus, conflicts, co_access_log, JSON export |

F-tags: `F-memory`, `F-compression`, `F-retrieval`, `F-storage`
