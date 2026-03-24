// Living Memory v1.0 — Reference Implementation
// github.com/demos-ra/living-memory
// License: Modified MIT

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type MLayer = 0 | 1 | 2 | 3 | 4;

interface MemNode {
  id: string;
  embedding: number[];
  sourceText: string | null;
  cachedLayer: MLayer;
  cachedClosure: number;
  cachedAsymmetry: number;
  cachedDominance: number;
}

interface MemEdge {
  fromId: string;
  toId: string;
  weight: number;
}

interface MemGraph {
  nodes: Map<string, MemNode>;
  edges: Map<string, MemEdge>; // key: `${fromId}->${toId}`
}

interface GraphMutation {
  edgesUpserted: MemEdge[];
  edgesRemoved: string[];      // edge keys
  nodesRemoved: string[];      // node ids
  layersChanged: Map<string, MLayer>;
}

interface TraverseResult {
  mutations: GraphMutation;
  activation?: Map<string, number>;
}

// ---------------------------------------------------------------------------
// Math primitives
// ---------------------------------------------------------------------------

function cosine(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

function stddev(values: number[], mu: number): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += (v - mu) * (v - mu);
  return Math.sqrt(sum / values.length);
}

// ---------------------------------------------------------------------------
// Graph queries
// ---------------------------------------------------------------------------

function outgoing(graph: MemGraph, nodeId: string): MemEdge[] {
  const result: MemEdge[] = [];
  for (const edge of graph.edges.values()) {
    if (edge.fromId === nodeId) result.push(edge);
  }
  return result;
}

function incoming(graph: MemGraph, nodeId: string): MemEdge[] {
  const result: MemEdge[] = [];
  for (const edge of graph.edges.values()) {
    if (edge.toId === nodeId) result.push(edge);
  }
  return result;
}

function degree(graph: MemGraph, nodeId: string): number {
  let d = 0;
  for (const edge of graph.edges.values()) {
    if (edge.fromId === nodeId || edge.toId === nodeId) d++;
  }
  return d;
}

function edgeKey(fromId: string, toId: string): string {
  return `${fromId}->${toId}`;
}

// ---------------------------------------------------------------------------
// §7.2 Bind
// ---------------------------------------------------------------------------

interface BindResult {
  scores: Map<string, number>;
  threshold: number;
}

function bind(ping: number[], graph: MemGraph): BindResult {
  const scores = new Map<string, number>();
  const allScores: number[] = [];

  for (const node of graph.nodes.values()) {
    const sim = cosine(ping, node.embedding);
    const neighborCount = degree(graph, node.id);
    const score = sim * Math.log(neighborCount + 1 + 1); // log(count + 2) avoids log(1)=0 for degree-0
    scores.set(node.id, score);
    allScores.push(score);
  }

  const mu = mean(allScores);
  const sigma = stddev(allScores, mu);
  const k = 1; // standard deviation multiplier
  const threshold = mu + k * sigma;

  return { scores, threshold };
}

// ---------------------------------------------------------------------------
// §7.3 Propagate + §7.4 Learn (inline)
// ---------------------------------------------------------------------------

interface PropagateResult {
  activation: Map<string, number>;
  edgeMutations: MemEdge[];
}

function propagateAndLearn(
  seeds: Map<string, number>,
  graph: MemGraph,
  steps: number = 3,
  topK: number = 20
): PropagateResult {
  let activation = new Map(seeds);
  const edgeMutations: MemEdge[] = [];

  // Normalize seeds to conserve energy (MA1): Σ = 1
  let total = 0;
  for (const v of activation.values()) total += v;
  if (total > 0) {
    for (const [id, v] of activation) activation.set(id, v / total);
  }

  for (let step = 0; step < steps; step++) {
    const next = new Map<string, number>();

    for (const [nodeId, energy] of activation) {
      if (energy <= 0) continue;
      const out = outgoing(graph, nodeId);
      if (out.length === 0) {
        // Energy stays at node if no outgoing edges
        next.set(nodeId, (next.get(nodeId) ?? 0) + energy);
        continue;
      }

      // Compute effective weights with policy bias
      const node = graph.nodes.get(nodeId)!;
      let sumEffective = 0;
      const effective: Array<{ edge: MemEdge; w: number }> = [];

      for (const edge of out) {
        const target = graph.nodes.get(edge.toId);
        if (!target) continue;

        // Policy bias: centrality (degree) × asymmetry × 1.0 (arm affinity skipped per prompt)
        const centrality = Math.log(degree(graph, edge.toId) + 1 + 1);
        const asymmetry = 1 + target.cachedAsymmetry;
        const policyBias = centrality * asymmetry;
        const w = edge.weight * policyBias;

        effective.push({ edge, w });
        sumEffective += w;
      }

      if (sumEffective === 0) {
        next.set(nodeId, (next.get(nodeId) ?? 0) + energy);
        continue;
      }

      // Distribute energy proportionally, learn inline
      const nodeDegree = degree(graph, nodeId);
      const lr = 1 / (nodeDegree + 1); // learning rate inversely proportional to degree

      const sumOutgoingWeight = out.reduce((s, e) => s + e.weight, 0);

      for (const { edge, w } of effective) {
        const flow = energy * (w / sumEffective);
        next.set(edge.toId, (next.get(edge.toId) ?? 0) + flow);

        // §7.4 Learn: Δw = lr × flow × (1 - w_ij / Σ_outgoing)
        const ratio = sumOutgoingWeight > 0 ? edge.weight / sumOutgoingWeight : 0;
        const delta = lr * flow * (1 - ratio);
        const newWeight = edge.weight + delta;

        edgeMutations.push({
          fromId: edge.fromId,
          toId: edge.toId,
          weight: newWeight,
        });
      }
    }

    // Lateral inhibition: top-K competition
    const sorted = [...next.entries()].sort((a, b) => b[1] - a[1]);
    activation = new Map<string, number>();
    let kept = 0;
    let keptTotal = 0;

    for (const [id, v] of sorted) {
      if (kept < topK) {
        activation.set(id, v);
        keptTotal += v;
        kept++;
      }
    }

    // Renormalize to conserve energy (MA1)
    if (keptTotal > 0) {
      for (const [id, v] of activation) activation.set(id, v / keptTotal);
    }
  }

  return { activation, edgeMutations };
}

// ---------------------------------------------------------------------------
// §7.6 Sparsify
// ---------------------------------------------------------------------------

interface SparsifyResult {
  edgesRemoved: string[];
  nodesRemoved: string[];
}

function sparsify(graph: MemGraph): SparsifyResult {
  const edgesRemoved: string[] = [];
  const pruneFraction = 0.1; // prune edges below 10% of mean outgoing weight from same node

  // Compute mean outgoing weight per node
  const nodeOutWeights = new Map<string, number[]>();
  for (const edge of graph.edges.values()) {
    if (!nodeOutWeights.has(edge.fromId)) nodeOutWeights.set(edge.fromId, []);
    nodeOutWeights.get(edge.fromId)!.push(edge.weight);
  }

  // Edge pruning: below fraction of mean outgoing from same source
  for (const edge of graph.edges.values()) {
    const weights = nodeOutWeights.get(edge.fromId);
    if (!weights || weights.length === 0) continue;
    const meanOut = mean(weights);
    const threshold = pruneFraction * meanOut;
    if (edge.weight < threshold) {
      const key = edgeKey(edge.fromId, edge.toId);
      edgesRemoved.push(key);
      graph.edges.delete(key);
    }
  }

  // Isolate removal: degree 0 after pruning → deleted (MA3)
  const nodesRemoved: string[] = [];
  for (const node of graph.nodes.values()) {
    if (degree(graph, node.id) === 0 && node.cachedLayer === 4) {
      nodesRemoved.push(node.id);
    }
  }

  return { edgesRemoved, nodesRemoved };
}

// ---------------------------------------------------------------------------
// §7.5 Compute Layer
// ---------------------------------------------------------------------------

function computeLayer(nodeId: string, graph: MemGraph): MLayer {
  const node = graph.nodes.get(nodeId);
  if (!node) return 4;

  const deg = degree(graph, nodeId);

  // 1. degree 0 → M4
  if (deg === 0) return 4;

  // 2. has edges → at least M3
  // 3. closure + stability → M2
  //    Closure: cycle participation ratio
  //    Stability: fraction of neighbors that can reactivate this node
  const out = outgoing(graph, nodeId);
  const neighborIds = new Set(out.map((e) => e.toId));
  let cycleCount = 0;

  for (const nId of neighborIds) {
    const nOut = outgoing(graph, nId);
    for (const e of nOut) {
      if (neighborIds.has(e.toId) || e.toId === nodeId) {
        cycleCount++;
        break;
      }
    }
  }

  const closure = neighborIds.size > 0 ? cycleCount / neighborIds.size : 0;

  // Stability: what fraction of neighbors have edges back toward this node
  const inc = incoming(graph, nodeId);
  const incomingFrom = new Set(inc.map((e) => e.fromId));
  let reactivators = 0;
  for (const nId of neighborIds) {
    if (incomingFrom.has(nId)) reactivators++;
  }
  const stability = neighborIds.size > 0 ? reactivators / neighborIds.size : 0;

  // Graph-relative thresholds for M2
  const m2Nodes = [...graph.nodes.values()].filter((n) => n.cachedLayer <= 2);
  const closures = m2Nodes.map((n) => n.cachedClosure);
  const thetaClose = closures.length > 0 ? mean(closures) : 0.5;
  const thetaStable = 0.5; // stability baseline

  if (closure >= thetaClose && stability >= thetaStable) {
    // 4. Directional asymmetry → M1
    const outWeights = out.map((e) => e.weight);
    const incWeights = inc.map((e) => e.weight);
    const outMean = mean(outWeights);
    const incMean = mean(incWeights);
    const asymmetry = Math.abs(outMean - incMean) / (outMean + incMean + 1e-10);

    const allAsymmetries = [...graph.nodes.values()].map((n) => n.cachedAsymmetry);
    const thetaAsym = mean(allAsymmetries);

    if (asymmetry >= thetaAsym && thetaAsym > 0) {
      // 5. Attractor dominance → M0 (read from cache, never computed on hot path)
      const allDominance = [...graph.nodes.values()].map((n) => n.cachedDominance);
      const thetaDom = mean(allDominance);

      if (node.cachedDominance >= thetaDom && thetaDom > 0) {
        return 0;
      }
      return 1;
    }
    return 2;
  }

  return 3;
}

// ---------------------------------------------------------------------------
// §6.1 Traverse — full pipeline
// ---------------------------------------------------------------------------

export function traverse(
  graph: MemGraph,
  ping: number[],
  options?: { returnActivation?: boolean }
): TraverseResult {
  const mutations: GraphMutation = {
    edgesUpserted: [],
    edgesRemoved: [],
    nodesRemoved: [],
    layersChanged: new Map(),
  };

  // Step 1: encode is already done — ping is the embedding vector

  // Step 2: bind
  const { scores, threshold } = bind(ping, graph);

  // Step 3: gate — filter to nodes above threshold
  const seeds = new Map<string, number>();
  for (const [nodeId, score] of scores) {
    if (score > threshold) {
      seeds.set(nodeId, score);
    }
  }

  // No binding → return empty
  if (seeds.size === 0) {
    return { mutations, activation: options?.returnActivation ? new Map() : undefined };
  }

  // Step 4: connect — create edges from ping node to bound nodes
  // In reference implementation, ping is not persisted as a node.
  // Edges are created between bound nodes to represent co-activation.
  const boundIds = [...seeds.keys()];
  for (let i = 0; i < boundIds.length; i++) {
    for (let j = i + 1; j < boundIds.length; j++) {
      const key = edgeKey(boundIds[i], boundIds[j]);
      const existing = graph.edges.get(key);
      const weight = existing ? existing.weight : scores.get(boundIds[j])! / seeds.size;
      const edge: MemEdge = { fromId: boundIds[i], toId: boundIds[j], weight };
      graph.edges.set(key, edge);
      mutations.edgesUpserted.push(edge);
    }
  }

  // Record degree before traversal for layer update
  const degreeBefore = new Map<string, number>();
  for (const node of graph.nodes.values()) {
    degreeBefore.set(node.id, degree(graph, node.id));
  }

  // Steps 5+6: propagate with inline learning
  const topK = Math.max(10, Math.floor(graph.nodes.size * 0.1));
  const { activation, edgeMutations } = propagateAndLearn(seeds, graph, 3, topK);

  // Apply learned edge weights
  for (const edge of edgeMutations) {
    const key = edgeKey(edge.fromId, edge.toId);
    graph.edges.set(key, edge);
    mutations.edgesUpserted.push(edge);
  }

  // Step 7: sparsify
  const { edgesRemoved, nodesRemoved } = sparsify(graph);
  mutations.edgesRemoved.push(...edgesRemoved);
  mutations.nodesRemoved.push(...nodesRemoved);

  // Step 8: update layers locally — only nodes whose degree changed
  for (const node of graph.nodes.values()) {
    const before = degreeBefore.get(node.id) ?? 0;
    const after = degree(graph, node.id);
    if (before !== after) {
      const newLayer = computeLayer(node.id, graph);
      if (newLayer !== node.cachedLayer) {
        mutations.layersChanged.set(node.id, newLayer);
        node.cachedLayer = newLayer;
      }
    }
  }

  // Step 9: return
  return {
    mutations,
    activation: options?.returnActivation ? activation : undefined,
  };
}
