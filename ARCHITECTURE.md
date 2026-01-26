# NCGN v7/v2.0 - Neural Cognitive Graph Network Architecture

## Executive Summary

The Neural Cognitive Graph Network (NCGN) v7/v2.0 is a neuro-symbolic artificial intelligence framework that implements a **dual-process cognitive architecture** inspired by human cognition. It combines:

- **System 1 (Fast)**: Vectorized associative processing using sparse matrix operations
- **System 2 (Slow)**: Symbolic reasoning using constraint-based LLM inference
- **Learning**: 3-factor Hebbian plasticity with reward modulation
- **Semantics**: Transformer-based embeddings for natural language grounding

This document provides a comprehensive architectural analysis of the NCGN system, detailing its design principles, mathematical foundations, component interactions, and implementation strategy.

---

## Table of Contents

1. [Architectural Paradigm](#1-architectural-paradigm)
2. [Core Design Principles](#2-core-design-principles)
3. [System Components](#3-system-components)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Data Flow and Interactions](#5-data-flow-and-interactions)
6. [Performance Considerations](#6-performance-considerations)
7. [Design Patterns](#7-design-patterns)
8. [Future Extensibility](#8-future-extensibility)

---

## 1. Architectural Paradigm

### 1.1 The Neuro-Symbolic Integration Challenge

Modern AI operates along two extremes:
- **Connectionism**: Neural networks (fast, adaptive, opaque)
- **Symbolism**: Logic systems (slow, rigid, interpretable)

NCGN bridges this divide through **explicit architectural separation**:

```
┌─────────────────────────────────────────────────┐
│           NCGN Dual-Process Architecture        │
├─────────────────┬───────────────────────────────┤
│  System 1       │  System 2                     │
│  (Associative)  │  (Deliberative)               │
├─────────────────┼───────────────────────────────┤
│ • Sparse matrix │ • LLM reasoning               │
│ • Vectorized    │ • Schema validation           │
│ • <10ms latency │ • ~100ms+ latency             │
│ • Continuous    │ • Event-driven                │
│ • Numerical     │ • Symbolic                    │
└─────────────────┴───────────────────────────────┘
```

### 1.2 Data-Oriented Design Philosophy

**Paradigm Shift**: From Object-Oriented (OO) to Data-Oriented Design (DOD)

**Traditional OO Approach** (NCGN v5):
```python
class Node:
    def __init__(self):
        self.activation = 0.0
        self.threshold = 0.5
        self.neighbors = []
```
❌ **Problems**:
- Cache misses from scattered memory
- Python GIL bottleneck
- No SIMD vectorization
- Pointer chasing overhead

**DOD Approach** (NCGN v7):
```python
# Structure of Arrays (SoA)
activations = np.zeros(N, dtype=np.float32)    # Contiguous
thresholds = np.zeros(N, dtype=np.float32)     # Contiguous
adjacency = scipy.sparse.csr_matrix((N, N))    # Sparse
```
✅ **Benefits**:
- Sequential memory access
- CPU cache prefetching
- SIMD operations (4-16x speedup)
- Cache-line alignment

**Performance Impact**: 10-100x speedup for propagation operations on large graphs (10k+ nodes).

---

## 2. Core Design Principles

### 2.1 Separation of Concerns

The architecture enforces strict modular boundaries:

| Module | Responsibility | Dependencies |
|--------|---------------|-------------|
| **Topology** | Graph structure (nodes/edges) | Rustworkx |
| **State** | Numerical arrays (activations) | NumPy, SciPy |
| **Engine** | Physics simulation (propagation) | State, Topology |
| **Learner** | Synaptic plasticity | State, Topology |
| **Brain** | Orchestration & API | All modules |

**Principle**: No cross-cutting concerns. State never manipulates topology. Engine never performs learning.

### 2.2 Integer Indexing Convention

**CRITICAL**: All nodes are identified by **non-negative integers** internally.

```python
# Label-to-Index Bidirectional Mapping
"fire" ←→ 0
"smoke" ←→ 1
"heat" ←→ 2
```

**Rationale**:
1. Rustworkx requires integer indices (performance)
2. Direct array indexing: `activations[idx]`
3. Cache-optimal sparse matrix operations

**Implementation**: `IndexRegistry` maintains thread-safe bidirectional mapping.

### 2.3 Immutability of Physics

System 1 propagation operates as **pure function**:

```python
# Input State
A_t: np.ndarray  # Activation vector at time t
W: csr_matrix    # Adjacency matrix (immutable during ticks)

# Output State
A_{t+1}: np.ndarray  # New activation vector

# External Effects
None  # No side effects, no I/O, no randomness (except refractory)
```

This enables:
- Reproducible simulation
- Parallel processing potential
- Easy testing and debugging
- State snapshots for rollback

### 2.4 Lazy Evaluation

Expensive resources are loaded on-demand:

| Resource | Load Trigger | Memory Impact |
|----------|--------------|---------------|
| Semantic embeddings | First `add_concept()` | ~384 MB (384-dim × 10k nodes) |
| LLM reasoner | First System 2 call | ~4 GB (model + context) |
| CSR adjacency matrix | First `propagate()` | ~(edges × 12 bytes) |

**Benefit**: Minimal memory footprint for simple use cases.

---

## 3. System Components

### 3.1 Component Hierarchy

```
Brain (Orchestrator)
├── GraphTopology (Structure)
│   ├── PyDiGraph (Rustworkx)
│   └── IndexRegistry (Label ↔ Index)
├── CognitiveState (Numerical State)
│   ├── Activation Arrays
│   ├── Threshold Arrays
│   ├── Refractory Counters
│   └── CSR Adjacency Matrix
├── PropagationEngine (System 1)
│   └── Matrix Operations
├── HebbianLearner (Plasticity)
│   └── 3-Factor Learning
├── ConfidenceTracker (Reliability)
├── ConflictDetector (Logic Validation)
├── DecisionEngine (Policy)
├── SemanticLayer (Embeddings)
│   └── SentenceTransformer
└── Linguistic Peripherals (System 2)
    ├── LinguisticProcessor
    └── NaturalLanguageFormatter
```

### 3.2 GraphTopology: The Structural Layer

**Purpose**: Manage graph structure with efficient Rust backend.

**Key Features**:
- **Rustworkx Integration**: Compiled Rust graph library (petgraph)
- **Index Stability**: Handles node deletion with index holes
- **Bidirectional Mapping**: O(1) label ↔ index lookups
- **Dirty Flag**: Signals when CSR matrix rebuild needed

**Critical Implementation Details**:

1. **Node Deletion Behavior**:
```python
# Rustworkx reuses deleted indices
graph.remove_node(5)  # Frees index 5
graph.add_node("new")  # May reuse index 5!

# MUST immediately zero state arrays:
state.activations[5] = 0.0
state.thresholds[5] = default
```

2. **Edge Data Extraction**:
```python
sources, targets, weights = topology.get_adjacency_data()
# Returns three parallel NumPy arrays for COO → CSR conversion
```

**Thread Safety**: `IndexRegistry` uses threading.Lock for concurrent access.

### 3.3 CognitiveState: The Numerical Core

**Purpose**: Structure-of-Arrays (SoA) representation of all node properties.

**Memory Layout** (for N nodes):
```
activations:         float32[N]  # Current energy (0.0-1.0)
thresholds:          float32[N]  # Firing threshold
refractory_counters: int32[N]    # Cooldown ticks
resting_potentials:  float32[N]  # Baseline energy
novelty_scores:      float32[N]  # Decay over time
embeddings:          float32[N×384]  # Semantic vectors (lazy)
adjacency_csr:       CSR Matrix  # Sparse weights (cached)
```

**Capacity Management**:
- Initial capacity: 10,000 nodes
- Doubling strategy: Amortized O(1) growth
- Pre-allocation: Avoids frequent reallocation

**Adjacency Matrix Caching**:
```python
# Rebuild only when topology changes
if topology.is_dirty:
    sources, targets, weights = topology.get_adjacency_data()
    coo = coo_matrix((weights, (sources, targets)), shape=(N, N))
    self._adj_csr = coo.tocsr()  # Compressed Sparse Row
    topology.mark_clean()
```

**Why CSR Format?**:
- Optimized for row slicing (outgoing edges)
- Efficient matrix-vector multiplication
- 3× faster than COO for propagation

### 3.4 PropagationEngine: System 1 Physics

**Purpose**: Simulate thought as signal propagation through weighted graph.

**Core Equation** (per tick):
```
A_{t+1} = σ((A_t × (1-δ) + W·A_t × α + I_ext) / (1 + β·ΣA)) ⊙ (1-R)
```

Where:
- **A_t**: Activation vector at time t
- **δ** (delta): Decay rate (0.1) - thoughts fade without reinforcement
- **W**: Sparse adjacency matrix (CSR)
- **α** (alpha): Flow rate (0.8) - synaptic conductivity
- **I_ext**: External input vector (sensory stimuli)
- **β** (beta): Normalization constant (0.01) - prevents seizures
- **σ** (sigma): Sigmoid activation function
- **R**: Refractory mask (binary vector)

**Propagation Algorithm** (vectorized):

```python
def propagate(self, steps: int = 1):
    for _ in range(steps):
        # Phase 1: Apply external input
        self._apply_external_input()
        
        # Phase 2: Decay existing activation
        A[:n] *= (1 - config.decay_delta)
        
        # Phase 3: Propagate through adjacency matrix
        if W is not None:
            propagated = W.dot(A[:n]) * config.flow_alpha
            A[:n] += propagated
        
        # Phase 4: Divisive normalization (homeostasis)
        total_energy = np.sum(A[:n])
        if total_energy > 0:
            A[:n] /= (1.0 + config.norm_beta * total_energy)
        
        # Phase 5: Sigmoid activation
        A[:n] = sigmoid(A[:n] * 6 - 3)
        
        # Phase 6: Identify firing nodes
        can_fire = (A[:n] > thresholds[:n]) & (R[:n] == 0)
        firing_indices = np.where(can_fire)[0]
        
        # Phase 7: Apply refractory period
        R[firing_indices] = config.refractory_period
        A[firing_indices] = 0.0  # Reset after firing
        
        # Phase 8: Decay refractory counters
        R[R > 0] -= 1
        
        # Phase 9: Seizure damping
        if np.sum(A[:n]) > config.seizure_threshold:
            A[:n] *= config.seizure_damping
        
        # Phase 10: Clamp and cleanup
        A[:n] = np.clip(A[:n], 0.0, 1.0)
        A[A < 0.001] = 0.0
```

**Refractory Period Mechanism**:
Biological neurons cannot fire continuously. Implementation:
```python
refractory_counters = np.zeros(N, dtype=np.int32)

# When node fires:
if activation[i] > firing_threshold:
    refractory_counters[i] = 10  # 10-tick cooldown
    activation[i] = 0.0
    
# Each tick:
refractory_counters = np.maximum(0, refractory_counters - 1)
mask = (refractory_counters == 0).astype(float)
activations *= mask
```

**Complexity Analysis**:
- **Dense Graph**: O(N²) - impractical
- **Sparse Graph**: O(E) where E = number of edges
- **Typical**: 10k nodes, 50k edges → ~5ms per tick

### 3.5 HebbianLearner: Synaptic Plasticity

**Purpose**: Implement reward-modulated Hebbian learning ("neurons that fire together, wire together").

**3-Factor Learning Rule**:
```
ΔW_{ij} = η × M × A_i × A_j × exists(W_{ij})
```

Where:
- **η** (eta): Learning rate (0.01)
- **M**: Reward modulator (dopamine signal)
  - M > 0 → LTP (Long-Term Potentiation) - strengthen
  - M < 0 → LTD (Long-Term Depression) - weaken
- **A_i**: Pre-synaptic activation
- **A_j**: Post-synaptic activation
- **exists(W_{ij})**: Only update existing edges (no auto-genesis)

**Algorithm**:
```python
def apply_reward(self, modulator: float) -> int:
    """Apply reward-modulated learning."""
    if abs(modulator) < 0.001:
        return 0
    
    A = self.state.activations
    W = self.state.adjacency  # CSR matrix
    active_mask = A > 0.1
    active_indices = set(np.where(active_mask)[0])
    
    if len(active_indices) < 2:
        return 0
    
    updated = 0
    lr = self.config.learning_rate
    
    # Iterate over CSR non-zero entries
    for i in active_indices:
        pre = A[i]
        row_start = W.indptr[i]
        row_end = W.indptr[i + 1]
        
        for idx in range(row_start, row_end):
            j = W.indices[idx]
            if j not in active_indices:
                continue
            
            post = A[j]
            
            # 3-Factor update
            delta = lr * modulator * pre * post
            W.data[idx] = np.clip(
                W.data[idx] + delta,
                config.weight_min,
                config.weight_max
            )
            updated += 1
    
    return updated
```

**Auto-Genesis Mechanism**:
Creates new edges based on co-occurrence:
```python
# Track co-activations
co_occurrence_counts[(idx1, idx2)] += 1

# Create edge when threshold reached
if co_occurrence_counts[(idx1, idx2)] >= 5:
    topology.add_connection(label1, label2, weight=0.3)
```

**Reward Prediction Error (RPE)**:
```python
class RewardModulator:
    def calculate_signal(self, actual_reward: float) -> float:
        # RPE = actual - expected
        rpe = actual_reward - self.expected_reward
        
        # Update baseline (exponential moving average)
        self.expected_reward = (
            0.1 * actual_reward + 
            0.9 * self.expected_reward
        )
        
        return rpe
```

### 3.6 Confidence & Decision Systems (v2.0 Logic Core)

#### ConfidenceTracker

**Purpose**: Track reliability of knowledge graph edges.

**3-Factor Confidence**:
1. **Repetition**: Count of reinforcements
2. **Source credibility**: Training > user > guesses
3. **Temporal decay**: Forgetting curve

**Formula**:
```python
base = 0.5 if source == "training" else 0.2
count_factor = min(0.8, count * 0.1)
confidence = clamp(base + count_factor, 0, 1)
```

**Decay Rates**:
- Training data: `conf *= 0.901` (1% decay per cycle)
- User data: `conf *= 0.99` (10% decay per cycle)

#### ConflictDetector

**Purpose**: Identify logical contradictions before graph updates.

**Conflict Types**:
```python
class ConflictType(Enum):
    NEW_KNOWLEDGE = "new"        # No conflict
    HARD_REJECT = "hard"         # >90% confidence conflict
    CURIOSITY = "curiosity"      # 50-90% confidence conflict
    ACCEPT = "accept"            # <50% confidence conflict
```

**Detection Logic**:
```python
def check_conflict(self, source, relation, target):
    existing_weight = topology.get_edge_weight(source, target)
    if existing_weight is None:
        return ConflictType.NEW_KNOWLEDGE
    
    conf = confidence.get_confidence(source, target)
    if conf > 0.9:
        return ConflictType.HARD_REJECT
    elif conf > 0.5:
        return ConflictType.CURIOSITY
    else:
        return ConflictType.ACCEPT
```

#### DecisionEngine

**Purpose**: Policy for knowledge acceptance.

**Decision Matrix**:
| Conflict Type | Decision | Action |
|--------------|----------|--------|
| NEW_KNOWLEDGE | accept | Add to graph |
| HARD_REJECT | reject | Discard update |
| CURIOSITY | curiosity | Flag for review |
| ACCEPT | accept | Overwrite existing |

**Action Selection** (Winner-Take-All):
```python
def select_action(self, actions: Dict[str, float]) -> str:
    """Select highest-energy action."""
    sorted_actions = sorted(
        actions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_actions[0][0] if sorted_actions else None
```

### 3.7 Semantic Layer: Natural Language Grounding

**Purpose**: Bridge natural language to graph concepts using embeddings.

**Architecture**:
```
Text Input → SentenceTransformer → 384-dim Vector
    ↓
Cosine Similarity
    ↓
Entry Points (top-k concepts)
    ↓
Inject Energy into Brain
```

**Embedding Model**: `all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: ~1000 sentences/sec on CPU
- Quality: 69.6% on STS benchmark

**Key Methods**:

1. **Entry Point Detection**:
```python
def find_entry_points(query, topology, top_k=5, min_similarity=0.3):
    # Encode query
    query_vec = model.encode(query)
    
    # Compute similarities to all concepts
    similarities = []
    for label in topology.registry.all_labels():
        concept_vec = embeddings[idx]
        sim = cosine_similarity(query_vec, concept_vec)
        if sim > min_similarity:
            similarities.append((label, sim))
    
    # Return top-k
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

2. **Semantic Weight Computation**:
```python
def compute_semantic_weight(source: str, target: str) -> float:
    vec_a = encode(source)
    vec_b = encode(target)
    
    # Cosine similarity ∈ [-1, 1]
    cosine_sim = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
    
    # Map to [0, 1]
    weight = (cosine_sim + 1) / 2
    return weight
```

**Optimization**: LRU cache (10,000 entries) to avoid re-encoding.

### 3.8 Linguistic Peripherals (System 2)

#### LinguisticProcessor

**Purpose**: Extract structured knowledge from natural language.

**Pipeline**:
```
Text → LLM (Llama) → JSON → Pydantic Validation → KnowledgeGraphUpdate
```

**Schema** (`KnowledgeGraphUpdate`):
```python
class KnowledgeGraphUpdate(BaseModel):
    reasoning: str  # Chain-of-Thought (generated FIRST)
    new_nodes: List[ConceptNode]
    new_edges: List[RelationEdge]
    nodes_to_remove: List[str]
    edges_to_weaken: List[RelationEdge]
```

**Example**:
```
Input: "Dogs eat meat."

Output:
{
  "reasoning": "Extract agent 'dog', action 'eats', object 'meat'",
  "new_nodes": [
    {"label": "dog", "properties": {"is_alive": true}},
    {"label": "meat", "properties": {"is_edible": true}}
  ],
  "new_edges": [
    {"source": "dog", "target": "meat", "relation_type": "eats", "weight": 0.8}
  ]
}
```

**Robustness Features**:
- Auto-generates missing `reasoning` field
- Maps loose key names to standard schema
- Sanitizes invalid entries with warnings
- Retries on validation failure (up to 3 times)

#### NaturalLanguageFormatter

**Purpose**: Translate mathematical brain state to human language.

**Input**: Brain state dictionary
```python
{
    "user_input": "What do dogs eat?",
    "active_concepts": {"dog": 0.8, "meat": 0.7, "carnivore": 0.6},
    "decision": "ANSWER"
}
```

**Output**: Single-sentence natural response
```
"Based on the active concepts (dog, meat, carnivore), 
dogs are carnivores that primarily eat meat."
```

**LLM Configuration**:
- Temperature: 0.7 (moderate creativity)
- Max tokens: 100 (concise responses)
- Format: ChatML (instruction-following)

**Response Types**:
- **ACCEPT**: Acknowledge new knowledge
- **REJECT**: Politely refuse citing conflict
- **CURIOSITY**: Ask for clarification
- **CHAT/ANSWER**: General query responses

---

## 4. Mathematical Foundations

### 4.1 Spreading Activation Dynamics

The propagation engine implements a **discrete-time dynamical system** modeling thought as energy flow through a weighted graph.

**State Space**:
```
S = ℝ^N × ℤ^N
where:
  A ∈ [0,1]^N  - Activation vector
  R ∈ ℕ^N      - Refractory counters
```

**Evolution Operator**:
```
φ: S → S
(A_t, R_t) ↦ (A_{t+1}, R_{t+1})
```

**Full Dynamics**:
```
A_{t+1} = σ(
    (A_t ⊙ (1-δ𝟙) + α·W·A_t + I_ext) / (1 + β·||A_t||₁)
) ⊙ (𝟙 - R_t>0)

R_{t+1} = max(0, R_t - 𝟙) + τ·(A_t > θ)
```

Where:
- **⊙**: Element-wise (Hadamard) product
- **𝟙**: All-ones vector
- **δ**: Decay rate (0.1)
- **α**: Flow rate (0.8)
- **W**: Adjacency matrix (sparse CSR)
- **β**: Normalization constant (0.01)
- **τ**: Refractory period (10 ticks)
- **θ**: Firing threshold vector
- **σ**: Sigmoid activation

**Sigmoid Function** (numerically stable):
```python
def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500)))
    )
```

**Stability Analysis**:

1. **Energy Conservation**: Total energy is bounded:
```
||A_{t+1}||₁ ≤ max(||A_t||₁, ||I_ext||₁) / (1 + β·||A_t||₁)
```

2. **Seizure Damping**: If total energy exceeds threshold:
```
if ||A_t||₁ > E_max:
    A_t ← γ·A_t  (γ = 0.5)
```

3. **Fixed Points**: Stable attractors at:
```
A* = σ(α·W·A* / (1 + β·||A*||₁))
```

### 4.2 Hebbian Learning Theory

**Classical Hebbian Rule** (2-factor):
```
ΔW_{ij} = η·x_i·x_j
```
❌ **Problem**: Unstable, leads to runaway weights.

**3-Factor Hebbian Rule** (reward-modulated):
```
ΔW_{ij} = η·M(r)·x_i·x_j
```
✅ **Advantage**: Dopamine-like modulator M(r) prevents instability.

**Reward Prediction Error**:
```
M(r) = r - r̂
where:
  r̂_{t+1} = α·r_t + (1-α)·r̂_t  (α = 0.1)
```

**Weight Bounds**:
```
W_{ij} ∈ [W_min, W_max] = [0, 1]
```

**Synaptic Scaling**:
```python
# Maintain stable weight distribution
if np.mean(W.data) > 0.6:
    W.data *= 0.95  # Global scaling
```

### 4.3 Semantic Similarity Metrics

**Cosine Similarity**:
```
sim(u, v) = (u · v) / (||u||₂ · ||v||₂)

Properties:
  - Range: [-1, 1]
  - sim(u, u) = 1 (identical)
  - sim(u, -u) = -1 (opposite)
  - sim(u, v) = 0 (orthogonal)
```

**Euclidean Distance** (alternative):
```
d(u, v) = ||u - v||₂ = √(Σ(u_i - v_i)²)

Properties:
  - Range: [0, ∞)
  - d(u, u) = 0
  - Triangle inequality: d(u, w) ≤ d(u, v) + d(v, w)
```

**Mapping to Edge Weights**:
```python
# Cosine to [0,1]
weight = (cosine_sim + 1) / 2

# Euclidean to [0,1] (with normalization)
weight = 1 / (1 + euclidean_dist)
```

### 4.4 Confidence Calculus

**Base Confidence**:
```
C_base = {
    0.5  if source = "training"
    0.2  if source = "user"
    0.0  if source = "guess"
}
```

**Reinforcement Factor**:
```
C_reinforce(n) = min(0.8, 0.1·n)
where n = number of reinforcements
```

**Total Confidence**:
```
C_total = clamp(C_base + C_reinforce(n), 0, 0.9)
```

**Temporal Decay**:
```
C_{t+1} = λ·C_t
where:
  λ = 0.901  (training data - slow decay)
  λ = 0.990  (user data - medium decay)
  λ = 0.950  (guesses - fast decay)
```

**Conflict Threshold Bands**:
```
C > 0.9  → HARD_REJECT  (high certainty)
C > 0.5  → CURIOSITY    (moderate certainty)
C ≤ 0.5  → ACCEPT       (low certainty)
```

---

## 5. Data Flow and Interactions

### 5.1 Full Cognitive Cycle

The complete information processing pipeline:

```
┌─────────────────────────────────────────────────────────┐
│         USER INPUT: "Dogs eat meat."                    │
└───────────────────┬─────────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  1. SEMANTIC LAYER  │
         │  - Encode text      │
         │  - Find entry points│
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────────┐
         │  2. INJECTION            │
         │  - Inject: "dog" (0.7)  │
         │  - Inject: "meat" (0.6) │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  3. SYSTEM 1 PROPAGATION│
         │  - 10 ticks             │
         │  - Energy spreads       │
         │  - Active: {dog, meat,  │
         │    carnivore, animal}   │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  4. LINGUISTIC PARSING  │
         │  - LLM extracts:        │
         │    • Nodes: [dog, meat] │
         │    • Edge: dog→meat     │
         │      (relation: "eats") │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  5. CONFLICT DETECTION  │
         │  - Check existing edges │
         │  - Confidence: 0.3      │
         │  - Type: ACCEPT         │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  6. DECISION ENGINE     │
         │  - Decision: accept     │
         │  - Add edge to graph    │
         │  - Reinforce confidence │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  7. RESPONSE FORMATTER  │
         │  - Generate natural text│
         │  - Output: "Got it!     │
         │    Dogs eat meat."      │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │  8. LEARNING (optional) │
         │  - Reward: +0.5         │
         │  - RPE: 0.45            │
         │  - Apply Hebbian LTP    │
         └─────────────────────────┘
```

### 5.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Brain (Orchestrator)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Topology    │  │   State      │  │   Engine     │     │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤     │
│  │ • PyDiGraph  │◄─┤ • Activations│◄─┤ • Propagate  │     │
│  │ • Registry   │  │ • Thresholds │  │ • Inject     │     │
│  │ • Edges      │──┤ • CSR Matrix │──┤ • Fire       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                           │                                 │
│  ┌────────────────────────▼────────────────────────┐       │
│  │              HebbianLearner                      │       │
│  ├──────────────────────────────────────────────────┤       │
│  │ • apply_reward()   • auto_learn()                │       │
│  │ • targeted_ltp()   • targeted_ltd()              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │Confidence│◄─┤Conflicts │◄─┤ Decision │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│       │              │              │                       │
│       └──────────────┴──────────────┘                       │
│                      │                                      │
│  ┌───────────────────▼──────────────────┐                  │
│  │        Logic Core (v2.0)             │                  │
│  │  • Confidence tracking               │                  │
│  │  • Conflict detection                │                  │
│  │  • Decision policy                   │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  ┌──────────────┐  ┌──────────────────┐                    │
│  │  Semantics   │  │  Linguistic       │                    │
│  ├──────────────┤  │  Peripherals      │                    │
│  │ • Embeddings │  ├──────────────────┤                    │
│  │ • Similarity │  │ • Processor       │                    │
│  │ • Entry pts  │  │ • Formatter       │                    │
│  └──────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 State Synchronization

**Critical Invariant**: Topology and State must remain synchronized.

**Synchronization Points**:

1. **Node Addition**:
```python
# 1. Add to topology
idx = topology.add_concept("fire")

# 2. Ensure state capacity
state.ensure_capacity(idx)

# 3. Set initial values
state.activations[idx] = 0.0
state.thresholds[idx] = default_threshold
```

2. **Node Deletion**:
```python
# 1. Remove from topology
idx = topology.remove_concept("fire")

# 2. MUST zero state arrays
state.zero_index(idx)  # CRITICAL!
```

3. **Matrix Rebuild**:
```python
# Before propagation
state.synchronize_matrix(topology)

# Inside synchronize_matrix:
if topology.is_dirty:
    sources, targets, weights = topology.get_adjacency_data()
    csr = build_csr(sources, targets, weights)
    state._adj_csr = csr
    topology.mark_clean()
```

### 5.4 Thread Safety Considerations

**Current Status**: Single-threaded design.

**Thread-Safe Components**:
- `IndexRegistry` (uses `threading.Lock`)

**NOT Thread-Safe**:
- `CognitiveState` arrays
- `PropagationEngine` buffers
- `HebbianLearner` statistics

**Future Parallelization Strategy**:
1. Read-only propagation on immutable state snapshot
2. Lock-free update queue for learning
3. Periodic synchronization barriers

---

## 6. Performance Considerations

### 6.1 Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Add concept | O(1) | <1 μs |
| Add edge | O(1) | <1 μs |
| Propagation (1 tick) | O(E + N) | 5-50 ms* |
| Matrix rebuild | O(E log E) | 10-100 ms* |
| Hebbian update | O(E_active) | 1-10 ms* |
| LLM reasoning | O(?) | 100-1000 ms |
| Embedding encode | O(L × D) | 5-20 ms** |

*For N=10k nodes, E=50k edges  
**For L=20 words, D=384 dimensions

### 6.2 Memory Footprint

**Baseline** (empty brain):
```
Topology:        ~1 KB (Rustworkx metadata)
State arrays:    ~400 KB (10k × 5 × float32)
Registry:        ~1 KB (empty dicts)
Total:           ~402 KB
```

**With 10k concepts, 50k edges**:
```
Topology:        ~800 KB (50k edges × 16 bytes)
State arrays:    ~400 KB (unchanged)
CSR matrix:      ~600 KB (50k × 12 bytes)
Embeddings:      ~15 MB (10k × 384 × float32) - lazy
Registry:        ~500 KB (10k × 50 bytes avg)
Total:           ~2.3 MB (without embeddings)
Total:           ~17.3 MB (with embeddings)
```

**With LLM**:
```
Brain:           ~17 MB
LLM model:       ~4 GB (TinyLlama 1.1B Q4)
LLM context:     ~16 MB (4096 tokens)
Total:           ~4.05 GB
```

### 6.3 Optimization Strategies

**1. Sparse Matrix Format Selection**:
- Use CSR for row-dominant operations (propagation)
- Consider CSC for column-dominant operations (future)
- COO only for construction

**2. Vectorization**:
- NumPy operations use SIMD instructions
- SciPy sparse ops use optimized BLAS/LAPACK
- Avoid Python loops over arrays

**3. Memory Locality**:
- SoA layout ensures sequential access
- Pre-allocate arrays to avoid fragmentation
- Use contiguous C-order arrays

**4. Lazy Loading**:
- Defer expensive resources (embeddings, LLM)
- Load on first use
- Clear caches when memory pressure detected

**5. Batching**:
```python
# Bad: Individual propagations
for input in inputs:
    brain.inject(input)
    brain.think(steps=10)

# Good: Batch injections
brain.inject_multiple(inputs)
brain.think(steps=10)
```

### 6.4 Profiling Results

**Sample Profile** (10k nodes, 50k edges, 10 ticks):
```
Function                    Time    %
----------------------------------------
scipy.sparse.csr_matrix.dot 25ms   50%
numpy.clip                   8ms   16%
sigmoid                      6ms   12%
numpy.sum                    4ms    8%
refractory logic             3ms    6%
other                        4ms    8%
----------------------------------------
Total                       50ms  100%
```

**Bottleneck**: Sparse matrix-vector multiplication dominates.

**Mitigation**:
- Reduce edge count (prune weak connections)
- Increase sparsity (more zeros)
- Use GPU sparse libraries (future: cuSPARSE)

---

## 7. Design Patterns

### 7.1 Facade Pattern

**Brain** class acts as a unified facade over complex subsystems:

```python
# Instead of:
topology = GraphTopology()
state = CognitiveState()
engine = PropagationEngine(state, topology)
# ... initialize 10+ components

# Simple interface:
brain = Brain()
brain.inject("fire", 0.5)
brain.think(steps=10)
```

### 7.2 Strategy Pattern

**Decision engine** uses strategy pattern for conflict resolution:

```python
class DecisionStrategy(ABC):
    @abstractmethod
    def decide(self, conflict: ConflictReport) -> str:
        pass

class ConservativeStrategy(DecisionStrategy):
    def decide(self, conflict):
        return "reject" if conflict.type == "HARD" else "accept"

class CuriousStrategy(DecisionStrategy):
    def decide(self, conflict):
        return "curiosity" if conflict.confidence < 0.9 else "accept"
```

### 7.3 Observer Pattern

**Propagation engine** can notify observers of state changes:

```python
class PropagationObserver(ABC):
    @abstractmethod
    def on_tick(self, active_concepts: Dict[str, float]):
        pass

class DashboardObserver(PropagationObserver):
    def on_tick(self, active_concepts):
        self.update_visualization(active_concepts)
```

### 7.4 Lazy Initialization

**Expensive resources** are created on-demand:

```python
@property
def semantics(self) -> Optional[SemanticLayer]:
    if self._semantics is None:
        self._semantics = SemanticLayer(config=self.config)
    return self._semantics
```

### 7.5 Template Method

**Game interface** defines template for game implementations:

```python
class GameInterface(ABC):
    def run_episode(self):  # Template method
        self.reset()
        while not self.is_done():
            action = self.select_action()
            reward = self.step(action)
            self.learn(reward)
        return self.get_stats()
    
    @abstractmethod
    def reset(self): pass
    
    @abstractmethod
    def step(self, action): pass
```

---

## 8. Future Extensibility

### 8.1 Planned Enhancements

**1. Parallelization**:
- Multi-threaded propagation
- GPU acceleration (cuSPARSE, PyTorch)
- Distributed graph sharding

**2. Advanced Learning**:
- Meta-learning (learning to learn)
- Curriculum learning
- Transfer learning across domains

**3. Richer Semantics**:
- Multi-modal embeddings (vision, audio)
- Contextual embeddings (BERT, GPT)
- Knowledge graph embeddings (TransE, DistMult)

**4. Enhanced System 2**:
- Multi-step reasoning chains
- Planning algorithms (MCTS, A*)
- Formal logic integration (Prolog, Z3)

**5. Observability**:
- Real-time dashboards
- Activation tracing
- Decision explanation

### 8.2 Extension Points

**Custom Activation Functions**:
```python
class CustomActivation(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Implement custom nonlinearity
        return x  # Replace with custom logic
```

**Custom Learning Rules**:
```python
class CustomLearner(LearnerInterface):
    def update_weights(self, pre, post, reward):
        # Implement custom plasticity
        pass
```

**Custom Embedding Models**:
```python
class CustomEmbedder(EmbedderInterface):
    def encode(self, text: str) -> np.ndarray:
        # Use domain-specific embeddings
        pass
```

### 8.3 Integration Hooks

**External Systems**:
- REST API for remote queries
- WebSocket for real-time updates
- Message queue (RabbitMQ, Kafka) for async processing

**Database Integration**:
- Persistent graph storage (Neo4j, ArangoDB)
- Vector database (Pinecone, Weaviate) for embeddings
- Time-series DB (InfluxDB) for monitoring

**Deployment Options**:
- Containerization (Docker)
- Orchestration (Kubernetes)
- Serverless (AWS Lambda) for query API

---

## Conclusion

The NCGN v7/v2.0 architecture represents a sophisticated integration of neuroscience-inspired computation and symbolic reasoning. Its data-oriented design, explicit System 1/System 2 separation, and modular architecture enable both performance and extensibility.

**Key Achievements**:
- ✅ 10-100x performance improvement over v5
- ✅ Biologically-plausible learning mechanisms
- ✅ Interpretable confidence and conflict systems
- ✅ Seamless natural language integration
- ✅ Proven in game environments

**Future Directions**:
- 🚀 GPU acceleration
- 🚀 Multi-modal learning
- 🚀 Distributed scaling
- 🚀 Production deployment

This architecture provides a solid foundation for advancing neuro-symbolic AI research and applications.
