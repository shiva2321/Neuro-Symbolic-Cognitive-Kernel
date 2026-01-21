# NCGN Architecture Documentation

Comprehensive technical documentation of the Neuromorphic Cognitive Graph Network architecture.

## Table of Contents
- [Overview](#overview)
- [System Layers](#system-layers)
- [Core Components](#core-components)
- [Data Structures](#data-structures)
- [Execution Model](#execution-model)
- [Surprise Mechanism](#surprise-mechanism)
- [Schema System](#schema-system)
- [Memory Management](#memory-management)
- [Performance Characteristics](#performance-characteristics)

## Overview

NCGN is a **neuromorphic cognitive architecture** that implements dual-process theory through:

1. **System 1 (Fast)**: Reactive, associative, energy-based graph dynamics
2. **System 2 (Slow)**: Deliberative, symbolic, schema-based validation

### Design Principles

```
╔════════════════════════════════════════════════════════════╗
║                    DESIGN PRINCIPLES                       ║
╠════════════════════════════════════════════════════════════╣
║ 1. SPARSITY         │ Only active nodes consume resources ║
║ 2. CAUSALITY        │ Predictions must be verifiable      ║
║ 3. SURPRISE-DRIVEN  │ Expensive reasoning only when needed║
║ 4. DETERMINISM      │ Serialized execution, reproducible  ║
║ 5. EXPLAINABILITY   │ All decisions have causal chains    ║
╚════════════════════════════════════════════════════════════╝
```

### Key Characteristics

| Feature | Implementation | Biological Analog |
|---------|----------------|-------------------|
| **Energy Dynamics** | 0.0-1.0 float per node | Membrane potential |
| **Firing Threshold** | Default 0.75 | Action potential |
| **Refractory Period** | 3 ticks | Neural refractory |
| **Attention** | k-WTA (k=10) | Selective attention |
| **Memory Decay** | α=0.9 per tick | Working memory fade |
| **Surprise Detection** | RMS violation | Prediction error |

## System Layers

### Three-Layer Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    LAYER 3: UI                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  Dashboard  │  │    Chat     │  │  Visualizer │      │
│  │   (Flask)   │  │     CLI     │  │ (NetworkX)  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└──────────────────────────┬────────────────────────────────┘
                           │ HTTP/WebSocket/STDIO
┌──────────────────────────▼────────────────────────────────┐
│                  LAYER 2: CORTEX                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Dialogue Manager (State Machine)                  │    │
│  │  • IDLE → PROCESSING → CLARIFICATION              │    │
│  │  • Contradiction resolution                       │    │
│  │  • Proposal generation                            │    │
│  └────┬──────────────────────────────────────┬───────┘    │
│       │                                      │            │
│  ┌────▼──────────┐              ┌───────────▼──────┐     │
│  │  Ingestion    │              │  Staging Buffer  │     │
│  │  • Text→Triple│              │  • Validation    │     │
│  │  • NLP parsing│              │  • Conflict det. │     │
│  └───────────────┘              └──────────────────┘     │
└──────────────────────────┬────────────────────────────────┘
                           │ Python API
┌──────────────────────────▼────────────────────────────────┐
│                   LAYER 1: CORE                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │            Graph Memory (Central Hub)              │  │
│  │  • Nodes: Dict[str, ConceptNode]                   │  │
│  │  • Edges: Dict[Tuple[str,str], Synapse]            │  │
│  │  • Schemas: Dict[str, EventSchema]                 │  │
│  │  • Properties: Dict[Tuple[str,str], bool]          │  │
│  └──────┬──────────────────────────────────┬──────────┘  │
│         │                                  │             │
│  ┌──────▼──────────┐              ┌───────▼──────────┐  │
│  │  System 1       │◄─Interrupt──┤  System 2        │  │
│  │  (Physics)      │              │  (Logic)         │  │
│  │                 ├──Surprise──►│                  │  │
│  │  • 8-Phase Tick │              │  • Schema Check  │  │
│  │  • Energy Flow  │              │  • Diagnosis     │  │
│  │  • K-WTA        │              │  • Intervention  │  │
│  │  • Surprise     │              │  • LTD/LTP       │  │
│  └─────────────────┘              └──────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Graph Memory (`core/memory.py`)

Central data structure storing all knowledge.

#### ConceptNode
```python
class ConceptNode:
    energy: float           # Current activation [0.0, 1.0]
    threshold: float        # Firing threshold (default 0.75)
    novelty_score: float    # How "new" this concept is
    refractory_timer: int   # Ticks until can fire again
    last_fired: int         # Tick when last fired
```

**Lifecycle**:
```
Created → Activated → Fired → Refractory → Decay → Pruned
  (E=0)    (E>0)      (E>T)   (timer=3)    (E<0.001)  (removed)
```

#### Synapse
```python
class Synapse:
    source: str             # Source node ID
    target: str             # Target node ID
    weight: float           # Strength [0.0, 1.0]
    confidence: float       # Epistemic certainty
    type: str               # Semantic label (e.g., "eats")
```

**Energy Transmission**:
```
Signal = Weight × SourceEnergy × FiringFlag
TargetEnergy += Signal
```

#### EventSchema
```python
class EventSchema:
    id: str                 # Unique identifier
    action: str             # Action verb (e.g., "eat")
    roles: Dict             # Required participants
    constraints: Dict       # Property requirements
    confidence: float       # Schema certainty
```

**Example** (`eat.json`):
```json
{
  "id": "schema_eat",
  "action": "eat",
  "confidence": 0.95,
  "roles": {
    "agent": "animate_object",
    "target": "edible_object"
  },
  "constraints": {
    "agent": ["is_animate"],
    "target": ["is_edible"]
  }
}
```

### 2. System 1 Engine (`core/system1.py`)

The physics engine executing graph dynamics.

#### The 8-Phase Tick Pipeline

```
PHASE EXECUTION ORDER (IMMUTABLE)
═══════════════════════════════════════════════════════════

Phase 0: TRANSDUCTION
─────────────────────
Purpose:    Inject external energy
Input:      sensory_buffer: Dict[str, float]
Output:     Updated node energies
Algorithm:  E_node = min(1.0, E_node + E_buffer)
Timing:     O(|buffer|)

Phase 1: PASSIVE DECAY
──────────────────────
Purpose:    Simulate membrane leak
Algorithm:  E_node = E_node × decay_alpha
            if E_node < 0.001: deactivate
Timing:     O(|active_nodes|)
Biology:    Resting potential restoration

Phase 2: FIRING DETERMINATION
──────────────────────────────
Purpose:    Identify firing nodes
Criteria:   E > threshold AND refractory_timer == 0
Output:     firing_set: Set[str]
Timing:     O(|active_nodes|)
Biology:    Action potential threshold

Phase 3: REFRACTORY & RESET
────────────────────────────
Purpose:    Prevent immediate re-firing
Algorithm:  For node in firing_set:
              E_node = 0.0
              refractory_timer = 3
Timing:     O(|firing_set|)
Biology:    Absolute refractory period

Phase 4: PROPAGATION
────────────────────
Purpose:    Transmit spikes via synapses
Algorithm:  For (src, tgt) in edges:
              if src in firing_set:
                signal = weight × 1.0
                pending_energy[tgt] += signal
Timing:     O(|firing_set| × avg_degree)
Note:       Signals BUFFERED, not applied yet

Phase 5: INTEGRATION
────────────────────
Purpose:    Apply buffered signals
Algorithm:  For tgt, signal in pending_energy:
              E_tgt = min(1.0, E_tgt + signal)
Timing:     O(|pending_energy|)
Biology:    Postsynaptic potential summation

Phase 6: K-WTA INHIBITION
──────────────────────────
Purpose:    Attention/competition
Algorithm:  top_k = heapq.nlargest(k, active, key=energy)
            for node not in top_k:
              E_node = 0.0
              deactivate
Timing:     O(N log K)
Biology:    Lateral inhibition, winner-take-all

Phase 7: SURPRISE MONITOR
──────────────────────────
Purpose:    Detect prediction failures
Algorithm:  surprise = sqrt(Σ((E_pred - E_obs) × conf)²)
            if surprise > threshold:
              trigger_system2()
Timing:     O(|predicted_nodes|)
See:        Surprise Mechanism section

Phase 8: MAINTENANCE
────────────────────
Purpose:    Housekeeping
Tasks:      • Decrement refractory timers
            • Decay novelty scores
            • Prune inactive nodes
Timing:     O(|active_nodes|)
```

### 3. System 2 Controller (`core/system2.py`)

The deliberative validator and planner.

#### Intervention Protocol

```
SYSTEM 2 ACTIVATION FLOW
═══════════════════════════════════════════════════════════

1. INTERRUPT
   │
   ├─ Triggered by: Surprise > Threshold
   ├─ System 1 state: PAUSED
   └─ Context saved: Firing set, energy map

2. DIAGNOSIS
   │
   ├─ Load EventSchema for attempted action
   ├─ Extract triple: (Agent, Action, Object)
   └─ Check constraints:
      │
      ├─ CONSTRAINT_VIOLATION
      │  Example: metal.is_edible = False
      │  Action: Flag violation
      │
      ├─ MISSING_KNOWLEDGE
      │  Example: unknown_object has no properties
      │  Action: Request definition
      │
      └─ NOVEL_CONCEPT
         Example: High novelty score
         Action: Monitor for patterns

3. INTERVENTION
   │
   ├─ LTD (Long-Term Depression)
   │  Algorithm: weight = weight × 0.8
   │  Purpose: Weaken failed prediction
   │
   ├─ LTP (Long-Term Potentiation)  
   │  Algorithm: weight = min(1.0, weight × 1.2)
   │  Purpose: Strengthen confirmed pattern
   │
   ├─ Goal Injection
   │  Algorithm: E_goal_node = 1.0
   │  Purpose: Direct System 1 attention
   │
   └─ Query Generation
      Algorithm: Template + Context → String
      Purpose: Request clarification

4. RESUME
   │
   └─ System 1 state: RUNNING with modifications
```

## Data Structures

### Memory Layout

```
GraphMemory
│
├─ nodes: Dict[str, ConceptNode]
│  │
│  └─ "dog" → ConceptNode(
│       energy=0.85,
│       threshold=0.75,
│       novelty_score=0.1,
│       refractory_timer=0
│     )
│
├─ edges: Dict[Tuple[str, str], Synapse]
│  │
│  └─ ("dog", "meat") → Synapse(
│       weight=0.9,
│       confidence=0.9,
│       type="eats"
│     )
│
├─ schemas: Dict[str, EventSchema]
│  │
│  └─ "schema_eat" → EventSchema(
│       action="eat",
│       constraints={"target": ["is_edible"]}
│     )
│
├─ properties: Dict[Tuple[str, str], bool]
│  │
│  ├─ ("meat", "is_edible") → True
│  └─ ("metal", "is_edible") → False
│
├─ active_nodes: Set[str]
│  └─ {"dog", "meat", "eat"}
│
└─ type_hierarchy: Dict[str, Set[str]]
   └─ "dog" → {"robot_dog", "police_dog"}
```

### Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Add Node | O(1) | O(1) | Dictionary insert |
| Add Edge | O(1) | O(1) | Dictionary insert |
| Get Node | O(1) | O(1) | Dictionary lookup |
| Get Neighbors | O(d) | O(d) | d = out-degree |
| Activate Node | O(1) | O(1) | Set add |
| Fire Node | O(d) | O(d) | Traverse edges |
| K-WTA | O(N log K) | O(K) | Heap selection |
| Schema Check | O(c) | O(1) | c = constraints |

## Execution Model

### Tick-Based Discrete Time

```
Time Model
══════════════════════════════════════════════════════════

t=0: Initial state
│
├─ User injects energy
│
t=1: First tick
│  ├─ Phase 0: Energy applied
│  ├─ Phase 1: Decay
│  ├─ Phase 2: Firing check
│  ├─ Phase 3: Reset
│  ├─ Phase 4: Propagation
│  ├─ Phase 5: Integration
│  ├─ Phase 6: K-WTA
│  ├─ Phase 7: Surprise
│  └─ Phase 8: Maintenance
│
t=2: Second tick
│  └─ (repeat phases)
│
...
│
t=N: System reaches stable state or pauses
```

### Deterministic Execution

**Guarantees**:
1. Same input → Same output
2. Phases execute in order
3. No race conditions
4. Reproducible for debugging

**Non-determinism Sources** (None in core):
- Random number generators (not used)
- Thread interleavings (single-threaded)
- System clock (only for logging)

## Surprise Mechanism

### Mathematical Formulation

**Root Mean Square (RMS) Surprise**:

$$
S = \sqrt{\sum_{n \in \text{Predicted}} \left((E_{\text{pred}}(n) - E_{\text{obs}}(n)) \times C(n)\right)^2}
$$

Where:
- $E_{\text{pred}}(n)$: Expected energy of node $n$
- $E_{\text{obs}}(n)$: Observed energy of node $n$
- $C(n)$: Confidence of prediction for $n$

### Surprise Scenarios

```
SCENARIO 1: High Surprise (Violation)
───────────────────────────────────────
Predicted: {meat: 0.9} (confidence: 0.9)
Observed:  {meat: 0.0, metal: 1.0}

Calculation:
S = sqrt(((0.9 - 0.0) × 0.9)²)
  = sqrt((0.81)²)
  = 0.81

Result: S > 0.45 → TRIGGER SYSTEM 2


SCENARIO 2: Low Surprise (Confirmation)
────────────────────────────────────────
Predicted: {meat: 0.8}
Observed:  {meat: 0.85}

Calculation:
S = sqrt(((0.8 - 0.85) × 0.9)²)
  = sqrt((-0.045)²)
  = 0.045

Result: S < 0.45 → CONTINUE NORMALLY


SCENARIO 3: No Surprise (Novel Input)
──────────────────────────────────────
Predicted: {} (nothing expected)
Observed:  {unknown_thing: 1.0}

Calculation:
S = 0 (no predictions to violate)

Result: S = 0 → LEARN WITHOUT SURPRISE
```

### Surprise vs. Novelty

| Metric | Surprise | Novelty |
|--------|----------|---------|
| **Trigger** | Violated expectation | New concept |
| **Formula** | RMS prediction error | Time-based decay |
| **Range** | [0, ∞) | [0, 1] |
| **Threshold** | 0.45 | N/A |
| **Response** | System 2 interrupt | Attention boost |
| **Decay** | Immediate | Exponential |

## Schema System

### Schema Definition Language

Schemas are JSON files defining action logic:

```json
{
  "id": "schema_<action>",
  "action": "<verb>",
  "confidence": 0.0-1.0,
  "roles": {
    "<role_name>": "<type_requirement>"
  },
  "constraints": {
    "<role_name>": ["<property1>", "<property2>"]
  },
  "preconditions": ["<condition>"],
  "postconditions": ["<effect>"]
}
```

### Built-in Schemas

#### eat.json
```json
{
  "id": "schema_eat",
  "action": "eat",
  "confidence": 0.95,
  "roles": {
    "agent": "animate_object",
    "target": "edible_object"
  },
  "constraints": {
    "agent": ["is_animate"],
    "target": ["is_edible"]
  }
}
```

#### move.json
```json
{
  "id": "schema_move",
  "action": "move",
  "confidence": 0.90,
  "roles": {
    "agent": "movable_object",
    "destination": "location"
  },
  "constraints": {
    "agent": ["can_move"]
  }
}
```

### Schema Validation Algorithm

```python
def validate_triple(agent: str, action: str, object: str) -> DiagnosisResult:
    """
    Validate a triple against loaded schemas.
    
    Returns:
        DiagnosisResult with:
        - diagnosis_type: OK | VIOLATION | MISSING
        - violated_constraints: List[str]
        - explanation: str
    """
    # 1. Load schema
    schema = schemas.get(f"schema_{action}")
    if not schema:
        return DiagnosisResult(MISSING, [], "No schema for action")
    
    # 2. Check agent constraints
    for constraint in schema.constraints.get("agent", []):
        if not has_property(agent, constraint):
            return DiagnosisResult(
                VIOLATION,
                [constraint],
                f"{agent} lacks {constraint}"
            )
    
    # 3. Check target constraints
    for constraint in schema.constraints.get("target", []):
        if not has_property(object, constraint):
            return DiagnosisResult(
                VIOLATION,
                [constraint],
                f"{object} lacks {constraint}"
            )
    
    return DiagnosisResult(OK, [], "Valid action")
```

## Memory Management

### Activation Dynamics

```
Node Lifecycle State Machine
════════════════════════════════════════════════════════

                 ┌─────────┐
                 │ CREATED │
                 │ E = 0.0 │
                 └────┬────┘
                      │ inject_energy()
                      ▼
    ┌──────────►┌──────────┐
    │           │  ACTIVE  │
    │           │ E > 0.001│
    │           └────┬─────┘
    │                │ E > threshold
    │                ▼
    │           ┌─────────┐
    │           │ FIRING  │
    │           │ sending │
    │           └────┬────┘
    │                │ reset
    │                ▼
    │      ┌────────────────┐
    │      │  REFRACTORY    │
    │      │ timer = 3      │
    │      └────┬───────────┘
    │           │ timer--, decay
    └───────────┘
                      │ E < 0.001
                      ▼
                 ┌─────────┐
                 │ INACTIVE│
                 │ pruned  │
                 └─────────┘
```

### Pruning Strategy

```python
def prune_inactive_nodes(threshold: float = 0.001):
    """
    Remove nodes with energy below threshold.
    
    Rationale:
    - Saves memory
    - Improves cache locality
    - Simulates biological forgetting
    """
    for node_id in list(nodes.keys()):
        if nodes[node_id].energy < threshold:
            # Remove node
            del nodes[node_id]
            
            # Remove incident edges
            for edge in list(edges.keys()):
                if edge[0] == node_id or edge[1] == node_id:
                    del edges[edge]
```

### Memory Footprint

```
Per-Node Memory
═══════════════════════════════════════════════════════

ConceptNode:
  - energy: float (8 bytes)
  - threshold: float (8 bytes)
  - novelty_score: float (8 bytes)
  - refractory_timer: int (8 bytes)
  - last_fired: int (8 bytes)
  - dict overhead: ~240 bytes
  TOTAL: ~280 bytes/node

Synapse:
  - source: str ref (8 bytes)
  - target: str ref (8 bytes)
  - weight: float (8 bytes)
  - confidence: float (8 bytes)
  - type: str (~50 bytes)
  - dict overhead: ~240 bytes
  TOTAL: ~320 bytes/edge

Estimated for 1000 nodes, 3000 edges:
  Nodes: 280 KB
  Edges: 960 KB
  TOTAL: ~1.2 MB
```

## Performance Characteristics

### Benchmarks

```
Hardware: Intel i7-9750H @ 2.6GHz, 16GB RAM
Python: 3.11.0

Tick Execution (100 nodes, 300 edges, k=10)
═══════════════════════════════════════════════════════
Phase 0 (Transduction):    0.05 ms
Phase 1 (Decay):           0.15 ms
Phase 2 (Firing):          0.10 ms
Phase 3 (Reset):           0.05 ms
Phase 4 (Propagation):     0.45 ms
Phase 5 (Integration):     0.20 ms
Phase 6 (K-WTA):           0.30 ms
Phase 7 (Surprise):        0.15 ms
Phase 8 (Maintenance):     0.10 ms
─────────────────────────────────────────────────────
TOTAL:                     1.55 ms/tick

Throughput: ~645 ticks/second


Scalability (avg_degree=3, k=10)
═══════════════════════════════════════════════════════
     Nodes  │   Edges │  ms/tick │  ticks/sec
──────────────────────────────────────────────────────
       100  │     300 │     1.5  │      667
       500  │   1,500 │     6.2  │      161
     1,000  │   3,000 │    13.8  │       72
     5,000  │  15,000 │    82.5  │       12
    10,000  │  30,000 │   201.3  │        5


Memory Scaling
═══════════════════════════════════════════════════════
     Nodes  │   Edges │  Memory (MB)
──────────────────────────────────────────────────────
       100  │     300 │    0.12
     1,000  │   3,000 │    1.20
    10,000  │  30,000 │   12.50
   100,000  │ 300,000 │  125.00
```

### Optimization Tips

1. **Reduce K**: Fewer winners = faster K-WTA
   ```python
   engine = System1Engine(memory, k_winners=5)
   ```

2. **Increase Decay**: Faster pruning
   ```python
   engine = System1Engine(memory, decay_alpha=0.85)
   ```

3. **Batch Processing**: Group operations
   ```python
   for node in batch:
       memory.add_node(node)
   memory.rebuild_index()  # Once, not per-node
   ```

4. **Sparse Graphs**: Keep avg_degree < 5
   ```python
   # Good: Dog → Meat, Dog → Pet
   # Bad: Dog → Everything
   ```

---

*For implementation details, see source code in `/core` directory.*
*For usage examples, see [README.md](README.md) and [NCGN_User_Guide.md](docs/NCGN_User_Guide.md).*
