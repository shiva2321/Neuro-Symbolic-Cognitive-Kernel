# NCGN Technical Documentation v5.0

## 1. System Architecture: The Dual-Process Model

NCGN implements a neuro-symbolic architecture divided into two distinct but interacting systems:

### System 1: The Reactive Engine (`core/system1.py`)
A dynamical system where concepts are nodes in a graph. It operates on "ticks." Every tick, energy flows through synapses based on weights. It is "Thinking Fast"—associative, automatic, and energy-efficient.

### System 2: The Deliberative Controller (`core/system2.py`)
A symbolic validator that monitors System 1. It is "Thinking Slow"—expensive, logical, and only wakes up when System 1 is "surprised."

---

## 2. The Immutable Tick Pipeline

Every discrete time step ($t$) in System 1 follows a strictly serialized 8-phase execution order:

1.  **Transduction (Phase 0)**: External energy enters the system via the `sensory_buffer`.
2.  **Passive Decay (Phase 1)**: All nodes lose a fraction of their energy ($E = E \times 0.9$). This clears "working memory."
3.  **Firing Determination (Phase 2)**: Nodes exceeding their threshold (default 0.75) are marked for firing.
4.  **Refractory & Reset (Phase 3)**: Firing nodes spend their energy and enter a refractory period where they cannot fire again for $N$ ticks.
5.  **Propagation (Phase 4)**: Spike transmission is calculated ($Input = Weight \times Spike$). Updates are buffered to prevent infinite cascades.
6.  **Integration (Phase 5)**: Buffered energy is applied to target nodes.
7.  **Lateral Inhibition (k-WTA) (Phase 6)**: A "Winner-Take-All" mechanism using a Min-Heap. Only the top $K$ most energetic nodes survive; all others are set to zero. This forces the system to make a "decision."
8.  **Surprise Monitor (Phase 7)**: The system compares the current state to the state it predicted earlier.
9.  **Maintenance (Phase 8)**: Housekeeping tasks like decaying novelty and decrementing timers.

---

## 3. Data Structures

### ConceptNode (`core/memory.py`)
The atomic unit of state.
* `energy`: Current membrane potential (0.0 to 1.0).
* `threshold`: Limit at which the node "fires."
* `novelty_score`: High for new concepts, decays over time.

### Synapse (`core/memory.py`)
The unit of association.
* `weight`: Strength of association (dictates energy flow speed).
* `confidence`: Epistemic truth value (used by System 2 to weigh violations).

---

## 4. Bridge & Surprise Math

**PATCHED v5.1**: Uses Root Mean Square (RMS) for proper Euclidean distance calculation.

Surprise is the penalty for **Violated High-Confidence Expectations**. It is calculated as:

$$S = \sqrt{\sum_{n \in Predicted} ((E_{pred}(n) - E_{obs}(n)) \times Confidence(n))^2}$$

*   If the system predicts something with high confidence and it *doesn't* happen, Surprise goes up.
*   If the system sees something completely new (Novelty), Surprise stays low (because there was no prediction to violate).
*   **Gate Threshold**: 0.45 (Lowered from 0.5 for higher sensitivity).

---

## 5. Intervention Protocol

When Surprise exceeds a threshold, System 2 pauses System 1 and executes:
1.  **Diagnosis**: Fetches an `EventSchema` (e.g., `eat.json`) and checks if the current action triple (Agent, Action, Object) violates constraints.
2.  **Intervention**: Modifies System 1 state. It might:
    *   **LTD (Long-Term Depression)**: Weaken the synapse that made the bad prediction.
    *   **Goal Injection**: Inject energy into a `Goal_Node` (like "Query User").
3.  **Resume**: System 1 continues, now directed by the new goal energy.

---

## 6. How to Extend

To add new knowledge to the system:
1.  Add a JSON schema to `core/schemas/`.
2.  Define roles (agent, target) and constraints (is_edible, is_animate).
3.  Use the `System2Controller.set_property()` method to define what the system knows about specific objects.

---

## 7. Complete System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         NCGN SYSTEM                             │
│                   Three-Layer Architecture                       │
└─────────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │     USER     │
                        └──────┬───────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────┐              ┌──────────┐              ┌──────────┐
│Dashboard│              │   CLI    │              │  API     │
│ (Web)   │              │  Chat    │              │  Calls   │
└────┬────┘              └────┬─────┘              └────┬─────┘
     │                        │                         │
     └────────────────────────┼─────────────────────────┘
                              │
                ┌─────────────▼──────────────┐
                │      UI LAYER              │
                │  • Flask WebSocket         │
                │  • Graph Visualization     │
                │  • Real-time Updates       │
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │    CORTEX LAYER            │
                │                            │
                │  ┌────────────────────┐    │
                │  │ Dialogue Manager   │    │
                │  │  State Machine     │    │
                │  └─────────┬──────────┘    │
                │            │               │
                │  ┌─────────▼──────────┐    │
                │  │  Ingestion &       │    │
                │  │  Staging Buffer    │    │
                │  └─────────┬──────────┘    │
                └────────────┼───────────────┘
                             │
                ┌────────────▼───────────────┐
                │     CORE LAYER             │
                │                            │
                │  ┌───────────────────┐     │
                │  │   Graph Memory    │     │
                │  │  • Nodes          │     │
                │  │  • Synapses       │     │
                │  │  • Schemas        │     │
                │  └───────┬───────────┘     │
                │          │                 │
                │  ┌───────▼────┐  ┌───────┐│
                │  │ System 1   │◄─┤System2││
                │  │  (Fast)    │  │(Slow) ││
                │  │            ├─►│       ││
                │  │ 8-Phase    │  │Schema ││
                │  │  Tick      │  │Check  ││
                │  └────────────┘  └───────┘│
                └────────────────────────────┘
```

---

## 8. Detailed Component Interactions

### 8.1 System 1 ↔ System 2 Interaction

```
Time →

Tick 0: [System 1] Normal operation
        • Energy: {dog: 1.0}
        • Prediction: {meat: 0.85}

Tick 1: [System 1] Surprise detected!
        • Observation: {metal: 1.0}
        • Surprise: 0.81 > 0.45
        • Status: PAUSED
        
        [System 2] Interrupt received
        • Load schema: eat.json
        • Check: metal.is_edible = FALSE
        • Diagnosis: CONSTRAINT_VIOLATION
        
        [System 2] Intervention
        • Apply LTD to dog→metal
        • Inject Query_User goal
        • Generate: "Why metal?"
        
        [System 1] Resume with changes
        • Modified weights
        • New goal energy

Tick 2: [System 1] Continue normally
        • Process Query_User goal
```

### 8.2 Dialogue State Transitions

```
State Flow:

IDLE
  │ User: "Dogs eat meat"
  ▼
PROCESSING
  │ Parse, Add to graph
  ▼
IDLE
  │ Success
  
IDLE
  │ User: "Dogs eat metal"
  ▼
PROCESSING
  │ Surprise triggered!
  ▼
CLARIFICATION_PENDING
  │ Waiting for explanation
  │ User: "It's a robot dog"
  ▼
PROCESSING
  │ Parse explanation
  │ Create subclass
  ▼
IDLE
  │ Confirmed
```

---

## 9. API Reference

### 9.1 Core Memory API

```python
from core.memory import GraphMemory

# Initialize
memory = GraphMemory()

# Add nodes
memory.add_node(
    node_id: str,
    energy: float = 0.0,
    threshold: float = 0.75,
    novelty_score: float = 0.0
) -> None

# Add edges
memory.add_synapse(
    source: str,
    target: str,
    weight: float = 0.5,
    confidence: float = 0.5,
    type: str = "associates"
) -> None

# Query
node = memory.get_node(node_id: str) -> Optional[ConceptNode]
neighbors = memory.get_neighbors(node_id: str) -> List[str]
active = memory.get_active_nodes() -> Set[str]
```

### 9.2 System 1 API

```python
from core.system1 import System1Engine

# Initialize
engine = System1Engine(
    memory: GraphMemory,
    decay_alpha: float = 0.9,
    k_winners: int = 10,
    refractory_period: int = 3,
    surprise_threshold: float = 0.45
)

# Operations
engine.inject_energy(node_id: str, energy: float) -> None
engine.tick() -> bool
engine.pause() -> None
engine.resume() -> None

# State queries
firing = engine.get_firing_set() -> Set[str]
surprise = engine.surprise_level -> float
tick = engine.current_tick -> int
```

### 9.3 System 2 API

```python
from core.system2 import System2Controller, Triple

# Initialize
controller = System2Controller(memory: GraphMemory)

# Schema management
controller.add_schema(schema: EventSchema) -> None
controller.load_schema(filename: str) -> EventSchema

# Property management
controller.set_property(
    entity: str,
    property: str,
    value: bool
) -> None

# Diagnosis
result = controller.process_interrupt(
    triple: Triple,
    surprise_level: float,
    firing_set: Set[str]
) -> Tuple[DiagnosisResult, InterventionPlan]
```

---

## 10. Performance Optimization Guide

### 10.1 Memory Optimization

```python
# Prune inactive nodes periodically
if tick % 100 == 0:
    memory.prune_inactive_nodes(threshold=0.001)

# Use smaller K for attention
engine = System1Engine(memory, k_winners=5)  # Instead of 10

# Limit graph size
MAX_NODES = 10000
if len(memory.nodes) > MAX_NODES:
    memory.remove_oldest_nodes(count=1000)
```

### 10.2 Computation Optimization

```python
# Increase decay rate (faster pruning)
engine = System1Engine(memory, decay_alpha=0.85)

# Reduce tick frequency for real-time apps
import time
for _ in range(N):
    engine.tick()
    time.sleep(0.1)  # 10 Hz instead of max speed

# Batch operations
for node_id in batch:
    memory.add_node(node_id)
# Then: memory.rebuild_indices()  # Once
```

### 10.3 Benchmarking

```python
import time

# Time a single tick
start = time.time()
engine.tick()
duration = time.time() - start
print(f"Tick time: {duration*1000:.2f} ms")

# Profile phases
engine.enable_profiling()
for _ in range(100):
    engine.tick()
stats = engine.get_profiling_stats()
print(stats)
```

---

## 11. Troubleshooting Common Issues

### Issue 1: High Memory Usage

**Symptoms**: Memory grows unbounded

**Causes**:
- Not pruning inactive nodes
- Creating too many nodes
- Not cleaning up old edges

**Solutions**:
```python
# Enable auto-pruning
memory.enable_auto_prune(threshold=0.001, interval=100)

# Limit node creation
if len(memory.nodes) < MAX_NODES:
    memory.add_node(node_id)
```

### Issue 2: System 2 Never Triggers

**Symptoms**: No queries, no surprise

**Causes**:
- Threshold too high
- Confidence too low
- No schemas loaded

**Solutions**:
```python
# Lower threshold
engine = System1Engine(memory, surprise_threshold=0.3)

# Increase confidence
memory.add_synapse("A", "B", confidence=0.9)

# Verify schemas
print(controller.schemas)  # Should not be empty
```

### Issue 3: Slow Execution

**Symptoms**: Many seconds per tick

**Causes**:
- Too many active nodes
- Dense graph (high average degree)
- Large K in K-WTA

**Solutions**:
```python
# Reduce K
engine = System1Engine(memory, k_winners=5)

# Increase decay
engine.decay_alpha = 0.85

# Profile to find bottleneck
import cProfile
cProfile.run('engine.tick()', sort='cumulative')
```

---

## 12. Advanced Topics

### 12.1 Custom Learning Rules

Implement custom synaptic plasticity:

```python
class HebbianEngine(System1Engine):
    """
    System 1 with Hebbian learning.
    Synapses strengthen when both nodes are active.
    """
    
    def _phase_9_hebbian_update(self):
        """Apply Hebbian learning rule."""
        for (src, tgt), synapse in self.memory.edges.items():
            src_node = self.memory.get_node(src)
            tgt_node = self.memory.get_node(tgt)
            
            if src_node.energy > 0.5 and tgt_node.energy > 0.5:
                # Strengthen: "Cells that fire together wire together"
                synapse.weight = min(1.0, synapse.weight * 1.01)
            else:
                # Decay unused connections
                synapse.weight = max(0.0, synapse.weight * 0.999)
```

### 12.2 Custom Schemas

Create domain-specific schemas:

```json
{
  "id": "schema_breathe",
  "action": "breathe",
  "confidence": 0.99,
  "roles": {
    "agent": "living_organism",
    "medium": "gas"
  },
  "constraints": {
    "agent": ["is_alive", "has_respiratory_system"],
    "medium": ["is_breathable"]
  },
  "preconditions": ["agent_is_conscious"],
  "postconditions": ["agent_oxygenated"]
}
```

### 12.3 Multi-Modal Integration

Extend to multiple modalities:

```python
class MultiModalMemory(GraphMemory):
    """
    Graph memory with visual, auditory, and textual nodes.
    """
    
    def add_visual_node(self, image_features: np.ndarray):
        """Add node from visual input."""
        node_id = f"visual_{hash(image_features.tobytes())}"
        self.add_node(node_id, modality="visual")
        self.node_features[node_id] = image_features
    
    def cross_modal_association(self, visual_id: str, text_id: str):
        """Link visual and textual representations."""
        self.add_synapse(
            visual_id, text_id,
            weight=0.7,
            type="cross_modal"
        )
```

---

## 13. References and Further Reading

### Academic Papers
- Kahneman, D. (2011). *Thinking, Fast and Slow*. (Dual-process theory)
- Mead, C. (1990). *Neuromorphic Electronic Systems*. (Neuromorphic computing)
- Laird, J. E. (2012). *The Soar Cognitive Architecture*. (Cognitive architectures)

### Implementation Details
- **Surprise Calculation**: Based on prediction error minimization
- **K-WTA**: Inspired by competitive learning in cortex
- **Schema Validation**: From rule-based expert systems

### Related Projects
- SOAR: Symbolic cognitive architecture
- ACT-R: Cognitive architecture for modeling
- Nengo: Neural engineering framework

---

*For complete code examples, see [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)*
*For workflow diagrams, see [WORKFLOW.md](WORKFLOW.md)*
*For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)*
