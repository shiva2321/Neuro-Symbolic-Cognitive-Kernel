# Phase 1: Core Architecture Refactor

**Date:** January 17, 2026  
**Status:** COMPLETE ✅  
**Lines Changed:** 500+ new code  
**Tests Passing:** 11/11 invariant tests

## Problem Statement

The original `BioNode` class violates separation of concerns by bundling:
- Neural state integration (computation)
- Spike emission (control)
- Plasticity rules (learning)
- Homeostatic regulation (adaptation)

This makes it hard to:
- Test components in isolation
- Disable learning without breaking inference
- Reason about data flow
- Swap implementations
- Enforce architectural invariants

## Solution: Modular Interface Contracts

We've extracted five clean interface boundaries:

### 1. Spike Event System (`spike.py`)
**Responsibility:** Define immutable spike events with causality guarantees

```python
@dataclass(frozen=True)
class Spike:
    source_id: int      # What neuron fired?
    timestamp: int      # When did it fire?
    magnitude: float    # How strong was the spike?
```

**Guarantees:**
- Immutable: once created, cannot be modified
- Validated: constructor enforces non-negativity
- Causality-safe: can be ordered temporally

### 2. Event Queue (`event_queue.py`)
**Responsibility:** Maintain time-ordered spike queue, enforce causality

```python
class EventQueue:
    def inject(spike: Spike) -> None
        # Raises ValueError if spike.timestamp < current_time
    
    def pop_next() -> Optional[Spike]
        # Returns next spike in temporal order
```

**Invariant:**
- **Event Causality:** No spike can be injected with timestamp in the past
- **Determinism:** Same injection sequence always produces same pop order

### 3. Dispatcher (`dispatcher.py`)
**Responsibility:** Route spikes from sources to targets via synapses

```python
class Dispatcher:
    def register_outgoing(source_id, target_id, synapse_index)
    
    def fan_out(spike, get_synapse_func) -> List[(target_id, WeightedSpike)]
        # Routes spike through synapses to all targets
```

**Invariant:**
- **Locality:** Dispatcher only routes; doesn't compute
- **Separation:** Spike arrival decoupled from weight modification

### 4. Neuron (`neuron.py`)
**Responsibility:** Integrate weighted inputs and emit spikes

```python
class Neuron:
    def integrate(current_tick, weighted_spikes) -> Optional[Spike]
        # Steps: refractory → leak → input → integrate → threshold → fire
    
    def update_traces(network_state)
        # Update eligibility traces for all incoming synapses
```

**Invariant:**
- **Locality:** Only accesses own state + incoming synapses
- **Determinism:** Same inputs always produce same spike
- **No Side Effects on Synapses:** integrate() doesn't modify weights

### 5. Plasticity (`plasticity.py`)
**Responsibility:** Manage all learning rules and adaptation

```python
class PlasticityController:
    def update_learning(synapse, post_fired, dopamine)
        # Apply STDP learning
    
    def update_homeostasis(synapses, firing_rate, node_type)
        # Apply homeostatic scaling
    
    def enable_learning() / disable_learning()
        # Gate learning independently from inference
```

**Invariant:**
- **Plasticity Isolation:** Learning can be disabled without affecting spike computation
- **Bounded Growth:** Synaptic weights stay bounded [0, 2.0]
- **Stability:** Homeostasis is always active for stability

### 6. Graph Store (`graph_store.py`)
**Responsibility:** Persistent network topology (separate from computation state)

```python
class GraphStore:
    def add_node(node_id, node_type, threshold, refractory_period)
    
    def add_edge(source_id, target_id, weight, is_inhibitory)
    
    def get_incoming_edges(node_id) -> Iterable[EdgeDesc]
    
    def get_nodes() -> Iterable[NodeDesc]
```

**Invariant:**
- **Topology-Only:** Stores structure, not state (no potentials, firing, traces)
- **Consistency:** Every edge connects valid nodes
- **Serializability:** Can be persisted and restored independently

## Separation of Concerns Matrix

| Component | Computes? | Stores State? | Modifies Weights? | Access Network? |
|-----------|-----------|---------------|-------------------|-----------------|
| Spike | ❌ | ❌ | ❌ | ❌ |
| EventQueue | ❌ | ✅ (queue) | ❌ | ❌ |
| Dispatcher | ❌ | ❌ | ❌ | ✅ (routing) |
| Neuron | ✅ | ✅ (potential) | ❌ | ✅ (inputs only) |
| Synapse | ❌ | ✅ (weight, trace) | ❌ | ❌ |
| PlasticityController | ❌ | ❌ | ✅ | ❌ |
| GraphStore | ❌ | ✅ (topology) | ❌ | ❌ |

## Invariants Enforced (11 tests, 100% passing)

### Event Causality
- ✅ Cannot inject spikes into the past
- ✅ Same injection sequence always produces same order

### Locality
- ✅ Neurons only access their inputs
- ✅ Synapses have no side effects on transmission

### Plasticity Isolation
- ✅ Learning can be disabled without affecting spike computation
- ✅ Inference works identically with/without learning gate

### Bounded Growth
- ✅ Synaptic weights never exceed safe bounds
- ✅ Homeostasis prevents saturation

### Graph Consistency
- ✅ All edges reference valid nodes
- ✅ Incoming/outgoing edges stay in sync

## Data Flow Example (Learning Trial)

```
EventQueue                     Dispatcher
┌─────────────┐               ┌─────────────┐
│ Spike(0, 5) │──inject────→  │ Route to    │
│             │  t=5          │ targets 1,2 │
└─────────────┘               └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
                Neuron 1         Neuron 2         Neuron 3
                ┌─────────┐     ┌─────────┐     ┌─────────┐
                │integrate│     │integrate│     │integrate│
                │  ws(1)  │     │  ws(2)  │     │  ws(3)  │
                │V+=0.5   │     │V+=0.3   │     │V-=0.7   │
                └────┬────┘     └────┬────┘     └────┬────┘
                     │               │               │
                  fires?          fires?          fires?
                   /                /                /
                 YES             NO              NO
                  │               │               │
                  ▼               ▼               ▼
            emit Spike(1,6)   emit none       emit none
                  │
                  └─→ PlasticityController
                      update_learning(
                          synapse=(0→1),
                          post_fired=True,
                          dopamine=0.5
                      )
                      → weight += 0.01 × trace × dopamine
                      → weight: 0.50 → 0.505
                      
                      update_homeostasis(
                          firing_rate=0.3,
                          target_rate=0.25
                      )
                      → scale ↑ to increase excitability
```

## Backward Compatibility

**Legacy code (old experiments) continues to work:**
- `BioNode`, `NeuromorphicNetwork` still exist
- `Synapse.get_current()` still works (deprecated but functional)
- Experiments can run without modification

**Migration path (future phases):**
1. ✅ Phase 1: Extract modules, enforce interfaces
2. 🔄 Phase 2: Build event-driven orchestrator
3. 🔄 Phase 3: Migrate experiments to new API
4. 🔄 Phase 4: Deprecate legacy code

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `core/spike.py` | 70 | Immutable spike events |
| `core/event_queue.py` | 110 | Temporal ordering |
| `core/dispatcher.py` | 100 | Spike routing |
| `core/neuron.py` | 180 | Refactored neuron |
| `core/plasticity.py` | 220 | Learning rules |
| `core/graph_store.py` | 180 | Topology storage |
| `tests/test_invariants.py` | 235 | Invariant verification |

## Verification Checklist

- ✅ All new modules are importable
- ✅ All interface contracts are explicit (docstrings + type hints)
- ✅ All invariants have tests
- ✅ 11/11 invariant tests pass
- ✅ No circular imports
- ✅ Backward compatibility preserved
- ✅ Separation of concerns achieved
- ✅ No global state introduced

## Next Steps (Phase 2)

1. **Build Neural Engine:** Compose modules into cohesive simulator
   - EventQueue → Dispatcher → Neurons → Plasticity pipeline
   - Handle multiple neurons firing simultaneously

2. **Event-Driven Update:** Replace synchronous "tick" with event loop
   - Only update neurons that receive spikes
   - 10-100x speedup for sparse networks

3. **Migrate Experiments:** Update existing experiments to use new API
   - PavlovExperiment
   - SequenceLearningExperiment
   - XORExperiment

4. **Performance Monitoring:** Add probes and metrics
   - Per-neuron statistics
   - Synaptic weight tracking
   - Learning rate telemetry

## Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│                   Neural Engine (Phase 2)              │
│  EventQueue → Dispatcher → Neurons → PlasticityCtrl   │
└────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    ┌─────────┐        ┌─────────┐     ┌──────────┐
    │GraphStore│       │Neuron   │     │PlasticCtrl
    │          │       │(compute)│     │(learning)│
    │(topology)│       │         │     │          │
    └─────────┘       └─────────┘     └──────────┘
        ▲                  ▲                  ▲
        │                  │                  │
        └──────────────────┼──────────────────┘
                      ┌────────────┐
                      │   Synapse  │
                      │(transmission)
                      └────────────┘
```

---

**Maintainers' Notes:**
- All changes are **incremental and reversible**
- Existing tests in `/experiments` and `/tests` still pass
- No behavioral changes to learning algorithms
- This is **not** a breaking change for users
