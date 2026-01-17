# Implementation Notes & Technical Details

**Last Updated:** January 16, 2026

This document consolidates technical implementation notes from various development phases.

---

## Binary Synapse Storage (Phase 2)

### Problem Solved
Moved from Python dictionary-based synapse storage (2GB RAM) to memory-mapped binary files (160MB).

### Architecture: Event-Driven Push

**Old (Pull-Based)** ❌
- Every tick, scan ALL connections
- O(total_connections) complexity
- Wastes cycles on silent neurons

**New (Push-Based)** ✅
- Only process neurons that fired
- O(firing_neurons × avg_out_degree)
- Zero compute if neuron doesn't fire

### Binary Layout

**Synapse Pool Strategy:**
```
Each node gets a fixed 160-byte block for max 10 synapses
Block offset = POOL_START + (NodeID × 160)
Synapse offset = Block + (Index × 16)
```

**Synapse Structure (16 bytes):**
```c
struct Synapse {
    uint32_t target_id;      // 4 bytes
    float weight;            // 4 bytes
    float eligibility_trace; // 4 bytes
    float permanence;        // 4 bytes
}
```

**Node Structure (32 bytes):**
```c
struct Node {
    uint32_t id;             // 4 bytes
    float potential;         // 4 bytes
    float threshold;         // 4 bytes
    float avg_firing_rate;   // 4 bytes
    uint64_t edge_ptr;       // 8 bytes (offset to synapse block)
    uint32_t edge_count;     // 4 bytes
    uint32_t node_type;      // 4 bytes (0=input, 1=hidden, 2=output)
}
```

### Performance Results
- **Memory savings:** 99% (2GB → 160MB for 10M synapses)
- **Cold start:** Instant (mmap loads on-demand)
- **Crash recovery:** Automatic (state always on disk)
- **Scale:** Tested with 1M neurons, 10M synapses

**Implementation:** `flash_colony.py`, `flash_manager.py`

---

## Homeostatic Plasticity (Synaptic Scaling)

### Problem Solved
Networks suffered from weight saturation and runaway excitation, particularly in the XOR experiment.

### Solution: Biological Homeostasis

Each neuron tracks its average firing rate and scales synaptic weights to maintain target activity.

**Mechanism:**
```python
# Track firing rate (exponential moving average)
avg_firing_rate = 0.99 * avg_firing_rate + 0.01 * (1 if fired else 0)

# Compare to target
rate_diff = avg_firing_rate - target_rate  # target = 0.25

# Scale all incoming weights
scaling_factor = 1.0 - (rate_diff * homeostasis_rate)
for synapse in inputs:
    synapse.weight *= scaling_factor
    synapse.weight = clamp(synapse.weight, 0.1, 3.0)
```

### Results

**Sequence Learning:**
- ✅ No regression - still achieves 100% accuracy
- Weights stabilize without saturation

**XOR Problem:**
- ⚠️ Partial improvement - accuracy 50% → 50-75%
- Weights no longer saturate at 2.0 ceiling
- Root cause persists: hidden neurons converge to similar weights

### Tuning Parameters
```python
target_rate = 0.25          # Fire 25% of time (selective coding)
homeostasis_rate = 0.005    # Speed of adjustment (slow)
```

**Implementation:** `bionode.py` (lines 29-31, 145-158)

---

## Semantic Parser Enhancement

### Problem Solved
Natural language query parser was too strict, rejecting common questions like "Who invented the compiler?"

### Changes Made

**1. Expanded Verb Vocabulary**
Added 20+ common verbs:
```python
KNOWN_VERBS = {
    "IS", "ARE", "EAT", "WRITE", "FLY", "SWIM",
    "INVENT", "INVENTED", "CREATE", "CREATED",
    "BUILD", "MAKE", "DEVELOP", "DESIGN", "DISCOVER",
    # ... more
}
```

**2. Verb Normalization**
```python
def _normalize_verb(verb):
    # WRITES → WRITE
    if verb.endswith('ES'): return verb[:-1]
    
    # INVENTED → INVENT
    if verb.endswith('ED'): return verb[:-2]
    
    # FLIES → FLY
    if verb.endswith('IES'): return verb[:-3] + 'Y'
    
    return verb
```

**3. Intelligent Entity Extraction**
- Fuzzy verb matching (handles word stems)
- Prioritizes longer/more specific entities
- Graceful fallback to "IS" for ambiguous cases

**4. Intransitive Verb Support**
```python
# "BIRDS FLY." → BIRDS --[CAN]--> FLY_ACTION
if subject and not object:
    verb_action = f"{verb}_ACTION"
    self.add_triple(subject, "CAN", verb_action)
```

### Results
| Query | Before | After |
|-------|--------|-------|
| "Who invented the compiler?" | ❌ Failed | ✅ HOPPER |
| "Who created Python?" | ❌ Failed | ✅ ROSSUM |
| "What flies?" | ❌ Failed | ✅ BIRDS |
| Success Rate | 20% | 100% |

**Implementation:** `context_driver.py`, `semantic_assistant.py`

---

## Load-Balanced Projection Wiring

### Problem Solved
When creating projections between brain regions, random wiring could overload target neurons (some getting 100+ inputs while others got 0).

### Solution: Smart Target Selection

**Algorithm:**
```python
def _smart_target_selection(self, targets, k):
    """Select k targets, preferring those with fewer existing inputs"""
    
    # Sort targets by in-degree (ascending)
    targets_with_degree = [
        (t, self.colony.get_in_degree(t))
        for t in targets
    ]
    targets_with_degree.sort(key=lambda x: x[1])
    
    # Prefer low-degree targets (70% from bottom quartile)
    quartile_size = len(targets) // 4
    low_degree_pool = targets_with_degree[:quartile_size]
    high_degree_pool = targets_with_degree[quartile_size:]
    
    selected = []
    for _ in range(k):
        if random.random() < 0.7 and low_degree_pool:
            target, _ = random.choice(low_degree_pool)
            low_degree_pool.remove((target, _))
        else:
            target, _ = random.choice(high_degree_pool)
            high_degree_pool.remove((target, _))
        selected.append(target)
    
    return selected
```

### Results
- More uniform degree distribution
- Prevents bottlenecks and dead neurons
- Improves learning stability

**Implementation:** `ncgn_anatomy.py`

---

## Teacher Forcing for Sequence Learning

### Problem Solved
Sequence learning was slow to converge without guidance on correct outputs.

### Solution: Supervised Spike Injection

Force the correct output neuron to fire during training:

```python
def teacher_forcing(network, correct_output_id):
    """Inject spike into correct output neuron"""
    if correct_output_id in network.nodes:
        network.nodes[correct_output_id].external_current = 5.0
        network.nodes[correct_output_id].is_firing = True
```

### Usage
```python
# Training step
network.set_input(current_input, 1.0)
teacher_forcing(network, correct_output)  # Guide learning
network.step(dopamine=1.0, learning_rate=0.01)
```

### Results
- Epoch to 100% accuracy: 300+ → 50-100
- More reliable learning
- Faster convergence

**Implementation:** Used in `sequence_experiment.py`, `unified_learner.py`

---

## Eligibility Traces for Delayed Reward

### Biological Basis
Real neurons need to know which synapses were responsible for a reward that arrives milliseconds/seconds later.

### Mechanism
```python
class Synapse:
    def __init__(self):
        self.eligibility_trace = 0.0
    
    def update_trace(self, pre_fired, decay=0.9):
        # Decay existing trace
        self.eligibility_trace *= decay
        
        # Boost if presynaptic neuron fired
        if pre_fired:
            self.eligibility_trace += 1.0
```

### 3-Factor STDP Learning
```python
if post_fired and dopamine > 0:
    for synapse in inputs:
        delta_w = learning_rate * synapse.eligibility_trace * dopamine
        synapse.weight += delta_w
```

### Timing Requirements
- Trace decay: 0.9 (keeps memory for ~10 ticks)
- Dopamine delay: Must arrive within trace lifetime
- Pre-fire timing: Must occur before post-fire + dopamine

**Implementation:** `synapse.py`, used throughout all networks

---

## Refractory Period Implementation

### Biological Basis
After firing, real neurons enter an absolute refractory period where they cannot fire again (sodium channel inactivation).

### Implementation
```python
class BioNode:
    def tick(self, current_tick, network_state):
        # Check if in refractory period
        if current_tick - self.last_spike_tick < self.refractory_period:
            self.is_firing = False
            self.potential = self.resting_potential
            return
        
        # ... normal update ...
        
        if self.potential >= self.threshold:
            self.is_firing = True
            self.last_spike_tick = current_tick
```

### Effects
- Prevents seizure-like runaway excitation
- Limits maximum firing rate
- Improves temporal coding
- More biologically realistic

**Default Value:** 3 ticks (adjustable per neuron)

---

## Shunting Inhibition

### Biological Basis
GABAergic inhibitory synapses don't just subtract from excitation—they shunt (clamp) the membrane potential.

### Implementation
```python
# Calculate inhibitory influence
inhibitory_sum = sum(
    synapse.weight * network_state.get(src_id, False)
    for src_id, synapse in self.inputs.items()
    if synapse.is_inhibitory
)

# Apply shunting (multiplicative suppression)
if inhibitory_sum > 0:
    self.potential *= (1.0 / (1.0 + inhibitory_sum))
```

### Effects
- Stronger suppression than subtractive inhibition
- More realistic lateral competition
- Essential for XOR and other non-linear problems

**Implementation:** `bionode.py` (lines 100-110)

---

## Memory-Mapped File I/O

### Benefits
1. **Instant cold start**: OS lazy-loads pages on demand
2. **Automatic persistence**: Changes written directly to disk
3. **Crash recovery**: State always consistent
4. **Shared memory**: Multiple processes can access same brain
5. **Exceeds RAM**: Can work with brain files larger than physical RAM

### Usage
```python
import mmap
import os

# Create/open brain file
f = open("brain.dat", "r+b")
mm = mmap.mmap(f.fileno(), 0)

# Direct binary read/write
node_offset = HEADER_SIZE + (node_id * NODE_SIZE)
data = struct.unpack('I f f f Q I I', mm[node_offset:node_offset+32])

# Changes automatically sync to disk
mm.close()
f.close()
```

**Implementation:** `flash_manager.py`, `flash_colony.py`

---

## Event-Driven Computation

### Philosophy
Only compute for neurons that need computation (those receiving spikes).

### Implementation
```python
def step(self):
    # 1. Detect firing neurons
    firing_nodes = [nid for nid, node in self.nodes.items() if node.is_firing]
    
    # 2. For each firing neuron, read its synapses
    for source_id in firing_nodes:
        synapses = self.read_synapses(source_id)
        
        # 3. Push spikes to targets (event-driven!)
        for target_id, weight in synapses:
            self.inject_current(target_id, weight)
    
    # 4. Update only affected neurons
    for node in affected_nodes:
        node.update()
```

### Benefits
- O(active_neurons) instead of O(all_neurons)
- 10-100x speedup for sparse activity
- Scales naturally to huge networks
- Matches biological brain efficiency

**Implementation:** `flash_colony.py`

---

## Continuous Learning Without Forgetting

### Strategy
Use separate, non-overlapping node regions for different tasks:

```python
# Task 1: Pavlov (nodes 0-2)
net.add_node(0, "input")   # Bell
net.add_node(1, "input")   # Food
net.add_node(2, "output")  # Salivate

# Task 2: Sequence (nodes 10-15)
net.add_node(10, "input")  # A
net.add_node(11, "input")  # B
# ... etc

# NO OVERLAP = NO INTERFERENCE
```

### Trace Clearing Between Tasks
```python
network.reset()  # Clear eligibility traces, keep weights
```

### Results
✅ **No catastrophic forgetting** in unified learner experiment
- Pavlov: 100% retained after sequence + XOR training
- Sequence: 100% retained after XOR training
- XOR: Partial learning (50-75%) but doesn't damage prior tasks

**Implementation:** `unified_learner.py`

---

## File Format Specification

### Brain File Structure

```
[HEADER: 256 bytes]
  - Magic: "NCGN" (4 bytes)
  - Version: 1 (4 bytes)
  - NodeCount: current nodes (4 bytes)
  - MaxNodes: capacity (4 bytes)
  - MaxEdgesPerNode: 10 (4 bytes)
  - Reserved: (236 bytes)

[NODE POOL: max_nodes × 32 bytes]
  - See Node Structure above

[SYNAPSE POOL: max_nodes × max_edges_per_node × 16 bytes]
  - See Synapse Structure above
```

### Semantic Brain File Structure

```
[HEADER: 256 bytes]
  - Magic: "SEMB" (4 bytes)
  - Version: 1 (4 bytes)
  - NodeCount: (4 bytes)
  - MaxNodes: (4 bytes)

[STRING POOL: variable]
  - Null-terminated concept strings

[TRIPLE POOL: variable]
  - (subject_id, predicate_id, object_id) triples
```

**Implementation:** `flash_manager.py`, `semantic_brain.py`

---

## Known Issues & Workarounds

### 1. XOR Weight Convergence
**Issue:** Hidden neurons converge to similar weights, losing selectivity.

**Workarounds:**
- Increase lateral inhibition between hidden nodes
- Use more hidden neurons
- Different weight initialization schemes
- Separate dopamine signals per output neuron

### 2. Learning Rate Sensitivity
**Issue:** Too high → unstable, too low → slow convergence.

**Recommendation:** 0.01 for most tasks, 0.005 for complex problems.

### 3. Memory-Mapped File Locking (Windows)
**Issue:** Files can't be deleted while mmap is open.

**Workaround:**
```python
mm.close()
f.close()
import time
time.sleep(0.1)  # Allow OS to release handle
os.remove("brain.dat")
```

### 4. Epoch Requirements Vary
**Issue:** Sequence learning sometimes needs 300+ epochs, other times converges in 50.

**Reason:** Random weight initialization causes variance.

**Solution:** Run for sufficient epochs (300-500) or implement early stopping.

---

## Performance Optimization Tips

1. **Use Flash Colony for large networks** (>10K neurons)
2. **Enable sparse connectivity** (max 10-20 edges per node)
3. **Batch multiple steps** before checking results
4. **Use event-driven architecture** (only update firing neurons)
5. **Memory-map brain files** (avoid load/save overhead)
6. **Disable debug prints** in inner loops
7. **Profile with `cProfile`** to find bottlenecks

---

## Testing Best Practices

1. **Always call `network.reset()`** between training phases
2. **Use fixed random seeds** for reproducible tests
3. **Test both learning AND retention** in multi-task scenarios
4. **Verify weight values**, not just accuracy
5. **Check for weight saturation** (all weights hitting limits)
6. **Monitor firing rates** (should be sparse, ~10-30%)

---

## References

- **Binary File I/O:** Python struct module documentation
- **Memory Mapping:** Python mmap module documentation  
- **STDP:** Bi & Poo (1998), Markram et al. (1997)
- **Eligibility Traces:** Sutton & Barto (2018) - Reinforcement Learning
- **Homeostatic Plasticity:** Turrigiano & Nelson (2004)
- **Event-Driven Computing:** SpiNNaker, BrainScaleS projects

