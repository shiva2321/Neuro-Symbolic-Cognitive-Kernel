# NCGN Architecture Overview

**Last Updated:** January 16, 2026

## System Overview

NCGN (Neuromorphic Cognitive Graph Network) is a bio-inspired neural network system that uses biologically plausible learning mechanisms without traditional deep learning methods (no backpropagation, no gradient descent, no matrix multiplication).

---

## Core Architecture Layers

### Layer 1: BioNode Network (Object-Based)
**Files:** `bionode.py`, `network.py`, `synapse.py`

The foundational neuromorphic network using Python objects:

- **BioNode**: Leaky Integrate-and-Fire (LIF) neurons with:
  - Membrane potential dynamics
  - Refractory periods
  - Homeostatic plasticity (synaptic scaling)
  - Shunting inhibition
  
- **Synapse**: Connection between neurons with:
  - Weight (connection strength)
  - Eligibility traces (for delayed learning)
  - Inhibitory/excitatory type
  
- **NeuromorphicNetwork**: Graph-based network manager using adjacency lists (dictionaries)

**Learning Mechanism:** 3-Factor STDP (Spike-Timing-Dependent Plasticity)
```
ΔWeight = LearningRate × EligibilityTrace × Dopamine
```

**Use Case:** Experiments, prototyping, smaller networks (<100K neurons)

---

### Layer 2: Flash Colony (Binary Memory-Mapped)
**Files:** `flash_colony.py`, `flash_manager.py`, `block_manager.py`

Production-grade neuromorphic system using memory-mapped binary files:

- **FlashColony**: Event-driven push-based architecture
  - Direct binary I/O with zero Python object overhead
  - Memory-mapped persistence (crash-proof, instant load)
  - Scales to millions of neurons (192 MB for 1M neurons)
  
- **Binary Layout:**
  ```
  [HEADER: 256 bytes]
  [NODE POOL: max_nodes × 32 bytes]
  [SYNAPSE POOL: max_nodes × max_edges × 16 bytes]
  ```

**Key Achievement:** 99% memory savings vs Python objects (2GB → 160MB)

**Use Case:** Large-scale networks (1M+ neurons), production systems

---

### Layer 3: Anatomy & Topology
**Files:** `ncgn_anatomy.py`, `neuro_kernel.py`, `flash_dynamic.py`

High-level brain architecture management:

- **NcgnAnatomy**: Semantic brain region management
  - Allocate neuron ID address spaces into functional regions
  - Create structured projections between regions
  - Load-balanced connectivity
  - Export topology for visualization

- **BrainRegion**: Contiguous block of neurons with semantic meaning
- **Projection**: Connection pattern between regions

**Use Case:** Building complex multi-region brain architectures

---

### Layer 4: Cognitive Systems
**Files:** `semantic_brain.py`, `context_driver.py`, `semantic_assistant.py`

Higher-level cognitive capabilities:

- **SemanticBrain**: RDF triple store for knowledge representation
  - Natural language parsing
  - Question answering
  - Knowledge graph queries
  - Binary persistence of semantic networks

**Use Case:** Knowledge-based reasoning, Q&A systems

---

## Experiment Suite

### Core Experiments

1. **Pavlov's Dog** (`pavlov_experiment.py`)
   - Classical conditioning
   - Bell + Food → Salivation
   - Demonstrates 3-Factor STDP learning
   - **Status:** ✅ Working (100% success)

2. **Sequence Learning** (`sequence_experiment.py`)
   - Temporal pattern recognition (A→B→C→A)
   - Predictive coding
   - Teacher forcing
   - **Status:** ✅ Working (100% in epoch 1)

3. **XOR Problem** (`xor_experiment.py`)
   - Non-linear classification
   - Tests neural selectivity
   - **Status:** ⚠️ Partial (50-75% accuracy, weight saturation issue)

4. **Unified Learner** (`unified_learner.py`)
   - ONE network learns all three tasks
   - Tests continuous learning
   - No catastrophic forgetting
   - **Status:** ✅ Working (Pavlov + Sequence retained)

### Support Scripts

- **launcher.py**: Interactive experiment menu
- **learning_dashboard.py**: Real-time Pavlov visualization
- **stress_test.py**: Large-scale performance testing
- **persistence_test.py**: Binary file I/O validation

---

## Key Features

### ✅ What This System DOES Use

- Sparse graph architecture (adjacency lists)
- Leaky Integrate-and-Fire neurons
- Refractory periods
- Shunting inhibition
- 3-Factor STDP learning
- Eligibility traces (delayed reward)
- Homeostatic plasticity
- Binary memory-mapped persistence
- Event-driven computation

### ❌ What This System Does NOT Use

- No matrix multiplication
- No backpropagation
- No gradient descent
- No layers (Conv, Dense, etc.)
- No traditional optimizers (Adam, SGD, RMSprop)
- No NumPy, PyTorch, or TensorFlow

---

## Performance Characteristics

| Metric | BioNode Network | Flash Colony |
|--------|-----------------|--------------|
| Memory per neuron | ~200 bytes | 32 bytes |
| Memory per synapse | ~50 bytes | 16 bytes |
| Neurons per second | ~100,000 | ~500,000+ |
| Max practical size | 100K neurons | 10M+ neurons |
| Persistence | Manual save/load | Automatic (mmap) |
| Cold start time | Seconds | Instant |

---

## Data Flow

```
User Application
    ↓
NcgnAnatomy (topology design)
    ↓
FlashColony (physics engine)
    ↓
Binary Memory-Mapped File
    ↓
Neuromorphic Computation (event-driven)
```

---

## Future Directions

1. **Multi-threaded simulation**: Parallel neuron updates
2. **GPU acceleration**: CUDA kernels for spike propagation
3. **Hierarchical learning**: Deep neuromorphic networks
4. **Multiple neuromodulators**: Dopamine, serotonin, norepinephrine
5. **Sensorimotor integration**: Camera/audio inputs, motor outputs
6. **Online learning**: Continuous adaptation to new data

---

## References

- **STDP**: Markram et al. (1997)
- **3-Factor Learning**: Reynolds & Wickens (2002)
- **LIF Neurons**: Lapicque (1907)
- **Neuromorphic Computing**: Mead (1990)

