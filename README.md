# NCGN: Neuromorphic Cognitive Graph Network

**A Biologically-Inspired "Synthetic Life" AI Architecture**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)]()

---

## 🧠 Overview

NCGN is a neuromorphic computing system that **mimics biological brain function** without using traditional deep learning methods. No matrix multiplication, no backpropagation, no layers—just a sparse graph of interconnected neurons that learn through biologically plausible mechanisms.

This project demonstrates that complex learning behaviors can emerge from simple, local, bio-inspired rules.

---

## 🚫 What This Project Does NOT Use

- ❌ **No Matrix Multiplication**: No NumPy matrices, PyTorch, or TensorFlow
- ❌ **No Backpropagation**: No gradient descent or chain rule
- ❌ **No Layers**: No sequential layer stacks (Conv, Dense, etc.)
- ❌ **No Traditional Optimizers**: No Adam, SGD, or RMSprop
- ❌ **No Heavy Dependencies**: Pure Python with standard library only

---

## ✅ What This Project DOES Use

- ✅ **Sparse Graph Architecture**: Adjacency lists (dictionaries) for scalability
- ✅ **Leaky Integrate-and-Fire Neurons**: Membrane potential with decay
- ✅ **Refractory Period**: Prevents seizure-like activity
- ✅ **Shunting Inhibition**: Active suppression of neural activity
- ✅ **3-Factor STDP Learning**: Spike-Timing-Dependent Plasticity with neuromodulation
- ✅ **Eligibility Traces**: Allows delayed reward learning
- ✅ **Homeostatic Plasticity**: Self-regulating synaptic scaling
- ✅ **Binary Memory-Mapped Persistence**: Flash-native, crash-proof storage
- ✅ **Event-Driven Computation**: Only active neurons compute

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Node_network.git
cd Node_network

# No dependencies to install! Pure Python.
```

### Run Your First Experiment

```bash
# Interactive launcher with menu
python launcher.py

# Or run experiments directly:
python pavlov_experiment.py      # Classical conditioning (2 seconds)
python sequence_experiment.py    # Sequence learning (30 seconds)
python unified_learner.py        # All experiments in one network (5 minutes)
```

### Try the Semantic Assistant

```bash
python simple_demo.py
```

```
Q: Who invented the compiler?
A: HOPPER

Q: Who created Python?
A: ROSSUM
```

---

## 🎯 Demonstrated Capabilities

### ✅ Classical Conditioning (Pavlov's Dog)
- **Network**: 3 neurons, 2 synapses
- **Training**: 15 trials (~2 seconds)
- **Success Rate**: 100%
- **Achievement**: Bell → Salivation learned through dopamine-modulated STDP

### ✅ Sequence Learning (Temporal Prediction)
- **Network**: 6 neurons, 9 synapses  
- **Training**: 50-500 epochs (~30 seconds)
- **Success Rate**: 100%
- **Achievement**: Learns A→B→C→A pattern with predictive coding

### ⚠️ XOR Problem (Non-Linear Classification)
- **Network**: 5 neurons, 8 synapses
- **Training**: 400-800 epochs (~60 seconds)
- **Success Rate**: 50-75% (partial)
- **Status**: Hidden neurons converge to similar weights (known issue)

### ✅ Unified Multi-Task Learning
- **Network**: 30 neurons, multiple task regions
- **Training**: All three tasks sequentially
- **Success Rate**: 100% Pavlov, 100% Sequence retained
- **Achievement**: **No catastrophic forgetting!**

### ✅ Semantic Question Answering
- **Network**: Flash-based RDF triple store
- **Capability**: Natural language query parsing
- **Success Rate**: 100% on knowledge base queries
- **Achievement**: "Who invented X?" style reasoning

---

## 🏗️ Architecture

### Two Implementation Layers

#### 1. BioNode Network (Object-Based)
**Files**: `bionode.py`, `network.py`, `synapse.py`

Python object-based neuromorphic network:
- Perfect for experiments and prototyping
- ~200 bytes per neuron, ~50 bytes per synapse
- Suitable for networks up to ~100K neurons
- Full introspection and debugging

#### 2. Flash Colony (Binary Memory-Mapped)
**Files**: `flash_colony.py`, `flash_manager.py`

Production-grade binary implementation:
- 32 bytes per neuron, 16 bytes per synapse (99% memory savings)
- Memory-mapped I/O (instant cold start, automatic persistence)
- Event-driven push architecture (only active neurons compute)
- Scales to millions of neurons (tested with 1M)
- Crash-proof (state always on disk)

---

## 📂 Project Structure

```
Node_network/
├── README.md                    # This file
├── requirements.txt             # No dependencies needed!
├── launcher.py                  # Interactive experiment menu
│
├── Core Library/
│   ├── bionode.py              # LIF neuron with homeostasis
│   ├── synapse.py              # Connection with eligibility traces
│   ├── network.py              # Graph-based network manager
│   ├── flash_colony.py         # Binary neuromorphic engine
│   ├── flash_manager.py        # Memory-mapped file I/O
│   ├── block_manager.py        # Synapse block allocator
│   └── ncgn_anatomy.py         # Brain region management
│
├── Experiments/
│   ├── pavlov_experiment.py    # Classical conditioning
│   ├── sequence_experiment.py  # Temporal pattern learning
│   ├── xor_experiment.py       # Non-linear classification
│   └── unified_learner.py      # Multi-task continuous learning
│
├── Semantic System/
│   ├── semantic_brain.py       # RDF knowledge graph
│   ├── context_driver.py       # NLP parser & reasoning
│   ├── semantic_assistant.py   # Q&A interface
│   └── simple_demo.py          # Demo script
│
├── Visualization/
│   └── learning_dashboard.py   # Real-time training visualization
│
├── Tests/
│   └── tests/                  # Unit tests (pytest)
│
└── Documentation/
    ├── docs/ARCHITECTURE.md         # System design deep dive
    ├── docs/EXPERIMENTS.md          # Experiment guide
    ├── docs/IMPLEMENTATION_NOTES.md # Technical details
    └── docs/archive/                # Historical docs
```

---

## 🧪 How It Works

### 3-Factor STDP Learning

The core learning mechanism combines three factors:

```
ΔWeight = LearningRate × EligibilityTrace × Dopamine

Where:
- EligibilityTrace: Did the presynaptic neuron fire recently?
- Dopamine: Is there a global reward signal?
- Post-spike: Did this neuron fire now?
```

This allows **delayed reward learning** without backpropagation!

### Leaky Integrate-and-Fire Dynamics

```python
# 1. Decay (leak toward resting potential)
potential *= decay_rate

# 2. Integrate inputs
potential += sum(synaptic_currents)

# 3. Fire if threshold reached
if potential >= threshold:
    spike()
    potential = 0
```

### Homeostatic Plasticity

Neurons self-regulate to maintain target firing rates:

```python
# Track average firing rate
avg_rate = 0.99 * avg_rate + 0.01 * (1 if fired else 0)

# Scale weights to reach target
if avg_rate > target:
    scale_down_all_weights()
elif avg_rate < target:
    scale_up_all_weights()
```

---

## 📊 Performance

*Tested on: Intel i7-10700K, 32GB RAM, Windows 11, Python 3.12*

| Metric | BioNode Network | Flash Colony |
|--------|-----------------|--------------|
| Memory per neuron | ~200 bytes | 32 bytes |
| Memory per synapse | ~50 bytes | 16 bytes |
| Neurons/second | ~100K | ~500K+ |
| Max practical size | 100K neurons | 10M+ neurons |
| Cold start time | Seconds | Instant (mmap) |
| Persistence | Manual | Automatic |

### Example: 1 Million Neurons
- **Memory**: 192 MB (vs 2GB for Python objects)
- **Disk I/O**: Zero-copy memory mapping
- **Crash recovery**: Automatic (state always on disk)

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Deep dive into system design, layers, and data flow
- **[EXPERIMENTS.md](docs/EXPERIMENTS.md)**: Complete guide to running and customizing experiments
- **[IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)**: Technical implementation details and optimizations

---

## 🔬 Biological Inspiration

### Neuroscience Principles
- **LIF Neurons**: Lapicque (1907)
- **STDP**: Markram et al. (1997), Bi & Poo (1998)
- **3-Factor Learning**: Reynolds & Wickens (2002)
- **Homeostatic Plasticity**: Turrigiano & Nelson (2004)
- **Eligibility Traces**: Sutton & Barto (2018)

### Neuromorphic Computing
- **Event-Driven Architecture**: SpiNNaker, BrainScaleS projects
- **Sparse Connectivity**: Biological cortex is ~1% connected
- **Local Learning**: No global error signals needed

---

## 🎓 Educational Value

This project is ideal for:
- **Computational Neuroscience**: See how biological principles translate to code
- **Neuromorphic Engineering**: Understand event-driven, spike-based computation
- **Alternative AI**: Explore learning without backpropagation
- **Systems Programming**: Learn binary I/O, memory mapping, performance optimization

---

## 🛠️ Development

### Adding New Experiments

```python
from network import NeuromorphicNetwork

# 1. Build network
net = NeuromorphicNetwork()
net.add_node(0, "input")
net.add_node(1, "output")
net.connect(0, 1, weight=0.1)

# 2. Training loop
for epoch in range(100):
    net.set_input(0, value=1.0)
    dopamine = 1.0 if correct else 0.0
    net.step(dopamine, learning_rate=0.01)
    
# 3. Test
if net.is_firing(1):
    print("Success!")
```

### Running Tests

```bash
cd tests
pytest
```

---

## 🐛 Known Issues

### XOR Weight Convergence
Hidden neurons converge to similar weights, losing the differential needed for XOR.

**Potential solutions:**
- Stronger lateral inhibition
- More hidden neurons  
- Different initialization schemes
- Separate dopamine signals per output

See [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for details.

---

## 🚀 Future Directions

- [ ] Multi-threaded simulation (parallel neuron updates)
- [ ] GPU acceleration (CUDA kernels)
- [ ] Hierarchical learning (deep neuromorphic networks)
- [ ] Multiple neuromodulators (serotonin, norepinephrine)
- [ ] Sensorimotor integration (camera/audio inputs)
- [ ] Evolutionary meta-learning
- [ ] Hardware deployment (Loihi, TrueNorth)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Performance optimization
- New neuron models (Izhikevich, adaptive LIF)
- Learning algorithms (BCM rule, Oja's rule)
- Visualization tools
- Hardware interfacing
- Documentation improvements

---

## 📄 License

MIT License - Feel free to use in your own neuromorphic projects!

---

## 🙏 Acknowledgments

Inspired by pioneers in:
- **Neuroscience**: Hodgkin, Huxley, Lapicque, Markram
- **Neuromorphic Engineering**: Carver Mead, Giacomo Indiveri
- **Computational Neuroscience**: Abbott, Dayan, Gerstner

---

## 📞 Contact

For questions, ideas, or collaboration:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Review [experiment examples](launcher.py)

---

**Built with biological principles, not gradients. 🧠⚡**

---

## 📖 Quick Examples

### Example 1: Simple Network

```python
from network import NeuromorphicNetwork

# Create network
net = NeuromorphicNetwork()
net.add_node(0, "input")
net.add_node(1, "output", threshold=1.0)
net.connect(0, 1, weight=0.5)

# Simulate
net.set_input(0, 1.0)
net.step(global_dopamine=0.0, learning_rate=0.01)

# Check result
print(f"Output fired: {net.is_firing(1)}")
```

### Example 2: Learning Loop

```python
from pavlov_experiment import main

# Run Pavlov's Dog experiment
main()

# Output:
# Phase 1: Baseline (no learning)
# Phase 2: Training (15 trials)
# Phase 3: Test (learned!)
# 🎉 SUCCESS! Classical conditioning achieved!
```

### Example 3: Semantic Q&A

```python
from context_driver import SemanticBrain

ai = SemanticBrain()
ai.learn_rdf("GRACE HOPPER INVENTED THE COMPILER.")
answer = ai.query("Who invented the compiler?")
print(answer)  # HOPPER
```

---

**Last Updated:** January 16, 2026

