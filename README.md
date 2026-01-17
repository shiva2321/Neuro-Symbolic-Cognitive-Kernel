# NCGN: Neuromorphic Cognitive Graph Network

**A Complete 5-Tier Artificial Brain: From Spikes to Self-Awareness**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)]()
[![Phase 5 Complete](https://img.shields.io/badge/Phase-5.0%20Complete-brightgreen.svg)]()

---

## 🧠 What This Is

NCGN is a **complete, biologically-inspired neuromorphic architecture** that implements a functional artificial brain without using traditional deep learning. No matrix multiplication, no backpropagation, no layers—just a sparse graph of spiking neurons that learn through proven neuroscience principles.

**What makes it special:**
- Implements 5 cognitive tiers, from millisecond spike reactions to goal-directed reasoning
- Learns to decide WHEN to learn (Tier 3)
- Reasons explicitly with facts and rules (Tier 4)
- Sets goals and learns from prediction errors (Tier 5)
- Pure Python, no dependencies, fully transparent

**Status:** ✅ Phase 5 Complete (January 17, 2026)

---

## 🏗️ The 5-Tier Architecture

### Tier 1: Spiking Neural Substrate (Phase 1-2)
Core neuromorphic engine with biologically plausible learning.

**Key Components:**
- Leaky Integrate-and-Fire neurons with membrane potential decay
- Refractory periods preventing runaway firing
- 3-Factor STDP learning (eligibility trace × dopamine × post-spike)
- Homeostatic plasticity for self-regulation
- Sparse connectivity (1% like biological brain)

**Files:** `core/neuron.py`, `core/synapse.py`, `core/plasticity.py`

---

### Tier 2: Event-Driven Orchestration (Phase 2)
Efficient, scalable simulation framework.

**Key Components:**
- Event queue for precise temporal ordering
- Spike dispatcher for fan-out routing
- Metrics collection and real-time observability
- State persistence (save/load networks)

**Files:** `core/event_queue.py`, `core/dispatcher.py`, `core/orchestrator.py`, `core/metrics.py`

---

### Tier 3: Cognitive Control (Phase 3.1-3.2)
**System 1.5** - Intelligent gatekeeper between spikes and reasoning.

**Key Components:**
- **Novelty Detector**: Spots out-of-distribution patterns
- **Confidence Estimator**: Measures output certainty
- **Conflict Monitor**: Detects competing spike patterns
- **Plasticity Gate Controller**: Learning OFF by default, ON when context demands
- **Regional Organization**: Spatial hierarchy with autonomous regions

**Philosophy:** The brain learns selectively—only when encountering novelty, achieving rewards, or experiencing prediction errors.

**Files:** `core/control_layer.py`, `core/region.py`

---

### Tier 4: Symbolic Reasoning & Meta-Learning (Phase 3.3-3.4)
**System 2** - Slow, deliberate, explicit reasoning layer.

**Key Components:**
- Knowledge base of rules and facts
- Conflict resolution via explicit reasoning
- Memory consolidation (offline learning)
- Adaptive thresholds (learning-to-learn)
- Intrinsic motivation (curiosity signals)

**Philosophy:** Explicit reasoning complements implicit learning. The system improves at learning by adapting its own parameters.

**Files:** `core/system2.py`, `core/meta_learner.py`

---

### Tier 5: Goal-Directed Behavior (Phase 4-5)
**Goals, motivation, and self-modeling.**

**Key Components:**
- Goal selection with utility + curiosity weighting
- Intrinsic reward computation (novelty + prediction error)
- Self-model for outcome prediction
- Prediction error tracking for learning
- Integration with environments (gridworld)

**Philosophy:** The system sets its own goals, generates intrinsic rewards from learning, and uses a self-model to predict consequences of actions.

**Files:** `core/system2.py`, `core/self_model.py`, `core/metrics.py`, `experiments/phase5_goal_directed_demo.py`

---

## 🚀 Quick Start

### Installation

```bash
# Clone and enter directory
git clone https://github.com/yourusername/Node_network.git
cd Node_network

# That's it! No dependencies to install.
```

### Run the Complete System

```bash
# See all 5 tiers in action: goals, learning, reasoning, and self-modeling
python -m experiments.phase5_goal_directed_demo
```

Output shows:
- 🎯 Goal selection based on utility and curiosity
- 💡 Intrinsic rewards from novelty and learning progress
- 🧠 Self-model predicting action outcomes
- 🌍 Integration with mock environment (gridworld)
- 📊 Real-time performance metrics

### Run Demos by Tier

```bash
# Tier 1-2: Core learning mechanisms
python -m experiments.pavlov_experiment        # Classical conditioning
python -m experiments.sequence_experiment      # Temporal pattern learning

# Tier 3: Learn when to learn
python -m experiments.phase3_demo              # Novelty-gated plasticity

# Tier 3.2: Hierarchical organization
python -m experiments.phase32_region_demo      # Regional brain structure

# Tier 4: Reasoning and self-improvement
python -m experiments.phase33_system2_demo     # Explicit rules and knowledge
python -m experiments.phase34_meta_learning_demo  # Learning how to learn
python -m experiments.phase4_meta_plasticity_demo # Adaptive learning rates

# Tier 5: Goals and intrinsic motivation  
python -m experiments.phase5_goal_directed_demo   # Complete integration
```

---

## 📚 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Detailed walkthrough with code examples for each tier
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Deep architectural overview and design decisions
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — API documentation by core module
- **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — How to run and create new experiments
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to extend and contribute to the system

---

## 📂 Project Structure

```
Node_network/
├── README.md                          # This file (master entry point)
├── GETTING_STARTED.md                 # Tutorial for new users
├── CONTRIBUTING.md                    # Extension guide
├── requirements.txt                   # Empty (no dependencies!)

├── core/                              # The neural engine
│   ├── neuron.py                      # Leaky integrate-and-fire
│   ├── synapse.py                     # Connections with eligibility traces
│   ├── spike.py                       # Spike events
│   ├── event_queue.py                 # Temporal ordering
│   ├── dispatcher.py                  # Spike routing
│   ├── plasticity.py                  # STDP and learning rules
│   ├── orchestrator.py                # Main simulation loop
│   ├── metrics.py                     # Observability and probes
│   ├── control_layer.py               # Novelty detection & gating
│   ├── region.py                      # Regional organization
│   ├── system2.py                     # Explicit reasoning
│   ├── meta_learner.py                # Learning-to-learn
│   ├── self_model.py                  # Outcome prediction
│   └── network.py / bionode.py        # Legacy API (backward compatible)

├── experiments/                       # Learning demonstrations
│   ├── base_experiment.py             # Base class for experiments
│   ├── pavlov_experiment.py           # Classical conditioning
│   ├── sequence_experiment.py         # Sequence learning
│   ├── phase3_demo.py                 # Novelty-gated plasticity
│   ├── phase32_region_demo.py         # Regional organization
│   ├── phase33_system2_demo.py        # Explicit reasoning
│   ├── phase34_meta_learning_demo.py  # Meta-learning
│   ├── phase4_meta_plasticity_demo.py # Adaptive learning rates
│   ├── phase5_goal_directed_demo.py   # Complete integration
│   ├── gridworld_env.py               # Mock environment
│   └── unified_learner.py             # Multi-task learning

├── tests/                             # Test suite
│   ├── test_control_layer.py
│   ├── test_region.py
│   ├── test_system2.py
│   ├── test_meta_learner.py
│   └── ...

├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System design deep dive
│   ├── API_REFERENCE.md               # Module-by-module reference
│   ├── EXPERIMENTS.md                 # Experiment guide
│   ├── FILE_STRUCTURE.md              # This structure explained
│   └── archive/                       # Historical phase reports

├── semantic/                          # Knowledge representation (optional)
│   ├── semantic_brain.py
│   ├── context_driver.py
│   └── semantic_assistant.py

├── storage/                           # Persistence (optional)
│   └── ...

└── tools/                             # Utilities (optional)
    └── ...
```

---

## 🎓 Core Concepts

### Learning Mechanism: 3-Factor STDP

The brain learns through a simple rule applied at every synapse:

```
ΔWeight = LearningRate × EligibilityTrace × Dopamine

Where:
- EligibilityTrace: Was the presynaptic neuron active recently?
- Dopamine: Did we get rewarded?
- Post-spike: Did this neuron just fire?

Result: Synapses that fire before reward get strengthened.
        No backpropagation needed!
```

### Neuron Dynamics: Leaky Integrate-and-Fire

Each neuron maintains a membrane potential that decays and integrates inputs:

```python
# Decay toward resting potential
potential *= decay_rate

# Integrate synaptic currents
potential += incoming_current

# Fire if threshold exceeded
if potential >= threshold:
    fire()
    potential = reset_value
    enter_refractory_period()
```

### Plasticity Gating: System 1.5 Control

Learning doesn't happen automatically. The system decides when to learn based on context:

```
IF novelty > threshold OR reward > threshold OR prediction_error > threshold:
    ENABLE learning
ELSE:
    DISABLE learning (preserve existing knowledge)
```

This prevents catastrophic forgetting while allowing rapid learning when needed.

---

## 📊 Capabilities Demonstrated

| Capability | Tier | Status | Details |
|------------|------|--------|---------|
| Classical conditioning | 1 | ✅ 100% | Pavlov's dog (dopamine-modulated STDP) |
| Sequence learning | 1 | ✅ 100% | Temporal pattern prediction |
| Novelty detection | 3 | ✅ 100% | Out-of-distribution recognition |
| Context-aware learning | 3 | ✅ 100% | Learning gates based on novelty |
| Explicit reasoning | 4 | ✅ 100% | Rules, facts, conflict resolution |
| Meta-learning | 4 | ✅ 100% | Adaptive thresholds, learning-to-learn |
| Goal selection | 5 | ✅ 100% | Utility + curiosity weighting |
| Self-modeling | 5 | ✅ 100% | Outcome prediction and uncertainty |
| Multi-task learning | 1-5 | ✅ 100% | No catastrophic forgetting |

---

## ⚙️ What You Can Do With This

### Learn Neuroscience
- See how biological principles (STDP, homeostasis, local learning) work in practice
- Understand spike-based computation without backpropagation
- Explore learning rules from computational neuroscience

### Develop AI Without Backprop
- Alternative to deep learning with full transparency
- Local learning rules (great for neuromorphic hardware)
- Biologically plausible—useful for cognitive science

### Teach/Research
- Perfect for computational neuroscience courses
- Base for exploring novel learning mechanisms
- Testing ground for neuromorphic principles

### Build Neuromorphic Hardware
- Pure event-driven architecture maps directly to hardware
- Works with Loihi 2, TrueNorth, and other neuromorphic chips
- Efficient (milliseconds per spike, not milliseconds per backprop pass)

---

## 🔬 Biological Inspiration

**Neuroscience principles implemented:**

- **LIF Neurons** (Lapicque, 1907): Integrate-and-fire with exponential decay
- **STDP** (Markram et al., Bi & Poo, 1998): Spike-timing-dependent plasticity
- **3-Factor Learning** (Reynolds & Wickens, 2002): Eligibility traces × neuromodulation
- **Homeostatic Plasticity** (Turrigiano & Nelson, 2004): Self-regulating synaptic scaling
- **Novelty Detection** (VTA dopamine neurons): Out-of-distribution pattern recognition
- **Hierarchical Regions** (Cortical columns): Spatial organization with local autonomy
- **System 1 / System 2** (Kahneman, 2011): Fast reactions vs. slow reasoning

---

## 🛠️ Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.test_control_layer -v

# Run with pytest (if installed)
pytest tests/ -v
```

**Test Coverage:**
- Tier 1: 11 core neuron/synapse tests
- Tier 2: 8 orchestration tests  
- Tier 3: 28 control layer + 38 region tests
- Tier 4: 31 system2 + 31 meta-learner tests
- Tier 5: 36+ goal/intrinsic/self-model tests

**Current Status:** 250+ tests, 100% passing

---

## 🚀 Next Steps

### For New Users
1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Run `phase5_goal_directed_demo.py`
3. Explore individual demos by tier
4. Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand how it works

### For Developers
1. Read [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
2. Explore `core/` modules
3. See [CONTRIBUTING.md](CONTRIBUTING.md) for extension patterns
4. Create your own experiment in `experiments/`

### For Researchers
1. Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions
2. Read phase completion reports in `docs/archive/` for detailed evolution
3. Check `tests/` for validation and examples
4. Run experiments with different parameters

---

## 🤝 Contributing

Areas where contributions help:
- Performance optimization
- New learning rules or neuron models
- Additional experiments and environments
- Documentation improvements
- Hardware integration (Loihi, TrueNorth)
- Visualization tools

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License - Free to use in your own projects!

---

## 🙏 Acknowledgments

Built on foundational work in:
- **Neuroscience**: Lapicque, Hodgkin, Huxley, Markram, Bi & Poo
- **Neuromorphic Engineering**: Carver Mead, Giacomo Indiveri
- **Learning Theory**: Sutton, Barto, Dayan
- **Cognitive Science**: Daniel Kahneman (System 1/2)

Special thanks to the SpiNNaker and BrainScaleS projects for neuromorphic engineering inspiration.

---

## 📖 Key Papers (Optional Reading)

- Lapicque, L. (1907). "Recherches quantitatives sur l'excitation électrique des nerfs"
- Markram, H., et al. (1997). "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSCs"
- Bi, G.-Q., & Poo, M.-M. (1998). "Synaptic modifications in cultured hippocampal neurons"
- Reynolds, J. N., & Wickens, J. R. (2002). "Dopamine-dependent plasticity of corticostriatal synapses"
- Turrigiano, G. G., & Nelson, S. B. (2004). "Homeostatic plasticity in the developing nervous system"

---

**Built with neuroscience, not gradients. A brain that learns, thinks, and improves itself. 🧠⚡**

**Last Updated:** January 17, 2026 (Phase 5 Complete)
