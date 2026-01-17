# Getting Started with NCGN

**A hands-on guide to understanding and using the neuromorphic brain architecture**

---

## Table of Contents

1. [Installation](#installation)
2. [Understanding the 5 Tiers](#understanding-the-5-tiers)
3. [Running Your First Demo](#running-your-first-demo)
4. [Exploring Each Tier](#exploring-each-tier)
5. [Running Tests](#running-tests)
6. [Creating Your Own Experiment](#creating-your-own-experiment)

---

## Installation

No special setup needed! NCGN uses pure Python with zero external dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/Node_network.git
cd Node_network

# Verify Python version (3.11+ required)
python --version

# That's it! No pip install needed.
```

---

## Understanding the 5 Tiers

NCGN implements a complete brain architecture organized in 5 cognitive tiers:

### Tier 1: The Fast Reactor (Spikes)
**Files:** `core/neuron.py`, `core/synapse.py`, `core/plasticity.py`

What it does:
- Neurons fire spikes when they're excited
- Spikes travel along synapses to other neurons
- Synapses strengthen/weaken based on dopamine (reward)
- All happens in milliseconds

**Key concepts:**
- **Neuron**: Integrates inputs, fires if threshold exceeded
- **Synapse**: Transmits spikes, has weight that changes with learning
- **STDP**: Spike-Timing-Dependent Plasticity (Hebbian learning with reward modulation)
- **Homeostasis**: Neurons self-regulate to maintain target firing rates

**Example:**
```python
from core.neuron import Neuron
from core.synapse import Synapse

# Create a neuron
neuron = Neuron(node_id=1, threshold=1.0)

# Create a synapse connecting another neuron to it
synapse = Synapse(source=0, target=1, weight=0.5)

# Fire the presynaptic neuron
synapse.fire(dopamine=1.0, learning_rate=0.01)
```

### Tier 2: The Orchestrator (Event-Driven Simulation)
**Files:** `core/event_queue.py`, `core/dispatcher.py`, `core/orchestrator.py`, `core/metrics.py`

What it does:
- Manages simulation timing (when things happen)
- Routes spikes efficiently (fan-out from one neuron to many)
- Collects metrics (what happened, how strong)
- Saves/loads network state

**Key concepts:**
- **Event Queue**: Precise temporal ordering
- **Dispatcher**: Routes spikes to target neurons
- **Metrics**: Real-time observability

**Example:**
```python
from core.orchestrator import Orchestrator

# Create orchestrator
orchestrator = Orchestrator(network)

# Run simulation for 100 time steps
orchestrator.simulate(steps=100, dopamine=0.5)

# Collect metrics
metrics = orchestrator.metrics
```

### Tier 3: The Gatekeeper (System 1.5)
**Files:** `core/control_layer.py`, `core/region.py`

What it does:
- Detects novelty (is this input new?)
- Estimates confidence (am I sure about this output?)
- Monitors conflict (are there competing interpretations?)
- **Gates learning** (only learns when it should)

**Key concepts:**
- **Learning is OFF by default**: Prevents catastrophic forgetting
- **Novelty opens the gate**: "This is new, I should learn it"
- **Reward opens the gate**: "I got rewarded, strengthen those synapses"
- **Regions**: Spatial organization with local autonomy

**Example:**
```python
from core.control_layer import ControlLayer

control = ControlLayer(network=net)

# Provide context
control.detect_novelty(network_state)
control.estimate_confidence(outputs)
control.should_learn()  # Returns True/False

# Regions: spatial organization
region = control.create_region(node_ids=[1, 2, 3])
region.set_learning_rule("stdp")
```

### Tier 4: The Thinker (System 2 + Meta-Learning)
**Files:** `core/system2.py`, `core/meta_learner.py`

What it does:
- Explicit reasoning with rules and facts
- Consolidates memory (offline learning)
- Adapts learning parameters (meta-learning)
- Generates intrinsic motivation

**Key concepts:**
- **Rules**: IF condition THEN action
- **Facts**: Explicit knowledge about the world
- **Consolidation**: Sleep-like processing to solidify learning
- **Meta-Learning**: Learning HOW to learn better

**Example:**
```python
from core.system2 import System2
from core.meta_learner import MetaLearner

# Explicit reasoning
system2 = System2()
system2.add_rule("IF novelty > 0.7 THEN learn")
system2.add_fact("SPIKE 5 CAUSED REWARD")

# Meta-learning
meta = MetaLearner()
meta.adapt_threshold("novelty", success=True)  # Lower threshold
meta.adapt_threshold("novelty", failure=True)   # Raise threshold
```

### Tier 5: The Self-Aware Brain (Goals + Self-Model)
**Files:** `core/system2.py`, `core/self_model.py`, `experiments/phase5_goal_directed_demo.py`

What it does:
- **Sets and selects goals** (what should I do?)
- **Generates intrinsic rewards** (learns because it's curious)
- **Predicts outcomes** (what will happen if I do this?)
- **Tracks prediction errors** (learns from being wrong)

**Key concepts:**
- **Goals**: Multiple objectives with utilities
- **Intrinsic motivation**: Reward from novelty and learning
- **Self-model**: "I predict action X leads to state Y"
- **Prediction error**: "I was wrong, adjust model"

**Example:**
```python
from core.system2 import System2, Goal

system2 = System2()

# Set goals
goal1 = Goal(goal_id=1, description="Reach target", utility=0.8)
goal2 = Goal(goal_id=2, description="Explore", utility=0.3)
system2.set_goals([goal1, goal2])

# Select goal (utility + curiosity)
selected = system2.select_goal(curiosity_weight=0.5)

# Self-model
from core.self_model import SelfModel
self_model = SelfModel()
prediction, uncertainty = self_model.predict_outcome(state=[1, 0, 1], action=2)
```

---

## Running Your First Demo

### The Complete System (All 5 Tiers)

```bash
# See everything in action: goals, learning, reasoning, self-modeling
python -m experiments.phase5_goal_directed_demo
```

**What you'll see:**
- The system selects goals based on utility and curiosity
- As it learns, intrinsic rewards increase
- Self-model makes predictions about outcomes
- System updates based on prediction errors
- Real-time metrics showing learning progress

Expected output:
```
Episode 1:
  Goal: Reach target (utility=0.8)
  Intrinsic reward: 0.45 (novelty)
  Prediction error: 0.23
  Learning enabled: True

Episode 2:
  Goal: Explore (utility=0.3)
  Intrinsic reward: 0.52 (learning progress)
  Prediction error: 0.15
  Learning enabled: True
```

---

## Exploring Each Tier

### Tier 1-2: Classical Conditioning (2 seconds)

```bash
python -m experiments.pavlov_experiment
```

Learn how Pavlov's Dog works:
1. **Phase 1**: Bell rings, dog doesn't salivate (baseline)
2. **Phase 2**: Bell + Food for 15 trials, system learns association
3. **Phase 3**: Bell alone triggers salivation (conditioned response)

Demonstrates:
- Spike propagation (spikes travel)
- STDP learning (dopamine from reward strengthens synapses)
- Network learning (weights increase from 0.1 → 0.3)

---

### Tier 1-2: Sequence Learning (30 seconds)

```bash
python -m experiments.sequence_experiment
```

Learn temporal patterns:
- Network learns A→B→C→A sequence
- Predicts next element based on current
- 100% success rate

Demonstrates:
- Temporal prediction (memory of recent spikes)
- Pattern completion (given A, predict B)

---

### Tier 3: Novelty-Gated Learning (10 seconds)

```bash
python -m experiments.phase3_demo
```

See System 1.5 in action:
1. **Familiar patterns**: Learning GATE is CLOSED (don't retrain on known stuff)
2. **Novel patterns**: Learning GATE OPENS (learn this new thing!)
3. **Reward signals**: Gate opens (something good happened, remember it!)

Demonstrates:
- Novelty detection (out-of-distribution recognition)
- Context-aware learning (only learn when appropriate)
- No catastrophic forgetting (old knowledge preserved)

---

### Tier 3.2: Hierarchical Regions (15 seconds)

```bash
python -m experiments.phase32_region_demo
```

See spatial brain organization:
- Brain divided into regions (like cortical columns)
- Each region can learn independently
- Regions coordinate through connections

Demonstrates:
- Regional autonomy (each region has own learning rules)
- Hierarchical structure (regions within regions)

---

### Tier 4: Explicit Reasoning (20 seconds)

```bash
python -m experiments.phase33_system2_demo
```

See explicit symbol manipulation:
1. Add facts: "SPIKE 5 CAUSES REWARD"
2. Add rules: "IF CONFLICT THEN CONSOLIDATE"
3. Reason: Resolve contradictions, consolidate memory

Demonstrates:
- Symbolic reasoning (rules and facts)
- Conflict resolution (what to do when signals disagree)
- Memory consolidation (offline learning during "sleep")

---

### Tier 4: Meta-Learning (20 seconds)

```bash
python -m experiments.phase34_meta_learning_demo
```

See learning-to-learn:
- System adapts its thresholds based on outcomes
- Successful strategies lower thresholds (learn more)
- Failed strategies raise thresholds (learn less)
- Intrinsic motivation emerges (system wants to learn)

Demonstrates:
- Adaptive thresholds (learning how to learn)
- Intrinsic motivation (curiosity signals)
- Self-improvement (system gets better at learning)

---

### Tier 4: Adaptive Learning Rates (15 seconds)

```bash
python -m experiments.phase4_meta_plasticity_demo
```

See learning rates adapt:
- System speeds up learning when making progress
- Slows down learning when destabilized
- Different regions learn at different rates

Demonstrates:
- Meta-plasticity (learning rate adapts)
- Regional differentiation (hippocampus fast, cortex slow)

---

## Running Tests

### All Tests

```bash
# Run everything
python -m unittest discover -s tests -p "test_*.py" -v
```

### Specific Tier Tests

```bash
# Tier 3 tests (control layer)
python -m unittest tests.test_control_layer -v

# Tier 3.2 tests (regions)
python -m unittest tests.test_region -v

# Tier 4 tests (system2)
python -m unittest tests.test_system2 -v

# Tier 4 tests (meta-learning)
python -m unittest tests.test_meta_learner -v
```

**Expected Output:**
```
test_novelty_detection ... ok
test_confidence_estimation ... ok
test_learning_gate ... ok
test_region_creation ... ok
...

Ran 250 tests in 5.234s

OK (250 tests)
```

---

## Creating Your Own Experiment

### Step 1: Understand the Base Class

```python
from experiments.base_experiment import ExperimentBase

class MyExperiment(ExperimentBase):
    """Your learning task here."""
    
    def __init__(self, name="my_experiment"):
        super().__init__(name)
    
    def build_network(self):
        """Create neurons and connections."""
        # Add neurons for your task
        self.network.add_node(0, "input")
        self.network.add_node(1, "output")
        self.network.connect(0, 1, weight=0.1)
    
    def create_stimuli(self):
        """Define training inputs."""
        # Define inputs and expected outputs
        self.stimuli = [
            (1.0, 1.0),  # input, expected_output
            (0.0, 0.0),
        ]
    
    def run_training(self, duration=100):
        """Execute learning phase."""
        for epoch in range(duration):
            for stimulus, expected in self.stimuli:
                # Present stimulus
                self.network.set_input(0, stimulus)
                
                # Determine reward
                output_fired = self.network.nodes[1].is_firing
                reward = 1.0 if output_fired == expected else 0.0
                
                # Step the network
                self.network.step(global_dopamine=reward, learning_rate=0.01)
```

### Step 2: Run Your Experiment

```python
# Create instance
exp = MyExperiment()

# Build network
exp.build_network()

# Run training
exp.run_training(duration=100)

# Check results
print(f"Learning complete: {exp.name}")
```

### Step 3: Add to experiments/ directory

```bash
# Save as experiments/my_experiment.py
# Add imports to experiments/__init__.py
# Run with: python -m experiments.my_experiment
```

---

## Key Files You'll Need to Know

| File | Purpose | Learn To |
|------|---------|----------|
| `core/neuron.py` | LIF neuron implementation | Understand spike generation |
| `core/synapse.py` | Synaptic transmission | See how weights change |
| `core/plasticity.py` | STDP learning rules | Understand dopamine modulation |
| `core/control_layer.py` | Novelty detection & gating | See when learning happens |
| `core/region.py` | Regional organization | Understand hierarchical structure |
| `core/system2.py` | Explicit reasoning | See symbolic knowledge |
| `core/meta_learner.py` | Learning-to-learn | Watch thresholds adapt |
| `core/self_model.py` | Outcome prediction | See planning capability |
| `experiments/base_experiment.py` | Experiment template | Create your own tasks |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'core'"
Make sure you're running from the `Node_network` directory:
```bash
cd Node_network
python -m experiments.phase3_demo
```

### "ValueError: Node not found"
Check that you created nodes before connecting them:
```python
net.add_node(0)  # Create first
net.connect(0, 1)  # Connect second
```

### Experiment runs but shows no output
Some demos require visualization. Check that you're not suppressing output:
```bash
python -m experiments.phase3_demo  # Use -m to run as module
```

### All tests fail
Make sure Python 3.11+ is being used:
```bash
python --version  # Should be 3.11+
```

---

## Next Steps

1. **Understand the architecture**: Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. **Learn the API**: Check [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
3. **Run all demos**: Execute each tier's demo
4. **Run tests**: Verify everything works in your environment
5. **Create an experiment**: Implement your own task
6. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## FAQ

**Q: Do I need special hardware?**
A: No! NCGN runs on any Python 3.11+ system. Neuromorphic hardware (Loihi, TrueNorth) can run NCGN, but it's not required.

**Q: How is this different from deep learning?**
A: No gradients, no backprop, no matrix multiplication. Instead: local learning rules, sparse connectivity, biological plausibility.

**Q: Can this replace neural networks?**
A: For different use cases. Great for interpretability, hardware efficiency, continuous learning. Less efficient for large-scale supervised learning.

**Q: How do I extend this?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md). You can add new experiments, new neuron models, new learning rules, etc.

**Q: Why 5 tiers?**
A: Each tier solves a different problem:
1. Spike reactions (raw computation)
2. Orchestration (efficiency)
3. Control (when to learn)
4. Reasoning (symbolic knowledge)
5. Goals (self-awareness)

---

## Resources

- **Papers**: See [README.md](README.md) references
- **Architecture Deep Dive**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **API Documentation**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Experiment Guide**: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

---

**Happy learning! 🧠⚡**

Start with `python -m experiments.phase5_goal_directed_demo` and explore from there.

**Last Updated:** January 17, 2026
