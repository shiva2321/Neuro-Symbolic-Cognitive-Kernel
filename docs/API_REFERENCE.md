# NCGN API Reference

**Complete reference for all public APIs in the neuromorphic system**

---

## Table of Contents

1. [Core Tier 1: Neural Substrate](#tier-1-neural-substrate)
2. [Core Tier 2: Orchestration](#tier-2-orchestration)
3. [Core Tier 3: Control Layer](#tier-3-control-layer)
4. [Core Tier 4: Reasoning & Meta-Learning](#tier-4-reasoning--meta-learning)
5. [Core Tier 5: Goals & Self-Modeling](#tier-5-goals--self-modeling)
6. [Utilities](#utilities)

---

## Tier 1: Neural Substrate

### `core.neuron.Neuron`

Leaky Integrate-and-Fire neuron with homeostatic plasticity.

```python
from core.neuron import Neuron

neuron = Neuron(
    node_id: int,                    # Unique identifier
    threshold: float = 1.0,           # Firing threshold
    resting_potential: float = 0.0,   # Baseline membrane potential
    decay_rate: float = 0.95,         # Leak rate (0.95 = 5% decay per step)
    refractory_period: int = 2        # Steps to stay silent after firing
)
```

**Key Methods:**

```python
# Set external input
neuron.set_external_input(value: float) -> None

# Process one simulation step
neuron.integrate_and_fire(input_current: float, dopamine: float) -> bool

# Get/set membrane potential
potential = neuron.get_potential() -> float
neuron.set_potential(value: float) -> None

# Check if firing
is_firing = neuron.is_firing() -> bool

# Get neuron info
info = neuron.get_info() -> Dict
```

**Example:**

```python
# Create neuron
neuron = Neuron(node_id=1, threshold=1.0)

# Apply input and fire
neuron.set_external_input(0.6)
fired = neuron.integrate_and_fire(input_current=0.5, dopamine=0.0)
print(f"Fired: {fired}")  # True if potential >= threshold
```

---

### `core.synapse.Synapse`

Synaptic connection with STDP learning and eligibility traces.

```python
from core.synapse import Synapse

synapse = Synapse(
    source: int,                    # Source neuron ID
    target: int,                    # Target neuron ID  
    weight: float = 0.1,            # Initial synaptic strength
    is_inhibitory: bool = False,    # Excitatory (False) or inhibitory (True)
    eligibility_trace_tau: float = 0.1  # Eligibility trace decay rate
)
```

**Key Methods:**

```python
# Transmit spike
current = synapse.transmit() -> float

# Apply STDP learning
synapse.apply_stdp(
    presynaptic_fired: bool,
    postsynaptic_fired: bool,
    dopamine: float,
    learning_rate: float
) -> None

# Apply homeostasis
synapse.apply_homeostasis(
    postsynaptic_rate: float,
    target_rate: float
) -> None

# Get/set weight
weight = synapse.get_weight() -> float
synapse.set_weight(value: float) -> None

# Get info
info = synapse.get_info() -> Dict
```

**Example:**

```python
# Create synapse
synapse = Synapse(source=0, target=1, weight=0.1)

# Transmit spike
current = synapse.transmit()

# Apply learning
synapse.apply_stdp(
    presynaptic_fired=True,
    postsynaptic_fired=True,
    dopamine=1.0,  # Reward
    learning_rate=0.01
)
```

---

### `core.plasticity.PlasticityController`

Manages learning rules and plasticity across the network.

```python
from core.plasticity import PlasticityController

controller = PlasticityController(
    network,
    base_learning_rate: float = 0.01,
    eligibility_trace_tau: float = 0.1
)
```

**Key Methods:**

```python
# Apply STDP to a synapse
controller.apply_stdp(
    synapse_id: str,
    presynaptic_fired: bool,
    postsynaptic_fired: bool,
    dopamine: float
) -> None

# Apply homeostatic scaling
controller.apply_homeostasis(
    neuron_id: int,
    firing_rate: float,
    target_rate: float
) -> None

# Set learning rate
controller.set_learning_rate(rate: float) -> None

# Get effective learning rate
rate = controller.get_effective_learning_rate() -> float
```

---

## Tier 2: Orchestration

### `core.event_queue.EventQueue`

Manages temporally ordered spike events.

```python
from core.event_queue import EventQueue

queue = EventQueue()
```

**Key Methods:**

```python
# Add event
queue.add_event(
    timestamp: int,
    neuron_id: int,
    event_type: str  # "spike", "dopamine", etc.
) -> None

# Get next event
event = queue.get_next_event() -> Dict | None

# Check if empty
empty = queue.is_empty() -> bool

# Get event count
count = queue.get_event_count() -> int
```

---

### `core.dispatcher.Dispatcher`

Routes spikes from source neurons to targets efficiently.

```python
from core.dispatcher import Dispatcher

dispatcher = Dispatcher(network)
```

**Key Methods:**

```python
# Register spike for routing
dispatcher.route_spike(
    source_id: int,
    timestamp: int
) -> List[int]  # Target neurons

# Get fan-out count
count = dispatcher.get_fan_out(source_id: int) -> int

# Dispatch all queued spikes
targets = dispatcher.dispatch_all() -> List[int]
```

---

### `core.orchestrator.Orchestrator`

Main simulation loop coordinating all components.

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator(
    network,
    learning_rate: float = 0.01
)
```

**Key Methods:**

```python
# Simulate one step
state = orchestrator.step(
    global_dopamine: float = 0.0,
    external_inputs: Dict[int, float] | None = None
) -> SimulationState

# Simulate multiple steps
states = orchestrator.simulate(
    steps: int,
    global_dopamine: float = 0.0,
    external_inputs_sequence: List[Dict] | None = None
) -> List[SimulationState]

# Set learning enabled/disabled
orchestrator.set_learning_enabled(enabled: bool) -> None

# Get metrics
metrics = orchestrator.get_metrics() -> MetricsCollector
```

---

### `core.metrics.MetricsCollector`

Observability and performance tracking.

```python
from core.metrics import MetricsCollector

metrics = MetricsCollector(network)
```

**Key Methods:**

```python
# Attach probes to neurons
metrics.attach_spike_probe(neuron_id: int) -> SpikeProbe
metrics.attach_weight_probe(synapse_id: str) -> WeightProbe
metrics.attach_state_probe(neuron_id: int) -> StateProbe

# Collect current snapshot
snapshot = metrics.collect_snapshot() -> ProbeSnapshot

# Get history
history = metrics.get_history(probe_id: str) -> List[float]

# Clear history
metrics.clear_history() -> None
```

---

## Tier 3: Control Layer

### `core.control_layer.ControlLayer`

System 1.5 monitoring and plasticity gating.

```python
from core.control_layer import ControlLayer

control = ControlLayer(
    network,
    novelty_threshold: float = 0.5,
    confidence_threshold: float = 0.7,
    conflict_threshold: float = 0.4
)
```

**Key Methods:**

```python
# Detect novelty in current state
novelty = control.detect_novelty(
    state: Dict[int, float]
) -> float  # [0, 1]

# Estimate output confidence
confidence = control.estimate_confidence(
    outputs: Dict[int, float]
) -> float  # [0, 1]

# Detect competing patterns
conflict = control.detect_conflict(
    state: Dict[int, float]
) -> float  # [0, 1]

# Determine if learning should be active
should_learn = control.should_learn(
    novelty: float,
    confidence: float,
    reward: float
) -> bool
```

**Example:**

```python
control = ControlLayer(network)

# Get current state
state = {1: 0.8, 2: 0.3, 3: 0.1}

# Check learning gate
novelty = control.detect_novelty(state)
confidence = control.estimate_confidence(state)
reward = 1.0  # Got positive feedback

should_learn = control.should_learn(novelty, confidence, reward)
print(f"Learning enabled: {should_learn}")  # True if novel or rewarded
```

---

### `core.region.Region`

Spatial brain organization with regional autonomy.

```python
from core.region import Region

region = Region(
    region_id: str,
    node_ids: List[int],
    parent_region: Region | None = None
)
```

**Key Methods:**

```python
# Set learning rule for region
region.set_learning_rule(
    rule_id: str  # "stdp", "hebbian", "oja", etc.
) -> None

# Set region-specific learning rate
region.set_learning_rate_multiplier(
    multiplier: float
) -> None

# Apply learning to a connection
region.apply_learning(
    target_id: int,
    postsynaptic_fired: bool,
    dopamine: float,
    learning_rate: float
) -> None

# Get region info
info = region.get_info() -> Dict

# Add sub-region
subregion = region.create_subregion(
    region_id: str,
    node_ids: List[int]
) -> Region
```

---

## Tier 4: Reasoning & Meta-Learning

### `core.system2.System2`

Explicit reasoning layer with knowledge base.

```python
from core.system2 import System2, Rule, RuleType

system2 = System2()
```

**Key Methods:**

```python
# Add rule
rule = system2.add_rule(
    rule_type: RuleType,      # IF_THEN, ASSOCIATION, CAUSAL, etc.
    condition: str,            # "novelty > 0.7"
    action: str,               # "enable_learning"
    confidence: float = 0.8
) -> Rule

# Query knowledge base
matching_rules = system2.query_rules(
    condition_substring: str
) -> List[Rule]

# Consolidate memory (offline learning)
system2.consolidate_memory(
    min_firing_rate: float = 0.1,
    consolidation_steps: int = 100
) -> Dict

# Get system state
state = system2.get_state() -> Dict

# Resolve conflict
winner = system2.resolve_conflict(
    rule1: Rule,
    rule2: Rule
) -> Rule
```

**Example:**

```python
system2 = System2()

# Add rules
system2.add_rule(
    rule_type=RuleType.IF_THEN,
    condition="novelty > 0.7",
    action="enable_learning",
    confidence=0.9
)

# Query
rules = system2.query_rules("novelty")
print(f"Found {len(rules)} rules about novelty")

# Consolidate
system2.consolidate_memory(steps=50)
```

---

### `core.meta_learner.MetaLearner`

Learning-to-learn through adaptive thresholds.

```python
from core.meta_learner import MetaLearner, AdaptationSignal

meta = MetaLearner()
```

**Key Methods:**

```python
# Adapt a threshold based on outcome
meta.adapt_threshold(
    threshold_name: str,  # "novelty", "reward", "error"
    signal: AdaptationSignal  # SUCCESS, FAILURE, BREAKTHROUGH, etc.
) -> float  # New threshold value

# Get current threshold
value = meta.get_threshold(threshold_name: str) -> float

# Set intrinsic motivation strength
meta.set_intrinsic_motivation_weight(
    weight: float  # [0, 1]
) -> None

# Compute intrinsic reward
reward = meta.compute_intrinsic_reward(
    novelty: float,
    prediction_error: float,
    learning_progress: float
) -> float  # [0, 1]

# Get adaptive learning rate
rate = meta.get_adaptive_learning_rate(
    step: int,
    performance_delta: float,
    stability: float
) -> float
```

---

## Tier 5: Goals & Self-Modeling

### `core.self_model.SelfModel`

Outcome prediction for planning and self-understanding.

```python
from core.self_model import SelfModel

self_model = SelfModel(
    input_dim: int = 10,
    hidden_dim: int = 16,
    num_actions: int = 4
)
```

**Key Methods:**

```python
# Predict outcome of action
predicted_state, uncertainty = self_model.predict_outcome(
    state: List[float],
    action: int
) -> Tuple[List[float], float]
# uncertainty: [0, 1] (0 = very sure, 1 = very uncertain)

# Update model from actual outcome
self_model.update(
    state: List[float],
    action: int,
    actual_next_state: List[float]
) -> float  # Prediction error

# Get model confidence
confidence = self_model.get_model_confidence() -> float

# Get model info
info = self_model.get_info() -> Dict
```

**Example:**

```python
self_model = SelfModel()

# Predict action outcome
state = [1.0, 0.5, 0.2, ...]
action = 2  # Move right
predicted_next, uncertainty = self_model.predict_outcome(state, action)

# Get actual outcome
actual_next = [1.0, 1.0, 0.2, ...]  # Moved right

# Update model
error = self_model.update(state, action, actual_next)
print(f"Prediction error: {error:.3f}")
print(f"Model confidence: {self_model.get_model_confidence():.3f}")
```

---

### `core.system2.Goal` & Goal Selection

Goal representation and selection.

```python
from core.system2 import System2, Goal

system2 = System2()

# Create goals
goal1 = Goal(
    goal_id=1,
    description="Reach target",
    utility=0.8  # How valuable
)

goal2 = Goal(
    goal_id=2,
    description="Explore",
    utility=0.3
)

# Set goals
system2.set_goals([goal1, goal2])

# Select goal
selected_goal = system2.select_goal(
    curiosity_weight: float = 0.5  # How much to explore vs. exploit
) -> Goal

# Update goal progress
system2.evaluate_goal_progress(
    goal_id: int,
    new_state: Dict
) -> float  # Progress [0, 1]

# Mark goal as complete
system2.mark_goal_failed(goal_id: int) -> None
```

---

## Utilities

### `experiments.base_experiment.ExperimentBase`

Base class for creating learning tasks.

```python
from experiments.base_experiment import ExperimentBase

class MyExperiment(ExperimentBase):
    def build_network(self):
        # Create neurons and synapses
        pass
    
    def create_stimuli(self):
        # Define training inputs
        pass
    
    def run_training(self, duration):
        # Execute learning
        pass
```

**Key Methods:**

```python
# Execute full experiment
results = experiment.execute(
    train_duration: int = 100,
    verbose: bool = True
) -> Dict

# Get network
network = experiment.get_network() -> NeuromorphicNetwork

# Record result
experiment.record_result(
    metric_name: str,
    value: float
) -> None
```

---

### `experiments.gridworld_env.GridWorldEnvironment`

Mock environment for testing goal-directed behavior.

```python
from experiments.gridworld_env import GridWorldEnvironment

env = GridWorldEnvironment(
    width: int = 10,
    height: int = 10,
    num_goals: int = 3,
    num_obstacles: int = 5
)
```

**Key Methods:**

```python
# Get current state
state = env.get_state() -> List[float]

# Take action
reward, next_state, done = env.step(
    action: int  # 0=up, 1=down, 2=left, 3=right
) -> Tuple[float, List[float], bool]

# Reset environment
initial_state = env.reset() -> List[float]

# Get goal positions
goals = env.get_goals() -> List[Tuple[int, int]]

# Render
env.render() -> str  # ASCII visualization
```

---

## Common Patterns

### Creating a Simple Network

```python
from core.network import NeuromorphicNetwork

# Create network
net = NeuromorphicNetwork()

# Add neurons
net.add_node(0, "input", threshold=1.0)
net.add_node(1, "hidden", threshold=1.5)
net.add_node(2, "output", threshold=1.0)

# Connect
net.connect(0, 1, weight=0.5)
net.connect(1, 2, weight=0.3)
```

### Running a Learning Loop

```python
orchestrator = Orchestrator(net)

for epoch in range(100):
    # Present stimulus
    net.set_input(0, value=1.0)
    
    # Determine reward
    reward = 1.0 if net.nodes[2].is_firing else 0.0
    
    # Step network
    state = orchestrator.step(global_dopamine=reward)
    
    # Check metrics
    metrics = orchestrator.get_metrics()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {metrics}")
```

### Using Control Layer

```python
control = ControlLayer(net)

# Get current state as dict
state_dict = {i: node.get_potential() for i, node in net.nodes.items()}

# Detect novelty
novelty = control.detect_novelty(state_dict)

# Check learning gate
should_learn = control.should_learn(
    novelty=novelty,
    confidence=0.7,
    reward=1.0
)

# Only apply learning if gate is open
if should_learn:
    orchestrator.step(global_dopamine=1.0)
```

---

## Reference Quick Links

| Component | File | Purpose |
|-----------|------|---------|
| Neurons | `core/neuron.py` | Spike generation |
| Synapses | `core/synapse.py` | Weight update |
| Plasticity | `core/plasticity.py` | Learning rules |
| Events | `core/event_queue.py` | Temporal ordering |
| Dispatch | `core/dispatcher.py` | Spike routing |
| Orchestration | `core/orchestrator.py` | Main loop |
| Metrics | `core/metrics.py` | Observability |
| Control | `core/control_layer.py` | Plasticity gating |
| Regions | `core/region.py` | Spatial hierarchy |
| Reasoning | `core/system2.py` | Symbolic knowledge |
| Meta-Learning | `core/meta_learner.py` | Learning-to-learn |
| Self-Model | `core/self_model.py` | Prediction |

---

**Last Updated:** January 17, 2026 (Phase 5 Complete)

For complete examples, see the `experiments/` directory.
