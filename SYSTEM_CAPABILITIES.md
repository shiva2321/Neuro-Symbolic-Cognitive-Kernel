# NSCK System Capabilities and Usage Guide

## Executive Summary

The NSCK (Neuro-Symbolic Cognitive Kit) is a hybrid AI system combining neural networks with symbolic reasoning, implementing cutting-edge cognitive architectures including:
- **Active Inference** (Free Energy Minimization)
- **Global Workspace Theory** (Attention & Module Competition)
- **Metacognition** (Confidence Monitoring & Executive Control)

## What Can Be Built

### 1. Adaptive Game-Playing Agents

**Supported Games:**
- **Snake**: Grid-based navigation with food collection
- **Pong**: Paddle control for ball interception
- **Maze**: Goal-directed pathfinding

**Capabilities:**
- Learn from experience through rule induction
- Transfer knowledge across games (zero-shot learning)
- Explain decisions in natural language
- Exhibit curiosity-driven exploration

### 2. Core System Components

#### Neural Processing Layer
- **TaskAwareSNN** (`snn_qat.py`): Task-aware Spiking Neural Network with ternary quantization
- **Universal Encoder** (`universal_encoder.py`): Multimodal encoding (visual, audio, text)
- **World Model** (`world_model.py`): Neural dynamics predictor for mental simulation

#### Symbolic Reasoning Layer
- **Rule Learner** (`rule_learner.py`): Inductive Logic Programming (ILP) for symbolic rule extraction
- **Causal Reasoning** (`causal_reasoning.py`): Causal graph learning with forward/backward chaining
- **STRIPS Planner** (`planner.py`): Goal-directed action planning
- **Spatial Reasoning** (`spatial_reasoning.py`): Grid-based pathfinding

#### Memory Systems
- **Episodic Memory** (`episodic_memory.py`): VSA-based long-term memory with LSH indexing
- **Staged Recall** (`staged_recall.py`): 4-tier memory access (Cache → Graph → LSH → Scan)
- **Intelligent Buffer** (`intelligent_buffer.py`): Tiered replay buffer with priority eviction

#### Executive Control
- **Metacognition** (`metacognition.py`): Confidence estimation, conflict detection, safe fallbacks
- **Global Workspace** (`global_workspace.py`): Module competition for attention
- **Agency** (`agency.py`): Active Inference implementation (Free Energy Minimization)

#### Transfer Learning
- **Analogy Engine** (`analogy.py`): Structure mapping for zero-shot transfer
- **Brain Fusion** (`brain_fusion.py`): Knowledge consolidation across tasks

#### Verification & Explainability
- **Grounding Verifier** (`grounding_verifier.py`): Reality check for symbolic predicates
- **Semantic Coherence** (`semantic_coherence.py`): Logical contradiction detection
- **Explanation Generator** (`explanation.py`): Natural language explanation generation
- **Saliency Visualizer** (`saliency.py`): Grad-CAM for neural decision visualization

### 3. User Interfaces

- **Dashboard** (`dashboard.py`): Tkinter UI for visualization and control
- **Python Server** (`python_server.py`): Central hub with ZMQ communication
- **Game UIs**: `snake_ui.py`, `pong_ui.py`, `maze_ui.py`

## How to Use the System

### Basic Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build the Rust VSA engine (optional but recommended)
cd nsck-demo/rust_vsa
cargo build --release
cd ../..
```

### Running the System

#### Option 1: Start the Python Server (Headless Training)

```bash
cd nsck-demo/python
python python_server.py

# With options:
python python_server.py --no-teacher  # Self-supervised learning only
python python_server.py --task snake  # Train on specific game
```

#### Option 2: Run with Dashboard UI

```bash
cd nsck-demo/python
python dashboard.py
```

The dashboard provides:
- Real-time visualization of cognitive state
- Process control (start/stop training)
- Experiment management
- Performance metrics

### Training Workflows

#### 1. Single Task Learning

```python
from cognitive_engine import create_cognitive_engine
from simulation import sim_snake

# Initialize engine
engine = create_cognitive_engine(persistence_path="nsck_brain.db")

# Training loop
for episode in range(1000):
    state = reset_game()
    done = False
    
    while not done:
        # Make decision
        cognitive_state = engine.decide(state, task_tag="snake")
        action = cognitive_state.chosen_action
        
        # Execute action
        next_state, reward, done = sim_snake(state, action)
        
        # Learn from experience
        engine.learn(
            state=state,
            action=action,
            reward=reward,
            task_tag="snake",
            next_state=next_state
        )
        
        state = next_state
```

#### 2. Zero-Shot Transfer Learning

```python
# Train on Snake
for episode in range(500):
    train_episode(engine, "snake")

# Test on Pong without training
state = reset_pong()
cognitive_state = engine.decide(state, task_tag="pong")

# Engine uses analogy to transfer Snake knowledge to Pong
transferred_action = engine.transfer(
    source_task="snake",
    target_task="pong",
    state=state,
    active_predicates=cognitive_state.active_predicates
)
```

#### 3. Metacognitive Reasoning

```python
# Make a decision
cognitive_state = engine.decide(state, task_tag="snake")

# Get explanation
explanation = engine.explain(query_type="action")
print(f"Action: {cognitive_state.chosen_action}")
print(f"Reason: {explanation}")

# Query counterfactual
alternative = engine.counterfactual("ACTION_LEFT")
print(f"What if I went left? {alternative}")

# Explain rejection
why = engine.why_not("ACTION_RIGHT")
print(f"Why not right? {why}")
```

#### 4. Mental Simulation (World Model)

```python
# After training world model
if engine.world_model and engine.world_model.is_ready("snake"):
    # Simulate action sequence
    imagined_reward = engine.imagine_rollout(
        initial_hv=cognitive_state.situation_hv,
        action_sequence=["ACTION_UP", "ACTION_RIGHT"],
        task_tag="snake"
    )
    
    print(f"Predicted reward: {imagined_reward}")
```

#### 5. Generative Dreaming

```python
# Generate synthetic training data from memory
dreams = engine.dream(num_samples=10, task_tag="snake")

for dream in dreams:
    # Use dream experiences to train SNN
    train_snn(
        state=dream['state'],
        action=dream['action'],
        reward=dream['reward']
    )
```

### Advanced Features

#### Causal Discovery

The system automatically builds causal graphs from experience:

```python
# Get causal graph statistics
telemetry = engine.get_causal_telemetry("snake")
print(f"Discovered {telemetry['link_count']} causal links")

# Export causal graph
graph = engine.causal_graphs["snake"]
for link in graph.all_links:
    print(f"{link.cause} → {link.effect} (strength: {link.strength:.2f})")
```

#### Curiosity-Driven Exploration

```python
# System automatically explores when:
# 1. Confidence is low (entropy > threshold)
# 2. State is novel (not in episodic memory)
# 3. Performance is stagnant

# Monitor exploration statistics
stats = engine.get_stats()
print(f"Explorations: {stats['explorations']}")
print(f"Curiosity stats: {stats['curiosity_stats']}")
```

#### Rule Induction

```python
# Rules are automatically induced every 50 episodes
# Get learned rules
rules = engine.rule_learner.get_rules("snake")

for rule in rules:
    print(f"IF {rule.condition} THEN {rule.consequence}")
    print(f"  Support: {rule.support}, Success Rate: {rule.success_rate:.2f}")
```

### Monitoring & Debugging

#### Get System Statistics

```python
stats = engine.get_stats()
print(f"Decisions: {stats['decisions']}")
print(f"Episodes: {stats['episodes_recorded']}")
print(f"Rules induced: {stats['rules_induced']}")
print(f"Rules per task: {stats['rules_per_task']}")
```

#### Export Decision Traces

```python
# Export traces for analysis
engine.export_traces("decision_traces.json")

# Traces include:
# - Timestamp
# - Task
# - Winning module (SNN/RULES/PLANNER/EXPLORATION)
# - Action taken
# - Proposals from all modules
```

#### Visualize Transfer Learning

```bash
# Generate transfer learning plots
python visualize_transfer.py

# Verify transfer statistics
python verify_transfer_stats.py
```

## What Needs to Be Done

### Immediate Requirements (Fixed)

- ✅ **Critical Fix**: Removed unused `PerceptionEngine` and `calculate_entropy` imports from `cognitive_engine.py`

### Optional Enhancements

1. **Complete PerceptionEngine Implementation** (if needed in future)
   - Currently, `perception.py` only has `CleanupMemory` and `FusionEngine`
   - If audio/visual fusion is needed, implement `PerceptionEngine` class

2. **Restore Missing Legacy Files** (if backward compatibility needed)
   - `brain.py` → replaced by `brain_fusion.py`
   - `homeostasis.py` → needs implementation for drive systems
   - `hippocampus.py` → replaced by `episodic_memory.py`
   - `symbolic_vsa.py` → replaced by `hypervec_shim.py`

3. **Testing & Validation**
   - Run full test suite: `pytest nsck-demo/tests/`
   - Validate zero-shot transfer empirically
   - Benchmark performance across games

4. **Documentation**
   - API documentation for all modules
   - Tutorial notebooks for common workflows
   - Architecture diagrams

5. **Performance Optimization**
   - Profile memory usage during long training runs
   - Optimize LSH indexing in episodic memory
   - Cache commonly used hypervectors

## Research Directions

### Potential Extensions

1. **Multi-Agent Systems**: Extend to cooperative/competitive scenarios
2. **Language Grounding**: Use `lingua_cortex.py` for natural language interaction
3. **Continual Learning**: Prevent catastrophic forgetting across task sequences
4. **Hierarchical Planning**: Multi-level goal decomposition
5. **Theory Formation**: Discover abstract principles from experience

### Publications & Evaluation

The system implements concepts from:
- Active Inference (Karl Friston)
- Global Workspace Theory (Bernard Baars)
- Metacognition (Dunlosky & Metcalfe)
- Analogical Transfer (Gentner's Structure Mapping)

Potential evaluation metrics:
- Sample efficiency (episodes to reach performance threshold)
- Transfer efficiency (zero-shot vs. from-scratch performance)
- Rule quality (precision, recall, interpretability)
- Explanation quality (human evaluation)

## Getting Help

- **Specifications**: See `docs/NSCK_SPECIFICATION.md`
- **Roadmap**: See `docs/ROADMAP.md`
- **Agent Protocols**: See `docs/AGENT_PROTOCOLS.md`
- **Issues**: Check existing issues in the repository

## License & Citation

[Add license information]
[Add citation format]
