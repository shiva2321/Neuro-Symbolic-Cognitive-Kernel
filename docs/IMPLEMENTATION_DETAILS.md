# NSCK IMPLEMENTATION DETAILS
## Comprehensive Technical Reference

**Date**: 2025-02-11  
**Scope**: Implementation-level documentation for nsck-demo/python modules  
**Purpose**: Developer reference with class signatures, methods, algorithms, data structures

This document provides detailed technical information extracted from the codebase, including:
- Complete class signatures and method parameters
- Data structures and their relationships
- Algorithm implementations and pseudocode
- Integration patterns between modules
- Usage examples with actual code

Companion to:
- [COMPLETE_MODULE_ANALYSIS.md](COMPLETE_MODULE_ANALYSIS.md) - Module overview and status
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and workflows
- [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md) - Mathematical foundations

---

## QUICK START

For immediate reference, key implementation details are available in:

1. **Module signatures and methods** - See the "Implementation Reference" section in [COMPLETE_MODULE_ANALYSIS.md](COMPLETE_MODULE_ANALYSIS.md)
2. **System workflows** - See detailed decision flow in [ARCHITECTURE.md](ARCHITECTURE.md) 
3. **API examples** - See code examples in [README.md](../README.md)
4. **Mathematical formulas** - See [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md)

---

## MODULE REFERENCE

### Core Cognitive Modules

**cognitive_engine.py**
- **Class:** `CognitiveEngine` - Central orchestrator integrating 30+ subsystems
- **Key Methods:**
  - `decide(state, task_tag, metacognition_result)` → `CognitiveState`
    - Multi-source proposal generation (SNN, RULES, PLANNER, etc.)
    - Global Workspace competition with mental rehearsal
    - Returns action, confidence, emotion, trace
  - `learn(state, action, reward, task_tag, outcome, next_state)` → `None`
    - Updates memory, rules, emotions, causal models
  - `transfer(source_task, target_task, analogy_type)` → `Dict`
    - Cross-domain knowledge transfer
- **Integration:** Connects rule_learner, episodic_memory, global_workspace, emotion_system, theory_of_mind, self_model, world_model, planner, causal_reasoners

**global_workspace.py**
- **Class:** `GlobalWorkspace` - GWT decision arbitration with mental rehearsal
- **Key Methods:**
  - `compete(proposals: List[Coalition])` → `Optional[Coalition]`
    - Standard LIDA competition
  - `compete_with_rehearsal(proposals, state_hv, world_model, get_action_hv_fn, n_cycles)` → `Optional[Coalition]`
    - Phase 8: Mental simulation with danger veto
    - Simulates outcomes, vetoes dangerous actions
  - `register_danger(danger_hv: HyperVector)` → `None`
    - Adds danger vector to veto registry
- **Algorithm:** Activation = salience + relevance + affect + (confidence * 0.5); Veto if similarity ≥ 0.75

**emotion_system.py**
- **Class:** `EmotionSystem` - Plutchik (8 emotions) + Russell (valence/arousal)
- **Key Methods:**
  - `update_from_drives(drives: Dict[str, float], reward: float)` → `None`
    - Maps homeostatic drives to emotions
    - Valence decay: 0.95, Arousal EMA: 0.3 blend rate
  - `get_mood(window: int = 10)` → `Dict`
    - Returns avg_valence, avg_arousal, dominant_emotion, stability
- **Formulas:** intensity = sqrt(valence² + arousal²); emotion_blend via inverse distance softmax

**theory_of_mind.py**
- **Class:** `TheoryOfMind` - Perspective-taking and false-belief detection
- **Key Methods:**
  - `detect_false_belief(agent_id, reality)` → `List[str]`
    - Sally-Anne Test core logic
  - `predict_action(agent_id)` → `str`
    - Predicts behavior from beliefs + desires
- **Data:** `agent_models: Dict[str, MentalStateModel]` with beliefs, desires, intentions

### Memory Systems

**episodic_memory.py**
- **Class:** `EpisodicMemory` - VSA + LSH experience storage
- **Key Methods:**
  - `record(episode: LiveEpisode)` → `None`
  - `recall_similar(query_hv, task_tag, k=5)` → `List[LiveEpisode]`
    - LSH-based similarity search with 1-bit expansion
  - `sample(task_tag, n=32)` → `List[LiveEpisode]`
- **Architecture:** Hot (RAM deque, 1000/task) + Cold (SQLite) + LSH index (64-bit hashes)

**semantic_memory.py**
- **Class:** `SemanticMemory` - NetworkX concept graph with spreading activation
- **Key Methods:**
  - `add_concept(name, properties, hv_override)` → `None`
    - Role-filler binding: prop_hv XOR value_hv, then bundle
  - `spread_activation(start_concepts, steps=3, decay=0.7)` → `Dict[str, float]`
    - Associative retrieval
  - `query(query_hv, k=5)` → `List[Tuple[str, float]]`

### Learning Systems

**rule_learner.py**
- **Class:** `RuleLearner` - Frequency-based ILP
- **Key Methods:**
  - `observe(state, action, reward, task_tag, outcome)` → `None`
  - `induce_rules(task_tag)` → `List[Rule]`
    - Threshold: support ≥ 5, success_rate ≥ 0.7
  - `get_applicable_rules(active_preds, task_tag)` → `List[Tuple[Rule, float]]`
    - Score = success_rate * (0.5 + 0.5 * specificity)
- **Constants:** min_support=5, min_success_rate=0.7, tenure_threshold=1000

**causal_reasoning.py**
- **Class:** `CausalReasoner` - Causal graph + counterfactuals
- **Key Methods:**
  - `counterfactual(actual, alternative, state, task_tag)` → `CounterfactualResult`
    - Risk/benefit scoring from causal chains
  - `forward_chain(start, max_depth=5)` → `List[CausalChain]`
  - `backward_chain(end, max_depth=5)` → `List[CausalChain]`
- **Formula:** Delta-P = P(E|C) - P(E|¬C)

**analogy.py**
- **Class:** `AnalogyEngine` - Structural domain mapping
- **Key Methods:**
  - `find_analogy(source_domain, target_domain)` → `Analogy`
    - Similarity = matched_mappings / total_source_concepts
  - `transfer_rule(condition, action, source, target)` → `Tuple[Set, str]`
  - `zero_shot_action(state, known_domain, new_domain, rules, preds)` → `Optional[str]`
- **Abstractions:** AGENT (SNAKE_HEAD, PLAYER_PADDLE), TARGET (FOOD, BALL), DANGER (BODY, MISS, WALL)

### Planning & Simulation

**planner.py**
- **Class:** `STRIPSPlanner` - BFS forward chaining
- **Key Methods:**
  - `plan(initial_state, goal, max_depth=10)` → `Optional[List[str]]`
    - Uses heapq, goal check: goal.issubset(state)
  - `plan_hierarchical(initial, goal, max_depth=50)` → `Optional[List[str]]`
- **Data:** State = FrozenSet[str] predicates; Actions = [UP, DOWN, LEFT, RIGHT]

**world_model.py**
- **Class:** `WorldModel` - Dynamics learning with bottleneck projection
- **Key Methods:**
  - `imagine(state, action_hv)` → `Tuple[np.ndarray, float]`
    - Returns predicted state + reward
  - `sample_hypothetical_trajectories(initial_hv, action_hvs, horizon=5, num_paths=10)`
- **Architecture:** 10240 → 128 (sparse projection) → MLP(256→64→64) → 128+1
- **Loss:** ||pred_state - target||² + 10.0 * ||pred_reward - target_reward||²

### Self-Awareness

**self_model.py**
- **Class:** `SelfModel` - Performance tracking + calibration
- **Key Methods:**
  - `predict_success(task_tag, state=None)` → `float`
    - Context-aware confidence (0-1)
    - Cold-start: 0.5 if attempts < 10
    - Blends base_rate, context_rate, recent_rate
  - `get_calibration_error(task_tag)` → `float`
    - MAE between predicted/actual
- **Data:** task_stats, context_stats, recent_window (size=20), identity_hv

**metacognition.py**
- **Class:** `MetacognitiveEngine` - Safety + escalation
- **Key Methods:**
  - `compute_confidence(top_k)` → `Tuple[float, str]`
    - EMA tracking of spread + margin
  - `compute_severity(conf, margin, action_dist, is_critical)` → `float`
    - Formula: 0.5*(1-conf) + 0.3*(1-margin) + 0.2*dist + (0.3 if critical)
- **Policy:** ALLOW (high conf) | FALLBACK (low conf) | BLOCK (severe conflict)

### VSA & Perception

**hypervec_py.py**
- **Class:** `HyperVectorPy` - 10,240-bit VSA operations
- **Key Methods:**
  - `xor(other)` → `HyperVector` - Binding
  - `bundle(other)` → `HyperVector` - Superposition (majority rule)
  - `similarity(other)` → `float` - Hamming distance normalized
  - `permute(shift)` → `HyperVector` - Circular rotation
- **Constants:** DIMENSION = 10,240

**universal_input.py**
- **Class:** `UniversalInput` - Heterogeneous data grounding
- **Key Methods:**
  - `ground(data, domain, min_val=0.0, max_val=1.0)` → `HyperVector`
    - Auto-detects type, recursively grounds nested structures
  - `ground_scalar(value, min, max, domain)` - Thermometer encoding (100 bins)
  - `ground_category(label, domain)` - Deterministic codebook (LRU, max 10K)
  - `ground_dict(data, domain)` - Role-filler binding
  - `ground_sequence(data, domain)` - Positional encoding
- **Constants:** n_bins=100, max_codebook=10000

---

## USAGE EXAMPLES

### Complete Cognitive Cycle

```python
from cognitive_engine import CognitiveEngine
from config import NSCKConfig

# Initialize
config = NSCKConfig()
engine = CognitiveEngine(config, persistence_path="nsck_brain.db")

# Decision
state = {"head": (5, 5), "food": (8, 3), "body": [(5,5), (5,4)]}
cognitive_state = engine.decide(state=state, task_tag="snake")

print(f"Action: {cognitive_state.chosen_action}")
print(f"Confidence: {cognitive_state.confidence}")
print(f"Emotion: {cognitive_state.emotion}")
print(f"Winner: {cognitive_state.trace['winner']}")

# Learning
engine.learn(
    state=state,
    action="ACTION_RIGHT",
    reward=1.0,
    task_tag="snake",
    outcome="success",
    next_state={"head": (6, 5), "food": (8, 3)}
)
```

### Memory Retrieval

```python
from episodic_memory import EpisodicMemory, LiveEpisode
from hypervec_py import HyperVectorPy
import time

memory = EpisodicMemory(recent_capacity=1000)

# Record
episode = LiveEpisode(
    timestamp=time.time(),
    task_tag="snake",
    situation_hv=HyperVectorPy(),
    state={"head": (5,5)},
    action="ACTION_UP",
    outcome="success",
    reward=1.0,
    emotion="joy"
)
memory.record(episode)

# Recall similar
query_hv = HyperVectorPy()
similar = memory.recall_similar(query_hv, "snake", k=5)
print(f"Found {len(similar)} similar episodes")
```

### Transfer Learning

```python
from analogy import AnalogyEngine

engine = AnalogyEngine()

# Find analogy
analogy = engine.find_analogy("snake", "pong")
print(f"Similarity: {analogy.overall_similarity}")

# Transfer rule
condition = {"AGENT_NEAR_TARGET", "NO_OBSTACLE_AHEAD"}
action = "MOVE_FORWARD"
new_cond, new_action = engine.transfer_rule(
    condition, action, "snake", "pong"
)
print(f"Transferred: {new_cond} → {new_action}")
```

---

## CONFIGURATION REFERENCE

| Module | Parameter | Default | Purpose |
|--------|-----------|---------|---------|
| **global_workspace** | attention_threshold | 0.5 | Min activation to broadcast |
| | veto_threshold | 0.75 | Danger similarity trigger |
| | max_rehearsal_cycles | 3 | Max deliberation rounds |
| **emotion_system** | valence_decay | 0.95 | Per-step decay rate |
| | arousal_blend_rate | 0.3 | EMA weight for arousal |
| | HISTORY_LIMIT | 200 | Trajectory bound |
| **episodic_memory** | recent_capacity | 1000 | Per-task buffer size |
| | consolidation_threshold | 500 | Trigger point |
| **rule_learner** | min_support | 5 | Min observations |
| | min_success_rate | 0.7 | Min confidence |
| | tenure_threshold | 1000 | High-support protection |
| **world_model** | bottleneck_dim | 128 | Latent space size |
| | hidden_dim | 64 | MLP layer size |
| | reward_weight | 10.0 | Loss multiplier |
| **hypervec_py** | DIMENSION | 10,240 | VSA bits |
| **universal_input** | n_bins | 100 | Thermometer bins |
| | max_codebook | 10,000 | LRU cache size |

---

## NEXT STEPS

For further details, consult:
- **Module Analysis**: [COMPLETE_MODULE_ANALYSIS.md](COMPLETE_MODULE_ANALYSIS.md)
- **System Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Mathematical Formulas**: [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md)
- **Getting Started**: [README.md](../README.md)
- **Test Evidence**: [RUN_LOGS_AND_EVIDENCE.md](RUN_LOGS_AND_EVIDENCE.md)

