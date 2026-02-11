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

## ADDITIONAL MODULES REFERENCE

### Continual Learning & Meta-Learning

**continual_learning.py**
- **Classes:** `ContinualLearner`, `PackNetManager`, `ProgressiveNetworks`
- **Key Methods:**
  - `ContinualLearner.compute_fisher_information()` → `Dict[str, Tensor]`
  - `ContinualLearner.ewc_loss(model)` → `float`
    - Formula: (λ/2) * Σ F_i * (θ_i - θ*_i)²
  - `PackNetManager.prune_and_allocate(model, task_id, prune_ratio=0.5)` → `None`
- **Constants:** ewc_lambda=1000, fisher_sample_size=200

**multi_task_learning.py**
- **Classes:** `SharedEncoder`, `TaskHead`, `MultiTaskLearner`
- **Key Methods:**
  - `MultiTaskLearner.gradient_surgery(gradients)` → `Dict[str, Tensor]`
    - PCGrad algorithm: projects conflicting gradients
  - `forward_multitask(x, task_id)` → `Tuple[action_logits, value]`
- **Architecture:** Input(state_dim) → Encoder(128) → Latent(64) → Heads(per-task)

**meta_learning.py**
- **Classes:** `MAML` - Model-Agnostic Meta-Learning
- **Key Methods:**
  - `meta_train(tasks, inner_lr, outer_lr, inner_steps)` → `None`
    - Bi-level optimization for few-shot adaptation
- **Algorithm:** θ' = θ - α∇L_task(θ); θ = θ - β∇Σ L_task(θ')

### Perception & Grounding

**perception.py**
- **Classes:** `CleanupMemory`, `FusionEngine`
- **Key Methods:**
  - `CleanupMemory.retrieve(query_hv, k=3)` → `List[Tuple[str, float]]`
    - K-nearest neighbors via Hamming distance
  - `FusionEngine.fuse_multimodal(audio_hv, visual_hv, text_hv)` → `HyperVector`
    - Role-based binding: audio⊗AUDIO_ROLE | visual⊗VISUAL_ROLE | text⊗TEXT_ROLE
- **Constants:** cleanup_capacity=500

**multimodal_processor.py**
- **Classes:** `MultimodalProcessor`
- **Key Methods:**
  - `process_text(text)` → `HyperVector`
  - `process_image(image_array)` → `HyperVector`
  - `process_audio(audio_waveform)` → `HyperVector`
  - `fuse_all(text, image, audio)` → `HyperVector`
- **Integration:** Uses universal_input for grounding, role-filler binding for fusion

**grounding_verifier.py**
- **Classes:** `GroundingVerifier`
- **Key Methods:**
  - `get_active_predicates(state, task_tag)` → `Set[str]`
    - Snake: REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT, DANGER_AHEAD, etc.
    - Pong: BALL_ABOVE, BALL_BELOW, BALL_APPROACHING, PADDLE_ALIGNED, etc.
    - Maze: AT_EXIT, WALL_AHEAD, PATH_CLEAR, etc.
  - `verify_grounding(action, state, task_tag)` → `bool`
- **Purpose:** Task-specific state→predicates mapping

**symbol_grounding.py**
- **Classes:** `SymbolGrounder`
- **Key Methods:**
  - `ground_state(state, task_tag)` → `HyperVector`
    - Binds predicates to VSA space
  - `ground_action(action)` → `HyperVector`
  - `unground_symbol(hv, candidates)` → `str`
    - Nearest-neighbor in symbol space
- **Constants:** predicate_codebook (deterministic seeding per predicate)

### Language & Dialogue

**language_module.py**
- **Classes:** `LanguageModule`
- **Key Methods:**
  - `understand(text)` → `Dict[str, Any]`
    - Intent extraction via LLM (Phi3) or mock fallback
  - `generate(system_state, context)` → `str`
    - Natural language explanation generation
  - `ground_to_vsa(parsed_intent)` → `HyperVector`
- **Dependencies:** llama-cpp-python (optional), fallback to template-based

**dialogue_manager.py**
- **Classes:** `DialogueManager`
- **Key Methods:**
  - `process_turn(user_input, cognitive_engine)` → `str`
    - Context window management (10 turns)
  - `resolve_anaphora(text, history)` → `str`
    - Simple pronoun→entity replacement
  - `route_to_handler(intent)` → `str`
    - Routes: question, command, why/explain
- **Context:** Deque of (role, text) tuples, max_size=10

**voice_interface.py**
- **Classes:** `VoiceInterface`
- **Key Methods:**
  - `listen()` → `str` - Speech-to-text
  - `speak(text)` → `None` - Text-to-speech
  - `conversational_loop(cognitive_engine)` → `None`
- **Dependencies:** speech_recognition, pyttsx3 (optional)

### Consciousness & Monitoring

**consciousness_metrics.py**
- **Classes:** `ConsciousnessMonitor`
- **Key Methods:**
  - `compute_phi(global_workspace)` → `float`
    - Integrated Information Theory approximation
  - `measure_broadcast_reach(winner, modules)` → `float`
  - `track_attention_stability(gw_history, window=10)` → `float`
- **Metrics:** phi (integration), reach (broadcast %), stability (winner consistency)

**explanation.py**
- **Classes:** `ExplanationGenerator`
- **Key Methods:**
  - `explain_decision(cognitive_state)` → `str`
    - Traces: winner, alternatives, confidence, emotion
  - `explain_rule(rule, examples)` → `str`
  - `explain_transfer(analogy, source_rule, target_rule)` → `str`
- **Output:** Human-readable natural language explanations

### Learning Support

**curiosity.py**
- **Classes:** `CuriosityModule`
- **Key Methods:**
  - `compute_novelty(situation_hv, task_tag)` → `float`
    - Formula: 1 - max_similarity(situation, prototypes)
  - `update_prototypes(situation_hv, task_tag)` → `None`
    - K-means-like clustering, max 100 prototypes/task
  - `detect_learning_stagnation(window=100)` → `bool`
    - Improvement < 0.01 threshold
- **Constants:** max_prototypes=100, window_size=100

**intrinsic_motivation.py**
- **Classes:** `IntrinsicCuriosityModule` (ICM)
- **Key Methods:**
  - `compute_intrinsic_reward(state, action, next_state)` → `float`
    - Forward model error: ||predicted_next - actual_next||²
  - `train_models(batch)` → `Dict[str, float]`
    - Losses: forward_loss, inverse_loss
- **Architecture:** Feature network (CNN) → latent(64) → forward/inverse models
- **Scaling:** intrinsic_reward * 0.01 (typical)

**learning.py**
- **Classes:** `ReplayBuffer`
- **Key Methods:**
  - `add(episode, quadrant)` → `None`
    - Quadrants: agree_success, agree_fail, disagree_success, disagree_fail
  - `sample_balanced(batch_size)` → `List[Episode]`
    - Samples equally from all 4 quadrants
  - `sleep_cycle(episodes, rule_learner)` → `None`
    - Offline consolidation: rule induction, pattern extraction
- **Capacity:** 2500 per quadrant (10K total)

**learning_progress.py**
- **Classes:** `LearningProgressTracker`
- **Key Methods:**
  - `track_progress(task_tag, metric_value)` → `float`
    - Returns delta from moving average
  - `detect_plateau(task_tag, threshold=0.01, window=50)` → `bool`
- **Purpose:** Triggers curriculum advancement or exploration boost

### Planning & Spatial Reasoning

**spatial_reasoning.py**
- **Classes:** `GridPlanner` (extends STRIPSPlanner)
- **Key Methods:**
  - `plan(initial_state, goal, max_depth=20)` → `List[str]`
    - Parses AT_X_Y predicates, computes Manhattan distance heuristic
  - `extract_position(state)` → `Tuple[int, int]`
  - `compute_movement(from_pos, to_pos)` → `str`
- **Heuristic:** A* with Manhattan distance to goal

**curriculum.py**
- **Classes:** `CurriculumManager`
- **Key Methods:**
  - `select_next_task(performance_history)` → `str`
    - Difficulty progression based on success rate
  - `adjust_difficulty(task_tag, direction)` → `None`
- **Strategy:** Gradual increase, fallback on failure

### Homeostasis & Drives

**homeostasis.py**
- **Classes:** `HomeostaticSystem`
- **Key Methods:**
  - `update_drives(elapsed_time, actions)` → `Dict[str, float]`
    - Drives: hunger, energy, curiosity, safety
    - Decay rates: hunger +0.01/step, energy -0.02/step
  - `compute_urgency()` → `float`
    - Max drive level (0-1)
  - `generate_drive_goals()` → `List[str]`
    - Maps drives to goal predicates (e.g., hunger → FIND_FOOD)
- **Constants:** hunger_decay=0.01, energy_decay=0.02, safety_baseline=0.8

### Agency & Active Inference

**agency.py**
- **Classes:** `ActiveInferenceAgent`
- **Key Methods:**
  - `infer_hidden_state(observations)` → `BeliefState`
  - `plan_actions(belief, preferences)` → `List[str]`
    - Minimizes expected free energy
  - `update_generative_model(outcome)` → `None`
- **Framework:** Partially Observable Markov Decision Process (POMDP)

**ai_controller.py**
- **Classes:** `PymdpController`
- **Key Methods:**
  - `step(observation)` → `int`
    - Uses pymdp library for active inference
  - `update_beliefs(obs, action)` → `None`
- **Dependencies:** pymdp (optional)

### Social Intelligence

**empathy.py**
- **Classes:** `EmpathyModule`
- **Key Methods:**
  - `simulate_other_emotion(agent_id, their_state, tom_model)` → `Dict`
    - Projects self onto other's situation
  - `generate_empathic_response(emotion)` → `str`
- **Algorithm:** Belief projection + self-emotion simulation

**teaching.py**
- **Classes:** `TeacherInterface`, `HumanTeacher`
- **Key Methods:**
  - `provide_feedback(state, action, correct_action)` → `None`
  - `correct_misconception(rule, counterexample)` → `None`
  - `suggest_exploration(area)` → `str`
- **Mode:** Interactive learning from human demonstrations

### Utilities & Infrastructure

**concept_mapper.py**
- **Classes:** `ConceptMapper`
- **Key Methods:**
  - `decode_hypervector(hv, codebook)` → `str`
    - Nearest-neighbor concept retrieval
  - `visualize_concept_space(hvs, labels)` → `None`
    - t-SNE or UMAP projection
- **Purpose:** Debugging and visualization

**latent_probe.py**
- **Classes:** `LatentProbe`
- **Key Methods:**
  - `train_probe(latents, labels)` → `LinearProbe`
    - Linear readout for latent space analysis
  - `interpret_dimension(dim_idx, examples)` → `str`
- **Purpose:** Neural representation interpretability

**context_engine.py**
- **Classes:** `ContextEngine`
- **Key Methods:**
  - `update_context(new_info)` → `None`
  - `retrieve_relevant(query, k=5)` → `List[str]`
    - Temporal decay + recency bias
  - `prune_stale(age_threshold=3600)` → `int`
- **Storage:** Temporal facts with timestamps

**lifecycle.py**
- **Classes:** `ConceptLifecycle`
- **Key Methods:**
  - `promote_concept(name, usage_count)` → `None`
    - Moves from short-term to long-term storage
  - `decay_unused(threshold=1000)` → `List[str]`
  - `merge_similar(similarity_threshold=0.9)` → `int`
- **Purpose:** Knowledge base hygiene

**brain_fusion.py**
- **Classes:** `BrainFusion`
- **Key Methods:**
  - `fuse_knowledge(brains, strategy="tagged_conservative")` → `FusedBrain`
    - Global layer: shared primitives (deterministic seeding)
    - Task layers: isolated task-specific knowledge
  - `query_fused(query_hv, task_tag)` → `Dict`
    - Returns: concept, action, similarity, provenance
- **Strategies:** tagged_conservative (isolate task rules), merge_all (global rules)

### Dashboards & Monitoring

**testing_dashboard.py**
- **Flask App:** Port 5051, 18 API endpoints
- **Features:**
  - `/api/chat` - Text input → cognitive response
  - `/api/game/start/<game>` - Launch Snake/Pong/Maze
  - `/api/monitor/emotion` - Real-time emotion state
  - `/api/export/logs` - Export TXT/JSON
- **Purpose:** Comprehensive testing interface
- **Tests:** 25 passing tests in test_testing_dashboard.py

**cognitive_dashboard.py**
- **Flask App:** Port 5050, 12 API endpoints
- **Features:**
  - `/api/input/multimodal` - Text/image/audio input
  - `/api/reasoning/trace` - Explanation generation
  - `/api/knowledge/query` - Semantic memory search
  - `/api/export/state` - JSON/ZIP export
- **Purpose:** Knowledge inspection and reasoning traces

**dashboard.py**
- **Flask App:** Port 5000, operational interface
- **Features:** Process management, ZMQ telemetry, AGI report generation
- **Purpose:** Main operational dashboard
- **Status:** Production-ready

### Game Environments

**snake_ui.py, pong_ui.py, maze_ui.py**
- **Purpose:** Pygame-based UI for human play and visualization
- **Key Methods:**
  - `render()` - Draw game state
  - `handle_input()` - Keyboard controls
  - `step(action)` - Physics update
- **Integration:** ZMQ communication with python_server.py

**snake_headless.py**
- **Purpose:** Headless Snake environment for batch testing
- **Key Methods:**
  - `reset()` → `state`
  - `step(action)` → `(state, reward, done, info)`
- **Compliance:** OpenAI Gym-like interface

**simulation.py**
- **Classes:** `SnakePhysics`, `PongPhysics`, `MazePhysics`
- **Key Methods:**
  - `simulate_action(state, action)` → `(next_state, reward, done)`
  - `get_valid_actions(state)` → `List[str]`
- **Purpose:** Pure Python physics for planning/imagination

### Training Scripts

**train_phase1_demo.py**
- **Purpose:** Multi-task learning + rule extraction demo
- **Key Features:**
  - Trains on Snake, Pong, Maze simultaneously
  - Extracts symbolic rules from neural policies
  - Dual inference: neural (fast) + symbolic (safe)
- **Run:** `python train_phase1_demo.py`

**train_phase2_demo.py**
- **Purpose:** Perception systems demo (vision, audio, language)
- **Key Features:**
  - Multimodal VSA binding
  - Cleanup memory associative retrieval
  - Grounding verification
- **Run:** `python train_phase2_demo.py`

**train_phase3_demo.py**
- **Purpose:** Continual learning demo (EWC, PackNet, Meta-learning)
- **Key Features:**
  - EWC prevents catastrophic forgetting (7.3% retention improvement)
  - Progressive Networks for task isolation
  - MAML for few-shot adaptation
- **Run:** `python train_phase3_demo.py`

**train_phase4_demo.py**
- **Purpose:** Causal discovery + planning demo
- **Key Features:**
  - Delta-P causal discovery
  - Counterfactual reasoning
  - Model Predictive Control (MPC)
  - Monte Carlo Tree Search (MCTS)
- **Run:** `python train_phase4_demo.py`

**train_phase5_demo.py**
- **Purpose:** Self-model + metacognition demo
- **Key Features:**
  - Performance tracking per task/context
  - Confidence calibration
  - Self-explanation
  - Autonomous learning rate adjustment
- **Run:** `python train_phase5_demo.py`

**train_phase6_demo.py**
- **Purpose:** Social & emotional intelligence demo
- **Key Features:**
  - Emotion generation (Plutchik + Russell)
  - Theory of Mind (Sally-Anne test)
  - Empathy simulation
  - Social learning
- **Run:** `python train_phase6_demo.py`

**train_phase7_demo.py**
- **Purpose:** Full integration + transfer learning demo
- **Key Features:**
  - KnowledgeStore for cross-session persistence
  - Cross-domain transfer via analogy
  - Abstract rule consolidation
  - LLM translator (Phi3) for natural language
- **Run:** `python train_phase7_demo.py`

### Neural Network Training

**train_snn.py**
- **Purpose:** Standalone SNN training script
- **Key Features:**
  - Task-aware SNN training
  - Quantization-aware training (QAT)
  - Multi-task gradient surgery
  - Model checkpointing
- **Run:** `python train_snn.py --task snake --epochs 100`

**snn_training_pipeline.py**
- **Classes:** `SNNTrainingPipeline`
- **Key Methods:**
  - `train_task(task_id, episodes)` → `None`
  - `evaluate_task(task_id, test_episodes)` → `Dict[str, float]`
  - `save_checkpoint(path)` → `None`
- **Purpose:** Modular training orchestration

### Persistence

**persistence.py**
- **Classes:** `BrainStore` - SQLite backend with WAL
- **Key Methods:**
  - `store_rule(rule, task_tag)` → `None`
  - `store_concept(name, hv_bytes)` → `None`
  - `store_episode(episode)` → `None`
  - `query_rules(task_tag, min_support=5)` → `List[Rule]`
  - `flush()` → `None` - Write-behind buffering (100 episodes or 30s)
- **Schema:** Tables for rules, concepts, episodes with indices on task_tag, timestamp
- **Constants:** buffer_size=100, flush_interval=30s

### Configuration

**config.py**
- **Classes:** `NSCKConfig` - Centralized hyperparameters
- **Key Attributes:**
  - `hv_dimension`: 10240
  - `ewc_lambda`: 1000
  - `attention_threshold`: 0.5
  - `min_rule_support`: 5
  - `world_model_bottleneck`: 128
- **Purpose:** Single source of truth for all module parameters

---

## COMPLETE MODULE INDEX

### By Category (91 Total Modules)

**Core Architecture (6)**
- cognitive_engine.py, global_workspace.py, python_server.py, config.py, system_launcher.py, agency.py

**Memory Systems (5)**
- episodic_memory.py, semantic_memory.py, intelligent_buffer.py, staged_recall.py, persistence.py

**Learning Systems (9)**
- rule_learner.py, causal_reasoning.py, analogy.py, continual_learning.py, multi_task_learning.py, meta_learning.py, curiosity.py, intrinsic_motivation.py, learning.py

**Neural Networks (5)**
- snn_qat.py, plastic_snn.py, universal_encoder.py, train_snn.py, snn_training_pipeline.py

**Planning & Reasoning (5)**
- planner.py, spatial_reasoning.py, world_model.py, brain_fusion.py, semantic_coherence.py

**Self-Awareness (4)**
- self_model.py, metacognition.py, explanation.py, consciousness_metrics.py

**Social Intelligence (4)**
- emotion_system.py, theory_of_mind.py, empathy.py, value_alignment.py

**Perception & Grounding (6)**
- perception.py, multimodal_processor.py, grounding_verifier.py, symbol_grounding.py, universal_input.py, hypervec_py.py

**Language & Dialogue (5)**
- language_module.py, dialogue_manager.py, voice_interface.py, voice_hd.py, lingua_cortex.py

**Dashboards & Monitoring (4)**
- testing_dashboard.py, cognitive_dashboard.py, dashboard.py, logger_service.py

**Game Environments (5)**
- snake_ui.py, snake_headless.py, pong_ui.py, maze_ui.py, maze_game.py, simulation.py

**Training Demos (7)**
- train_phase1_demo.py through train_phase7_demo.py

**Support Systems (7)**
- homeostasis.py, curriculum.py, learning_progress.py, lifecycle.py, context_engine.py, teaching.py, teacher_interface.py

**Utilities (12)**
- concept_mapper.py, latent_probe.py, saliency.py, character_dataset.py, char_offline_eval.py, debug_char_preprocess.py, visualize_transfer.py, build_codebook.py, download_model.py, ai_controller.py, voice_chatbot.py, train_semantic_folding.py

**Infrastructure (2)**
- hypervec_shim.py, __init__.py

---

## NEXT STEPS

For further details, consult:
- **Module Analysis**: [COMPLETE_MODULE_ANALYSIS.md](COMPLETE_MODULE_ANALYSIS.md)
- **System Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Mathematical Formulas**: [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md)
- **Getting Started**: [README.md](../README.md)
- **Test Evidence**: [RUN_LOGS_AND_EVIDENCE.md](RUN_LOGS_AND_EVIDENCE.md)

