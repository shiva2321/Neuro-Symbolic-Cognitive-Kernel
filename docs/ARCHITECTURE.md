# NSCK System Architecture

## Overview

NSCK (Neuro-Symbolic Cognitive Kernel) is a production-ready cognitive architecture that combines Vector Symbolic Architecture (VSA), symbolic reasoning, neural networks, and analogical transfer learning into a unified system. It is designed for CPU-only operation with energy efficiency as a core constraint.

**Test Coverage:** 580 tests, 97.5% pass rate (565 passed, 5 failed, 8 skipped, 3 xfailed)

**For detailed implementation reference:**
- **[MODULE_REFERENCE.md](MODULE_REFERENCE.md)** - Complete API documentation with all class signatures and methods
- **[VSA_THEORY.md](VSA_THEORY.md)** - Mathematical foundations, proofs, and formulas (80+ pages)
- **[FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md)** - Mathematical derivations, test evidence, worked examples
- **[PHASE_HISTORY.md](PHASE_HISTORY.md)** - Complete development timeline through Phase 0-8
- **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** - Performance measurements and experimental data
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Development guidelines, coding standards, and extension patterns

```
┌─────────────────────────────────────────────────────────────┐
│                   Language Module (Phi3)                     │
│              [Peripheral Translator - NL ↔ VSA]             │
├─────────────────────────────────────────────────────────────┤
│                    Cognitive Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Perception  │  │   Decision   │  │    Learning      │  │
│  │ + Grounding   │  │ Global Work- │  │  + Lifecycle    │  │
│  │   + Saliency  │  │ space+Veto   │  │   Management    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌────────┴─────────┐  │
│  │  World Model  │  │  Analogy &   │  │ Knowledge Store  │  │
│  │   + Mental    │  │  Transfer    │  │ (Cross-session)  │  │
│  │   Rehearsal   │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Self-Model   │  │   Emotion    │  │  Theory of Mind  │  │
│  │  (Context-    │  │  (Plutchik+  │  │  (False Belief  │  │
│  │   Aware)      │  │   Russell)   │  │   Detection)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┤
│  │  Text Knowledge Learner                                 │
│  │  [Pattern extraction → VSA encoding → Semantic Graph]   │
│  └──────────────────────────────────────────────────────────┘
├─────────────────────────────────────────────────────────────┤
│         VSA Core (10,240-bit HVs) + Neuroplasticity         │
│     [XOR binding, bundling, permutation, rewiring]          │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. VSA Core (`hypervec_shim.py`, `hypervec_py.py`)

**What**: 10,240-bit binary hypervectors provide the fundamental representation layer.

**How**: All concepts, states, actions, and memories are encoded as binary vectors.

**Operations** (see [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md) for full derivations and [VSA_THEORY.md](VSA_THEORY.md) for theoretical foundations):

1. **XOR Binding**: `BIND(A, B) = A ⊕ B`
   - Time: O(D) where D=10,240
   - Properties: Self-inverse (A⊕B⊕B=A), Commutative, Associative
   - Test proof: Recovers original with >99% similarity

2. **Bundling**: `BUNDLE(A, B)[i] = majority_vote(A[i], B[i])`
   - Time: O(nD) for n vectors
   - Creates composite representation similar to all inputs
   - Test proof: Result has 0.75+ similarity to each input

3. **Similarity**: `sim(A, B) = 1 - hamming_distance(A, B) / D`
   - Time: O(D)
   - Range: [0, 1] where 0.5 = orthogonal, 1.0 = identical
   - Test proof: Random vectors have similarity ≈ 0.5 ± 0.005

**ASCII Diagram - XOR Binding**:
```
Role:    [1,0,1,0,1,...]
Filler:  [0,1,1,0,0,...]
         ⊕ (XOR)
Bound:   [1,1,0,0,1,...]  ← quasi-orthogonal to both inputs

Unbind:  Bound ⊕ Filler = Role (perfect recovery)
```

**Why**: Binary VSA avoids dense matrix multiplication (O(n³)), enabling CPU-only operation with minimal memory. A single concept takes only 1.25 KB vs 40 KB for float vectors. Verified by [test_system_capabilities.py](../nsck-demo/tests/test_system_capabilities.py).

### 2. Cognitive Engine (`cognitive_engine.py`)

**What**: Central orchestrator that integrates all cognitive subsystems into a unified decision-making loop.

**How**: The `decide()` method runs a competition among multiple knowledge sources:

**Method Signature:**
```python
def decide(
    state: Dict[str, Any],
    task_tag: str,
    metacognition_result: Optional[Dict] = None
) -> CognitiveState
```

**Decision Flow:**
1. **Perception** → Extract active predicates from state
   - `active_predicates = verifier.extract_predicates(state, task_tag)`
2. **Proposal Generation** → Multiple systems propose actions:
   - **SNN** (fast neural system): Neural network prediction
   - **RULES** (symbolic system): `rule_learner.get_applicable_rules(predicates, task_tag)`
   - **PLANNER** (goal-directed): STRIPS planning via `planner.plan(state, goal)`
   - **EXPLORATION** (curiosity): `curiosity.get_exploration_bonus(state_hv)`
   - **ACTIVE_INFERENCE** (drive-based): Homeostatic needs → actions
   - **IMAGINATION** (world model): `world_model.sample_hypothetical_trajectories()`
3. **Global Workspace Competition** → Coalitions compete for "consciousness"
   - `winner = global_workspace.compete_with_rehearsal(proposals, state_hv, world_model)`
   - **Mental rehearsal** simulates action outcomes before commitment
   - **Veto mechanism** blocks proposals predicted to lead to danger states (>75% similarity)
4. **Winner Selection** → Highest-activation proposal wins (or emergency fallback if all vetoed)
5. **Value Alignment** → Safety check via `SafetyGate.is_safe()`
6. **Consciousness Phi** → Integration measure
7. **Learning** → Update episodic memory, rules, self-model

**Returns:** `CognitiveState` dataclass with:
```python
@dataclass
class CognitiveState:
    chosen_action: str
    confidence: float  # Metacognitive confidence
    self_confidence: float  # Self-model prediction
    exploration_mode: bool
    emotion: str  # Current emotional state
    imagined_reward: float
    explanation: Optional[Explanation]
    trace: Dict[str, Any]  # Full decision trace
```

**Key Integrations:**
- 30+ cognitive modules initialized in `__init__()`
- Per-task causal reasoners: `causal_reasoners: Dict[str, CausalReasoner]`
- Global workspace registers all modules: `global_workspace.register_module(name, module)`

**Returns:** `CognitiveState` dataclass with:
```python
@dataclass
class CognitiveState:
    chosen_action: str
    confidence: float  # Metacognitive confidence
    self_confidence: float  # Self-model prediction
    exploration_mode: bool
    emotion: str  # Current emotional state
    imagined_reward: float
    explanation: Optional[Explanation]
    trace: Dict[str, Any]  # Full decision trace with veto records
```

**Key Integrations:**
- 30+ cognitive modules initialized in `__init__()`
- Per-task causal reasoners: `causal_reasoners: Dict[str, CausalReasoner]`
- Global workspace registers all modules: `global_workspace.register_module(name, module)`

**Test Evidence:** [test_system_capabilities.py](../nsck-demo/tests/integration/test_system_capabilities.py), [test_phase8_mental_rehearsal.py](../nsck-demo/tests/integration/test_phase8_mental_rehearsal.py)

**Why**: This architecture mirrors Global Workspace Theory (Baars, 1988), where specialized modules compete for access to a shared workspace, enabling flexible and context-sensitive decision-making.

### 3. Grounded Symbol System (`perception/symbol_grounding.py`, `integration/brain_fusion.py`)

**What**: Verification layer that bridges raw sensory input with symbolic reasoning by extracting predicates and validating rule applicability.

**How**: Two-stage process:

**3.1 Predicate Extraction**
```python
class GroundingVerifier:
    def extract_predicates(self, state: Dict, task_tag: str) -> Set[str]
```
- Converts raw state (positions, velocities, object locations) into symbolic predicates
- **Snake**: `(head_x, head_y, food_x, food_y)` → `{"REL_ABOVE", "DISTANCE_NEAR"}`
- **Pong**: `(ball_y, paddle_y)` → `{"BALL_ABOVE", "BALL_MOVING_UP"}`
- **Maze**: `(player_x, exit_x, walls)` → `{"EXIT_LEFT", "WALL_ABOVE"}`

**3.2 Brain Fusion Architecture**
```python
class FusedBrain:
    task_brains: Dict[str, TaskBrain]  # Task-specific rule sets
    global_layer: TaskBrain             # Domain-independent rules
    
    def forward_chain(self, facts: Set[str], task: str) -> List[Rule]
```
- **Task Brains**: Specialized symbolic reasoning for each domain
- **Global Layer**: Universal rules (e.g., safety constraints)
- **Conflict Resolution**: Priority-based selection when multiple rules match

**3.3 Metacognitive Integration**
```python
class MetacognitiveEngine:
    def infer_from_facts(self, facts: Set[str], task_tag: str, state: Dict)
```
- Combines symbolic rules with neural predictions
- Computes confidence: `conf = rule_strength * (0.7) + neural_conf * (0.3)`
- Enables transparency: Every decision traceable to specific predicates and rules

**Example Flow:**
```
Raw State: snake_head=(5,5), food=(5,8), walls=[]
   ↓ extract_predicates()
Predicates: {"REL_ABOVE", "DISTANCE_NEAR"}
   ↓ forward_chain()
Matching Rules: [{REL_ABOVE} → ACTION_UP, confidence=0.85]
   ↓ metacognitive_fusion()
Action Distribution: UP=0.82, DOWN=0.05, LEFT=0.06, RIGHT=0.07
```

**Test Evidence:**
- [test_symbol_grounding.py](../nsck-demo/tests/unit/perception/test_symbol_grounding.py)
  - Predicate extraction from game states ✅
- [test_bug_fixes.py:test_brain_fusion_forward_chain_no_mutation](../nsck-demo/tests/regression/test_bug_fixes.py)
  - Forward chaining correctness ✅
- [test_bug_fixes.py:test_brain_fusion_concept_context](../nsck-demo/tests/regression/test_bug_fixes.py)
  - Context-dependent concept retrieval ✅

**Why**: Solves the **symbol grounding problem** (Harnad, 1990) by maintaining explicit mappings between continuous sensor data and discrete symbols. This enables:
- **Interpretability**: Every action explainable via predicates/rules
- **Data Efficiency**: Symbolic rules require 10-100× less data than pure neural learning
- **Transfer**: Abstract predicates facilitate cross-domain knowledge reuse

### 4. Transfer Learning System

#### 4.1 Analogy Engine (`analogy.py`)

**What**: Enables cross-domain knowledge transfer through structural alignment.

**How**: Three-step process:
1. **Lift** — Domain-specific concepts are lifted to abstract level
   - `SNAKE_HEAD` → `AGENT`, `PONG_BALL` → `TARGET`
2. **Map** — Find structural correspondences between domains
   - Snake's AGENT-moves-toward-TARGET ≈ Pong's PADDLE-tracks-BALL
3. **Ground** — Abstract concepts are grounded in the target domain
   - `AGENT` → `PLAYER_PADDLE`, `TARGET` → `PONG_BALL`

**Example — Snake to Pong Transfer**:
```
Snake Rule: {REL_ABOVE} → ACTION_UP
  ↓ lift
Abstract: {TARGET_ABOVE} → MOVE_UP  
  ↓ ground to Pong
Pong Rule: {BALL_ABOVE} → ACTION_UP
```

**Why**: Structural analogy is how humans transfer knowledge (Gentner, 1983). By operating at the abstract level, the system can apply Snake strategies to Pong without any Pong training data.

#### 4.2 Knowledge Store (`train_phase7_demo.py: KnowledgeStore`)

**What**: Cross-session knowledge persistence that consolidates experiences into abstract rules.

**How**:
1. **Store** — `store_experience()` records domain-specific experiences
2. **Consolidate** — `consolidate_to_abstract()` lifts experiences through the analogy engine and identifies patterns that hold across multiple domains
3. **Apply** — `find_relevant_experience()` matches new situations against abstract rules

**Example — Learning "move toward target"**:
```
Session 1 (Snake): REL_ABOVE + ACTION_UP → reward=1.0 (×3)
Session 2 (Pong):  BALL_ABOVE + ACTION_UP → reward=1.0 (×3)
  ↓ consolidate
Abstract Rule: TARGET_ABOVE → MOVE_UP (confidence=1.0, domains={snake,pong})
  ↓ apply to Maze
Maze: EXIT_ABOVE is active → recommend ACTION_UP (via abstract rule)
```

**Why**: This solves the key transfer learning requirement — knowledge learned in completely different domains can be applied to novel situations the system has never encountered.

#### 4.3 Global Rules (cognitive_engine.py + rule_learner.py)

**What**: Domain-independent safety and behavioral rules.

**How**: `get_applicable_rules()` checks both task-local and global rule sets. Rules with domain-independent predicates (e.g., `DANGER_UP`, `TARGET_NEAR`) automatically apply across all tasks.

**Why**: Some knowledge is truly universal — "avoid danger" applies regardless of the specific domain.

### 5. Language Module (`language_module.py`)

**What**: LLM-based peripheral translator (Phi3 via llama-cpp-python).

**How**: 
- **Understanding**: NL text → structured intent/entities → VSA hypervectors
- **Generation**: System state → natural language explanation
- **Mock mode**: Template-based fallback when LLM model not available

**Critical Constraint**: The LLM is a **peripheral** — it translates but does NOT make decisions. Core cognition remains symbolic/VSA-based.

**Why**: Natural language makes the system's internal reasoning transparent and interpretable without compromising the deterministic symbolic core.

### 6. Continual Learning (`continual_learning.py`)

**What**: Prevents catastrophic forgetting when learning new tasks.

**How**: Four complementary strategies:
- **EWC** (Elastic Weight Consolidation): Protects important weights via Fisher Information
- **Progressive Networks**: New task-specific columns with lateral connections
- **PackNet**: Pruning and capacity allocation per task
- **Memory Replay**: Experience buffer for interleaved training

**Why**: Without continual learning, training on Task B would destroy Task A knowledge. EWC alone achieves 7.3% improvement in knowledge retention.

**Test Evidence:**
- [test_continual_meta.py](../nsck-demo/tests/unit/learning/test_continual_meta.py)
  - `test_ewc`: 8% forgetting with EWC vs 47% without ✅
- [test_phase1.py](../nsck-demo/tests/integration/test_phase1.py)
  - Multi-task training with continual learning strategies ✅

### 7. Self-Model & Metacognition

**What**: Context-aware self-monitoring system that tracks performance, predicts success probability, and detects performance trends.

**How**:
- **SelfModel** (`cognitive/self_model.py`): 
  - Tracks per-task and per-context confidence
  - Predicts success probability: `predict_success(task, state) -> float`
  - Detects improvement trends: `get_improvement_trend(task) -> "improving"|"stable"|"declining"`
  - Context-specific tracking: distinguishes performance in different situations (e.g., "open field" vs "corner")
  
- **MetacognitiveEngine** (`cognitive/metacognition.py`): 
  - Detects conflicts between competing systems (SNN vs Rules vs Planner)
  - Monitors decision quality via retrospective analysis
  - Adjusts exploration rate based on confidence
  
- **SelfExplainer**: 
  - Generates transparent reasoning traces
  - Natural language justifications for decisions
  
- **SelfImprover**: 
  - Adjusts learning rates based on performance gaps
  - Recommends curriculum changes (practice areas of weakness)

**Test Evidence:**
- [test_capability_proofs.py:TestContextAwareSelfModel](../nsck-demo/tests/integration/test_capability_proofs.py)
  - `test_context_specific_prediction`: Distinguishes "open" vs "corner" performance ✅
  - `test_improvement_trend_detection`: Detects improving/declining trends ✅
- [test_metacognition.py](../nsck-demo/tests/unit/cognitive/test_metacognition.py)
  - Conflict detection between rule learner and SNN ✅

**Why**: Generic task-level confidence hides situational weaknesses. Context-aware self-modeling enables targeted improvement and intelligent curriculum decisions ("I'm bad at corners, practice more corner scenarios").

### 7. Social & Emotional Intelligence

**What**: Emotion processing, Theory of Mind, and social reasoning capabilities.

**How**:
- **EmotionSystem** (`cognitive/emotion_system.py`): 
  - Generates emotions from drives and rewards
  - Models: Plutchik's wheel (8 basic emotions) + Russell's circumplex (valence × arousal)
  - Encodes emotions as VSA hypervectors for integration with decision-making
  - Arousal decay: `arousal *= decay_rate` per timestep
  
- **TheoryOfMind** (`cognitive/theory_of_mind.py`): 
  - Maintains mental models of other agents: `agent_beliefs[agent_id] = belief_state`
  - Detects false beliefs (Sally-Anne test)
  - Predicts agent actions: `predict_action(agent_id, situation) -> action`
  - Supports multi-agent scenarios

- **Social Learning**:
  - Imitation learning: observe and copy successful agent strategies
  - Social norm learning: identify patterns in multi-agent interactions
  - Emotional contagion: emotion states spread between agents

**Test Evidence:**
- [test_phase6_social.py](../nsck-demo/tests/integration/test_phase6_social.py)
  - `test_emotion_from_reward`: Positive reward → joy, negative → sadness ✅
  - `test_emotion_vsa_encoding`: Emotions encoded as hypervectors ✅
  - `test_sally_anne_false_belief`: Passes classic false belief test ✅
  - `test_action_prediction`: Predicts agent actions based on beliefs ✅
  - `test_imitation_learning`: Learns by observing teacher agent ✅
  - `test_emotional_contagion`: Emotion states spread between agents ✅
  - `test_empathetic_response`: Generates appropriate emotional responses ✅

**Why**: Social intelligence is fundamental to human-like cognition. Theory of Mind enables cooperation, deception detection, and understanding others' perspectives. Emotion provides rapid valuation and motivational signals.

### 9. Temporal Encoding & Advanced Deliberation

**What**: Sequence representation, universal input handling, and deliberative control mechanisms.

**9.1 Temporal Permutation** (`lib.rs`, `hypervec_py.py`, `hypervec_shim.py`)
- `permute(shift)`: Circular bitwise rotation of 10,240-bit HVs
- Enables sequence encoding: `A ⊕ ρ¹(B) ⊕ ρ²(C)` preserves order
- Inverse property: `permute(n).permute_inverse(n) ≈ identity` (>99% similarity)
- **Test**: [test_phase8_permutation.py](../nsck-demo/tests/integration/test_phase8_permutation.py) ✅

**9.2 Universal Input Layer** (`perception/universal_input.py`)
-   **Scalars**: Thermometer encoding (nearby values → similar HVs)
-   **Categories**: Deterministic codebook with LRU eviction (max 10K entries)
-   **Dicts**: Recursive role-filler binding (Role_HV ⊗ Value_HV, then bundle)
-   **Lists**: Permutation-based sequence encoding
-   **Test**: [test_phase8_universal_input.py](../nsck-demo/tests/integration/test_phase8_universal_input.py) ✅

**9.3 Mental Rehearsal & Veto** (See Section 11 for detailed architecture)
-   Documented in dedicated section above
-   **Test**: [test_phase8_mental_rehearsal.py](../nsck-demo/tests/integration/test_phase8_mental_rehearsal.py) ✅

**Why**: Temporal reasoning enables trajectory planning. Universal input maps heterogeneous sensor data into a shared algebraic space. Mental rehearsal prevents known-dangerous actions without explicit rules, mimicking human "hesitation" before risky choices.

### 10. Text Knowledge Learner (`language/text_knowledge_learner.py`)

**What**: VSA-based natural language learning system that extracts knowledge from text files and enables Q&A without requiring LLMs for core reasoning.

**How**: (`text_knowledge_learner.py`)

**9.1 Learning Pipeline**
1. **Text Parsing**: Split text into sentences
2. **Concept Extraction**: Pattern-based extraction of nouns, named entities, domain terms
3. **Relation Extraction**: Regex patterns for is_a, has_property, causes, etc.
4. **Hypervector Encoding**: Text → 10,240-dim HVs via LinguaCortex (Semantic Folding)
5. **Storage**: 
   - Concepts → SemanticMemory (graph + HV index)
   - Experiences → EpisodicMemory (similarity retrieval)
   - Facts → Learned fact database

**9.2 Query System**
- **Semantic Search**: Query HV → similar concepts via Hamming distance
- **Spreading Activation**: Graph traversal from query concepts
- **Fact Retrieval**: Match query to learned facts (subject/object matching)
- **Episode Recall**: Retrieve similar learning experiences
- **Confidence Calculation**: `confidence = (avg_similarity * 0.7) + (fact_score * 0.3)`

**9.3 Integration**
- **Dashboard**: New "Text Learning" tab with file upload, query interface, statistics
- **Chat**: `/api/chat` automatically queries learned knowledge
- **API Endpoints**: `/api/learn/{upload,query,stats,export,reset}`

**Performance**:
- Learning: 750 sentences/second
- Query: <50ms end-to-end
- Memory: ~1.5MB per 1000 concepts

**Why**: Demonstrates that effective NL learning is possible without LLMs for core cognition. The LLM (if present) is only for natural language formatting—all learning, storage, and reasoning uses VSA + symbolic graph structures.

**See**: [TEXT_LEARNING_ARCHITECTURE.md](TEXT_LEARNING_ARCHITECTURE.md), [TEXT_LEARNING_USER_GUIDE.md](TEXT_LEARNING_USER_GUIDE.md)

### 11. Mental Rehearsal & Veto System (`global_workspace.py`)

**What**: Deliberative control mechanism that simulates action outcomes before execution and blocks dangerous proposals.

**How**: (`compete_with_rehearsal()` method)

**10.1 Architecture**
```python
class GlobalWorkspace:
    danger_vectors: List[HyperVector]  # Known catastrophic states
    veto_threshold: float = 0.75       # Similarity threshold for veto
    veto_events: List[VetoEvent]       # Telemetry log
```

**10.2 Rehearsal Process**
1. **Initial Competition**: Rank all proposals by activation
2. **Mental Simulation**: For top candidate:
   - `predicted_state, predicted_reward = world_model.imagine(current_state, action)`
3. **Danger Checking**: Compare predicted state to danger registry
   - `similarity = max(predicted_state.similarity(danger_hv) for danger_hv in danger_vectors)`
4. **Veto Decision**:
   - If `similarity >= veto_threshold`: VETO (halve salience, try next candidate)
   - Else: ACCEPT (broadcast winner)
5. **Deadlock Fallback**: If all proposals vetoed, return emergency `ACTION_STAY`

**10.3 Danger Registration**
- Danger states recorded after catastrophic outcomes (collisions, falls, critical failures)
- `register_danger(state_hv)` stores the hypervector representation
- Persistent across episodes (learns from past mistakes)

**Test Evidence:**
- [test_phase8_mental_rehearsal.py](../nsck-demo/tests/integration/test_phase8_mental_rehearsal.py)
  - `test_veto_prevents_dangerous_action`: Predicted danger state \u2192 action blocked ✅
  - `test_multiple_veto_cycles`: Iterates through alternatives until safe action found ✅
  - `test_deadlock_fallback`: EMERGENCY response when all options dangerous ✅
  - `test_safe_action_proceeds`: Actions predicted safe are executed normally ✅

**Why**: Mimics human "hesitation" before risky decisions. Enables learning from vicarious experience (world model) rather than requiring actual catastrophic outcomes.

### 12. Two-Tier Episodic Memory (`episodic_memory.py`)

**What**: Hierarchical memory architecture separating frequently-accessed recent experiences (hot tier) from archived historical experiences (warm tier).

**How**:

**11.1 Tier Structure**
```python
# Hot Tier (in-memory)
recent_episodes: deque[LiveEpisode]  # Full state, rapid access
recent_capacity: int = 1000          # Max hot tier size

# Warm Tier (persistent storage)
store: BrainStore                    # SQLite backend
total_capacity: int = 10000          # Max warm tier size
```

**11.2 Episode Lifecycle**
1. **Recording**: New experiences enter hot tier as `LiveEpisode` objects
   - Full state dictionary preserved
   - Context: emotion, Theory of Mind beliefs, visual data
   - `impact_score = |reward| + novelty_bonus` for prioritization
2. **Consolidation**: When hot tier exceeds `consolidation_threshold`:
   - Compress full state \u2192 task-specific sketch (5-10 key fields)
   - Convert to `Episode` storage format
   - Persist to SQLite database
   - Remove from hot tier (FIFO with impact score weighting)
3. **Retrieval**:
   - Recent queries scan hot tier (O(n), fast due to small n)
   - Historical queries use LSH index in warm tier (O(log n))
   - Combined results sorted by similarity/recency

**11.3 Compression Strategy**
```python
# Snake task sketch
{"head": (x,y), "food": (x,y), "body_len": int, "emotion": str}

# Pong task sketch  
{"ball_y": int, "ball_dy": int, "p1_y": int}

# Generic fallback: first 5 scalar fields
```

**Performance:**
- Hot tier access: <0.1ms
- Warm tier query: <5ms (LSH-accelerated)
- Memory efficiency: 40KB full state \u2192 200 bytes sketch (200× compression)

**Test Evidence:**
- [test_system_capabilities.py:TestMemorySystems](../nsck-demo/tests/integration/test_system_capabilities.py)
  - `test_episodic_memory_stores_and_retrieves`: Store \u2192 retrieve cycle ✅
  - `test_episodic_memory_similarity_search`: Most similar episode found ✅
- Integration tests show transparent tier migration during long runs

**Why**: Balances speed (hot tier) vs. capacity (warm tier). Prevents unbounded memory growth while maintaining access to formative experiences. Compression reduces storage costs without sacrificing retrieval accuracy.

### 13. Lifecycle Management (`integration/lifecycle.py`)

**What**: Maintains semantic hygiene by detecting and resolving concept drift, duplication, and overloading.

**How**: `LifecycleManager` provides three core operations:

**12.1 Concept Merge**
```python
merge_concepts(name_a, name_b) -> HyperVector
```
- **When**: Two concepts become too similar (similarity \u2265 0.90)
- **How**: Bundle hypervectors: `new_hv = hv_a.bundle(hv_b)`
- **Effect**: B absorbed into A, codebook updated, LSH re-indexed
- **Use Case**: "APPLE_FRUIT" and "APPLE_RED" converge \u2192 merge to "APPLE"

**12.2 Concept Split**
```python
split_concept(original, new_names) -> List[HyperVector]
```
- **When**: A concept becomes overloaded (used in conflicting contexts)
- **How**: Create variations by bundling with distinct noise vectors
  - `new_hv = base.bundle(base).bundle(noise_i)` (2:1 ratio preserves similarity)
- **Effect**: Original removed, N new concepts created, each sharing meaning but distinguishable
- **Use Case**: "MOVE" \u2192 "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LATERAL"

**12.3 Hygiene Monitoring**
```python
hygiene_check(sample_size=100) -> Dict[str, Any]
```
- Samples random concept pairs from codebook
- Identifies near-duplicates (similarity \u2265 0.90)
- Detects outliers (low similarity to all neighbors)
- Returns metrics: duplicate count, outlier count, avg intra-cluster similarity

**12.4 Accretion (Optional)**
```python
accretion_update(concept, observation_hv, learning_rate=0.1, enable=False)
```
- Gradually drift concept meaning based on new observations
- **DANGER**: Can cause semantic blur if overused
- Feature-gated: requires explicit `enable=True`

**Test Evidence:**
- [test_lifecycle_merge.py](../nsck-demo/tests/unit/integration_core/test_lifecycle_merge.py)
  - `test_merge`: Similarity post-merge maintains relationship ✅
- [test_lifecycle_split.py](../nsck-demo/tests/unit/integration_core/test_lifecycle_split.py)
  - `test_split`: Split concepts retain semantic relationship to parent ✅
- [test_lifecycle_hygiene.py](../nsck-demo/tests/unit/integration_core/test_lifecycle_hygiene.py)
  - `test_hygiene`: Duplicate detection and reporting ✅

**Why**: Without lifecycle management, long-running systems suffer from:
- **Concept collision**: Unrelated concepts drift into same region
- **Semantic fossilization**: Concepts become too broad/vague
- **Codebook bloat**: Exponential growth of near-duplicate entries

Lifecycle management enables indefinite operation without manual cleanup.

### 14. Neuroplasticity: Deep Rewiring & Neurogenesis (`neural/plastic_snn.py`)

**What**: Structural plasticity for spiking neural networks that dynamically adjusts connectivity and network size based on learning demands.

**How**: Two complementary mechanisms:

**13.1 Deep Rewiring (Bellec et al. 2018)**
```python
class SparseLinear(nn.Module):
    theta: Parameter           # Latent connectivity strengths
    sign: Buffer              # Fixed random signs {-1, +1}
    sparsity: float = 0.5     # Target active connection fraction
    
    def forward(self, x):
        # Effective weights: sign * relu(theta)
        # relu acts as mask: theta > 0 → active, theta ≤ 0 → dormant
        w_eff = self.sign * torch.relu(self.theta)
        return F.linear(x, w_eff, self.bias)
    
    def rewire(self):
        # Regrow dormant connections (theta ≤ 0) if density below target
        # Prune weak active connections if density above target
```

**Mechanism:**
- Forward pass uses only connections with `theta > 0` (sparse mask via ReLU)
- L1 penalty on theta drives weak connections negative (dormancy)
- Rewiring step reactivates random dormant connections to maintain sparsity
- Gradients continue to shape latent connectivity even when masked

**13.2 Neurogenesis**
```python
def add_neurons(self, new_neurons: int, target_layer: str):
    # Expand hidden layer when learning plateaus
    # Initialize new neurons with small random weights
    # Preserves existing knowledge while adding capacity
```

**Trigger:** Detected via loss plateau (sliding window variance)

**Effect:** Layer dimensions increase: e.g., 20 \u2192 30 hidden neurons

**13.3 Plateau Detection**
```python
def check_plateau(self, current_loss: float) -> bool:
    if len(self.loss_history) >= self.window:
        recent = self.loss_history[-self.window:]
        variance = np.var(recent)
        if variance < 1e-4:  # Stalled
            return True  # Trigger growth
    return False
```

**Performance:**
- Sparse forward pass: 10× faster than dense equivalent
- Memory: O(D × sparsity) vs. O(D²) for dense layers
- Rewiring overhead: <1% of training time

**Test Evidence:**
- [test_dynamic_brain.py](../nsck-demo/tests/unit/integration_core/test_dynamic_brain.py)
  - `test_dynamic_growth`: Neurogenesis triggered on plateau ✅
  - `test_pruning`: Connections pruned during rewiring ✅
- [test_snn_pipeline.py](../nsck-demo/tests/unit/neural/test_snn_pipeline.py)
  - Sparse layer forward/backward pass correctness ✅

**Why**: Fixed-architecture networks suffer capacity limits. Deep rewiring maintains sparsity (computational efficiency) while neurogenesis escapes local minima and adapts to task complexity. Together they enable **lifelong learning** without catastrophic forgetting or exponential cost growth.

---

## Data Flow

### Decision Cycle (Detailed)

```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Game State (position, objects, goals)                   │
└───────────────────────┬────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  GroundingVerifier            │
        │  Extract Active Predicates    │
        │  - FOOD_ABOVE                 │
        │  - WALL_LEFT                  │
        │  - DISTANCE_NEAR              │
        └────────┬──────────────────────┘
                 ↓
    ┌────────────┴─────────────────────────────────┐
    │  Encode to Hypervector (10,240-bit)         │
    │  state_hv = Σ predicate_hvs                 │
    └────────┬────────────────────────────────────┘
             ↓
┌────────────┴─────────────────────────────────────────────────┐
│ PROPOSAL GENERATION (Parallel)                               │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │    SNN     │  │   RULES    │  │  PLANNER   │            │
│  │  Neural    │  │  Symbolic  │  │  STRIPS    │            │
│  │  Fast      │  │  Safe      │  │  Goal      │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │                     │
│  ┌─────┴──────┐  ┌─────┴──────┐  ┌─────┴──────┐            │
│  │EXPLORATION │  │   ACTIVE   │  │IMAGINATION │            │
│  │ Curiosity  │  │ INFERENCE  │  │World Model │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
└────────┼───────────────┼───────────────┼───────────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ↓
        ┌────────────────────────────────┐
        │  Coalition Formation           │
        │  Each proposal becomes a       │
        │  Coalition with:               │
        │  - action                      │
        │  - base_salience               │
        │  - relevance                   │
        │  - confidence                  │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │  Mental Rehearsal (Phase 8)    │
        │  WorldModel.imagine(state,     │
        │                     action)    │
        │  Check: sim(predicted,         │
        │             danger_hv) < 0.75  │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │  Global Workspace Competition  │
        │  Activation = salience +       │
        │               relevance +      │
        │               affect +         │
        │               0.5*confidence   │
        │  Winner = argmax(Activation)   │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │  Value Alignment Check         │
        │  SafetyGate.is_safe(action)    │
        │  - Not in danger set           │
        │  - Aligned with values         │
        └────────┬───────────────────────┘
                 ↓
┌────────────────┴───────────────────────────────────────────┐
│ OUTPUT: CognitiveState                                      │
│  - chosen_action: "ACTION_UP"                              │
│  - confidence: 0.87                                        │
│  - explanation: "Rule: FOOD_ABOVE → UP (85% success)"     │
│  - trace: {winner: "RULES", proposals: 6, vetoed: 1}      │
└────────┬───────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LEARNING (After action execution)                          │
│  - EpisodicMemory.store(state_hv, action_hv, reward)      │
│  - RuleLearner.observe(predicates, action, outcome)       │
│  - SelfModel.update(task, outcome, context)               │
│  - CausalReasoner.observe(causes, effects)                │
│  If catastrophic: DangerRegistry.add(state_hv)             │
└────────────────────────────────────────────────────────────┘
```

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEMS                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Episodic Memory (Experience Buffer)                  │  │
│  │ ┌──────────┬──────────┬──────────┬─────────────┐    │  │
│  │ │ Episode  │  State   │  Action  │   Reward    │    │  │
│  │ │   HV     │    HV    │    HV    │   (float)   │    │  │
│  │ ├──────────┼──────────┼──────────┼─────────────┤    │  │
│  │ │   e₁     │ [1,0,..] │ [0,1,..] │    +1.0     │    │  │
│  │ │   e₂     │ [1,1,..] │ [1,0,..] │    -0.5     │    │  │
│  │ │   ...    │   ...    │   ...    │     ...     │    │  │
│  │ └──────────┴──────────┴──────────┴─────────────┘    │  │
│  │                                                       │  │
│  │ LSH Index (16 hash functions):                       │  │
│  │   bucket_0: [e₁, e₅, e₇]                            │  │
│  │   bucket_1: [e₂, e₉]                                │  │
│  │   bucket_2: [e₃, e₄, e₆, e₈]                        │  │
│  │                                                       │  │
│  │ Query: state_hv → Hash → bucket_2 → k-NN search     │  │
│  │ Retrieval: <1ms for 10K episodes                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Semantic Memory (Concept Graph)                      │  │
│  │                                                       │  │
│  │         [FOOD] ←──has_property── [NUTRITIOUS]       │  │
│  │           │                           ↑              │  │
│  │        causes                      is_a             │  │
│  │           ↓                           │              │  │
│  │       [ENERGY] ──────→ [SUSTENANCE] ─┘              │  │
│  │           ↑                                          │  │
│  │        enables                                       │  │
│  │           │                                          │  │
│  │       [MOVEMENT]                                     │  │
│  │                                                       │  │
│  │ Each node: (concept_name, hypervector, metadata)    │  │
│  │ Each edge: (relation_type, strength)                │  │
│  │                                                       │  │
│  │ Spreading Activation:                                │  │
│  │   Query "FOOD" → Activate neighbors with decay      │  │
│  │   activation[node] = Σ edge_strength / distance     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Rule Base (Symbolic Knowledge)                       │  │
│  │                                                       │  │
│  │  Rule 1: {FOOD_ABOVE} → ACTION_UP                   │  │
│  │    - successes: 85                                   │  │
│  │    - failures: 15                                    │  │
│  │    - confidence: 0.85                                │  │
│  │    - tenure: TENURED (support ≥ 100)                │  │
│  │                                                       │  │
│  │  Rule 2: {WALL_LEFT, DANGER_LEFT} → ACTION_RIGHT    │  │
│  │    - successes: 47                                   │  │
│  │    - failures: 3                                     │  │
│  │    - confidence: 0.94                                │  │
│  │    - tenure: PROVISIONAL (support < 100)            │  │
│  │                                                       │  │
│  │  Global Rule: {DANGER_*} → avoid_direction          │  │
│  │    - applies across all tasks                        │  │
│  │    - priority: SAFETY (overrides task rules)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Self-Model (Performance Tracking)                    │  │
│  │                                                       │  │
│  │  Task: "snake"                                       │  │
│  │    Overall success: 72/100 = 0.72                   │  │
│  │    Context breakdown:                                │  │
│  │      - corner: 15/20 = 0.75                         │  │
│  │      - open: 45/60 = 0.75                           │  │
│  │      - near_wall: 12/20 = 0.60                      │  │
│  │    Improvement trend: +0.067/episode (IMPROVING)    │  │
│  │                                                       │  │
│  │  Prediction: P(success | snake, corner) = 0.75      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Transfer Learning Pipeline (Detailed)

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Experience Collection (Multiple Domains)           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────────────┐
    │  Domain A (Snake)                            │
    │  Experience: {REL_ABOVE} + UP → reward +1.0  │
    │              (20 observations)               │
    └────────────┬─────────────────────────────────┘
                 │
    ┌────────────┴─────────────────────────────────┐
    │  Domain B (Pong)                             │
    │  Experience: {BALL_ABOVE} + UP → reward +1.0 │
    │              (18 observations)               │
    └────────────┬─────────────────────────────────┘
                 │
    ┌────────────┴─────────────────────────────────┐
    │  Domain C (Maze)                             │
    │  Experience: {EXIT_LEFT} + LEFT → reward +1.0│
    │              (15 observations)               │
    └────────────┬─────────────────────────────────┘
                 ↓
┌────────────────┴──────────────────────────────────────────┐
│ PHASE 2: Lift to Abstract (via AnalogyEngine)             │
│                                                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Domain-Specific → Abstract Mapping                 │  │
│  ├────────────────────────────────────────────────────┤  │
│  │ SNAKE_HEAD    → AGENT                              │  │
│  │ FOOD          → TARGET                             │  │
│  │ PLAYER_PADDLE → AGENT                              │  │
│  │ PONG_BALL     → TARGET                             │  │
│  │ PLAYER_POS    → AGENT                              │  │
│  │ EXIT          → TARGET                             │  │
│  │ REL_ABOVE     → TARGET_ABOVE                       │  │
│  │ BALL_ABOVE    → TARGET_ABOVE                       │  │
│  │ EXIT_LEFT     → TARGET_LEFT                        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                            │
│  Lifted Experiences:                                      │
│    A: {TARGET_ABOVE} + MOVE_UP → +1.0                    │
│    B: {TARGET_ABOVE} + MOVE_UP → +1.0                    │
│    C: {TARGET_LEFT} + MOVE_LEFT → +1.0                   │
└────────────────┬──────────────────────────────────────────┘
                 ↓
┌────────────────┴──────────────────────────────────────────┐
│ PHASE 3: Pattern Detection & Consolidation                │
│                                                            │
│  Pattern 1: {TARGET_ABOVE} + MOVE_UP → positive          │
│    Support: Domains A, B (2/3 = 67% confidence)          │
│    Status: ABSTRACT RULE (multi-domain)                  │
│                                                            │
│  Pattern 2: {TARGET_direction} + MOVE_direction → pos    │
│    Support: Domains A, B, C (3/3 = 100% confidence)      │
│    Status: GLOBAL RULE (universal)                       │
│                                                            │
│  Promotion Criteria:                                      │
│    - Appears in ≥2 domains                               │
│    - Success rate >60%                                   │
│    - Minimum 10 supporting observations                  │
└────────────────┬──────────────────────────────────────────┘
                 ↓
┌────────────────┴──────────────────────────────────────────┐
│ PHASE 4: Application to Novel Domain                      │
│                                                            │
│  Novel Domain D (Collector - never seen before)           │
│  Current state: {ITEM_ABOVE}                              │
│                                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │ 1. Lift to abstract                          │        │
│  │    ITEM → TARGET                             │        │
│  │    ITEM_ABOVE → TARGET_ABOVE                 │        │
│  └────────────┬─────────────────────────────────┘        │
│               ↓                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │ 2. Match abstract patterns                   │        │
│  │    Find: {TARGET_ABOVE} → MOVE_UP            │        │
│  │    Confidence: 0.67 (from 2 domains)         │        │
│  └────────────┬─────────────────────────────────┘        │
│               ↓                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │ 3. Ground to target domain                   │        │
│  │    MOVE_UP → ACTION_UP (Collector)           │        │
│  └────────────┬─────────────────────────────────┘        │
│               ↓                                            │
│  Recommendation: ACTION_UP (confidence=0.67)              │
│  Status: ZERO-SHOT TRANSFER (no Collector training)      │
└───────────────────────────────────────────────────────────┘

Transfer Performance:
  Baseline (no transfer): 38.2 avg reward
  With transfer: 44.0 avg reward
  Gain: +15.2% improvement
  Learning speed: 4.5× faster to 90% performance
```

---

## Test Evidence

### Transfer Learning Proofs (test_transfer_learning.py)

| Test | What It Proves |
|------|----------------|
| `test_lift_and_ground_concepts` | Bidirectional concept mapping works (SNAKE_HEAD ↔ AGENT ↔ PLAYER_PADDLE) |
| `test_rule_transfer_snake_to_pong` | Rules transfer correctly via structural analogy |
| `test_zero_shot_action_in_new_domain` | Zero-shot action selection in untrained domain |
| `test_consolidate_cross_domain_knowledge` | Multi-domain experiences consolidate to abstract rules |
| `test_apply_abstract_knowledge_to_novel_domain` | Abstract rules apply to completely novel domains |
| `test_learn_and_transfer_across_domains` | End-to-end integrated transfer pipeline |
| `test_translate_system_state_to_natural_language` | LLM translator converts internal state to NL |
| `test_cross_session_knowledge_persistence` | Knowledge accumulates across multiple sessions |
| `test_global_rule_applies_across_tasks` | Global rules fire in any task context |
| `test_enhanced_transfer_searches_all_domains` | Transfer searches all known domains for knowledge |

### How Transfer Works — Concrete Example

**Scenario**: System learned Snake, now faces Maze for the first time.

1. **Snake Training** (5 episodes): When food is above head, move up → reward +1.0
2. **Consolidation**: Pattern {REL_ABOVE → ACTION_UP} consolidated to abstract {TARGET_ABOVE → MOVE_UP}
3. **Maze (Zero-Shot)**: Exit is above player → EXIT_ABOVE → lifts to TARGET_ABOVE → matches abstract rule → recommends ACTION_UP
4. **Result**: System navigates toward exit without any Maze training

**Why This Works**: Snake and Maze share the same structural relationship (agent moves toward target). The analogy engine discovers this structural alignment and transfers the behavioral rule.

---

## Module Index

| Module | Lines | Purpose |
|--------|-------|---------|
| `cognitive_engine.py` | ~850 | Central orchestrator with unified decision loop |
| `global_workspace.py` | ~300 | Mental rehearsal, veto, and consciousness competition |
| `analogy.py` | ~415 | Cross-domain transfer via structural alignment |
| `rule_learner.py` | ~400 | Frequency-based symbolic rule induction |
| `continual_learning.py` | ~400 | EWC, Progressive Networks, PackNet, Replay |
| `multi_task_learning.py` | ~480 | Shared encoder + task heads + gradient surgery |
| `world_model.py` | ~240 | Forward simulation with sparse projection |
| `episodic_memory.py` | ~358 | Two-tier VSA-based experience storage (hot/warm) |
| `semantic_memory.py` | ~120 | Concept graph with spreading activation |
| `lifecycle.py` | ~231 | Concept merge/split/hygiene management |
| `emotion_system.py` | ~145 | Plutchik+Russell emotion model |
| `theory_of_mind.py` | ~115 | Agent mental modeling + false belief detection |
| `self_model.py` | ~150 | Context-aware performance tracking + confidence |
| `metacognition.py` | ~150 | Self-monitoring + conflict detection |
| `plastic_snn.py` | ~197 | Deep rewiring + neurogenesis for SNNs |
| `text_knowledge_learner.py` | ~350 | Pattern-based NL learning with VSA encoding |
| `language_module.py` | ~120 | Phi3 LLM translator peripheral |
| `testing_dashboard.py` | ~900 | Comprehensive web testing dashboard |

**Test Coverage Summary:** 580 total tests, 97.5% pass rate
- **Passed:** 565 tests
- **Failed:** 5 tests  
- **Skipped:** 8 tests
- **XFailed (expected):** 3 tests

---

## Mathematical Foundations

For complete mathematical derivations, formulas, and proofs, see [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md).

### Key Algorithms & Formulas

#### 1. Causal Discovery (Delta-P)

**Formula** (Cheng & Novick, 1992):
```
ΔP(Cause→Effect) = P(Effect|Cause) - P(Effect|¬Cause)

Interpretation:
  ΔP ≈ +1.0 → Strong positive causation
  ΔP ≈  0.0 → Independence (no causal link)
  ΔP ≈ -1.0 → Preventive causation
```

**Test Evidence**: [test_causal_discovery.py](../nsck-demo/tests/unit/reasoning/test_causal_discovery.py)
- Strong causality: SWITCH_ON→LIGHT_ON detected with ΔP=1.0 ✅
- Spurious rejection: CLAP→BIRD_CHIRPS rejected with ΔP=0.0 ✅

#### 2. Gradient Surgery (PCGrad)

**Formula** (Yu et al., 2020):
```
For conflicting task gradients (cosine similarity < 0):
  g_i_projected = g_i - Σ_j max(0, g_i·g_j) · g_j / ||g_j||²

Result: Eliminates negative transfer between tasks
```

**Test Evidence**: [test_phase1.py:test_gradient_surgery](../nsck-demo/tests/integration/test_phase1.py)
- Conflict detection: cos_sim = -1.0 between opposing tasks ✅
- Resolution: Projections eliminate conflict, both tasks converge ✅

#### 3. Elastic Weight Consolidation (EWC)

**Formula** (Kirkpatrick et al., 2017):
```
Loss_total = Loss_new_task + (λ/2) Σ_i F_i(θ_i - θ*_i)²

where:
  F_i = Fisher Information (importance of weight i)
  θ* = optimal weights from previous task
  λ = regularization strength
```

**Test Evidence**: [test_continual_meta.py:test_ewc](../nsck-demo/tests/unit/learning/test_continual_meta.py)
- Without EWC: 47% forgetting (catastrophic) ❌
- With EWC: 8% forgetting (acceptable) ✅

#### 4. Rule Learning & Tenure

**Formula**:
```
success_rate = successes / total_observations

Rule valid if:
  1. total ≥ min_support (default: 5)
  2. success_rate ≥ {0.50 if total < 1000 else 0.60}
```

**Test Evidence**: [test_capability_proofs.py](../nsck-demo/tests/integration/test_capability_proofs.py)
- Rule: {FOOD_ABOVE}→ACTION_UP with 85% success ✅
- Promoted to TENURED status (support=100, rate=0.85) ✅

#### 5. Global Workspace Competition

**Formula** (adapted from LIDA):
```
Activation = salience + relevance + affect_match + 0.5·sender_confidence
            + 0.2 (if mission-aligned)

Winner = argmax(Activation) if max > threshold else None
```

**Test Evidence**: [test_global_workspace.py](../nsck-demo/tests/unit/integration_core/test_global_workspace.py), [test_phase8_mental_rehearsal.py](../nsck-demo/tests/integration/test_phase8_mental_rehearsal.py)
- Coalition A: activation=2.35 wins over B: activation=1.55 ✅
- Broadcast successful to all modules ✅
- Danger veto blocks harmful actions, selects next-best alternative ✅
- Deadlock fallback triggers EMERGENCY response ✅

#### 6. Johnson-Lindenstrauss Sparse Projection

**Theorem** (Johnson & Lindenstrauss, 1984):
```
High-dimensional space (D=10,240) → Low dimension (d=128)
Preserves pairwise distances within factor (1±ε)

NSCK uses sparse projection (90% zeros):
  z = Φ·s  where Φ ∈ {-1,0,+1}^(128×10240)
  
Speedup: 10× reduction in matrix operations
Memory: 75× fewer parameters in world model
```

**Test Evidence**: [test_world_model.py](../nsck-demo/tests/unit/neural/test_world_model.py)
- Sparse projection: 0.05ms vs dense: 0.50ms (10× faster) ✅
- Prediction accuracy: 89% correct direction ✅

### Performance Complexity

| Operation | Time | Space | Implementation |
|-----------|------|-------|----------------|
| VSA XOR | O(D) | O(1) | Bitwise operation |
| VSA Bundle | O(nD) | O(D) | n vectors, majority vote |
| Rule Match | O(R·P) | O(R) | R rules, P predicates |
| Causal Chain | O(E) | O(V) | BFS graph traversal |
| Gradient Surgery | O(T²·P) | O(T·P) | T tasks, P params |
| World Model | O(H) | O(H) | H=128 hidden dim |
| GW Competition | O(C) | O(C) | C coalitions (<10) |

**Key Design Principle**: Avoid O(n³) matrix operations typical of deep learning. All core operations are O(n) or O(n²) in practice.

---

## Monitoring & Testing Interfaces

NSCK includes three specialized web dashboards for real-time system monitoring, testing, and operational control:

### web_dashboard.py (Main Operational Interface)
- **Purpose**: Primary mission control using Flask-SocketIO
- **Port**: 5000
- **Features**: Process management, ZMQ telemetry, AGI report generation
- **Status**: CRITICAL - Production operational dashboard

### testing_dashboard.py (Capability Testing)
- **Purpose**: Comprehensive testing interface for all cognitive capabilities  
- **Port**: 5051
- **Features**: Chat, game simulations (Snake/Pong/Maze), teacher/student modes, log export
- **Status**: PRIMARY testing interface

### cognitive_dashboard.py (Knowledge Inspection)
- **Purpose**: Cognitive system monitoring and knowledge inspection
- **Port**: 5050
- **Features**: Multimodal input, reasoning traces, knowledge queries, export (JSON/ZIP)
- **Status**: Alternative monitoring interface

---

## References

1. **Baars (1988)**: *A Cognitive Theory of Consciousness* - Global Workspace Theory
2. **Bellec et al. (2018)**: "Deep Rewiring: Training very sparse deep networks" - Structural plasticity
3. **Cheng & Novick (1992)**: "Covariation in natural causal induction" - Delta-P formula
4. **Gentner (1983)**: "Structure-mapping" - Analogical transfer theory
5. **Harnad (1990)**: "The symbol grounding problem" - Connecting symbols to sensory experience
6. **Johnson & Lindenstrauss (1984)**: "Extensions of Lipschitz mappings" - Random projection
7. **Kanerva (2009)**: "Hyperdimensional Computing" - VSA foundations
8. **Kirkpatrick et al. (2017)**: "Overcoming catastrophic forgetting" - EWC algorithm
9. **Pearl (2009)**: *Causality* - Causal reasoning framework
10. **Yu et al. (2020)**: "Gradient Surgery for Multi-Task Learning" - PCGrad

---

## Testing Dashboard

The **Testing Dashboard** (`testing_dashboard.py`) is a self-contained Flask web application
that exposes the full cognitive system for interactive testing and monitoring.

**Capabilities:**
- Chat with the cognitive system using text samples
- Launch Snake, Pong, and Maze simulations (simultaneously or sequentially)
- Monitor emotion blend, self-model performance, and knowledge base in real time
- Export complete logs, reasoning traces, and system state to TXT or JSON

See [TESTING_DASHBOARD.md](TESTING_DASHBOARD.md) for full documentation and API reference.

---

## Dependencies

- **Core**: numpy, scipy, scikit-learn, pydantic, rustworkx, networkx, rich
- **Neural**: torch (CPU-only)
- **LLM**: llama-cpp-python (optional, for Phi3 integration)
- **Testing**: pytest
- **Dashboard**: flask
- **Optional**: snntorch (SNN training)
