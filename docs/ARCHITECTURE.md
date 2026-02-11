# NSCK System Architecture

## Overview

NSCK (Neuro-Symbolic Cognitive Kernel) is a cognitive architecture that combines Vector Symbolic Architecture (VSA), symbolic reasoning, neural networks, and analogical transfer learning into a unified system. It is designed for CPU-only operation with energy efficiency as a core constraint.

```
┌─────────────────────────────────────────────────────────────┐
│                   Language Module (Phi3)                     │
│              [Peripheral Translator - NL ↔ VSA]             │
├─────────────────────────────────────────────────────────────┤
│                    Cognitive Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Perception  │  │   Decision   │  │    Learning      │  │
│  │   (Phase 2)   │  │   (GW/Rules) │  │   (Phase 1+3)   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌────────┴─────────┐  │
│  │  World Model  │  │  Analogy &   │  │ Knowledge Store  │  │
│  │  (Phase 4)    │  │  Transfer    │  │ (Cross-session)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Self-Model   │  │   Emotion    │  │  Theory of Mind  │  │
│  │  (Phase 5)    │  │  (Phase 6)   │  │   (Phase 6)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 VSA Core (10,240-bit HVs)                   │
│          [XOR binding, bundling, similarity search]         │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. VSA Core (`hypervec_shim.py`, `hypervec_py.py`)

**What**: 10,240-bit binary hypervectors provide the fundamental representation layer.

**How**: All concepts, states, actions, and memories are encoded as binary vectors.

**Operations** (see [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md) for full derivations):

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

**What**: Central orchestrator that integrates all cognitive subsystems.

**How**: The `decide()` method runs a competition among multiple knowledge sources:
1. **Perception** → Extract active predicates from state
2. **Proposal Generation** → Multiple systems propose actions:
   - SNN (fast neural system)
   - Rules (symbolic system, including global rules)
   - Planner (goal-directed STRIPS planning)
   - Active Inference (curiosity-driven)
   - Imagination (world model simulation)
3. **Global Workspace Competition** → Coalitions compete for "consciousness"
4. **Winner Selection** → Highest-activation proposal wins
5. **Value Alignment** → Safety check on chosen action
6. **Learning** → Update episodic memory, rules, self-model

**Why**: This architecture mirrors Global Workspace Theory (Baars, 1988), where specialized modules compete for access to a shared workspace, enabling flexible and context-sensitive decision-making.

### 3. Transfer Learning System

#### 3.1 Analogy Engine (`analogy.py`)

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

#### 3.2 Knowledge Store (`train_phase7_demo.py: KnowledgeStore`)

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

#### 3.3 Global Rules (cognitive_engine.py + rule_learner.py)

**What**: Domain-independent safety and behavioral rules.

**How**: `get_applicable_rules()` checks both task-local and global rule sets. Rules with domain-independent predicates (e.g., `DANGER_UP`, `TARGET_NEAR`) automatically apply across all tasks.

**Why**: Some knowledge is truly universal — "avoid danger" applies regardless of the specific domain.

### 4. Language Module (`language_module.py`)

**What**: LLM-based peripheral translator (Phi3 via llama-cpp-python).

**How**: 
- **Understanding**: NL text → structured intent/entities → VSA hypervectors
- **Generation**: System state → natural language explanation
- **Mock mode**: Template-based fallback when LLM model not available

**Critical Constraint**: The LLM is a **peripheral** — it translates but does NOT make decisions. Core cognition remains symbolic/VSA-based.

**Why**: Natural language makes the system's internal reasoning transparent and interpretable without compromising the deterministic symbolic core.

### 5. Continual Learning (`continual_learning.py`)

**What**: Prevents catastrophic forgetting when learning new tasks.

**How**: Four complementary strategies:
- **EWC** (Elastic Weight Consolidation): Protects important weights via Fisher Information
- **Progressive Networks**: New task-specific columns with lateral connections
- **PackNet**: Pruning and capacity allocation per task
- **Memory Replay**: Experience buffer for interleaved training

**Why**: Without continual learning, training on Task B would destroy Task A knowledge. EWC alone achieves 7.3% improvement in knowledge retention.

### 6. Self-Model & Metacognition (Phase 5)

**What**: Self-awareness, performance prediction, and autonomous improvement.

**How**:
- `SelfModel`: Tracks per-task confidence, predicts success probability
- `MetacognitiveEngine`: Detects conflicts between competing systems
- `SelfExplainer`: Generates transparent reasoning traces
- `SelfImprover`: Adjusts learning rates based on performance gaps

### 7. Social & Emotional Intelligence (Phase 6)

**What**: Emotion processing and Theory of Mind.

**How**:
- `EmotionSystem`: Generates emotions from drives and rewards (Plutchik + Russell models)
- `TheoryOfMind`: Maintains mental models of other agents, detects false beliefs
- Sally-Anne test passing (first-order false belief detection)

### 8. Temporal Encoding & Advanced Deliberation (Phase 8)

**What**: Three interrelated capabilities that bring the system closer to human-like cognition:

**8.1 Temporal Permutation** (`lib.rs`, `hypervec_py.py`, `hypervec_shim.py`)
- `permute(shift)`: Circular bitwise rotation of 10,240-bit HVs
- Enables sequence encoding: `A ⊕ ρ¹(B) ⊕ ρ²(C)` preserves order
- Inverse property: `permute(n).permute_inverse(n) ≈ identity` (>99% similarity)

**8.2 Universal Input Layer** (`universal_input.py`)
-   **Scalars**: Thermometer encoding (nearby values → similar HVs)
-   **Categories**: Deterministic codebook with LRU eviction (max 10K entries)
-   **Dicts**: Recursive role-filler binding (Role_HV ⊗ Value_HV, then bundle)
-   **Lists**: Permutation-based sequence encoding

**8.3 Mental Rehearsal & Veto** (`global_workspace.py`)
-   `compete_with_rehearsal()`: Simulates action outcomes via WorldModel before committing
-   Danger vector registry: States at catastrophic outcomes form a veto set
-   Veto threshold: predicted states with >75% similarity to danger vectors are blocked
-   Deadlock fallback: EMERGENCY ACTION_STAY when all proposals vetoed

**Why**: Temporal reasoning enables trajectory planning. Universal input maps heterogeneous sensor data into a shared algebraic space. Mental rehearsal prevents known-dangerous actions without explicit rules, mimicking human "hesitation" before risky choices.

---

## Data Flow

### Decision Cycle

```
State → Verifier → Active Predicates
                        │
     ┌──────────────────┼──────────────────────┐
     ↓                  ↓                      ↓
  SNN Fast           Rules+Global          Planner
  System             System                STRIPS
     │                  │                      │
     └──────────┬───────┴──────────────────────┘
                ↓
        Global Workspace Competition
                ↓
        Winner → Value Alignment Check
                ↓
        Action → Learn from Outcome
```

### Transfer Learning Pipeline

```
Domain A Experiences → KnowledgeStore.store_experience()
                              ↓
Domain B Experiences → KnowledgeStore.store_experience()
                              ↓
              KnowledgeStore.consolidate_to_abstract()
                     [Lift → Pattern → Promote]
                              ↓
                      Abstract Rules
                              ↓
Novel Domain C → find_relevant_experience()
                     [Match abstract patterns]
                              ↓
                   Recommended Action
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
| `cognitive_engine.py` | ~850 | Central orchestrator |
| `analogy.py` | ~415 | Cross-domain transfer via structural alignment |
| `rule_learner.py` | ~400 | Frequency-based symbolic rule induction |
| `continual_learning.py` | ~400 | EWC, Progressive Networks, PackNet, Replay |
| `multi_task_learning.py` | ~480 | Shared encoder + task heads + gradient surgery |
| `world_model.py` | ~240 | Forward simulation with sparse projection |
| `episodic_memory.py` | ~200 | VSA-based experience storage |
| `semantic_memory.py` | ~120 | Concept graph with spreading activation |
| `emotion_system.py` | ~145 | Plutchik+Russell emotion model |
| `theory_of_mind.py` | ~115 | Agent mental modeling + false belief |
| `self_model.py` | ~150 | Performance tracking + confidence |
| `metacognition.py` | ~150 | Self-monitoring + conflict detection |
| `language_module.py` | ~120 | Phi3 LLM translator peripheral |
| `train_phase7_demo.py` | ~500 | Integrated system + KnowledgeStore |
| `testing_dashboard.py` | ~900 | Comprehensive web testing dashboard |

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

**Test Evidence**: `test_causal_discovery.py`
- Strong causality: SWITCH_ON→LIGHT_ON detected with ΔP=1.0 ✅
- Spurious rejection: CLAP→BIRD_CHIRPS rejected with ΔP=0.0 ✅

#### 2. Gradient Surgery (PCGrad)

**Formula** (Yu et al., 2020):
```
For conflicting task gradients (cosine similarity < 0):
  g_i_projected = g_i - Σ_j max(0, g_i·g_j) · g_j / ||g_j||²

Result: Eliminates negative transfer between tasks
```

**Test Evidence**: `test_phase1.py::test_gradient_surgery`
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

**Test Evidence**: `test_continual_meta.py::test_ewc`
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

**Test Evidence**: `test_capability_proofs.py`
- Rule: {FOOD_ABOVE}→ACTION_UP with 85% success ✅
- Promoted to TENURED status (support=100, rate=0.85) ✅

#### 5. Global Workspace Competition

**Formula** (adapted from LIDA):
```
Activation = salience + relevance + affect_match + 0.5·sender_confidence
            + 0.2 (if mission-aligned)

Winner = argmax(Activation) if max > threshold else None
```

**Test Evidence**: `test_global_workspace.py`, `test_phase8_mental_rehearsal.py`
- Coalition A: activation=2.35 wins over B: activation=1.55 ✅
- Broadcast successful to all modules ✅
- Phase 8: Danger veto blocks harmful actions, selects next-best alternative ✅
- Phase 8: Deadlock fallback triggers EMERGENCY response ✅

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

**Test Evidence**: `test_world_model.py`
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
2. **Cheng & Novick (1992)**: "Covariation in natural causal induction" - Delta-P formula
3. **Gentner (1983)**: "Structure-mapping" - Analogical transfer theory
4. **Johnson & Lindenstrauss (1984)**: "Extensions of Lipschitz mappings" - Random projection
5. **Kanerva (2009)**: "Hyperdimensional Computing" - VSA foundations
6. **Kirkpatrick et al. (2017)**: "Overcoming catastrophic forgetting" - EWC algorithm
7. **Pearl (2009)**: *Causality* - Causal reasoning framework
8. **Yu et al. (2020)**: "Gradient Surgery for Multi-Task Learning" - PCGrad

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
