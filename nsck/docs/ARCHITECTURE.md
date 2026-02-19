# NSCK Architecture

Complete system architecture of the Neural-Symbolic Cognitive Kernel. All diagrams reflect the actual codebase — every arrow is a verified import or method call.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Subsystem Architecture](#subsystem-architecture)
  - [VSA Foundation](#1-vsa-foundation)
  - [Reasoning Layer](#2-reasoning-layer)
  - [Memory Systems](#3-memory-systems)
  - [Perception Layer](#4-perception-layer)
  - [Learning Subsystem](#5-learning-subsystem)
  - [Cognitive Layer](#6-cognitive-layer)
  - [Language Layer](#7-language-layer)
  - [Integration Layer](#8-integration-layer)
- [Central Orchestrator](#central-orchestrator)
- [GWT Broadcast Network](#gwt-broadcast-network)
- [Data Flow Summary](#data-flow-summary)
- [Directory Layout](#directory-layout)

---

## High-Level Overview

NSCK is organized into 8 subsystems, all orchestrated by the `CognitiveEngine`:

```mermaid
graph TB
    subgraph Foundation["1. VSA Foundation"]
        HV["HyperVector Engine<br/>hypervec_py.py (361 LOC)<br/>hypervec_shim.py (320 LOC)<br/>+ Rust accelerator (2,665 LOC)"]
    end

    subgraph Reasoning["2. Reasoning"]
        CE["CognitiveEngine<br/>cognitive_engine.py<br/>1,188 LOC"]
        GWT["GlobalWorkspace<br/>global_workspace.py<br/>312 LOC"]
        Plan["STRIPSPlanner<br/>planner.py<br/>296 LOC"]
        Causal["CausalDiscovery + CausalGraph<br/>causal_reasoning.py<br/>954 LOC"]
        RL["RuleLearner<br/>rule_learner.py<br/>624 LOC"]
        Ana["AnalogyEngine<br/>analogy.py<br/>528 LOC"]
        Ctx["ContextEngine<br/>context_engine.py<br/>440 LOC"]
    end

    subgraph Memory["3. Memory"]
        Epi["EpisodicMemory<br/>episodic_memory.py<br/>398 LOC"]
        Sem["SemanticMemory<br/>semantic_memory.py<br/>217 LOC"]
        Staged["StagedRecall<br/>staged_recall.py<br/>134 LOC"]
    end

    subgraph Perception["4. Perception"]
        SNN["SNNPerceptionModule<br/>snn_perception.py<br/>561 LOC"]
        Bridge["VSA-SNN Bridge<br/>vsa_snn_bridge.py<br/>461 LOC"]
        Ground["GroundingVerifier<br/>grounding_verifier.py<br/>585 LOC"]
        SNNInt["SNNWorkspaceAdapter<br/>snn_integration.py<br/>240 LOC"]
    end

    subgraph Learning["5. Learning"]
        Hebb["HebbianMatrix + VSAHebbianLearner<br/>hebbian.py<br/>506 LOC"]
        Cur["CuriosityModule<br/>curiosity.py<br/>336 LOC"]
    end

    subgraph Cognitive["6. Cognitive"]
        Meta["SafetyGate + MetacognitiveEngine<br/>metacognition.py<br/>504 LOC"]
        Self["SelfModel<br/>self_model.py<br/>255 LOC"]
        ToM["TheoryOfMind<br/>theory_of_mind.py<br/>312 LOC"]
        Emo["EmotionSystem<br/>emotion_system.py<br/>420 LOC"]
    end

    subgraph Lang["7. Language"]
        LM["LanguageModule<br/>language_module.py<br/>484 LOC"]
        DM["DialogueManager<br/>dialogue_manager.py<br/>145 LOC"]
        TKL["TextKnowledgeLearner<br/>text_knowledge_learner.py<br/>1,071 LOC"]
        UI["UniversalInput<br/>universal_input.py<br/>947 LOC"]
        LC["LinguaCortex<br/>lingua_cortex.py<br/>260 LOC"]
    end

    subgraph Integration["8. Integration"]
        Conf["NSCKConfig<br/>config.py<br/>73 LOC"]
        Persist["BrainStore<br/>persistence.py<br/>852 LOC"]
        Fusion["BrainFusion<br/>brain_fusion.py<br/>481 LOC"]
        Explain["ExplanationGenerator<br/>explanation.py<br/>433 LOC"]
    end

    HV --> CE
    CE --> GWT
    CE --> Plan
    CE --> Causal
    CE --> RL
    CE --> Ana
    CE --> Epi
    CE --> Sem
    CE --> Ground
    CE --> Cur
    CE --> Self
    CE --> Meta
    CE --> LM
    CE --> DM
    CE --> Explain
    CE --> Persist
```

---

## Subsystem Architecture

### 1. VSA Foundation

All modules encode knowledge as 10,240-bit binary hypervectors. The VSA layer provides three core operations:

```mermaid
graph LR
    subgraph Operations
        Bind["Bind (XOR)<br/>A ⊗ B = A ⊕ B<br/>Creates role-filler pairs"]
        Bundle["Bundle (Majority)<br/>A + B → superposition<br/>Creates sets"]
        Permute["Permute (Shift)<br/>ρᵏ(v) = circular shift<br/>Creates sequences"]
    end

    subgraph Implementation
        Py["hypervec_py.py<br/>Pure Python + NumPy<br/>HyperVectorPy class"]
        Shim["hypervec_shim.py<br/>Backend selector"]
        Rust["rust_vsa/<br/>7 Rust source files<br/>PyO3 bindings"]
    end

    Shim -->|"try import"| Rust
    Shim -->|"fallback"| Py
    Bind --> Shim
    Bundle --> Shim
    Permute --> Shim
```

**Rust Accelerator** (`rust_vsa/src/`):

| File | Lines | What It Provides |
|------|-------|-----------------|
| `lib.rs` | 280 | HyperVector type + PyO3 bindings |
| `semantic.rs` | 444 | SemanticMemoryConcurrent (thread-safe) |
| `episodic.rs` | 427 | EpisodicMemoryConcurrent (thread-safe) |
| `worker_pool.rs` | 488 | CognitiveWorkerPool (parallel ops) |
| `concurrent.rs` | 319 | Lock-free concurrent data structures |
| `persistence.rs` | 416 | PersistentStorage (disk I/O) |
| `async_runtime.rs` | 291 | AsyncCognitiveRuntime (tokio) |

---

### 2. Reasoning Layer

The reasoning layer is where decisions happen. All proposals compete in the **Global Workspace**:

```mermaid
graph TD
    subgraph Sources["Coalition Sources"]
        S1["EXTERNAL<br/>(from registered task)"]
        S2["RULES<br/>(from RuleLearner)"]
        S3["EXPLORATION<br/>(from CuriosityModule)"]
        S4["Q_LEARNING<br/>(from tabular Q-values)"]
        S5["MEMORY<br/>(from EpisodicMemory recall)"]
        S6["PLANNER<br/>(from STRIPSPlanner)"]
    end

    GWT["GlobalWorkspace.compete()<br/>Multi-factor scoring:<br/>salience + relevance +<br/>affect + 0.5×confidence +<br/>mission_bias"]
    
    Safety["SafetyGate.is_safe()<br/>Danger veto check"]
    
    Winner["Winning Action<br/>+ Explanation"]

    S1 --> GWT
    S2 --> GWT
    S3 --> GWT
    S4 --> GWT
    S5 --> GWT
    S6 --> GWT
    GWT -->|"threshold ≥ 0.5"| Safety
    Safety -->|"safe"| Winner
    Safety -->|"vetoed"| GWT
```

**Planner ↔ Causal Discovery Integration:**

```mermaid
graph LR
    Obs["Observations<br/>(state, action, effect)"]
    CD["CausalDiscovery<br/>ΔP statistics"]
    CG["CausalGraph<br/>forward/backward links"]
    Plan["STRIPSPlanner<br/>A* search"]
    CR["CausalReasoner<br/>forward chaining"]

    Obs -->|"observe()"| CD
    CD -->|"induce_graph()"| CG
    CG -->|"learn_operators_from_graph()"| Plan
    CG -->|"CausalReasoner(graph)"| CR
    Plan -->|"plan(init, goal)"| Actions["Action Sequence"]
```

---

### 3. Memory Systems

```mermaid
graph TD
    subgraph Episodic["Episodic Memory"]
        Hot["Hot Tier<br/>collections.deque<br/>Recent episodes in RAM"]
        LSH["LSH Index<br/>4 tables × 10-bit hashes<br/>Fast approximate lookup"]
        Warm["Warm Tier<br/>SQLite via BrainStore<br/>Consolidated episodes"]
    end

    subgraph Semantic["Semantic Memory"]
        Graph["NetworkX DiGraph<br/>Concepts as nodes<br/>Relations as edges"]
        HVIdx["VSA Index<br/>concept_name → HyperVector<br/>Cosine similarity search"]
    end

    subgraph Staged["Staged Recall"]
        L0["L0: Working Memory (LRU)"]
        L2["L2: LSH Index"]
        L3["L3: Brute-force scan"]
    end

    Hot -->|"consolidate<br/>at 500 episodes"| Warm
    Hot --> LSH
    Graph --> HVIdx
    L0 --> L2 --> L3
```

**Episodic Memory Flow:** New episodes enter the hot deque → indexed by LSH → when count exceeds 500, older half is consolidated to SQLite. Retrieval: query HV → LSH bucket match → 1-bit neighbor probing → exact similarity ranking → top-k results.

**Semantic Memory:** Dual representation — NetworkX graph for structural queries (inheritance, spreading activation) and VSA index for similarity-based retrieval. `spread_activation(concepts, steps, decay)` performs iterative activation through the graph.

---

### 4. Perception Layer

```mermaid
graph LR
    Input["Sensory Input<br/>(numpy array)"]
    LIF["LIF Neuron Layer<br/>τ_m = 20ms, V_th = 1.0<br/>V_reset = 0.0"]
    STDP["STDP Plasticity<br/>A+ = 0.01, A- = 0.012<br/>τ+ = 20ms, τ- = 20ms"]
    Spikes["Spike Train"]
    Rate["RateCoder<br/>rate → bundle of neuron HVs"]
    Temp["TemporalCoder<br/>spike timing → permuted HVs"]
    HVout["Concept HyperVector"]
    Hebb["VSAHebbianLearner<br/>Association formation"]
    Cleanup["ConceptMapper<br/>Pattern → known concept"]

    Input --> LIF
    LIF -->|"spikes"| STDP
    STDP -->|"weight update"| LIF
    LIF --> Spikes
    Spikes --> Rate --> HVout
    Spikes --> Temp --> HVout
    HVout --> Hebb
    HVout --> Cleanup
```

The SNN layer uses **Leaky Integrate-and-Fire** neurons with **STDP** (Spike-Timing-Dependent Plasticity). The `vsa_snn_bridge.py` module provides bidirectional conversion between spike trains and HyperVectors via rate coding or temporal coding.

---

### 5. Learning Subsystem

Two complementary learning systems:

| System | File | Mechanism | Purpose |
|--------|------|-----------|---------|
| **Hebbian** | `hebbian.py` | Oja's rule + reward modulation + eligibility traces | Association learning, weight adaptation |
| **Curiosity** | `curiosity.py` | VSA novelty detection + learning progress tracking | Exploration vs. exploitation balance |

**Hebbian Learning:** Three-factor rule: `ΔW = α × pre × post × (reward - baseline)`. Eligibility traces (`λ=0.9`) allow credit assignment across time delays.

**Curiosity Module:** Computes novelty as `1 - max_similarity(query, visited)`. When novelty exceeds threshold (0.5), triggers random exploration. Learning progress is tracked per-task to drive sustained curiosity in productive areas.

---

### 6. Cognitive Layer

```mermaid
graph TD
    subgraph Metacognition
        SG["SafetyGate<br/>Static safety check"]
        ME["MetacognitiveEngine<br/>Confidence monitoring<br/>Conflict detection<br/>Safety veto"]
    end

    subgraph SelfModel
        SM["SelfModel<br/>Per-task confidence tracking<br/>Calibration error (CE)<br/>Improvement trend"]
    end

    subgraph ToM
        TOM["TheoryOfMind<br/>Per-agent mental models<br/>False belief detection<br/>Action prediction"]
    end

    subgraph Emotion
        ES["EmotionSystem<br/>Valence-arousal model<br/>Plutchik 8 emotions<br/>Inverse-distance blending"]
    end
```

**Active in decide() loop:** SafetyGate (veto), SelfModel (confidence).  
**Instantiated but not currently wired into decide():** TheoryOfMind, EmotionSystem.

---

### 7. Language Layer

```mermaid
graph TD
    Text["Raw Text Input"]
    
    LM["LanguageModule<br/>Dual-mode NLU:<br/>1. LLM (llama_cpp) if available<br/>2. Rule-based NLU (300 lines)<br/>   POS tagging → chunking →<br/>   semantic frames"]
    
    DM["DialogueManager<br/>Intent routing:<br/>goal → set_mission_goal()<br/>status → get_stats()<br/>explain → explain()"]
    
    TKL["TextKnowledgeLearner<br/>1,071 LOC pipeline:<br/>Relation extraction →<br/>Semantic folding →<br/>Causal graph building"]
    
    UI["UniversalInput<br/>4-component VSA encoder:<br/>keyword (50%) +<br/>char n-gram (15%) +<br/>word-order (15%) +<br/>phrase-structure (20%)"]
    
    LC["LinguaCortex<br/>Semantic fingerprints<br/>128×128 SDR grid"]
    
    Text --> LM
    LM --> DM
    DM -->|"intent routing"| CE["CognitiveEngine"]
    Text --> TKL
    TKL -->|"facts"| Sem["SemanticMemory"]
    TKL -->|"episodes"| Epi["EpisodicMemory"]
    TKL -->|"causes"| Causal["CausalGraph"]
    Text --> UI -->|"HyperVector"| HV["10,240-bit HV"]
    LM --> LC
```

---

### 8. Integration Layer

| Module | File | LOC | Purpose |
|--------|------|-----|---------|
| **NSCKConfig** | `config.py` | 73 | Hyperparameter dataclass |
| **BrainStore** | `persistence.py` | 852 | SQLite persistence for rules, episodes, concepts |
| **BrainFusion** | `brain_fusion.py` | 481 | Multi-task brain management + rule resolution |
| **ExplanationGenerator** | `explanation.py` | 433 | Natural language explanation templates |
| **KnowledgeIntegration** | `knowledge_integration.py` | 530 | Alternative pipeline connecting perception → memory → reasoning |

---

## Central Orchestrator

`CognitiveEngine` is the hub that instantiates and connects all modules:

```mermaid
graph TD
    CE["CognitiveEngine.__init__()"]
    
    CE -->|"instantiates"| RL["RuleLearner"]
    CE -->|"instantiates"| Epi["EpisodicMemory"]
    CE -->|"instantiates"| Sem["SemanticMemory"]
    CE -->|"instantiates"| Cur["CuriosityModule"]
    CE -->|"instantiates"| SM["SelfModel"]
    CE -->|"instantiates"| ToM["TheoryOfMind"]
    CE -->|"instantiates"| GWT["GlobalWorkspace"]
    CE -->|"instantiates"| CD["CausalDiscovery"]
    CE -->|"instantiates"| Plan["STRIPSPlanner"]
    CE -->|"instantiates"| Ana["AnalogyEngine"]
    CE -->|"instantiates"| Exp["ExplanationGenerator"]
    CE -->|"instantiates"| BF["BrainFusion"]
    CE -->|"instantiates"| UI["UniversalInput"]
    CE -->|"instantiates"| LM["LanguageModule"]
    CE -->|"instantiates"| DM["DialogueManager"]
    CE -->|"instantiates"| BS["BrainStore"]
    
    RL -->|"verifier="| GV["GroundingVerifier"]
    RL -->|"store="| BS
    DM -->|"engine="| CE
    DM -->|"language="| LM
```

---

## GWT Broadcast Network

When the Global Workspace selects a winning coalition, it **broadcasts** to all registered `WorkspaceModule` subscribers:

```mermaid
sequenceDiagram
    participant CE as CognitiveEngine
    participant GWT as GlobalWorkspace
    participant RL as RuleLearner
    participant EpiA as EpisodicAdapter
    participant SemA as SemanticAdapter
    participant SNN as SNNAdapter

    CE->>GWT: compete(coalitions)
    GWT->>GWT: rank by activation score
    GWT->>GWT: apply attention threshold (≥0.5)
    GWT-->>RL: broadcast(winner_content)
    GWT-->>EpiA: broadcast(winner_content)
    GWT-->>SemA: broadcast(winner_content)
    GWT-->>SNN: broadcast(winner_content)
    GWT-->>CE: return winner Coalition
```

Registered subscribers:
- **RuleLearner** — observes broadcast for rule induction
- **EpisodicBroadcastAdapter** — updates episode statistics
- **SemanticBroadcastAdapter** — triggers spreading activation
- **SNNWorkspaceAdapter** — perceptual priming (when SNN is registered)

---

## Data Flow Summary

```mermaid
graph LR
    Input["Input State"]
    
    GP["GroundingVerifier<br/>get_active_predicates()"]
    SH["EpisodicMemory<br/>create_situation_hv()"]
    CU["CuriosityModule<br/>should_explore()"]
    
    C1["Coalition: EXTERNAL"]
    C2["Coalition: RULES"]
    C3["Coalition: EXPLORATION"]
    C4["Coalition: Q_LEARNING"]
    C5["Coalition: MEMORY"]
    C6["Coalition: PLANNER"]
    
    GWT["GWT compete()"]
    SG["SafetyGate"]
    
    RO["record_outcome()<br/>TD(0) Q-update"]
    RL["rule_learner.observe()"]
    EM["episodic_memory.record()"]
    CD["causal_discovery.observe()"]
    
    Output["Action + Explanation"]
    
    Input --> GP --> SH --> CU
    CU --> C1
    CU --> C2
    CU --> C3
    CU --> C4
    CU --> C5
    CU --> C6
    C1 --> GWT
    C2 --> GWT
    C3 --> GWT
    C4 --> GWT
    C5 --> GWT
    C6 --> GWT
    GWT --> SG --> Output
    Output --> RO
    Output --> RL
    Output --> EM
    Output --> CD
```

---

## Directory Layout

```
nsck/python/core/
├── vsa/                          # VSA Foundation
│   ├── hypervec_py.py            # Pure Python HyperVector (361 LOC)
│   └── hypervec_shim.py          # Rust/Python backend selector (320 LOC)
│
├── reasoning/                    # Reasoning Layer
│   ├── cognitive_engine.py       # Central orchestrator (1,188 LOC)
│   ├── global_workspace.py       # GWT coalition competition (312 LOC)
│   ├── planner.py                # A* STRIPS planner (296 LOC)
│   ├── causal_reasoning.py       # ΔP causal discovery + graph (954 LOC)
│   ├── rule_learner.py           # Rule induction from experience (624 LOC)
│   ├── analogy.py                # Analogical reasoning + transfer (528 LOC)
│   └── context_engine.py         # Polysemy disambiguation (440 LOC)
│
├── learning/                     # Learning Subsystem
│   ├── hebbian.py                # Oja's rule + reward modulation (506 LOC)
│   └── curiosity.py              # VSA novelty + exploration (336 LOC)
│
├── cognitive/                    # Cognitive Layer
│   ├── metacognition.py          # Safety gate + confidence (504 LOC)
│   ├── self_model.py             # Per-task calibration (255 LOC)
│   ├── theory_of_mind.py         # Agent mental models (312 LOC)
│   └── emotion_system.py         # Valence-arousal + Plutchik (420 LOC)
│
├── perception/                   # Perception Layer
│   ├── snn_perception.py         # LIF neurons + STDP (561 LOC)
│   ├── snn_integration.py        # GWT adapter for SNN (240 LOC)
│   ├── vsa_snn_bridge.py         # Spike ↔ HyperVector conversion (461 LOC)
│   ├── grounding_verifier.py     # Predicate grounding (585 LOC)
│   └── symbol_grounding.py       # Abstract symbol grounding (196 LOC)
│
├── memory/                       # Memory Systems
│   ├── episodic_memory.py        # Two-tier + LSH indexing (398 LOC)
│   ├── semantic_memory.py        # Graph + VSA dual repr (217 LOC)
│   └── staged_recall.py          # Multi-level retrieval (134 LOC)
│
├── language/                     # Language Layer
│   ├── language_module.py        # Dual-mode NLU/NLG (484 LOC)
│   ├── dialogue_manager.py       # Intent routing (145 LOC)
│   ├── text_knowledge_learner.py # Text → knowledge pipeline (1,071 LOC)
│   ├── universal_input.py        # 4-component VSA encoder (947 LOC)
│   └── lingua_cortex.py          # Semantic fingerprints (260 LOC)
│
├── integration/                  # Integration Layer
│   ├── config.py                 # Hyperparameters (73 LOC)
│   ├── persistence.py            # SQLite brain store (852 LOC)
│   ├── brain_fusion.py           # Multi-task management (481 LOC)
│   ├── explanation.py            # NL explanation templates (433 LOC)
│   └── knowledge_integration.py  # Alternative pipeline (530 LOC)
│
└── training/                     # Training
    ├── snn_training.py           # SNN training pipeline (589 LOC)
    └── snn_benchmarks.py         # SNN benchmark suite (483 LOC)
```
