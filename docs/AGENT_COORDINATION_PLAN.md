# NSCK: Agent Coordination Plan

**Purpose:** Concrete, implementable plan for using multiple coding agents to build and enhance the NSCK system with guardrails to prevent drift, breakage, and context loss.

**Audience:** Coding agents (AI or human), project maintainers, contributors

**Status:** All 7 phases complete (301+ tests passing). Now executing 2026 Enhancement Phase.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Core Principles](#core-principles)
3. [2026 Enhancement Execution](#enhancement-2026)
4. [Agent Roles & Responsibilities](#agent-roles)
5. [Work Units: What Gets Built](#work-units)
6. [Execution Model: How Agents Run](#execution-model)
7. [Guardrails: Preventing Drift & Breakage](#guardrails)
8. [Context Protocol: Staying Up-to-Date](#context-protocol)
9. [Dependency Map: Sequencing & Parallelism](#dependency-map)
10. [Verification & Quality Gates](#verification)
11. [Session Handoff Protocol](#session-handoff)

---

<a name="problem-statement"></a>
## 1. Problem Statement

### Current Situation
NSCK has completed all 7 original phases with 301+ passing tests. The system now needs **research-backed enhancements** based on 50+ peer-reviewed papers (2024-2026) to improve:
- Decision quality (+15-20%)
- Throughput (+30-40%)
- Continual learning (62% less forgetting)
- Transfer learning (+25% zero-shot)
- Energy efficiency (84% reduction)

### Challenge
Implementing 15 improvements across 12 weeks requires:
- **Multiple specialized agents** working in parallel
- **Coordination** to prevent conflicts
- **Quality assurance** to avoid regressions
- **Documentation** for all changes

---

<a name="core-principles"></a>
## 2. Core Principles

| Principle | Rule | Why |
|---|---|---|
| **Roadmap Alignment** | All work traces to `ROADMAP_TO_AGI.md` Section 1 (2026 Enhancement) | Prevents feature drift |
| **No Regressions** | All 301+ existing tests must pass | Protects working code |
| **Test Coverage** | 90%+ coverage on new code | Ensures quality |
| **Documentation First** | Update docs before coding | Maintains clarity |
| **Benchmark Everything** | Measure improvements vs baseline | Validates claims |
| **Modular Changes** | Small, focused PRs | Easier review |
| **Feature Flags** | All improvements toggleable | Safe rollout |

---

<a name="enhancement-2026"></a>
## 3. 2026 Enhancement Execution

### Phase 1: Core Improvements (Weeks 1-4) - CURRENT FOCUS

**Goal:** Implement 5 HIGH priority improvements with measurable impact

| # | Improvement | Agent | Duration | Status |
|---|-------------|-------|----------|--------|
| 1.1 | Task-Adaptive VSA Encoding | `vsa-core-agent` | 5 days | ☐ Ready |
| 1.2 | Distributed Global Workspace | `workspace-agent` | 7 days | ☐ Ready |
| 1.3 | Generalization-Preserved Learning | `continual-learning-agent` | 6 days | ☐ Ready |
| 1.4 | Meta-Analogical Transfer | `transfer-learning-agent` | 8 days | ☐ Ready |
| 1.5 | Dual-Process Architecture | `architecture-agent` | 8 days | ☐ Ready |

**Support Roles:**
- `testing-agent`: Continuous testing, benchmarking
- `documentation-agent`: Documentation updates

---

<a name="agent-roles"></a>
## 4. Agent Roles & Responsibilities

### Specialized Implementation Agents

#### vsa-core-agent
**Expertise:** Vector Symbolic Architecture, hyperdimensional computing

**Responsibilities:**
- Task-Adaptive VSA Encoding (Phase 1.1)
- Vector Hardware Acceleration (Phase 3.1)
- Adaptive Sparse Projection (Phase 3.5)

**Files:**
- `nsck-demo/python/hypervec_shim.py`
- `nsck-demo/python/hypervec_py.py`
- `nsck-demo/tests/test_adaptive_encoding.py` (new)

**Success Criteria:**
- Encoding strategies working (learning, symbolic, hybrid)
- +10-15% classification accuracy
- +12% reasoning accuracy
- All correlation targets met

---

#### workspace-agent
**Expertise:** Global Workspace Theory, cognitive architectures

**Responsibilities:**
- Distributed Global Workspace (Phase 1.2)
- Context-Sensitive Competition (Phase 2.1)
- Emotion-Guided Attention (Phase 2.5)

**Files:**
- `nsck-demo/python/distributed_workspace.py` (new)
- `nsck-demo/python/global_workspace.py`
- `nsck-demo/python/cognitive_engine.py`
- `nsck-demo/tests/test_distributed_workspace.py` (new)

**Success Criteria:**
- +30-40% throughput
- -25% latency
- 100% decision correctness maintained
- Handles 10× more coalitions

---

#### continual-learning-agent
**Expertise:** Continual learning, catastrophic forgetting prevention

**Responsibilities:**
- Generalization-Preserved Learning (Phase 1.3)
- Robust Policy Optimization (Phase 2.2)
- Self-Synthesized Rehearsal (Phase 3.2)

**Files:**
- `nsck-demo/python/hyperbolic_learning.py` (new)
- `nsck-demo/python/continual_learning.py`
- `nsck-demo/tests/test_gpl.py` (new)

**Success Criteria:**
- 3% forgetting vs 8% baseline (62% reduction)
- +5% memory overhead only
- Compatible with existing EWC

---

#### transfer-learning-agent
**Expertise:** Transfer learning, analogical reasoning

**Responsibilities:**
- Meta-Analogical Transfer (Phase 1.4)
- Category Theory Mapping (Phase 2.4)

**Files:**
- `nsck-demo/python/meta_analogy.py` (new)
- `nsck-demo/python/analogy.py`
- `nsck-demo/tests/test_meta_analogy.py` (new)

**Success Criteria:**
- +25% zero-shot performance
- 3× sample efficiency
- Transfer to distant domains

---

#### architecture-agent
**Expertise:** System architecture, integration

**Responsibilities:**
- Dual-Process Architecture (Phase 1.5)
- System integration
- Architecture coordination

**Files:**
- `nsck-demo/python/dual_process.py` (new)
- `nsck-demo/python/cognitive_engine.py`
- `nsck-demo/tests/test_dual_process.py` (new)

**Success Criteria:**
- +15% overall accuracy
- +40% error detection
- 5× faster on easy cases
- S1 usage 70-80%

---

#### snn-agent
**Expertise:** Spiking neural networks, neuromorphic computing

**Responsibilities:**
- NeuroNAS (Phase 2.3)
- Surrogate Gradient Enhancement (Phase 3.3)
- Adversarial Robustness (Phase 3.4)

**Files:**
- `nsck-demo/python/snn_training_pipeline.py`
- `nsck-demo/python/plastic_snn.py`

**Success Criteria:**
- 84% energy reduction
- 92% area savings
- Match ANN accuracy

---

### Support Agents

#### testing-agent
**Role:** Testing, benchmarking, quality assurance

**Continuous Responsibilities:**
1. Maintain test infrastructure
2. Run continuous benchmarks
3. Monitor test coverage (target: 90%+)
4. Generate performance reports
5. Sign off on all deliverables

**Phase-Specific:**
- Week 1: Set up baseline benchmarks
- Week 2-3: Review unit tests as they're created
- Week 3: Run integration tests
- Week 4: Generate Phase 1 performance report

---

#### documentation-agent
**Role:** Documentation, knowledge management

**Continuous Responsibilities:**
1. Update technical documentation
2. Maintain API documentation
3. Create implementation guides
4. Generate visual diagrams
5. Write user-facing docs

**Phase-Specific:**
- Week 1-3: Document each improvement as implemented
- Week 4: Update ARCHITECTURE.md, README.md
- Week 4: Create Phase 1 summary

---

## 5. Work Units: Phase 1 Tasks

### Task 1.1: Task-Adaptive VSA Encoding
**Owner:** vsa-core-agent  
**Duration:** 5 days  
**Dependencies:** None

**Subtasks:**
- [ ] Day 1-2: Design and implement `EncodingStrategyManager`
- [ ] Day 2-3: Implement encoding algorithms (learning, symbolic, hybrid)
- [ ] Day 3-4: Integrate with `CognitiveEngine`
- [ ] Day 4-5: Write 10+ tests and run benchmarks

**Deliverables:**
- `EncodingStrategyManager` class
- 10+ unit tests
- Performance benchmarks showing +10-15% improvement
- Integration complete

**Verification:**
- [ ] All tests pass
- [ ] Benchmarks meet targets
- [ ] Code review approved
- [ ] Documentation complete

---

### Task 1.2: Distributed Global Workspace
**Owner:** workspace-agent  
**Duration:** 7 days  
**Dependencies:** None

**Subtasks:**
- [ ] Day 1-2: Design architecture (DistributedGlobalWorkspace, ContextRouter, MetaArbiter)
- [ ] Day 3-5: Implement core components
- [ ] Day 5-6: Integrate with `CognitiveEngine`
- [ ] Day 6-7: Write 8+ tests and run benchmarks

**Deliverables:**
- `distributed_workspace.py` module
- 8+ comprehensive tests
- Performance benchmarks showing +30-40% throughput
- Configuration guide

**Verification:**
- [ ] All tests pass
- [ ] Throughput +30-40%
- [ ] Latency -25%
- [ ] No regressions

---

### Task 1.3: Generalization-Preserved Learning
**Owner:** continual-learning-agent  
**Duration:** 6 days  
**Dependencies:** None

**Subtasks:**
- [ ] Day 1-2: Implement `HyperbolicEmbedding` class
- [ ] Day 3-4: Implement `GeneralizationPreservedLearning` algorithm
- [ ] Day 4-5: Integrate with `ContinualLearner`
- [ ] Day 5-6: Write 12+ tests and run benchmarks

**Deliverables:**
- `hyperbolic_learning.py` module
- 12+ comprehensive tests
- Benchmark showing 3% vs 8% forgetting
- Integration guide

**Verification:**
- [ ] All tests pass
- [ ] Forgetting reduced by 62%
- [ ] Memory overhead < 5%
- [ ] Compatible with EWC

---

### Task 1.4: Meta-Analogical Transfer
**Owner:** transfer-learning-agent  
**Duration:** 8 days  
**Dependencies:** None

**Subtasks:**
- [ ] Day 1-2: Design `ReasoningStrategy` and `MetaAnalogicalTransfer` classes
- [ ] Day 3-5: Implement strategy extraction
- [ ] Day 5-6: Implement strategy transfer
- [ ] Day 6-7: Integrate with `AnalogyEngine`
- [ ] Day 7-8: Write 15+ tests and run benchmarks

**Deliverables:**
- `meta_analogy.py` module
- 15+ comprehensive tests
- Benchmark showing +25% zero-shot improvement
- Documentation with examples

**Verification:**
- [ ] All tests pass
- [ ] Zero-shot +25%
- [ ] Sample efficiency 3×
- [ ] Transfers to distant domains

---

### Task 1.5: Dual-Process Architecture
**Owner:** architecture-agent  
**Duration:** 8 days  
**Dependencies:** None

**Subtasks:**
- [ ] Day 1-2: Design System1, System2, DualProcessCognition
- [ ] Day 3: Implement System1 (fast path)
- [ ] Day 4: Implement System2 (slow path)
- [ ] Day 5-6: Implement coordinator with metacognitive monitoring
- [ ] Day 6-7: Integrate with `CognitiveEngine`
- [ ] Day 7-8: Write 12+ tests and run benchmarks

**Deliverables:**
- `dual_process.py` module
- 12+ comprehensive tests
- Benchmark showing +15% accuracy, +40% error detection
- Usage statistics showing 70-80% S1 usage

**Verification:**
- [ ] All tests pass
- [ ] Accuracy +15%
- [ ] Error detection +40%
- [ ] Easy cases 5× faster
- [ ] S1/S2 balance correct

---
| **Test-First** | Never merge code that reduces existing test pass rate | Protects the 165+ passing tests |
| **Minimal Surface** | Each agent works on ≤3 files per session | Reduces merge conflicts |
| **Efficiency Constraints** | O(n) VSA ops, dims ≤128 shared layers, no heavy matmuls | Matches project architecture |
| **LLM Isolation** | LLMs are peripherals for communication only, not core logic | Preserves neuro-symbolic design |
| **Honesty** | Document what actually works, not what you wish worked | Prevents capability inflation |

---

<a name="agent-roles"></a>
## 3. Agent Roles & Responsibilities

### 3.1 Role Definitions

Each agent is assigned a **role** that defines its scope. An agent must never work outside its role without explicit approval.

#### Role: `neural-core`
- **Scope:** SNN training, neural policies, intrinsic motivation, world model
- **Files:** `snn_qat.py`, `plastic_snn.py`, `train_snn.py`, `world_model.py`, `intrinsic_motivation.py`
- **Tests:** `test_snn_*.py`, `test_world_model.py`, `test_intrinsic_motivation.py`
- **Constraints:** Keep SNN dims ≤128, ICM params ≤30K, use sparse projections

#### Role: `symbolic-reasoning`
- **Scope:** Rule learning, causal reasoning, planning, spatial reasoning
- **Files:** `rule_learner.py`, `causal_reasoning.py`, `planner.py`, `spatial_reasoning.py`, `analogy.py`
- **Tests:** `test_causal_*.py`, `test_chaining.py`, `test_counterfactuals.py`, `test_planning.py`
- **Constraints:** Use VSA for all symbol representations, O(n) operations only

#### Role: `memory-systems`
- **Scope:** Episodic memory, semantic memory, persistence, sleep consolidation
- **Files:** `episodic_memory.py`, `semantic_memory.py`, `persistence.py`, `learning.py`, `staged_recall.py`, `intelligent_buffer.py`
- **Tests:** `test_memory_*.py`, `test_dreaming.py`, `test_sleep.py`
- **Constraints:** VSA-based hashing, LSH bucketing, SQLite for persistence

#### Role: `cognitive-integration`
- **Scope:** Cognitive engine, global workspace, metacognition, brain fusion
- **Files:** `cognitive_engine.py`, `global_workspace.py`, `metacognition.py`, `brain_fusion.py`, `self_model.py`
- **Tests:** `test_cross_module.py`, `test_global_workspace.py`, `test_brain_fusion.py`, `test_emergence.py`
- **Constraints:** Must integrate with all other subsystems, no standalone logic

#### Role: `perception-encoding`
- **Scope:** Universal encoder, symbol grounding, perception, concept mapping
- **Files:** `universal_encoder.py`, `symbol_grounding.py`, `perception.py`, `concept_mapper.py`, `grounding_verifier.py`
- **Tests:** `test_perception.py`, `test_grounding.py`, `test_encoder.py`
- **Constraints:** Multi-modal encoding, bind to VSA concepts

#### Role: `language-communication`
- **Scope:** Language module, dialogue, voice, chatbot, explanation
- **Files:** `language_module.py`, `lingua_cortex.py`, `dialogue_manager.py`, `voice_interface.py`, `explanation.py`, `chatbot.py`
- **Tests:** `test_language.py`, `test_dialogue.py`, `test_explanation.py`
- **Constraints:** LLM is peripheral only, meaning must ground to VSA

#### Role: `infrastructure`
- **Scope:** Config, logging, server, dashboard, CI/CD, testing infrastructure
- **Files:** `config.py`, `logger_service.py`, `python_server.py`, `dashboard.py`, `system_launcher.py`
- **Tests:** `test_config.py`, `test_server.py`, `test_dashboard.py`
- **Constraints:** No changes to core logic, infrastructure only

#### Role: `environments`
- **Scope:** Game environments, simulation, AI controller
- **Files:** `snake_headless.py`, `snake_ui.py`, `pong_ui.py`, `maze_game.py`, `maze_ui.py`, `simulation.py`, `ai_controller.py`
- **Tests:** `test_snake.py`, `test_pong.py`, `test_maze.py`, `test_simulation.py`
- **Constraints:** Environments must expose structured state dicts

#### Role: `vsa-core`
- **Scope:** HyperVector implementations (Python and Rust), shim layer
- **Files:** `hypervec_py.py`, `hypervec_shim.py`, `rust_vsa/`
- **Tests:** `test_hypervec.py`, `test_vsa_*.py`
- **Constraints:** 10,240-bit binary vectors, O(n) ops, import via `hypervec_shim`

### 3.2 Role Assignment Rules

1. **One role per agent per session** — no multi-role agents
2. **Roles can run in parallel** if they don't share files (see [Dependency Map](#dependency-map))
3. **Role conflicts** (two agents touching the same file) must be resolved by sequential execution
4. **Role escalation** (needing to touch files outside your scope) requires documenting the cross-role dependency

---

<a name="work-units"></a>
## 4. Work Units: What Gets Built

### 4.1 Work Unit Structure

Every piece of work is a **Work Unit (WU)** with this structure:

```
WU-<phase>-<number>: <Title>
├── What:   Exact deliverable (files, functions, classes)
├── Why:    Which roadmap item this addresses
├── How:    Technical approach (pseudocode or algorithm)
├── Inputs: What must exist before this WU starts
├── Outputs: What this WU produces (files, APIs, test results)
├── Tests:  Specific test cases that validate the WU
├── Role:   Which agent role handles this
├── Est:    Estimated effort (S/M/L)
└── Parallel: Can run alongside which other WUs
```

### 4.2 Phase 0 Work Units (Foundation Strengthening)

#### WU-0-01: CI/CD Pipeline Setup
```
What:   GitHub Actions workflow for pytest, linting, type checking
Why:    ROADMAP Phase 0 → "Add Missing Infrastructure" → "CI/CD pipeline"
How:    Create .github/workflows/ci.yml with pytest, mypy, ruff
Inputs: pytest.ini, requirements.txt (both exist)
Outputs: .github/workflows/ci.yml, passing CI on main branch
Tests:  CI runs and all 165+ existing tests pass
Role:   infrastructure
Est:    S
Parallel: WU-0-02, WU-0-03, WU-0-04
```

#### WU-0-02: Structured Logging
```
What:   Replace print statements with structured logging
Why:    ROADMAP Phase 0 → "Add Missing Infrastructure" → "Logging system"
How:    Use Python stdlib logging with JSON formatter in logger_service.py
Inputs: logger_service.py (exists, may need enhancement)
Outputs: Consistent logging across all modules
Tests:  Existing tests still pass, log output is parseable JSON
Role:   infrastructure
Est:    M
Parallel: WU-0-01, WU-0-03, WU-0-04
```

#### WU-0-03: Configuration Management Consolidation
```
What:   Centralize all hyperparams in config.py with validation
Why:    ROADMAP Phase 0 → "Add Missing Infrastructure" → "Configuration management"
How:    Extend config.py with dataclass-based config, env var overrides
Inputs: config.py (exists), scattered hardcoded values across modules
Outputs: Single source of truth for all configuration
Tests:  Config loading, validation, override tests
Role:   infrastructure
Est:    M
Parallel: WU-0-01, WU-0-02, WU-0-04
```

#### WU-0-04: SNN Training Pipeline
```
What:   Actual gradient-based SNN training on MNIST
Why:    ROADMAP Phase 0 → "Complete the Perception Engine" → "Implement actual SNN training"
How:    Use snnTorch with surrogate gradients, train on MNIST
Inputs: snn_qat.py (exists), train_snn.py (exists)
Outputs: Training script that achieves >90% MNIST accuracy
Tests:  test_snn_training.py — model trains, loss decreases, accuracy >90%
Role:   neural-core
Est:    L
Parallel: WU-0-01, WU-0-02, WU-0-03
```

#### WU-0-05: Error Handling Standardization
```
What:   Consistent exception handling across all 78 modules
Why:    ROADMAP Phase 0 → "Add Missing Infrastructure" → "Proper error handling"
How:    Define exception hierarchy in exceptions.py, use across modules
Inputs: All Python modules
Outputs: exceptions.py, updated error handling in critical modules
Tests:  Error cases produce meaningful exceptions, not raw tracebacks
Role:   infrastructure
Est:    M
Parallel: WU-0-04
```

#### WU-0-06: Test Coverage Improvement
```
What:   Increase test coverage for critical modules
Why:    ROADMAP Phase 0 → "Unit tests (>80% coverage)"
How:    Add tests for untested paths in the 18 critical modules
Inputs: Existing test suite (53 files)
Outputs: New test files covering edge cases
Tests:  Coverage report shows >80% on critical modules
Role:   infrastructure
Est:    L
Parallel: WU-0-04
```

#### WU-0-07: RL Foundation (A2C/PPO)
```
What:   Basic Actor-Critic and PPO implementation for Snake
Why:    ROADMAP Phase 0 → "Implement Actual Learning" → "Actor-Critic (A2C)"
How:    PyTorch-based A2C with discrete actions, integrate with snake_headless.py
Inputs: snake_headless.py (exists), snn_qat.py (exists)
Outputs: rl_agent.py with A2C/PPO, training script
Tests:  Agent learns Snake, reward increases over episodes
Role:   neural-core
Est:    L
Parallel: WU-0-01, WU-0-02, WU-0-03, WU-0-05
```

#### WU-0-08: Neural-Symbolic Rule Bridge
```
What:   Extract symbolic rules from learned neural policies
Why:    ROADMAP Phase 0 → "Connect RL to symbolic rules" → "Extract rules from learned policy"
How:    Decision tree extraction from policy network, convert to rule_learner format
Inputs: WU-0-07 (RL agent), rule_learner.py (exists)
Outputs: rule_extraction.py, integration with rule_learner.py
Tests:  Extracted rules match policy behavior on test states
Role:   symbolic-reasoning
Est:    M
Parallel: WU-0-05, WU-0-06
Dependencies: WU-0-07 must complete first
```

### 4.3 Phase 1 Work Units (Neural Learning Engine)

#### WU-1-01: Multi-Task Shared Encoder
```
What:   Shared convolutional encoder for Snake + Pong
Why:    ROADMAP Phase 1 → "Multi-Task Learning" → "Shared convolutional encoder"
How:    Common feature extractor with task-specific heads
Inputs: WU-0-04, WU-0-07 (trained SNN, RL agent)
Outputs: multi_task_encoder.py, task heads
Tests:  Both tasks learn without negative transfer
Role:   neural-core
Est:    L
Parallel: WU-1-02
Dependencies: WU-0-04, WU-0-07
```

#### WU-1-02: Meta-Learning (Reptile)
```
What:   Few-shot adaptation to new game environments
Why:    ROADMAP Phase 1 → "Meta-Learning for Fast Adaptation"
How:    Reptile algorithm (simpler than MAML), task distribution from game variants
Inputs: WU-0-07 (RL agent), game environments
Outputs: meta_learner.py
Tests:  Adapts to new game variant in <100 episodes
Role:   neural-core
Est:    L
Parallel: WU-1-01
Dependencies: WU-0-07
```

#### WU-1-03: Semantic Folding Enhancement
```
What:   Train semantic folding model on text corpus
Why:    IMPLEMENTATION_ROADMAP Phase 1 → "Semantic Folding Enhancement"
How:    Word co-occurrence → grid mapping → hypervector generation
Inputs: lingua_cortex.py (exists, skeleton)
Outputs: Enhanced lingua_cortex.py, training script
Tests:  Word similarity benchmark (SimLex-999 subset)
Role:   language-communication
Est:    L
Parallel: WU-1-01, WU-1-02
```

#### WU-1-04: LLM Integration (Peripheral Only)
```
What:   Integrate small LLM for language I/O (NOT core logic)
Why:    IMPLEMENTATION_ROADMAP Phase 1 → "LLM Integration"
How:    llama-cpp-python with Phi-3 for text understanding/generation, grounded to VSA
Inputs: language_module.py (exists), WU-1-03
Outputs: Enhanced language_module.py
Tests:  LLM extracts intents, entities ground to VSA concepts
Role:   language-communication
Est:    M
Parallel: WU-1-01, WU-1-02
Dependencies: WU-1-03
```

### 4.4 Phase 2+ Work Units

Detailed work units for Phase 2+ follow the same structure but are not expanded here because Phase 0 and Phase 1 must complete first. Each subsequent phase's work units will be defined when the prior phase reaches its milestone deliverables.

**Phase 2 areas (defined when Phase 1 milestones met):**
- Vision system (contrastive learning, ViT)
- Audio system (Whisper integration)
- Multimodal fusion
- Symbol grounding from raw perception

**Phase 3+ areas (defined when Phase 2 milestones met):**
- Continual learning (EWC, PackNet, replay)
- World models (DreamerV3-style)
- Hierarchical planning
- Self-model & metacognition enhancement
- Social/emotional intelligence

---

<a name="execution-model"></a>
## 5. Execution Model: How Agents Run

### 5.1 Parallel Execution Rules

Agents can run **simultaneously** when:
1. Their work units have no file overlap (different `Files` lists)
2. Their work units have no dependency relationship
3. They are working on different test files

Agents must run **sequentially** when:
1. One work unit depends on another's output (see `Dependencies`)
2. Two work units touch the same file
3. One work unit changes a shared API that another consumes

### 5.2 Parallelism Matrix (Phase 0)

| | WU-0-01 | WU-0-02 | WU-0-03 | WU-0-04 | WU-0-05 | WU-0-06 | WU-0-07 | WU-0-08 |
|---|---|---|---|---|---|---|---|---|
| **WU-0-01** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **WU-0-02** | ✅ | — | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **WU-0-03** | ✅ | ✅ | — | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **WU-0-04** | ✅ | ✅ | ✅ | — | ✅ | ✅ | ⚠️ | ✅ |
| **WU-0-05** | ✅ | ⚠️ | ⚠️ | ✅ | — | ✅ | ✅ | ✅ |
| **WU-0-06** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| **WU-0-07** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | — | ❌ |
| **WU-0-08** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | — |

✅ = Can run in parallel  ⚠️ = Caution — shared infrastructure files; agents should coordinate by reviewing each other's target files before starting and running sequentially if both touch the same file  ❌ = Must run sequentially (hard dependency)

### 5.3 Recommended Execution Order

```
Wave 1 (parallel):  WU-0-01, WU-0-02, WU-0-03, WU-0-04
Wave 2 (parallel):  WU-0-05, WU-0-06, WU-0-07
Wave 3 (sequential): WU-0-08 (depends on WU-0-07)
Wave 4 (parallel):  WU-1-01, WU-1-02, WU-1-03
Wave 5 (sequential): WU-1-04 (depends on WU-1-03)
```

---

<a name="guardrails"></a>
## 6. Guardrails: Preventing Drift & Breakage

### 6.1 Pre-Execution Checklist

Every agent MUST complete this before writing any code:

```
□ Read docs/AGENT_INSTRUCTIONS.md (governance rules)
□ Read this document's Work Unit for your assigned task
□ Run the full test suite: python -m pytest nsck-demo/tests/ -v
   (note: ignore tests requiring flask/cv2/snntorch if not installed)
□ Record baseline test count: _____ tests passing
□ Confirm your Work Unit's inputs exist
□ Confirm no other agent is modifying your target files
```

### 6.2 During-Execution Rules

| Rule | Enforcement | Why |
|---|---|---|
| **No file outside your role's scope** | Agent must check file list before editing | Prevents cross-role conflicts |
| **Run tests after every change** | `python -m pytest nsck-demo/tests/ -v --tb=short` | Catches breakage immediately |
| **No new dependencies without justification** | Must document why existing deps aren't sufficient | Keeps project lean |
| **No API signature changes to critical modules** | Unless the work unit explicitly requires it | Prevents cascading breakage |
| **All new code needs type hints** | Python 3.11+ type annotations required | Matches project convention |
| **Import HyperVectors correctly** | `import hypervec_shim as hypervec_rs` only | Never import `hypervec_rs` directly |
| **Max 128-dim for shared layers** | Neural components must use compact representations | Energy/memory efficiency |

### 6.3 Post-Execution Verification

```
□ All pre-existing tests still pass (count matches baseline)
□ New tests added for new functionality
□ New tests pass
□ No binary artifacts committed (*.db, *.pkl, *.pth, *.csv, *.png)
□ Documentation updated if architecture changed
□ Session handoff notes written (see Section 10)
```

### 6.4 Automatic Rejection Criteria

An agent's work is **automatically rejected** if:
1. Existing test count drops (any previously passing test now fails)
2. Files outside the assigned role's scope are modified without documented justification
3. A new dependency is added without security check and justification
4. Core VSA operations use O(n²) algorithms
5. An LLM is used for core decision-making logic (not just I/O)
6. Documentation claims capabilities that aren't backed by passing tests

---

<a name="context-protocol"></a>
## 7. Context Protocol: Staying Up-to-Date

### 7.1 Required Reading (Every Agent, Every Session)

Before starting any work, every agent must read these files in order:

```
1. docs/AGENT_INSTRUCTIONS.md          — governance rules
2. docs/AGENT_COORDINATION_PLAN.md     — this document, your work unit
3. ROADMAP_TO_AGI.md                   — phase definitions (skip code samples)
4. README.md                           — what currently works vs doesn't
5. docs/COMPLETE_MODULE_ANALYSIS.md    — module status & dependencies
```

**Total context load:** ~3,500 lines of markdown. Agents should parse structure and focus on their relevant sections.

### 7.2 Codebase Context Snapshots

Each agent should understand the current state of files it will touch. Use this pattern:

```bash
# 1. Check current module status
head -50 nsck-demo/python/<your-module>.py

# 2. Check existing tests
ls nsck-demo/tests/test_<your-module>*.py

# 3. Check who imports your module
grep -rl "import <your-module>" nsck-demo/python/

# 4. Check who your module imports
head -30 nsck-demo/python/<your-module>.py | grep "^import\|^from"

# 5. Run existing tests for your module
python -m pytest nsck-demo/tests/test_<your-module>.py -v
```

### 7.3 Cross-Agent Communication

When agents need to coordinate (e.g., one agent creates an API that another consumes):

1. **Define the interface first** — write the function signature, docstring, and type hints
2. **Write the test** — create the test that validates the interface contract
3. **Implement** — fill in the implementation
4. **Notify downstream** — document the interface in the session handoff

Example:
```python
# Agent A (neural-core) defines this interface:
def extract_policy_rules(policy: nn.Module, states: List[Dict]) -> List[Rule]:
    """Extract symbolic rules from a trained neural policy.

    Args:
        policy: Trained PyTorch policy network
        states: Sample states to evaluate the policy on

    Returns:
        List of Rule objects compatible with rule_learner.py
    """
    ...

# Agent B (symbolic-reasoning) can then code against this interface
```

### 7.4 Staying Current with Other Agents' Changes

```bash
# Before starting work, pull latest and check recent changes
git log --oneline -10
git diff HEAD~1 --stat

# Check if any of your target files were modified
git log --oneline -5 -- nsck-demo/python/<your-files>
```

---

<a name="dependency-map"></a>
## 8. Dependency Map: Sequencing & Parallelism

### 8.1 Module Dependency Graph (Simplified)

```
                    python_server.py
                          │
                  cognitive_engine.py
                    │     │     │
          ┌─────────┤     │     ├─────────┐
          │         │     │     │         │
    global_      meta-   brain_   self_   episodic_
    workspace  cognition fusion  model   memory
          │         │     │              │
          └────┬────┘     │         persistence
               │          │
         ┌─────┼─────┐    │
         │     │     │    │
    rule_   causal_  planner
    learner reasoning   │
         │     │     spatial_
         │     │     reasoning
         │     │
    ┌────┴─────┴────┐
    │               │
  snn_qat    universal_
    │        encoder
    │          │
  hypervec_shim ←── hypervec_py / rust_vsa
```

### 8.2 Safe Parallelism Zones

These groups of modules have **no shared dependencies** and can be worked on simultaneously:

| Zone | Modules | Role |
|---|---|---|
| **Zone A** | `snn_qat`, `plastic_snn`, `train_snn`, `world_model` | neural-core |
| **Zone B** | `rule_learner`, `causal_reasoning`, `planner`, `spatial_reasoning` | symbolic-reasoning |
| **Zone C** | `episodic_memory`, `semantic_memory`, `persistence`, `learning` | memory-systems |
| **Zone D** | `language_module`, `lingua_cortex`, `dialogue_manager`, `voice_interface` | language-communication |
| **Zone E** | `snake_headless`, `pong_ui`, `maze_game`, `simulation` | environments |
| **Zone F** | `.github/workflows/`, `config.py`, `logger_service.py` | infrastructure |

**Critical shared zone** (sequential access only):
- `cognitive_engine.py` — imports from Zones A, B, C; only `cognitive-integration` role touches this
- `hypervec_shim.py` / `hypervec_py.py` — only `vsa-core` role touches these
- `global_workspace.py` — only `cognitive-integration` role touches this

---

<a name="verification"></a>
## 9. Verification & Quality Gates

### 9.1 Per-Work-Unit Verification

Each work unit has specific acceptance criteria. The agent must demonstrate:

1. **Functional:** New code runs without errors
2. **Tested:** New tests pass and cover the happy path + at least 1 edge case
3. **Non-regressing:** All pre-existing tests still pass
4. **Documented:** Docstrings on all new public functions/classes
5. **Efficient:** No O(n²) operations in VSA paths, dims ≤128 for shared layers

### 9.2 Phase Gate Criteria

Before moving to the next phase, **all** work units in the current phase must pass:

| Gate | Phase 0 → Phase 1 | Phase 1 → Phase 2 |
|---|---|---|
| **Tests** | All 165+ existing tests pass + new tests | All Phase 0+1 tests pass |
| **Coverage** | >80% on critical modules | >80% maintained |
| **Deliverable** | SNN trains on MNIST >90%, RL agent plays Snake | Multi-task encoder works, meta-learning adapts |
| **Docs** | README updated with new capabilities | README updated |
| **CI** | GitHub Actions passing on main | CI still passing |

### 9.3 Test Categories

```
Unit Tests:       Test individual functions/methods in isolation
Integration Tests: Test module interactions (e.g., SNN → symbol grounding)
Regression Tests:  The existing 165+ tests — these must always pass
Benchmark Tests:   Performance/accuracy targets (MNIST >90%, Snake reward)
```

### 9.4 Running Tests

```bash
# Full test suite (required before and after each session)
python -m pytest nsck-demo/tests/ -v

# Tests safe to run without optional dependencies (flask, cv2, snntorch)
python -m pytest nsck-demo/tests/ -v \
  --ignore=nsck-demo/tests/test_dashboard.py \
  --ignore=nsck-demo/tests/test_deep_dreaming.py \
  --ignore=nsck-demo/tests/test_dynamic_brain.py \
  --ignore=nsck-demo/tests/test_saliency.py \
  --ignore=nsck-demo/tests/test_transfer.py

# Single module test
python -m pytest nsck-demo/tests/test_<module>.py -v

# With coverage
python -m pytest nsck-demo/tests/ -v --cov=nsck-demo/python --cov-report=term-missing
```

---

<a name="session-handoff"></a>
## 10. Session Handoff Protocol

### 10.1 Why Handoff Matters

Coding agents have limited context windows and session lifetimes. When one agent's session ends, the next agent (or the same agent in a new session) must pick up without losing progress. The handoff protocol ensures continuity.

### 10.2 Handoff Template

At the end of every session, the agent must produce a summary capturing:

```markdown
## Session Summary

**Agent Role:** <role name>
**Work Unit:** <WU-X-XX>
**Date:** <ISO 8601>

### Completed
- [ ] List specific files changed and what changed
- [ ] List tests added/modified
- [ ] List tests run and results

### In Progress
- [ ] What's partially done and where you stopped
- [ ] What the next step should be

### Blockers
- [ ] Any issues that prevented completion
- [ ] Dependencies on other agents/work units

### Test Results
- Baseline: X tests passing
- After changes: Y tests passing
- New tests added: Z

### Files Changed
- `nsck-demo/python/file.py` — description of change
- `nsck-demo/tests/test_file.py` — description of test added

### Context for Next Agent
- Key decisions made and why
- Gotchas or non-obvious behavior discovered
- API contracts established with other roles
```

### 10.3 Where to Record Handoffs

Session handoff notes should be included in PR descriptions or commit messages. This keeps the context in the version control history where any future agent can find it.

---

## Appendix A: Quick Reference Card

### For Any Agent Starting a Session

```
1. Read docs/AGENT_INSTRUCTIONS.md
2. Read docs/AGENT_COORDINATION_PLAN.md (find your Work Unit)
3. git log --oneline -10                  # Check recent changes
4. python -m pytest nsck-demo/tests/ -v   # Record baseline
5. Do your work (within role scope only)
6. python -m pytest nsck-demo/tests/ -v   # Verify no regression
7. Write session handoff summary
```

### File Naming Conventions

- Source: `nsck-demo/python/<module_name>.py`
- Tests: `nsck-demo/tests/test_<module_name>.py`
- Config: `nsck-demo/python/config.py`
- Docs: `docs/<DOCUMENT_NAME>.md`

### Import Conventions

```python
# VSA — ALWAYS use the shim
import hypervec_shim as hypervec_rs

# Internal modules — direct import (conftest.py adds paths)
from episodic_memory import EpisodicMemory
from rule_learner import RuleLearner
```

---

## Appendix B: Decision Log Template

When an agent makes a non-obvious technical decision, document it:

```markdown
### Decision: <title>
**Date:** <date>
**Agent Role:** <role>
**Work Unit:** <WU-X-XX>

**Context:** What situation required a decision
**Options Considered:**
1. Option A — pros/cons
2. Option B — pros/cons

**Chosen:** Option X
**Rationale:** Why this option best fits the project constraints
**Impact:** What this means for other agents/modules
```

---

*End of Agent Coordination Plan*
