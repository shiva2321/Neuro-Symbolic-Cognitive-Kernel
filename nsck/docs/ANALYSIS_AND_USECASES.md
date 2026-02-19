# NSCK — Architecture Analysis & Use Cases

> Generated 2026-02-17. Reference document for project planning.

---

## Part 1: What Is NSCK?

**NSCK** (Neural-Symbolic Cognitive Kernel) is a deterministic, CPU-first
cognitive architecture combining:

- **Vector Symbolic Architecture (VSA)** — 10,240-bit binary hypervectors as
  the universal representation substrate
- **Global Workspace Theory (GWT)** — consciousness-inspired competition where
  modules propose "coalitions" and the winner is broadcast to all
- **Symbolic reasoning** — causal graphs, rule induction, STRIPS planning,
  analogical transfer

**28 Python core files (~5,700 LOC)** organized into 8 layers + a Rust
accelerator (~3,070 LOC).

### Module Inventory

| # | Module | Lines | External Libs | Neural/Gradient | VSA Integration |
|---|--------|-------|---------------|-----------------|-----------------|
| 1 | vsa/hypervec_py.py | 314 | numpy | No | **IS** the VSA layer |
| 2 | vsa/hypervec_shim.py | 199 | numpy | No | **IS** the VSA layer |
| 3 | reasoning/cognitive_engine.py | 727 | numpy | No | Yes — situation HVs |
| 4 | reasoning/global_workspace.py | 313 | numpy | No | Yes — danger vector similarity |
| 5 | reasoning/causal_reasoning.py | 903 | stdlib only | No | No |
| 6 | reasoning/rule_learner.py | 606 | numpy | No | Yes — state HV proposals |
| 7 | memory/episodic_memory.py | 357 | numpy | No | Yes — situation HVs, LSH |
| 8 | memory/semantic_memory.py | 144 | numpy, networkx | No | Yes — concept HVs |
| 9 | language/universal_input.py | 947 | numpy | No | Yes — heavy VSA grounding |
| 10 | language/lingua_cortex.py | 260 | numpy, sklearn | No | No (parallel SDR) |
| 11 | learning/curiosity.py | 336 | stdlib only | No | Yes — novelty via HV sim |
| 12 | cognitive/metacognition.py | 472 | stdlib only | No | Yes — situation digest |
| 13 | integration/config.py | 73 | stdlib only | No | Config only |
| 14 | core/__init__.py | 6 | — | No | No |

**Total: ~5,700 lines. Zero neural/gradient components. Entirely symbolic + VSA.**

---

## Part 2: Capabilities

| Capability | How |
|---|---|
| **Decide** | Grounding verifier → VSA encoding → GWT competition → action (~0.1ms/cycle) |
| **Learn from experience** | Frequency-based rule induction from (state, action, reward) tuples |
| **Causal reasoning** | Forward/backward chaining on directed causal graphs, counterfactuals |
| **Episodic memory** | LSH-indexed episodes with O(1) approximate nearest-neighbor recall |
| **Semantic memory** | Concept graph with spreading activation + role-filler binding |
| **Cross-domain transfer** | Zero-shot via analogical HV mapping between task-tagged rules |
| **Natural language** | Regex-based NLU, heuristic POS tagging, VSA-grounded text encoding |
| **Metacognition** | Self-monitoring, confidence calibration, conflict detection |
| **Theory of Mind** | Models other agents' beliefs |
| **Explainability** | Every decision produces a human-readable trace |

---

## Part 3: Rust Integration Status

| Component | Status |
|---|---|
| `HyperVector` (core VSA ops) | **Working** — Rust backend active, ~10x faster |
| `SemanticMemoryConcurrent` | **Not exposed** — compiled but not accessible |
| `EpisodicMemoryConcurrent` | **Not exposed** |
| `CognitiveWorkerPool` | **Not exposed** |
| `PersistentStorage` | **Not exposed** |
| `AsyncCognitiveRuntime` | **Not exposed** |
| `HyperVectorRegistry` | **Not exposed** |

Rust crate is a **superset** of Python (3,070 LOC with concurrency, persistence,
async) but only `HyperVector` is wired through. Major integration gap.

---

## Part 4: Should NSCK Be a Neural Network?

**No.** NSCK is a **cognitive operating system**, not an application.

Neural modules should be **first-class GWT participants** — an LLM or SNN
proposes coalitions just like the rule learner. NSCK adjudicates.

```
┌─────────────────────────────────────────────────┐
│              Applications (tasks)                │
├─────────────────────────────────────────────────┤
│    Neural Modules (LLM, CNN, SNN, RL agent)     │  ← perception, language
│    ↕ VSA bridge (binary HV ↔ dense embedding)   │  ← translation layer
├─────────────────────────────────────────────────┤
│              NSCK Kernel                         │  ← reasoning, memory,
│    GWT · Causal Graphs · Rule Engine · Memory    │     planning, explanation
│    Metacognition · Theory of Mind · Curiosity    │     STAYS SYMBOLIC
├─────────────────────────────────────────────────┤
│         Rust VSA Runtime (hypervec_rs)           │  ← fast substrate
└─────────────────────────────────────────────────┘
```

### Improvement Priorities

| Priority | Improvement | Unlocks |
|---|---|---|
| P0 | Fix Rust integration (expose all Rust modules) | 10–100x memory ops, async, production readiness |
| P0 | VSA ↔ Dense embedding bridge (`nn.Linear`) | Bidirectional neural ↔ symbolic communication |
| P1 | LLM adapter (replace mock LanguageModule) | Real NLU/NLG with NSCK reasoning as grounding |
| P1 | Differentiable VSA (FHRR) | End-to-end gradient flow through symbolic layer |
| P2 | Neural rule refinement (MLP/GNN rule scoring) | Better generalization from fewer examples |
| P2 | SNN revival (archived modules) | True neural-symbolic hybrid via GWT |
| P3 | Attention-GWT bridge | Transformer attention ↔ workspace competition |

---

## Part 5: Use Cases

### 1. Autonomous Robotics / Embedded Agents

**Where:** Warehouse robots, drones, agricultural bots, resource-constrained agents.

**How:** `register_task()` with sensor verifier. NSCK decides actions, learns
rules, transfers skills across environments.

**Why NSCK:** 0.1ms CPU inference, learns from 5 observations, explainable,
deterministic (certifiable).

### 2. Game AI / NPC Decision-Making

**Where:** Strategy games, NPC behavior, procedural content generation.

**How:** GWT competition balances multiple NPC goals. Curiosity drives
exploration. Theory of Mind models player behavior. Cross-domain transfer
moves tactics between environments.

**Why NSCK:** NPCs that actually learn, no training data, designers can read
AI explanations, zero-shot transfer between levels.

### 3. Process Control & Industrial Automation

**Where:** Manufacturing, HVAC, chemical processing, energy grid management.

**How:** Sensor readings → predicates. Causal graph encodes domain physics.
Backward chaining for root cause analysis. Counterfactuals for "what if".

**Why NSCK:** Auditable (regulatory requirement), causal not correlational,
counterfactual analysis, deterministic (safety-certifiable).

### 4. Tutoring / Educational Systems

**Where:** Adaptive learning platforms, intelligent tutoring, student modeling.

**How:** Student state → predicates. Episodic memory tracks individual history.
Theory of Mind models student understanding. Metacognition adjusts difficulty.

**Why NSCK:** Per-student memory, metacognitive monitoring, Theory of Mind,
no training dataset required.

### 5. Diagnostic / Troubleshooting Systems

**Where:** IT helpdesk, medical preliminary assessment, vehicle diagnostics,
equipment fault analysis.

**How:** Symptoms → predicates. Backward chaining identifies root causes.
`why_not()` explains rejected diagnoses. Rule induction learns new fault patterns.

**Why NSCK:** Root cause analysis (not correlation), `why_not()` for trust,
learns from resolved incidents, transfer across environments.

### 6. Multi-Agent Coordination

**Where:** Swarm robotics, fleet management, distributed sensor networks.

**How:** Each agent runs NSCK instance. Theory of Mind models other agents.
Brain Fusion merges rules. Analogical transfer shares skills.

**Why NSCK:** Lightweight (165 MB), symbolic knowledge sharing (compact),
built-in Theory of Mind, Brain Fusion across agents.

### 7. Decision Support / Augmented Intelligence

**Where:** Cockpit aids, emergency C2, trading desks, clinical support.

**How:** Human provides situation. NSCK reasons through options, explains
trade-offs, flags risks. `why_not()` and `counterfactual()` answer human
"what if" questions.

**Why NSCK:** Legible explanations, counterfactual reasoning, `why_not()`,
fully offline.

### 8. Research Platform / Cognitive Science

**Where:** Universities, AI research labs studying cognitive architectures.

**How:** GWT, dual-process memory, metacognition, curiosity, Theory of Mind
— all implemented as testable modules.

**Why NSCK:** Five cognitive science frameworks in code, modular/swappable,
experiment test suite, fully reproducible.

---

## Common Requirements Across All Use Cases

| Requirement | Why Neural Fails | Why NSCK Fits |
|---|---|---|
| Few-shot learning | Needs thousands of examples | Rules from 5 observations |
| Explainability | Attention weights aren't explanations | Human-readable decision traces |
| Determinism | GPU FP is non-deterministic | Binary XOR is always the same |
| Low resources | GPUs + GBs of VRAM | 0.1ms on CPU, 165 MB RAM |
| Causal reasoning | Correlation approximation | Actual graph traversal |
| Trust/audit | Can't certify a black box | Full audit trail |
