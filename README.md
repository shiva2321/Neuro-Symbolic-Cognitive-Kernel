# NSCK — Neural-Symbolic Cognitive Kernel

> **A glass-box cognitive architecture: every decision is traceable, every reasoning step is explainable.**  
> No neural-network black boxes. No LLM API calls. All reasoning runs locally, symbolically, and transparently.

---

## What Is This?

**NSCK** (Neural-Symbolic Cognitive Kernel) is a domain-agnostic reasoning engine built on **Vector Symbolic Architecture (VSA)** — a mathematical framework where every concept, relation, and memory is a 10,240-bit binary hypervector. You register your domain (predicates + actions), feed it observations, and get back a chosen action *plus* a human-readable explanation of why.

This repository contains two things:

| Layer | What it is | Entry point |
|---|---|---|
| **[`nsck/`](nsck/)** | The cognitive kernel — memory, reasoning, perception, language, learning | [`nsck/README.md`](nsck/README.md) |
| **[`nsck_ai_model/`](nsck_ai_model/)** | Conversational AI wrapper around the kernel | [`nsck_ai_model/README.md`](nsck_ai_model/README.md) |

---

## Why Is It Different?

| Typical LLM / Neural approach | NSCK |
|---|---|
| Gradient-based black box | Every rule, causal link, and coalition score is inspectable at runtime |
| Needs GPU + billions of parameters | Runs on CPU; Rust accelerator optional (21–206× speedup) |
| Can't explain *why* it said something | 11-stage ThoughtTrace with confidence scores per step |
| Frozen knowledge unless retrained | Learns continuously from new text with no gradient update |
| One monolithic model | 8 independent subsystems — swap or extend any one |

---

## What Can It Do?

| Capability | How |
|---|---|
| **Symbolic reasoning** | Global Workspace Theory (LIDA-Lite) selects the winning cognitive coalition |
| **Causal inference** | ΔP discovery + multi-hop forward/backward causal chains |
| **Planning** | STRIPS A\* planner with operator preconditions and effects |
| **Cross-domain analogy** | Structural alignment lifts rules from a source domain to a target |
| **Conceptual blending** | `AnalogyEngine.blend()` merges two domains via shared generic space (V3) |
| **Episodic memory** | Two-tier store: in-memory hot tier + SQLite warm tier, LSH k-NN retrieval |
| **Semantic memory** | NetworkX directed graph + HV index + spreading activation |
| **Perception** | Leaky Integrate-and-Fire spiking neurons (LIF) with STDP; Rate/Temporal VSA bridge |
| **Language understanding** | SRL + Construction Grammar (V3) + Frame Semantics (V3) + Coreference (V3) |
| **Knowledge extraction** | TextKnowledgeLearner parses text → SVO triples + frame-filled relations → SemanticMemory |
| **Belief revision** | Free-energy belief scoring detects and resolves contradictions (V3) |
| **Distributional semantics** | Co-occurrence codebook for synonym-quality HVs (V3) |
| **Dual-process decisions** | System 1 (fast, threshold) vs System 2 (full GWT) routing (V3) |
| **Self-regulation** | Homeostatic memory pruning + stigmergic path preference during `sleep()` (V3) |
| **Natural language generation** | Discourse planner + anaphora resolution assembles multi-sentence responses |
| **Continuous learning** | Hebbian association, EWC (Elastic Weight Consolidation), curiosity-driven novelty |
| **Meta-learning** | MAML-style fast adaptation to new tasks |
| **Emotion & metacognition** | Plutchik 8-emotion circumplex; safety gate vetoes unsafe actions |
| **Self-model** | Calibrated per-task confidence; Theory of Mind for other agents |
| **Math reasoning** | Symbolic arithmetic, algebra, numeric HVs with FPE noise for monotone similarity |

---

## Architecture at a Glance

Everything flows through `CognitiveEngine` — a central orchestrator that calls each subsystem in sequence:

```
Input (text / state dict / image)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  1. VSA Foundation   HyperVector (10,240-bit binary)          │
│                      XOR bind · majority-vote bundle · permute │
├───────────────────────────────────────────────────────────────┤
│  2. Perception       LIF spiking neurons + STDP               │
│                      VSA-SNN Bridge (Rate/Temporal coding)    │
│                      Multimodal processor (HOG/color/LBP)     │
├───────────────────────────────────────────────────────────────┤
│  3. Memory           EpisodicMemory  hot-deque + SQLite + LSH │
│                      SemanticMemory  DiGraph + HV spreading   │
├───────────────────────────────────────────────────────────────┤
│  4. Reasoning        GlobalWorkspace  coalition competition   │
│                      CausalGraph      ΔP chains               │
│                      STRIPSPlanner    A* search               │
│                      AnalogyEngine    cross-domain alignment  │
│                      RuleLearner      ILP induction           │
├───────────────────────────────────────────────────────────────┤
│  5. Cognitive        EmotionSystem    Plutchik 8-emotion      │
│                      SafetyGate       metacognitive veto      │
│                      SelfModel        confidence calibration  │
│                      TheoryOfMind     belief modelling        │
├───────────────────────────────────────────────────────────────┤
│  6. Language         TextKnowledgeLearner → SemanticMemory    │
│                      SemanticRoleLabeler (AGENT/PATIENT/…)    │
│                      NLG discourse planner + anaphora         │
├───────────────────────────────────────────────────────────────┤
│  7. Rust Accelerator rust_vsa: 21–206× speedup (PyO3/Tokio)  │
│                      rust_snn: parallel LIF/STDP              │
├───────────────────────────────────────────────────────────────┤
│  8. Persistence      BrainStore (SQLite) · BrainFusion        │
│                      ExplanationGenerator (human traces)      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  chosen_action + confidence + explanation
```

> For full Mermaid subsystem diagrams see → **[`nsck/docs/ARCHITECTURE.md`](nsck/docs/ARCHITECTURE.md)**

---

## Key Numbers

| Metric | Value |
|---|---|
| HyperVector dimension | 10,240 bits |
| Python source files (nsck core) | ~53 files, ~26,000 LOC |
| Rust source files | 9 (rust_vsa + rust_snn) |
| Rust speedup over Python VSA | 21× (element-wise) – 206× (parallel k-NN over 1,000 vectors) |
| Test files / tests | ~50 files · 531 passing (Python-only) / **671 passing** (with Rust .so) |
| V3 feature flags | 14 (all off by default — zero regressions) |
| New V3 modules | 6 (construction_grammar, frame_semantics, coreference, distributional_semantics, belief_revision, homeostasis) |
| SNN layers | LIF + STDP + Hebbian + Rate/Temporal VSA bridge |
| Decision latency (Python) | ~0.001 ms (cached rules) |
| Memory query @ 1K concepts | ~19 ms Python / ~0.1 ms Rust |

---

## Quick Start

```bash
git clone https://github.com/shiva2321/Node_network.git
cd Node_network
pip install -r requirements.txt

# Optional: build Rust accelerator for 21-206× speedup
cd nsck/rust_vsa && pip install -e . && cd ../..

# Run the full test suite
python -m pytest nsck/tests/ nsck_ai_model/tests/ -q
```

### Use the Cognitive Engine directly

```python
import sys; sys.path.insert(0, 'nsck')
from python.core.reasoning.cognitive_engine import CognitiveEngine

engine = CognitiveEngine()
engine.register_task(
    task_tag="nav",
    predicates={"obstacle": lambda s: s.get("obstacle", False)},
    actions=["move", "wait"],
)
result = engine.decide({"obstacle": True}, ["move", "wait"], "nav")
print(result.chosen_action)   # e.g. "wait"
print(result.explanation)     # human-readable reasoning trace
```

### Use the conversational AI layer

```bash
# Start the Flask dashboard (trains on seed data, then serves on port 5090)
python -m nsck_ai_model.dashboard --train seed --port 5090
# Open http://localhost:5090
```

```python
from nsck_ai_model.ai_engine import NSCKAIEngine

engine = NSCKAIEngine()
engine.train_on_text("Paris is the capital of France.")
response = engine.chat("What is the capital of France?")
print(response['response'])   # "Paris"
print(response['trace'])      # full 11-stage ThoughtTrace dict
```

---

## Dive Deeper

| I want to understand… | Go here |
|---|---|
| The full system architecture + subsystem diagrams | [`nsck/docs/ARCHITECTURE.md`](nsck/docs/ARCHITECTURE.md) |
| The math — VSA algebra, ΔP causality, FPE numerics | [`nsck/docs/FORMULAS.md`](nsck/docs/FORMULAS.md) |
| Every class, method, and parameter | [`nsck/docs/MODULE_REFERENCE.md`](nsck/docs/MODULE_REFERENCE.md) |
| The Rust concurrent layer (API + build guide) | [`nsck/docs/RUST_API_REFERENCE.md`](nsck/docs/RUST_API_REFERENCE.md) |
| The full test suite — what each file tests and why | [`nsck/docs/TESTING.md`](nsck/docs/TESTING.md) |
| Data-flow walkthroughs (decision loop, learning loop) | [`nsck/docs/WORKFLOWS.md`](nsck/docs/WORKFLOWS.md) |
| Roadmap — what's done, what's next, research gaps | [`nsck/docs/NSCK_ROADMAP_AND_PLAN.md`](nsck/docs/NSCK_ROADMAP_AND_PLAN.md) |
| Analysis and real-world use cases | [`nsck/docs/ANALYSIS_AND_USECASES.md`](nsck/docs/ANALYSIS_AND_USECASES.md) |
| NSCK package quick start + module list | [`nsck/README.md`](nsck/README.md) |
| AI model API reference + dashboard | [`nsck_ai_model/README.md`](nsck_ai_model/README.md) |

---

## Design Principles

1. **Glass-box transparency** — every decision exposes Q-values, coalition scores, rule activations, and confidence at runtime. Nothing lives inside gradient tensors.
2. **Neuro-symbolic unity** — VSA is the shared language for both neural (SNN/Hebbian) and symbolic (rules/planning) processing. The same 10,240-bit vector represents a word, a memory, a rule, and a percept.
3. **Biological plausibility** — Global Workspace Theory for attention, Hebbian learning for association, LIF spiking neurons for perception, curiosity for self-directed exploration.
4. **Domain agnosticism** — register predicates and actions for any domain; the same kernel handles navigation, text Q&A, causal analysis, and conversation.
5. **No external AI dependencies** — no LLM APIs, no transformer inference, no embedding services. The Rust accelerator is the only optional component.

---

## License

See [LICENSE](LICENSE) for details.
