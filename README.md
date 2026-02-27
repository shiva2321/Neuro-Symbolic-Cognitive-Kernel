# Neuro-Symbolic Cognitive Kernel (NSCK)

**V13 · February 2026** &nbsp;|&nbsp; Python 3.11+ &nbsp;|&nbsp; Rust accelerators &nbsp;|&nbsp; MIT License

94 modules · 232 classes · ~37K LOC Python · ~4K LOC Rust · 1 437 tests (1 424 pass with Rust)

---

## What Is NSCK?

NSCK is a **neuro-symbolic cognitive architecture** built for research into
machine cognition. It fuses three computational paradigms — **Vector Symbolic
Architecture** (VSA), **Spiking Neural Networks** (SNN), and **Global Workspace
Theory** (GWT) — into a single, glass-box reasoning system where every decision
is fully traceable.

The kernel provides perception, memory, reasoning, learning, language
understanding, and metacognition **without external LLMs, deep learning, or
opaque neural networks**. All intelligence arises from symbolic manipulation of
10 240-bit binary hypervectors, spike-based plasticity, and structured rule
systems. Optional Rust extensions (compiled via PyO3 / maturin) accelerate the
hot paths by 5–85×.

NSCK is a **research framework**, not a production product. It is designed to
explore what a transparent, biologically motivated cognitive architecture can
achieve and where it falls short.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  6  EXECUTIVE          Metacognition · Self-Model · Safety       │
├──────────────────────────────────────────────────────────────────┤
│  5  LANGUAGE           Parser · ConstructionGrammar · NgramNLU   │
│                        FluentNLG · Dialogue · Pragmatics         │
├──────────────────────────────────────────────────────────────────┤
│  4  LEARNING           Hebbian · Q-Learning · RuleInduction      │
│                        ActiveInference · Curiosity · Conformal   │
├──────────────────────────────────────────────────────────────────┤
│  3  REASONING          Causal(ΔP) · Rules · Planning(BFS)       │
│                        Analogy · Spatial · Math · Counterfactual │
├──────────────────────────────────────────────────────────────────┤
│  2  MEMORY             Semantic(KG+HNSW) · Episodic(Rust)       │
│                        Procedural(V13) · CrossModal(V13)         │
├──────────────────────────────────────────────────────────────────┤
│  1  PERCEPTION         10 Modality Adapters · SignalIngestor     │
│                        UniversalHVEncoder · SNN Frontend          │
├──────────────────────────────────────────────────────────────────┤
│  0  SUBSTRATE          VSA Engine (10 240-bit HVs, FHRR mode)   │
│                        GWT Workspace · Dual-Process (Sys1/Sys2)  │
│                        Rust: vsa_rs · snn_rs                     │
└──────────────────────────────────────────────────────────────────┘
```

**Global Workspace coalitions:** RULES · MEMORY · EXPLORATION · Q_LEARNING · PLANNER · MATH · EXTERNAL
**V13 additions:** KLE uncertainty, conformal prediction, pattern generalisation, cross-modal associative memory, signal ingestor, procedural skill cache.

---

## Key Capabilities

| Area | Highlights |
|------|-----------|
| **VSA** | 10 240-bit binary HVs; bind, bundle, permute, similarity. FHRR complex-phasor mode. Rust backend: ~4.1 M ops/s bind, ~3.8 M ops/s similarity, ~1.4 M ops/s bundle (5–65× over Python). |
| **SNN** | LIF neurons, STDP learning, rate & temporal coding. Rust backend (`snn_rs`). |
| **GWT** | Coalition competition, broadcast, mental rehearsal with danger veto, KLE uncertainty (V13). |
| **Dual Process** | System 1 (fast pattern match) / System 2 (deliberate search). |
| **Memory** | Semantic (knowledge graph + HNSW), Episodic (Rust-backed, 10 K capacity), Procedural (V13 skill cache), Cross-Modal Associative (V13). |
| **Reasoning** | Causal (ΔP, MI confounder detection), rules (learning + neural scoring), planning (BFS), analogy (functoriality), spatial, math, belief revision, counterfactuals. |
| **Language** | Left-corner parser, construction grammar, frame semantics, coreference, distributional semantics, pragmatics, NgramNLU (151 K sent/s), FluentNLG, dialogue manager. |
| **Perception** | 10 modality adapters (text, dict, numeric, numeric sequence, SNN, multimodal, stream, image, audio, video), SignalIngestor (V13), UniversalHVEncoder (V13). |
| **Learning** | Hebbian, Q-learning, rule induction, active inference (free energy), curiosity (surprise-driven), cross-domain transfer, continual learning, conformal prediction (V13), pattern generalisation (V13). |
| **Cognitive** | Metacognition, self-model, theory of mind, emotion system, safety verifier. |
| **API** | FastAPI REST endpoints: `/decide`, `/learn`, `/sleep`, `/status`. |
| **Transparency** | Every decision is traceable — no black boxes. |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel.git
cd Neuro-Symbolic-Cognitive-Kernel

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. (Optional) Build Rust accelerators
cd nsck/rust_vsa && maturin develop --release && cd ../..
cd nsck/rust_snn && maturin develop --release && cd ../..

# 4. Run the basic example
python examples/01_cognitive_engine_basic.py
```

### Minimal Code

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nsck"))

from python.core.substrate import NSCKSubstrate

sub = NSCKSubstrate()
sub.register_task("navigation")

result = sub.process("move towards the goal", task_tag="navigation")
print(result.chosen_action, result.confidence, result.explanation)

# Learn from feedback
sub.feedback(result.chosen_action, reward=1.0, task_tag="navigation")

# Consolidate offline
sub.sleep("navigation")
```

---

## Performance (Rust enabled, x86-64)

| Benchmark | Throughput | Latency |
|-----------|-----------|---------|
| VSA bind | 4 109 142 ops/s | 0.24 μs/op |
| VSA similarity | 3 793 104 ops/s | 0.26 μs/op |
| VSA bundle | 1 354 342 ops/s | 0.74 μs/op |
| Memory query (1 K concepts) | — | 0.56 ms |
| Decision loop | — | p50 = 0.13 ms, p99 = 0.20 ms |
| NLU (NgramNLU) | 151 019 sent/s | — |
| Causal ΔP | 2 070 973 ops/s | — |
| SNN LIF step | — | 0.017 ms |

---

## Project Structure

```
Neuro-Symbolic-Cognitive-Kernel/
├── nsck/
│   ├── python/core/        # Core kernel (perception, memory, reasoning, …)
│   ├── rust_vsa/            # Rust VSA accelerator (PyO3/maturin)
│   ├── rust_snn/            # Rust SNN accelerator (PyO3/maturin)
│   ├── api/                 # FastAPI REST interface
│   ├── tests/               # 1 437 tests
│   ├── docs/                # Architecture, formulas, roadmap, references
│   ├── scripts/             # Utility scripts
│   └── data/                # Built-in datasets
├── examples/                # Runnable examples
├── research/papers/         # Research papers (foundation, causal, transfer)
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT
```

---

## Documentation

Detailed docs live in [`nsck/docs/`](nsck/docs/):

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](nsck/docs/ARCHITECTURE.md) | System design and layer breakdown |
| [MODULE_REFERENCE.md](nsck/docs/MODULE_REFERENCE.md) | API reference for all modules |
| [REPOMAP.md](nsck/docs/REPOMAP.md) | Navigation guide to the codebase |
| [FORMULAS.md](nsck/docs/FORMULAS.md) | Mathematical foundations |
| [RUST_API_REFERENCE.md](nsck/docs/RUST_API_REFERENCE.md) | Rust extension API |
| [TESTING.md](nsck/docs/TESTING.md) | Test suite guide |
| [NSCK_ROADMAP_AND_PLAN.md](nsck/docs/NSCK_ROADMAP_AND_PLAN.md) | Roadmap and future plans |
| [TRANSPARENCY_GUARANTEE.md](nsck/docs/TRANSPARENCY_GUARANTEE.md) | Explainability commitments |

---

## Research Papers

Companion papers are in [`research/papers/`](research/papers/):

1. **Foundation** — VSA fundamentals and the cognitive kernel design
2. **Causal Reasoning** — ΔP calculus with mutual-information confounder detection
3. **Cross-Domain Transfer** — Analogical transfer across task domains

---

## Known Limitations

NSCK is a research prototype. Be aware of these constraints:

- **No real NLU.** NgramNLU is fast but shallow — it uses n-gram heuristics, not deep semantic parsing. It will misinterpret complex or ambiguous sentences.
- **No deep learning.** There are no transformer models, CNNs, or gradient-based learners. All "neural" components are symbolic or spike-based.
- **SNN Rust FFI overhead.** Crossing the Python↔Rust boundary per spike step adds latency. Gains are clearest for batch workloads.
- **Cold-start analogy.** The analogy engine requires populated memory; it cannot generalise from a single example.
- **Perception adapters are schematic.** Image, audio, and video adapters provide the interface but not production-grade feature extraction.
- **Scale ceiling.** The architecture is tested up to ~10 K concepts in semantic memory. Behaviour at larger scales is unexplored.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

## License

[MIT](LICENSE) — see the LICENSE file for full text.
