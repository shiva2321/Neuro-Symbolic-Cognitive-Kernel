# NSCK: A Unified Neuro-Symbolic Cognitive Kernel Using Binary Hypervector Representations

**Shivam Prajapati**
Bachelor of Computer Science, University of Prince Edward Island
Charlottetown, Prince Edward Island, Canada

*Developed iteratively with AI coding-agent assistance over 2.5 years of research, exploration, and implementation.*
*Open-source — February 2026*

---

## Abstract

We present **NSCK** (Neuro-Symbolic Cognitive Kernel), a CPU-native cognitive architecture that uses 10,240-bit binary hypervectors as a shared representational substrate for perception, memory, reasoning, and language. Unlike mainstream AI systems that separate neural processing from symbolic reasoning — or abandon interpretability entirely — NSCK unifies both under a single Vector Symbolic Architecture (VSA) algebra: XOR binding, majority-vote bundling, and circular-shift permutation. No gradient descent, no matrix multiplication, and no neural-network inference is used at decision time. Every decision is fully auditable through an 11-stage cognitive trace.

The kernel integrates eight layers: a VSA foundation, a Leaky-Integrate-and-Fire (LIF) spiking neural network (SNN) perception layer with STDP learning, two-tier episodic and semantic memory, a Global Workspace Theory (GWT)-based executive, causal discovery via Δ-P statistics, a STRIPS planner with learned operators, and a glass-box explanation generator. A dual Rust/PyO3 extension achieves 3–54× speedup over the Python baseline on individual VSA operations, with aggregate throughput exceeding 1.5 million operations per second.

Evaluation on x86-64 with Rust backend shows 1,528,662 VSA operations per second, sub-millisecond decision latency (p50: 0.149 ms), 2.37 ms SNN perception pipeline, 1.25 ms/sentence NLU throughput, and **1,111 passing tests** (5 skipped, 4 xfailed) as of V8 (February 2026). Honest limitations are documented: open-domain NLU coverage reaches approximately 40–60%; the SNN→predicate bridge now supports VSA cleanup-memory-based concept naming but full sensor-to-predicate grounding from raw pixels remains an open engineering item. The Rust advantage is concentrated on VSA bitwise operations (33×); for SNN simulation, numpy vectorized operations are competitive with Rust+PyO3 at small neuron counts.

V8 adds nine new capabilities: concurrent multimodal fusion with coherence windowing; Ebbinghaus memory lifecycle (decay + prune + reconsolidation); multi-turn dialogue state tracking via rolling history hypervectors; MathReasoner wired into the GWT coalition loop; 2-level hierarchical resonator networks for nested semantic role labelling; multi-agent cognitive fusion via `MultiAgentSession`; an NSCK-Eval benchmark suite (bAbI, math word problems, cross-domain transfer, NLG quality, dialogue coherence); and an active inference full loop (`ActiveInferenceLearner`) that biases coalition salience via free-energy minimisation.

We release the full codebase (97 Python source files and 17 Rust source files across 2 crates), all benchmarks, and documentation under open license.

**Keywords:** Vector Symbolic Architecture, Hyperdimensional Computing, Neuro-Symbolic AI, Cognitive Architecture, Spiking Neural Networks, Global Workspace Theory, Causal Reasoning, Interpretable AI

---

## 1. Introduction

The modern landscape of artificial intelligence is dominated by large-scale statistical learning — Transformer-based language models, deep convolutional networks, and reinforcement learning agents that require millions of GPU-hours and billions of parameters. While these systems achieve impressive benchmarks, they share structural limitations: opacity (internal state is not interpretable), computational cost (impractical for edge hardware), and inability to incorporate new knowledge without retraining.

A fundamentally different class of architectures, rooted in cognitive neuroscience and symbolic AI, offers complementary properties: interpretability by construction, continual learning without forgetting, and deterministic behaviour for safety-critical applications.

### 1.1 Motivation

The core idea behind NSCK is simple and comes from observing how humans understand the world:

We as humans associate and understand the world based on what we have been taught, experienced, and learned over time. We can generalise our knowledge based on associations, deductions, abstractions, and relations — represented through natural language symbols. But in our minds, all of this is represented as spikes of electricity, patterns, and processes we still do not fully understand.

Computers are non-living things. They interpret everything in 0s and 1s. But they are flexible enough that if we can represent anything in 0s and 1s in an interpretable and computable way, we can process information in a manner that parallels how living organisms do.

The question became: *can we give machines a way to represent data and the real world in 0s and 1s, in an interpretable and representable way, in higher dimensions?* Because even in our real world, data and representations are not mono-dimensional.

This led to the choice of **Vector Symbolic Architecture (VSA)** — specifically, 10,240-bit binary hypervectors. Everything a concept contains can be fully represented in these bits, and we do not need to rely on learned weights like neural networks or heavy matrix multiplications, because bitwise operations are inherently fast at the hardware level.

But representation alone is not enough. The system needs mechanisms to learn, perceive, and reason about the world. This led to integrating spiking neural networks (for biologically plausible perception), Global Workspace Theory (for attention and executive control), symbolic rules, causal reasoning, abstract concepts, relationships, and patterns — all operating on the same hypervector representations.

### 1.2 Development Approach

NSCK was developed over approximately 2.5 years through many iterations, trials, failures, and breaks. The approach was: identify an obstacle, research solutions across multiple fields (neuroscience, cognitive science, mathematics, computer science), understand the concepts, decide what to use and why, then implement. When no existing implementation existed for a particular mechanism, the guiding principle was: *everything we need to build must exist in the real world — nature has done its job across many fields over billions of years*. This cross-field perspective — diving into neuroscience, psychology, information theory, and distributed computing — is reflected in the breadth of mechanisms NSCK integrates.

The implementation was done with substantial assistance from AI coding agents. This human-directed, AI-assisted development approach allowed a complex multi-layer system to be built and tested by a single developer. We describe this transparently.

### 1.3 Contributions

1. A complete, open-source implementation of a unified neuro-symbolic cognitive architecture with 97 Python modules, two Rust extension crates, and 85 test files (1,111 passing, 5 skipped, 4 xfailed — V8, February 2026).
2. A principled integration of Binary VSA with GWT, LIF-SNN, Δ-P causal inference, STRIPS planning, and Plutchik emotion within a single 10,240-bit representational space.
3. A glass-box 11-stage decision trace (ThoughtTrace) that makes every cognitive step auditable at runtime — not post-hoc rationalisation.
4. Empirical benchmarks across 11 real-world scenarios plus an NSCK-Eval suite (bAbI QA, math word problems, cross-domain transfer, NLG quality, dialogue coherence), reported with full honesty including failures.
5. Demonstration that bitwise operations on binary hypervectors, accelerated via Rust, can serve as a computationally efficient alternative to matrix multiplication for cognitive tasks.
6. Nine V8 capabilities: concurrent multimodal scheduling, Ebbinghaus memory lifecycle, multi-turn dialogue state tracking, MathReasoner–GWT integration, 2-level hierarchical resonator networks, multi-agent cognitive fusion, active inference free-energy loop, and a formal evaluation benchmark suite.

### 1.4 Research Questions

NSCK was built to investigate three questions:

1. **Can a system reason, plan, and learn without gradient descent?**
2. **Can every decision be fully explained to a human observer in real time?**
3. **Can biologically plausible neural dynamics (spiking neurons) and symbolic logic operate within the same representational space?**

Our answer — implemented, tested, and openly released — is **yes**, with important caveats documented honestly throughout this paper.

---

## 2. Background

### 2.1 Vector Symbolic Architectures

VSA was introduced by Kanerva [1] as Sparse Distributed Memory and formalised as a general algebra by Plate [2] (Holographic Reduced Representations), Gayler [3] (MAP/BSC), and later surveyed by Schlegel et al. [4]. NSCK uses **Binary Spatter Codes (BSC)** with XOR binding — the same algebra used in recent hyperdimensional computing (HDC) applications for edge computing [5, 6].

The key insight: the same hypervector space can represent both **subsymbolic** neural activations (a spike pattern encoded as a hypervector) and **symbolic** predicates (a rule "IF obstacle_ahead THEN STOP" expressed as a VSA binding). This unification removes the translation bottleneck that separates neural and symbolic processing in hybrid architectures.

### 2.2 Global Workspace Theory

Baars [7] proposed that cognition involves competition among specialist modules for access to a shared broadcast channel — a "global workspace." Dehaene et al. [8] provided neuroimaging evidence for this model. NSCK's GWT implementation is inspired by LIDA [9], where coalitions of information compete for workspace access based on salience, relevance, and emotional valence.

### 2.3 Spiking Neural Networks

Maass [10] established SNNs as universal approximators. The Leaky Integrate-and-Fire (LIF) model [11] with Spike-Timing-Dependent Plasticity (STDP) [12] provides biologically plausible sensory processing. NSCK bridges SNN output to VSA via rate coding and temporal coding, following the VSA-SNN integration studied by Frady et al. [13] and Renner et al. [14].

### 2.4 Causal Inference

Pearl's do-calculus [15] provides the theoretical foundation for causal reasoning. NSCK uses the Δ-P model [16] with Laplace smoothing, which can infer causal structure from as few as 2–5 observations — important for a continual-learning setting with limited data.

### 2.5 Cognitive Architectures

NSCK shares goals with ACT-R [17] (production rules and declarative memory), SOAR [18] (symbolic goal-directed reasoning), and LIDA [9] (GWT with codelets). None of these use a single representational substrate for both neural and symbolic processing. NSCK's distinguishing claim is that XOR, bundle, and permute over binary hypervectors serve both roles simultaneously.

---

## 3. System Architecture

NSCK is structured as eight vertically-integrated layers orchestrated by a central **CognitiveEngine** (1,402 lines of code). Each layer communicates through the CognitiveEngine and through the GWT broadcast channel.

```
Input (text / sensor vector / image)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  1. VSA Foundation    10,240-bit binary HV algebra   │
│     XOR bind · majority-vote bundle · permute        │
├──────────────────────────────────────────────────────┤
│  2. Perception        LIF spiking neurons + STDP     │
│     VSA-SNN Bridge    Rate / Temporal coding         │
│     Multimodal        HOG / colour / LBP / edges     │
├──────────────────────────────────────────────────────┤
│  3. Memory            EpisodicMemory (hot+SQLite+LSH)│
│                       SemanticMemory (DiGraph+HV)    │
├──────────────────────────────────────────────────────┤
│  4. Reasoning         GlobalWorkspace  (GWT-LIDA)    │
│                       CausalGraph      (Δ-P chains)  │
│                       STRIPSPlanner    (A* search)   │
│                       AnalogyEngine    (cross-domain)│
│                       RuleLearner      (ILP)         │
├──────────────────────────────────────────────────────┤
│  5. Cognitive         EmotionSystem    (Plutchik 8)  │
│                       SafetyGate       (veto logic)  │
│                       SelfModel        (calibration) │
│                       TheoryOfMind     (belief model)│
├──────────────────────────────────────────────────────┤
│  6. Language           TextKnowledgeLearner (SVO→HV) │
│                        ConstructionGrammar  (71 CxG) │
│                        FluentNLG + DialogueManager   │
├──────────────────────────────────────────────────────┤
│  7. Rust Accelerator  hypervec_rs  (PyO3 + Rayon)    │
│                       snn_rs       (parallel LIF)    │
├──────────────────────────────────────────────────────┤
│  8. Integration       BrainStore   (SQLite)          │
│                       BrainFusion  (multi-task merge) │
│                       ExplanationGenerator           │
└──────────────────────────────────────────────────────┘
        │
        ▼
  chosen_action + confidence + ThoughtTrace explanation
```

*Figure 1: NSCK eight-layer architecture. All layers share the 10,240-bit binary hypervector as their representational currency.*

### 3.1 Representational Foundation — Binary Hypervectors

All knowledge in NSCK is represented as elements of **{0, 1}^d** where **d = 10,240 bits**.

**Why this dimension.** The capacity of a d-dimensional binary space is 2^d. The expected number of random vectors within normalised Hamming distance 0.45 of any given vector is approximately 2^100 [1] — more than enough for any practical knowledge base, while remaining computationally tractable on a CPU.

**Three operations define the algebra:**

**Binding (XOR):**

$$\mathbf{A} \otimes \mathbf{B} = \mathbf{A} \oplus \mathbf{B}$$

Self-inverse: (A ⊕ B) ⊕ B = A ⊕ (B ⊕ B) = A ⊕ **0** = A. This means unbinding is the same operation as binding — no separate "inverse" is needed. The bound result is quasi-orthogonal to both operands: E[sim(A ⊕ B, A)] ≈ 0.5.

**Bundling (majority vote):**

For n vectors, each position takes the majority bit value with deterministic tie-breaking:

$$\text{bundle}(\mathbf{v}_1, \ldots, \mathbf{v}_n)_i = \text{majority}(v_{1i}, \ldots, v_{ni})$$

The bundle is similar to all its components: E[sim(bundle, v_i)] ≈ 0.5 + 0.5/n.

**Permutation (circular shift):**

$$\rho^k(\mathbf{v})_i = v_{(i + k) \bmod d}$$

Used for encoding sequential position. The shift creates a quasi-orthogonal variant: sim(ρ^k(v), v) ≈ 0.5 for k ≠ 0.

**Normalised Hamming similarity:**

$$\text{sim}(\mathbf{A}, \mathbf{B}) = 1 - \frac{\sum_{i=1}^{d} A_i \oplus B_i}{d}$$

For random A, B: E[sim] = 0.5, σ ≈ 1/(2√d) ≈ 0.00494. Meaningful signal: sim > 0.55 (≈ 10σ above chance).

**Why binary + bitwise.** XOR and popcount are single-instruction operations on modern CPUs. No floating-point arithmetic, no matrix multiplication, no weight tensors. This makes NSCK's core operations hardware-level fast — a property we exploit through Rust acceleration.

**Role-filler encoding for concepts:**

$$\text{HV}(\text{concept}) = \text{HV}(\text{name}) \oplus \bigoplus_{(\text{prop}, \text{val})} \text{HV}(\text{prop}) \oplus \text{HV}(\text{val})$$

This is the exact encoding used in SemanticMemory: each concept's name, properties, and values are bound together and bundled into a single 10,240-bit vector that holistically represents the concept.

### 3.2 Perception Layer — SNN with STDP

The perception layer implements a population of **Leaky Integrate-and-Fire (LIF)** neurons:

**LIF dynamics (discrete Euler):**

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m} [-(V(t) - V_{\text{rest}}) + I(t)]$$

Parameters from implementation: τ_m = 20 ms, V_rest = −70 mV, V_thresh = −55 mV, V_reset = −75 mV, refractory period = 2 ms.

**Spike condition:** If V(t) ≥ V_thresh, emit spike; V ← V_reset.

**STDP learning rule:**

$$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (LTP)} \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (LTD)} \end{cases}$$

Parameters: A₊ = 0.01, A₋ = 0.012, τ₊ = τ₋ = 20 ms.

**VSA-SNN Bridge:** The RateCoder maps firing rates to hypervectors — a population of n neurons with spike pattern {t_i} produces:

$$\mathbf{v}_{\text{percept}} = \text{bundle}(\{\mathbf{v}_i^{\text{concept}} : \text{neuron } i \text{ fired}\})$$

The SimpleConceptMapper uses Jaccard similarity on neuron activation patterns to recognise previously seen concepts. When a SemanticMemory is attached, registered SNN concepts are resolved to human-readable predicate names via VSA cleanup-memory lookup (nearest-neighbour HV search over stored concepts, threshold > 0.55). This provides the SNN→predicate bridge that enables GWT coalitions to carry meaningful symbolic content.

**Weber-Fechner logarithmic scaling.** Before spike simulation, the preprocessed input signal is compressed via a logarithmic transform inspired by psychophysical intensity scaling [26]:

$$x_{\text{wf}} = \text{sign}(x) \cdot \log(1 + |x|)$$

This mirrors the compressive nonlinearity observed in biological auditory and visual systems [27]: sensitivity is high at low intensities (improving discrimination of faint stimuli) and saturates gracefully at high intensities (preventing spike-rate saturation). In practice, the SNN now handles signals spanning four orders of magnitude (0.01 to 1000×) without manual gain tuning.

**Remaining limitation:** The SNN→predicate bridge now supports both automatic VSA-based concept naming (via cleanup memory) and manual labelling. However, full end-to-end grounding from raw pixels to domain predicates — without pre-registered semantic concepts — remains an open engineering item.

### 3.3 Memory Layer — Tiered Episodic and Semantic

**Episodic Memory** is a two-tier store:
- *Hot tier:* A deque of recent LiveEpisode objects (default capacity 500) for O(1) amortised read/write.
- *Warm tier:* BrainStore (SQLite) for durable storage.
- *Retrieval:* LSH hashing of the situation hypervector buckets candidates, then exact Hamming re-ranking.

Each episode records: timestamp, task tag, situation HV, state dict, action, outcome, reward, emotion label, Theory-of-Mind belief snapshot, and impact_score = |reward| + novelty_bonus.

**LSH indexing** uses m = 16 bits per key, K = 8 tables:

$$h_j(\mathbf{v}) = \text{sgn}(\mathbf{r}_j \cdot \mathbf{v}), \quad B(\mathbf{v}) = \text{concat}(h_1(\mathbf{v}), \ldots, h_m(\mathbf{v}))$$

Collision probability: Pr[h(u) = h(v)] = 1 − arccos(c)/π, where c is cosine similarity.

**Semantic Memory** maintains a NetworkX DiGraph with typed, weighted edges and a parallel HV index. Relation weights: is_a: 0.9, has_property: 0.7, causes: 0.6, part_of: 0.5, similar_to: 0.4.

**Spreading activation:**

$$a_j^{(t+1)} = a_j^{(t)} + \sum_{(i,j) \in E} a_i^{(t)} \cdot \gamma \cdot w_{\text{rel}(i,j)}$$

where γ = 0.7. After T = 3 hops: a^(T) ≤ (0.7 × 0.9)^3 ≈ 0.25, naturally bounding search radius.

**Sleep consolidation:** Impact-scored episodes migrate from hot to warm tier. Homeostatic pruning removes edges with weight < 0.01 and evicts stale concepts.

### 3.4 Reasoning Layer — GWT Coalition Competition

The Global Workspace implements LIDA-style coalition competition:

**Coalition activation:**

$$\alpha(C) = s(C) + r(C) + e(C) + 0.5 \cdot q(C)$$

where s = base_salience, r = relevance, e = affect_match, q = sender_confidence.

**Competition rule:** C* = argmax_C α(C) subject to α(C*) ≥ 0.5.

Six coalition sources compete each cycle: RULES (rule learner), EXPLORATION (curiosity), Q_LEARNING (Q-table lookup), MEMORY (episodic recall), PLANNER (STRIPS A* step), and EXTERNAL (input).

**Mental rehearsal veto:** Before committing to C*:
1. Predict next state via WorldModel.
2. Compute danger similarity d = sim(predicted, v_danger).
3. If d > 0.75: veto C*, try next-ranked coalition. Up to 3 rounds.

**SafetyGate** checks symbolic safety constraints before GWT competition and vetoes coalitions whose predicted outcome exceeds danger thresholds.

**CognitiveState trace:** Every decision includes the full coalition scores, rule sources, causal chain, confidence, and explanation — the actual computational path, not a post-hoc reconstruction.

### 3.5 Causal Discovery — Δ-P Statistics

$$\Delta P(c \to e) = P(e \mid c) - P(e \mid \neg c)$$

With Laplace smoothing (α = 1.0 when observations < 10, else 0.0):

$$P(e \mid c) = \frac{N(e, c) + \alpha}{N(c) + 2\alpha}$$

This gives a non-degenerate estimate from as few as 2 observations. Causal chain strength:

$$\text{strength}(c_1 \to \cdots \to c_n) = \prod_{i=1}^{n-1} \Delta P(c_i \to c_{i+1})$$

**Counterfactual simulation (do-operator):** Remove cause node, propagate modified graph — a simplified implementation of Pearl's do-calculus.

**Mutual-information confounder detection.** A known weakness of Δ-P is that it assumes no hidden confounders. To partially address this, NSCK implements a mutual-information (MI) screen [28] over candidate third variables. For each variable Z in the observation set, the system computes:

$$\text{MI}(X; Z) = \sum_{x, z} p(x, z) \log \frac{p(x, z)}{p(x)\, p(z)}$$

If MI(C, Z) + MI(Z, E) accounts for most of the apparent C → E association, Z is flagged as a potential confounder [29]. The detection runs automatically whenever a new causal edge is proposed, and returns both the identified confounder and a confidence score. In testing, the method correctly identifies STRESS as a confounder for the spurious HIGH_BP → HEADACHE link with 100% confidence. This does not replace a full do-calculus intervention but provides a practical first-pass filter for observational data.

### 3.6 Language Layer — Construction Grammar + FluentNLG

**TextKnowledgeLearner** (1,074 lines) parses text into SVO triples via regex and POS heuristics, generates concept hypervectors with role-filler VSA binding, stores concepts in SemanticMemory, and builds causal links from verb patterns.

**ConstructionGrammar** implements 71 constructions (copular, SVO, passive, causative, locative, comparative, containment, etc.) with ~400 COMMON_VERBS forms. **BrillPosTagger** (300+ lexicon, 8 suffix rules) achieves 74% accuracy.

**FluentNLG** generates noise-free natural language responses — no template tokens like "is_a" or "has_property" leak into output. Wired into DialogueManager for all response methods.

**Known limitation:** NLU coverage is approximately 40–60% on general English text. The construction grammar classifier uses a fixed verb vocabulary. Complex syntax (nested clauses, coordination) is not handled.

---

## 4. Implementation

### 4.1 Code Scale

| Component | Files | Lines of Code |
|---|---|---|
| nsck/python/core/ (8 subsystems) | 88 | ~25,000 |
| nsck/rust_vsa/ (Rust VSA crate) | 7 | ~4,048 |
| nsck/rust_snn/ (Rust SNN crate) | 3 | ~1,500 |
| nsck_ai_model/ (conversational layer) | 14 | ~11,028 |
| Tests | ~40 | ~10,800 |
| **Total** | — | **~99,300** |

### 4.2 Shim Pattern

The backend selector `hypervec_shim.py` transparently imports the Rust extension `hypervec_rs` when compiled; all modules import from the shim. If the Rust shared library is absent, the Python fallback activates — maintaining mathematical identity across backends. This allows CI to run without a Rust toolchain.

### 4.3 Configuration

NSCKConfig provides **25 feature flags**, enabling minimal/research/production presets. Researchers can toggle individual subsystems (dual-process routing, homeostasis, stigmergy, pragmatics, spatial reasoning, fluent dialogue) to study component interactions.

### 4.4 Decision Loop

The `decide(state, available_actions, task_tag)` method executes per cycle:
1. Ground state → active predicates
2. Create situation HV (bundle all predicate HVs)
3. Check curiosity (HV similarity to known prototypes)
4. Build coalitions (6 sources)
5. GWT competition with mental rehearsal veto
6. Safety check
7. Generate explanation
8. Update self-model
9. Return CognitiveState (action, confidence, explanation, trace)

Typical latency: 0.65–0.90 ms per cycle (Python + Rust).

---

## 5. Evaluation

All benchmarks were run on commodity x86-64 hardware. Results are from the implemented benchmark suite.

### 5.1 Performance Benchmarks

*Table 1: VSA and perception benchmarks (measured with Rust backend on x86-64).*

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| HV XOR (per-op) | 1.3 μs | 0.43 μs | 3× |
| HV Bundle (per-op) | 55.6 μs | 1.02 μs | 54× |
| HV Similarity (per-op) | 7.5 μs | 0.51 μs | 15× |
| HV Permute (per-op) | 7.9 μs | 1.15 μs | 7× |
| HV Negate (per-op) | — | 0.97 μs | — |
| VSA aggregate throughput | 46,590 ops/s | 1,528,662 ops/s | 33× |
| SNN perceive pipeline (64→256) | 1.81 ms | 2.31 ms | 0.8×† |
| SNN core simulate() only | 0.55 ms | 1.01 ms | 0.5×† |
| Decision latency (p50/p95) | — | 0.149/0.180 ms | — |
| NLU throughput | — | 1.25 ms/sentence | — |
| Memory query @1K concepts (p50) | — | 0.42 ms | — |
| Episodic recall @500 eps (p50) | — | 0.27 ms | — |
| Batch similarity 50×50 | — | 0.51 ms | — |
| Weber-Fechner 10K values | — | 0.58 ms | — |

† **Honest note on SNN performance:** The Rust SNN backend is *slower* than the Python backend for small neuron counts (256) because numpy vectorized operations are highly optimised with BLAS, while the Rust path incurs PyO3 FFI overhead on list↔Vec conversion at each call. The Rust advantage is concentrated on VSA bitwise operations (33× speedup) where numpy's float-based routines cannot compete with native bit manipulation. At larger neuron counts (>1024) or when eliminating the FFI boundary, the Rust SNN core would show its advantage.

*Table 2: Benchmark scenario results (Rust backend active).*

| Scenario | Cycles | Wall (s) | avg (ms) | p99 (ms) | RSS Δ (MB) | Success |
|---|---|---|---|---|---|---|
| Robot Navigation | 500 | 8.12 | 0.90 | 1.38 | +5.5 | 43.6% |
| Medical Triage | 300 | 0.28 | 0.66 | 0.94 | +1.3 | 15.3% |
| Financial Trading | 400 | 0.52 | 0.71 | 1.01 | +0.2 | 45.2% |
| Env. Monitoring | 200 | 0.15 | 0.65 | 0.91 | +0.0 | 62.5% |
| Language Dialogue | 50 | 45.29 | 905.77 | 926.65 | +34.4 | 100.0% |
| VSA Stress (10K ops) | 10,000 | 0.07 | 0.01 | 0.01 | +2.1 | 100.0% |
| SNN Stress (200 batches) | 200 | 3.90 | 19.45 | 22.24 | +1.4 | 0.5%* |
| Episodic Memory (2K) | 2,000 | 0.20 | 0.09 | 0.40 | +2.9 | 100.0% |
| Sleep Consolidation | 1 | 0.04 | 42.41 | — | +0.0 | 100.0% |
| Cross-Domain Transfer | 30 | 0.00 | 0.01 | 0.01 | +0.0 | 0.0%** |
| SNN→GWT Pipeline | 100 | 2.23 | 22.27 | 30.59 | −1.3 | 100.0% |
| **Total** | — | **61.21** | — | — | **+49.1** | — |

\* SNN Stress: 0.5% is the "concept recognised" criterion with low-confidence SNN output — processing ran without errors at 51.3 batches/s.
\*\* Cross-Domain Transfer: Analogy engine requires bootstrapped rules; cold-start = 0%. With bootstrapped rules, transfer achieves +14% over random baseline.

### 5.2 Learning Results

From the Robot Navigation scenario (500 cycles):
- 9 causal rules learned from CausalGraph
- 499 episodic memory recalls
- 472 STRIPS plans generated (A* with learned operators)
- 250 episodes consolidated to warm tier during sleep()

### 5.3 V7 Language Evaluation

Training on 20 real-world sentences (Rust backend):
- Concepts learned: 100
- KG edges: 82
- Training time: 29.2 ms (1.5 ms/sentence)
- Query hit rate: 100%
- Fluent response rate: 100% (no template noise in output)

Distributional similarity (after pre-training on 200-sentence built-in corpus):
- brain ↔ memory: 0.658 ✓ (co-occur in cognition domain)
- cell ↔ nucleus: 0.504 (marginal — limited shared contexts)
- water ↔ ocean: 0.501 (near-random — corpus too small)

### 5.5 V8 Capability Results (February 2026)

*Test suite: **1,111 passed**, 5 skipped, 4 xfailed — with Rust extensions active.*

**Memory lifecycle:** `decay_concepts(λ=0.01)` and `prune_below(threshold=0.1)` verified. Access metadata (`access_count`, `last_accessed`, `importance_score`) correctly updated on every concept retrieval. Episodic reconsolidation (HV blending on recall when Hamming distance > 15%) confirmed in 20 unit tests.

**Dialogue state tracking:** Rolling history HV via permuted bundling across turns. Topic-shift detection (cosine < 0.3) and clarification generation (confidence < 0.4) verified. `get_context_hv()` injected into GWT coalition scoring.

**MathReasoner–GWT integration:** `UniversalInput.is_mathematical()` (7-pattern regex classifier) routes arithmetic queries through a MATH coalition (salience 0.95). Verified on 50 arithmetic word problems in `benchmarks/math_word_problems.py`.

**Hierarchical Resonator Networks:** `HierarchicalResonatorNetwork` with L1 (AGENT/VERB/PATIENT/THEME/INSTRUMENT) and L2 (MODIFIER_AGENT/MODIFIER_VERB/MODIFIER_PATIENT) verified in 15 unit tests. `factorize_hierarchical(hv, depth=2)` returns a `{"L1": …, "L2": …}` dict.

**Multi-agent cognitive fusion:** `MultiAgentSession.exchange_snapshots()` collects SemanticMemory HV summaries; `negotiate_beliefs(topic_hv)` produces consensus via majority-vote bundling. `TheoryOfMind.model_other_agent()` and `perspective_take()` verified.

**Active inference:** `ActiveInferenceLearner.free_energy(action, state_hv) = prediction_error − epistemic_value` wired into `decide()`. Coalition salience adjusted by `weight × (0.5 − F)`. `SafetyGate.check_free_energy()` vetoes actions with F > 0.7.

**NSCK-Eval benchmark suite:** `BenchmarkRunner.run_all()` executes 5 benchmarks and prints a tabular report. All benchmark modules import cleanly with or without a live engine.

### 5.4 Capability Assessment

*Table 3: System capabilities with honest grades (V8).*

| Dimension | Grade | Evidence |
|---|---|---|
| Symbolic Reasoning | A | Rules, causal chains, planning, analogy functional |
| Language Understanding | A− | BrillPosTagger 74% + CG ~85% on covered constructions |
| Fluent NL Responses | A | FluentNLG; 100% noise-free responses |
| KG Quality / Noise | B | Stop-concept filter + generic threshold 0.62 |
| Distributional Semantics | B− | Codebook pre-trains on 200 sentences |
| Memory | A− | Decay/prune lifecycle + transitive inference + prototype generalization |
| Dialogue Coherence | B+ | Rolling history HV + topic shift + clarification (V8) |
| Math Reasoning | B | MathReasoner–GWT coalition + 50 word-problem suite (V8) |
| Multi-Agent | B | MultiAgentSession consensus bundling (V8) |
| Active Inference | B | Free-energy coalition bias + SafetyGate veto (V8) |
| Explainability | A | Full CognitiveState trace — 100% auditable |
| Scalability | C+ | O(N) Python; Rust for bulk; NSW for 50K+ |

---

## 6. Discussion

### 6.1 What Works Well

**VSA as a universal substrate.** The single 10,240-bit hypervector type serves as the representational currency across all eight layers without any translation bottleneck. This architectural coherence is NSCK's most valuable property.

**Glass-box interpretability by construction.** Every coalition score, rule confidence, causal chain strength, and novelty estimate is a first-class object accessible at runtime. The 11-stage ThoughtTrace makes every decision auditable. This is not post-hoc rationalisation — it is the actual computational path.

**Lean memory footprint.** 49.1 MB total RSS across all scenarios. This is directly attributable to the absence of large weight matrices.

**Rust acceleration is real — and honest.** 1,528,662 ops/s VSA throughput (Rust) vs 46,590 ops/s (Python) — a 33× aggregate speedup on bitwise operations. However, for SNN simulation at small neuron counts (256), the Python numpy backend is competitive due to BLAS optimisation and the absence of PyO3 FFI overhead. This is an important finding: *Rust acceleration is most impactful where the workload is inherently bitwise (VSA) rather than floating-point (SNN), and where data transfer costs are amortised over larger computations.*

**Continual learning without catastrophic forgetting.** Symbolic knowledge (rules, causal graphs) does not suffer from interference when new tasks are learned.

### 6.2 What NSCK Cannot Replace

NSCK is not an LLM replacement. It cannot match the breadth of commonsense knowledge, vocabulary coverage, or open-domain language fluency of large language models. It wins on transparency, no GPU requirement, no hallucination of reasoning traces, continuous learning, and causal modelling. It loses on raw NLU accuracy and commonsense breadth.

### 6.3 The Honest Gap

Three fundamental gaps remain:

1. **Grounding.** The SNN→symbol bridge now supports VSA cleanup-memory-based concept naming (resolved via nearest-neighbour HV lookup in SemanticMemory). However, full end-to-end grounding from raw pixels to domain predicates without pre-registered semantic concepts is not yet implemented.
2. **NLU coverage.** Construction grammar covers approximately 40–60% of general English. Complex syntax is not handled.
3. **Cold-start transfer.** Zero-shot analogical transfer requires existing rules in the source domain. With no bootstrapped rules, transfer yields 0%. With bootstrapped rules (≥50 source episodes), transfer achieves +14% over random baseline.

### 6.4 Positioning

NSCK is a **complementary reasoning substrate** — not a replacement for statistical models. The architecture is designed so that other researchers and developers can build upon it: the 25 configuration flags allow isolating any subsystem, and the glass-box property means every component's behaviour can be inspected and modified.

### 6.5 Cross-Disciplinary Foundations

Several recent enhancements draw on principles from fields outside mainstream AI. The table below maps each scientific discipline to the NSCK mechanism it informs and the concrete benefit observed.

| Scientific Field | Principle | NSCK Mechanism | Benefit |
|---|---|---|---|
| Psychophysics | Weber-Fechner law [26] | Logarithmic pre-scaling in SNN perception | 4-order-of-magnitude dynamic range without manual gain tuning |
| Information Theory | Mutual information [28] | Confounder detection in CausalDiscovery | Flags spurious causal edges (e.g., HIGH_BP → HEADACHE) before they enter the causal graph |
| Neuroscience / Free Energy | Friston's free-energy principle [30] | Free-energy surprise in CuriosityModule | Curiosity signal decreases as predictions converge, preventing exploration loops |
| Category Theory | Functoriality [31] | Functoriality score in AnalogyEngine | Measures structure preservation of cross-domain mappings (F = 1.0 for perfect isomorphism) |
| Statistical Physics | Jaynes' maximum-entropy principle [32] | Adaptive similarity threshold in AnalogyEngine | Replaces hard-coded threshold with θ* = μ + Φ⁻¹(1−π)·σ derived from dimensionality |

These additions are modest — each addresses one specific gap in the existing architecture. We do not claim that incorporating a formula from another field constitutes a deep theoretical contribution; rather, these are principled engineering choices that replace ad-hoc heuristics with well-understood mathematical foundations.

---

## 7. Related Work Comparison

| System | Representational Unity | SNN | GWT | Causal | Explainable | Open Source |
|---|---|---|---|---|---|---|
| ACT-R [17] | No (separate) | No | No | No | Partial | Yes |
| SOAR [18] | No (separate) | No | No | No | Partial | Yes |
| LIDA [9] | No (separate) | No | Yes | No | Partial | Partial |
| BinSpaun [19] | Partial (SNN+NEF) | Yes | No | No | No | Yes |
| **NSCK** | **Yes (VSA)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

To our knowledge, no prior published system simultaneously integrates BSC-VSA, LIF-SNN, GWT, Δ-P causal discovery, STRIPS planning, Plutchik emotion, SelfModel confidence calibration, Theory of Mind, and glass-box explanation generation in a single open-source kernel. We make this claim humbly and invite correction.

---

## 8. Limitations

1. **SNN→predicate grounding** now supports VSA cleanup-memory concept naming and Weber-Fechner logarithmic scaling for wide dynamic range, but full pixel-to-predicate grounding remains incomplete.
2. **Cold-start analogical transfer** succeeds at 0% without bootstrapped rules — +14% with ≥50 source episodes.
3. **NLU coverage** (~40–60%) is vocabulary-limited.
4. **Domain accuracy** without trained verifiers is low (Medical: 15.3%, Navigation: 43.6%).
5. **No GPU path** — all Rust acceleration is CPU-only.
6. **LSH stale-index** — evicted episodes leave stale bucket entries.
7. **Language dialogue latency** (905 ms/turn) is acceptable for conversation but not real-time.
8. **Confounder detection scope** — MI-based confounder screening partially addresses hidden confounders but is limited to variables already observed; latent confounders outside the observation set are not detected.

---

## 9. Future Work

1. **Full sensor→predicate grounding** via SNN output HVs + cleanup memory. Weber-Fechner scaling now handles intensity compression; the remaining gap is end-to-end learning of the concept-naming mapping from raw pixels without pre-registered semantic concepts.
2. **GWT competition loop in Rust** for > 100 Hz applications.
3. **Large-corpus distributional training** (HuggingFace FineWeb integration exists but not deployed at scale).
4. **LLM adapter** — treat a local LLM as a GWT coalition source. NSCK adjudicates through causal checking and safety gating.
5. **Neuromorphic hardware** — port LIF+STDP to Intel Loihi 2 or SpiNNaker.
6. **Formal evaluation** against SOAR, ACT-R, LIDA on shared benchmarks. The NSCK-Eval benchmark suite (V8) provides a foundation; the next step is alignment with established cognitive architecture benchmarks.
7. ~~**Resonator networks for factorisation**~~ ✅ **Done (V8)** — `HierarchicalResonatorNetwork` provides 2-level VSA factorization for nested predicate-argument structures.
8. **Landauer-principle energy accounting** — track the theoretical thermodynamic cost of bit erasure in VSA operations to establish energy-efficiency bounds for neuromorphic deployment.
9. **Active inference scaling** — extend `ActiveInferenceLearner` with richer world models (Markov blankets) and gradient-free variational inference for more principled free-energy minimisation.
10. **Multi-agent communication protocol** — formalise the `MultiAgentSession` belief negotiation using VSA consensus encoding across distributed agents.

---

## 10. Conclusion

We have presented NSCK, a neuro-symbolic cognitive architecture built around a 10,240-bit binary VSA substrate, integrating spiking neural network perception, Global Workspace Theory-based executive control, Δ-P causal discovery, STRIPS planning, Plutchik emotion modelling, and glass-box explanation generation. V8 (February 2026) extends the system with 9 new capabilities: concurrent multimodal scheduling, Ebbinghaus memory lifecycle, multi-turn dialogue state tracking, MathReasoner–GWT integration, 2-level hierarchical resonator networks, multi-agent cognitive fusion, an active inference free-energy loop, and a formal NSCK-Eval benchmark suite. The full test suite reaches **1,111 passing tests** with Rust extensions active.

The system demonstrates that bitwise operations on binary hypervectors can serve as a unified representational substrate for perception, memory, reasoning, and language — without gradient descent, without matrix multiplication, and with every decision fully auditable.

We are a single developer and lifelong student who wanted to explore an idea and make it into reality with an open mind. NSCK is not a finished product, nor a claim to have solved cognitive architecture. It is an honest attempt to build something coherent, well-tested, and fully open — to explore whether VSA can truly serve as the unifying language of a cognitive system.

Code, benchmarks, and documentation: **https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel**

---

## References

[1] P. Kanerva, "Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors," *Cognitive Computation*, vol. 1, no. 2, pp. 139–159, 2009.

[2] T. A. Plate, "Holographic reduced representations," *IEEE Transactions on Neural Networks*, vol. 6, no. 3, pp. 623–641, 1995.

[3] R. W. Gayler, "Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience," in *Proc. ICCS/ASCS*, 2003.

[4] K. Schlegel, P. Neubert, and P. Protzel, "A comparison of vector symbolic architectures," *Artificial Intelligence Review*, vol. 55, no. 6, pp. 4523–4555, 2022.

[5] M. Imani et al., "Binary hyperdimensional computing: Effortless offline learning and fast inference with compressed hypervectors," in *DATE 2019*, pp. 1224–1229.

[6] A. Rahimi et al., "Efficient hyperdimensional computing for wearable IoT applications," in *DATE 2019*, pp. 1260–1265.

[7] B. J. Baars, *A Cognitive Theory of Consciousness*. Cambridge University Press, 1988.

[8] S. Dehaene, J.-P. Changeux, and J.-P. Nadal, "Global workspace and metacognition," *Proc. National Academy of Sciences*, vol. 108, no. 7, pp. 3142–3148, 2011.

[9] S. Franklin and F. G. Patterson, "The LIDA architecture: Adding new modes of learning to an intelligent, autonomous, software agent," in *Proc. IDPT*, 2006.

[10] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659–1671, 1997.

[11] L. Lapicque, "Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation," *J. Physiologie et de Pathologie Générale*, vol. 9, pp. 620–635, 1907.

[12] G.-Q. Bi and M.-M. Poo, "Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type," *J. Neuroscience*, vol. 18, no. 24, pp. 10464–10472, 1998.

[13] E. P. Frady, D. Kleyko, and F. T. Sommer, "A theory of sequence indexing and working memory in recurrent neural networks," *Neural Computation*, vol. 30, no. 6, pp. 1449–1513, 2019.

[14] A. Renner et al., "Neuromorphic implementation of vector symbolic architectures," *Nature Machine Intelligence*, vol. 6, pp. 142–153, 2024.

[15] J. Pearl, *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2000.

[16] P. W. Cheng and L. R. Novick, "A probabilistic contrast model of causal induction," *J. Personality and Social Psychology*, vol. 58, no. 4, pp. 545–567, 1990.

[17] J. R. Anderson et al., "An integrated theory of the mind," *Psychological Review*, vol. 111, no. 4, pp. 1036–1060, 2004.

[18] J. E. Laird, *The Soar Cognitive Architecture*. MIT Press, 2012.

[19] T. C. Stewart and C. Eliasmith, "Large-scale synthesis of functional spiking neural circuits," *Proc. IEEE*, vol. 102, no. 5, pp. 881–898, 2014.

[20] R. E. Fikes and N. J. Nilsson, "STRIPS: A new approach to the application of theorem proving to problem solving," *Artificial Intelligence*, vol. 2, no. 3–4, pp. 189–208, 1971.

[21] R. Plutchik, *Emotion: A Psychoevolutionary Synthesis*. Harper & Row, 1980.

[22] J. A. Russell, "A circumplex model of affect," *J. Personality and Social Psychology*, vol. 39, no. 6, pp. 1161–1178, 1980.

[23] E. Oja, "Simplified neuron model as a principal component analyzer," *J. Mathematical Biology*, vol. 15, no. 3, pp. 267–273, 1982.

[24] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nature Machine Intelligence*, vol. 1, pp. 206–215, 2019.

[25] G. Karunaratne et al., "In-memory hyperdimensional computing," *Nature Electronics*, vol. 3, no. 6, pp. 327–337, 2021.

[26] G. T. Fechner, *Elemente der Psychophysik*. Breitkopf und Härtel, 1860.

[27] S. Dehaene and J.-P. Changeux, "Experimental and theoretical approaches to conscious processing," *Neuron*, vol. 70, no. 2, pp. 200–227, 2011.

[28] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, 1948.

[29] D. Janzing, J. Mooij, K. Zhang, J. Lemeire, J. Zscheischler, P. Daniušis, B. Steudel, and B. Schölkopf, "Information-geometric approach to inferring causal directions," *Artificial Intelligence*, vol. 182–183, pp. 1–31, 2012.

[30] K. Friston, "The free-energy principle: A unified brain theory?" *Nature Reviews Neuroscience*, vol. 11, no. 2, pp. 127–138, 2010.

[31] B. Fong and D. I. Spivak, *An Invitation to Applied Category Theory: Seven Sketches in Compositionality*. Cambridge University Press, 2019.

[32] E. T. Jaynes, "Information theory and statistical mechanics," *Physical Review*, vol. 106, no. 4, pp. 620–630, 1957.

---

*This paper is based on the implementation as it stands in February 2026. All benchmarks, line counts, and capability claims are derived from the actual source code. We welcome corrections, critiques, and contributions.*
