# NSCK Architecture Deep Dive
## How, What, Where, and Why Everything Works

*Written March 2026 — covers V1 through V27 (SoCL)*

---

## The One-Sentence Summary

> NSCK is a **cognitive architecture** that thinks in **10240-bit binary
> vectors** using bitwise XOR, majority-vote, and Hamming similarity —
> **not** with the floating-point matrix multiplications of deep learning —
> with a thin ring of conventional math (SVD, PCA, one tiny neural net)
> only where bitwise logic cannot do the job.

---

## Table of Contents

1. [What is NSCK?](#1-what-is-nsck)
2. [The Fundamental Primitive: 10 240-Bit Hypervectors](#2-the-fundamental-primitive-10240-bit-hypervectors)
3. [The 7 Cognitive Layers — How They Are Wired Together](#3-the-7-cognitive-layers)
4. [Where Matrix Multiplications Are Used](#4-where-matrix-multiplications-are-used)
5. [Where Deep Learning / Neural Nets Are Used](#5-where-deep-learning--neural-nets-are-used)
6. [The Societal Graph (V26) — Living Concepts](#6-the-societal-graph-v26)
7. [Societal-Guided EWC / SoCL (V27) — Forgetting Prevention](#7-societal-guided-ewc--socl-v27)
8. [Efficiency Analysis: Bitwise vs Float Cost Table](#8-efficiency-analysis)
9. [Benchmark Numbers](#9-benchmark-numbers)
10. [Design Decisions Explained](#10-design-decisions-explained)

---

## 1. What is NSCK?

NSCK stands for **Neuro-Symbolic Cognitive Kernel**.  The name captures the
design philosophy:

- **Neuro** — the bottom layer uses *Spiking Neural Networks* (LIF neurons,
  STDP plasticity) to model biological-style perception.
- **Symbolic** — the middle and upper layers use *symbolic rules*, causal
  graphs, working memory, language, and safety constraints that are fully
  human-readable.
- **Cognitive** — the whole architecture follows *Global Workspace Theory*
  (GWT), the cognitive-science model for consciousness that says different
  specialist modules compete to broadcast their result globally.
- **Kernel** — it is a small, embeddable core (~50 000 lines of Python,
  ~5 100 lines of Rust), not a finished application.

The **bridge** between the neural and symbolic layers is
**Vector Symbolic Architecture (VSA)**: every concept, rule, and memory is
encoded as a high-dimensional binary vector (a *hypervector*), so that
symbolic binding and retrieval can be done with bitwise operations that are
both very fast and statistically noise-tolerant.

---

## 2. The Fundamental Primitive: 10 240-Bit Hypervectors

### Why 10 240 bits?

The number 10 240 = 160 × 64 is chosen so that the vector fits into exactly
**160 unsigned 64-bit integers** in the Rust backend, enabling POPCNT-based
Hamming distance in a single tight loop.  In the Python fallback the vector is
stored as a `numpy.ndarray` of `int8` (one byte per bit).

**Probability theory guarantees**: for two *random* 10240-bit vectors the
expected Hamming similarity is 0.5 ± 0.005 (3σ deviation ~1.6%).  This means:
- Any two independently generated concept vectors are *quasi-orthogonal*
  (dissimilar) with overwhelming probability.
- Binding and bundling operations produce new vectors that are predictably
  close to or far from the inputs.

### The three VSA operations

All reasoning in NSCK reduces to three primitives:

| Operation | Symbol | Implementation | Meaning |
|-----------|--------|---------------|---------|
| **XOR / Bind** | `A ⊕ B` | `numpy.bitwise_xor` | "A is associated with B". Result is dissimilar to both — creates a unique binding. |
| **Bundle / Superpose** | `A ∪ B` | majority vote | "A and B together". Result is similar to both — creates a set or category. |
| **Similarity** | `sim(A,B)` | `1 - Hamming/10240` | How alike two concepts are. Range [0,1]. |

And two additional operators:

| Operation | Symbol | Implementation | Meaning |
|-----------|--------|---------------|---------|
| **Permute** | `ρ^k(A)` | `numpy.roll` / Rust cyclic shift | Encodes temporal position or sequence index. |
| **Negate** | `¬A` | `1 - A` (bitflip) | "Not A" in bipolar space. |

### No floating-point weights in the knowledge layer

A traditional neural network stores knowledge in floating-point weight
matrices (hundreds of millions of 32-bit floats).  NSCK stores knowledge in
the *structure* of the hypervector space: which vectors were bound/bundled
together, which are similar to which.  There are **no learnable weights** in
the core VSA layer.

---

## 3. The 7 Cognitive Layers

```
Input (image / audio / text / sensor)
         │
         ▼
 ┌───────────────────────────────────────┐
 │  Layer 1 — SNN Perception             │  (vsa_snn_bridge.py, snn_perception.py)
 │  Spiking neurons → spike train        │
 │  → Rate/Temporal coder → HV           │
 └──────────────────┬────────────────────┘
                    │  PerceptPacket (HV + concept_id + strength)
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 2 — VSA Working Memory         │  (memory/semantic_memory.py + episodic)
 │  Semantic memory (HNSW ANN index)     │
 │  Episodic memory (time-tagged HVs)    │
 │  Procedural memory (skill cache)      │
 └──────────────────┬────────────────────┘
                    │  situation_hv
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 3 — Causal Reasoning           │  (reasoning/causal_reasoner.py)
 │  Δ-P causal discovery (Pearl do-calc) │
 │  WorldModel (next-state prediction)   │
 │  CounterfactualSimulator              │
 └──────────────────┬────────────────────┘
                    │  CausalGraph
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 4 — Global Workspace Theory    │  (reasoning/cognitive_engine.py)
 │  Coalition competition                │
 │  GWT broadcast to all subscribers    │
 │  Rule learning + symbolic rules       │
 └──────────────────┬────────────────────┘
                    │  DecisionResult (action + trace)
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 5 — Language                   │  (language/)
 │  Template / fluent dialogue           │
 │  HV → natural language                │
 └──────────────────┬────────────────────┘
                    │
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 6 — Cognitive Safety           │  (cognitive/safety_verifier.py)
 │  Declarative rules (forbidden actions)│
 │  Confidence gating                    │
 │  Theory of Mind (agent modelling)     │
 └──────────────────┬────────────────────┘
                    │
                    ▼
 ┌───────────────────────────────────────┐
 │  Layer 7 — Societal Graph (V26)       │  (societal/)
 │  LivingHyperVectors with bonds        │
 │  Leiden community clustering          │
 │  SocietalContextRouter                │
 └───────────────────────────────────────┘
```

### Layer 1 — SNN Perception (`nsck/python/core/perception/`)

**What it does**: converts raw sensor input into a spike train, then encodes
the spike train as a hypervector.

**How it works step by step**:
1. Input (e.g. 64-dimensional pixel values) is multiplied by a random weight
   matrix `input_weights` (shape: `[snn_size, input_dim]`) to produce a
   synaptic current for each neuron.
2. Each neuron integrates using the **Leaky Integrate-and-Fire** (LIF) equation:
   ```
   τ · dv/dt = -(v - v_rest) + I_syn
   ```
   When `v ≥ v_thresh` (−55 mV), the neuron fires (spike = 1), resets to
   `v_reset` (−75 mV), and enters a 2 ms refractory period.
3. After `simulation_time_ms` (default 20 ms), the spike train (shape:
   `[time_steps, snn_size]`) is passed to the encoder.
4. **Rate Coder**: neurons that fired above a threshold rate each contribute
   their pre-assigned neuron HV, bundled with weight proportional to fire rate.
5. **Temporal Coder**: each spike event at time `t` contributes
   `permute(neuron_HV, k)` where `k ∝ t`, encoding *when* the spike occurred.
6. Result: a single 10240-bit HV representing the perceptual pattern.

**Matrix multiplication here**: only the single `input_weights @ x` per time
step (see §4 for full cost analysis).

### Layer 2 — VSA Working Memory (`nsck/python/core/memory/`)

Four memory stores, all using hypervectors:

| Store | What | Retrieval |
|-------|------|-----------|
| **SemanticMemory** | Concept → HV + metadata | HNSW approximate nearest-neighbour graph |
| **EpisodicMemory** | (time, situation_hv, action, reward) tuples | Cosine similarity scan |
| **ProceduralMemory** | (context_hv → action, reward) skill cache | Top-k similarity lookup |
| **CrossModalMemory** | vision HV ↔ text HV pairs | Bidirectional similarity |

**No matrix multiplications** in the memory layer.  Retrieval is
HV similarity (bitwise Hamming), HNSW graph walks.

### Layer 3 — Causal Reasoning (`nsck/python/core/reasoning/`)

**Causal engine**: uses **Δ-P** (delta-P) causal discovery.  Given observations
(antecedent A, consequent C, do-calc intervention), it estimates:
```
ΔP(A→C) = P(C|A) − P(C|¬A)
```
All stored as plain Python dicts of counts — no matrices at all.

**WorldModel**: predicts next situation HV via XOR binding of current
situation HV with an action HV.  Pure bitwise operation.

### Layer 4 — Global Workspace (`nsck/python/core/reasoning/cognitive_engine.py`)

**Coalition competition**: multiple modules propose (HV, action, confidence)
tuples.  A coalition with the highest GWT score wins and broadcasts.

The **attention re-weighting** (V13+) uses a small transformer-inspired
multi-head attention mechanism to compute a scalar weight for each coalition
(see §5 — this is the only dot-product attention in the system).

**Rule learning**: rules are `{condition_set} → consequence` horn clauses
stored as Python objects.  The `RuleNeuralScorer` (§5) re-ranks them.

**EWC** (V16): protects concept importances from being overwritten when the
system switches tasks (see §7).

### Layer 5 — Language (`nsck/python/core/language/`)

Template-based and fluent dialogue systems.  Concepts retrieved from semantic
memory are rendered as words using HV nearest-neighbour lookup.  No neural
language model is used by default.

### Layer 6 — Safety (`nsck/python/core/cognitive/safety_verifier.py`)

Declarative forbidden-action rules, Plutchik emotion HVs, Theory of Mind
(agent model HVs).  Confidence gating: if uncertainty > threshold, the safety
verifier vetoes the proposed action.  Everything is bitwise comparison.

### Layer 7 — Societal Graph (`nsck/python/core/societal/`)

See §6 for full explanation.

---

## 4. Where Matrix Multiplications Are Used

The following table lists **every place** in the production code (non-test)
where matrix multiplication (`@`, `np.dot`, `np.matmul`) appears, what it
does, how often it runs, and why it must be there.

### 4.1 SNN Input Projection (hot path)

**File**: `nsck/python/core/perception/snn_perception.py` line 720

```python
input_current = self.input_weights @ x_proc   # [snn_size] = [256 × input_dim] @ [input_dim]
```

- **Dimensions**: typically `[256 × 64]` (snn_size=256, input_dim=64)
- **When**: runs every timestep during every perception call (~20 times per
  20 ms simulation window)
- **Why unavoidable**: this is the synapse model — each neuron receives the
  weighted sum of all input signals, which is exactly a matrix-vector product.
- **Cost**: 256 × 64 = 16 384 multiply-adds per timestep × 20 timesteps =
  **327 K FLOPs per perception call**.

### 4.2 RandomProjector (Model Transplant, runs once at fit time)

**File**: `nsck/python/core/transplant/projector.py` line 71

```python
proj = embeddings @ self._P   # [N, 10240] = [N, dim_in] @ [dim_in, 10240]
```

- **Dimensions**: `[vocab_size × dim_in] @ [dim_in × 10240]` — e.g.
  `[10000 × 768] @ [768 × 10240]`
- **When**: runs **once** when a pretrained model is absorbed.  Never during
  inference.
- **Why**: Johnson-Lindenstrauss random projection preserves distances when
  mapping float embeddings to binary HV space.
- **Cost**: `10000 × 768 × 10240 ≈ 78.6 GFLOPs`.  Takes a few seconds once.

### 4.3 SVDFactoredProjector (Model Transplant, runs once)

**File**: `nsck/python/core/transplant/projector.py` lines 287, 302

```python
_U, _S, Vt = np.linalg.svd(E_c, full_matrices=False)  # SVD
Z = (E - self._mean) @ self._V_proj.T                  # project to k=128 dims
```

- **Dimensions**: SVD of `[min(10000, vocab) × dim_in]`, then project to
  `[N × k]` where `k = 128` principal components.
- **When**: once at transplant time.
- **Why**: SVD discards low-variance directions, improving FPE encoding
  quality from ~23% to ~58% on text tasks (V24 level-coding breakthrough).
- **Cost**: `SVD(10000×768) ≈ 7.9 GFLOPs` + projection `≈ 983 MFLOPs`.

### 4.4 Attention in GWT (`attention_gwt_bridge.py`)

**File**: `nsck/python/core/reasoning/attention_gwt_bridge.py` line 72

```python
scores.append(float(np.dot(q, kp)) / (scale + _EPSILON))
```

- **Dimensions**: `q` and `kp` are 64-dimensional vectors (key_dim=64).
- **When**: once per decision cycle, for each coalition (typically 2-10).
- **Cost**: 64 × n_coalitions ≈ **640 FLOPs per decision**.  Negligible.

### 4.5 RuleNeuralScorer forward pass

**File**: `nsck/python/core/learning/rule_neural_scorer.py` lines 98-99

```python
h = _sigmoid(x @ self._W1 + self._b1)  # [6] @ [6×16] = [16]
out = _sigmoid(h @ self._W2 + self._b2)  # [16] @ [16×1] = [1]
```

- **Dimensions**: 6-feature input → 16-hidden → 1-output.
- **When**: once per decision cycle when more than one rule is applicable
  (optional; off by default unless `rule_scorer` is set).
- **Cost**: 6×16 + 16×1 = **112 multiply-adds** per scoring call.  Negligible.

### 4.6 Hebbian Weight Update (optional)

**File**: `nsck/python/core/learning/hebbian.py` line 258

```python
outer = torch.einsum("bi,bj->ij", post, pre) / batch_size
```

- **When**: only if `HebbianMatrixTorch` is used and PyTorch is installed.
  **Not on the default cognitive engine path.**
- **Cost**: `batch × post_dim × pre_dim`.  Offline learning only.

### 4.7 Gabor Filter Bank (Image Features)

**File**: `nsck/python/core/adapters/image_adapter.py` line 458

```python
response = np.abs(np.einsum("hwkl,kl->hw", view, kernel))
```

- **When**: once per image during feature extraction.
- **Why**: Gabor filters extract biologically-plausible orientation features —
  this is scientifically motivated feature engineering, not end-to-end learning.
- **Cost**: small (128 orientations × small patch = fast on CPU).

### 4.8 Summary Table: How Much of NSCK is Matrix Multiplication?

| Component | FLOPs per inference | FLOPs one-time setup |
|-----------|--------------------|--------------------|
| SNN input projection | ~327 K | — |
| GWT attention | ~640 | — |
| RuleNeuralScorer (optional) | ~112 | — |
| **Total per-inference float** | **~328 K FLOPs** | — |
| RandomProjector | — | ~78.6 GFLOPs |
| SVDFactoredProjector | — | ~8.9 GFLOPs |
| **Total per-inference bitwise** | **~1.2 M bit ops** | — |

**For comparison**: ResNet-18 forward pass = ~1.8 GFLOPs = **5 500× more
FLOPs** than NSCK's per-inference float cost.  GPT-2 = ~50 GFLOPs = **152 000×
more FLOPs**.

---

## 5. Where Deep Learning / Neural Nets Are Used

### 5.1 Spiking Neural Network (SNN) — *Is* a neural net, but not deep learning

The LIF layer is technically a neural network.  However:
- It uses **no backpropagation**.
- Weights are initialised randomly and updated only via **STDP**
  (Spike-Timing Dependent Plasticity) — a local biologically-plausible rule:
  ```
  ΔW_ij = A_+ · exp(-Δt/τ)  if  t_pre < t_post   (potentiation)
  ΔW_ij = -A_- · exp(-Δt/τ) if  t_pre > t_post  (depression)
  ```
- It has **one layer** whose job is to *spike-encode* input, not classify it.
- No gradient flows backwards.

### 5.2 RuleNeuralScorer — a tiny 2-layer perceptron (129 parameters)

The only place **backpropagation** is used in production code.

- Architecture: 6 → 16 → 1 (sigmoid activations)
- Parameters: `6×16 + 16 + 16×1 + 1 = 129 weights`
- Training: `update(rule, reward)` does one online SGD step per feedback.
- Why: symbolic rule confidence is noisy.  Adding 6 hand-crafted features
  (fire-rate, trend, complexity) and a micro-perceptron gives better ranking.

### 5.3 Multi-Head Attention (GWT coalition scoring)

Uses `Q·K^T / √d` attention (transformer style) with 2-4 heads, key_dim=64.
**Random fixed projection matrices** (`W_q`, `W_k`) — not trained.
The attention is a scoring heuristic, not a learner.

### 5.4 Hebbian Learning (optional, offline)

`HebbianMatrixNumPy` (pure NumPy) or `HebbianMatrixTorch` (with PyTorch).
Implements Oja's Rule — prevents weights from growing unbounded.
Off by default.

### 5.5 PCA + LDA in NSCKHDVisionClassifier

Used only during **training** of the HD vision classifier:
1. **PCA(128)**: compress 644 image features → 128 principal components.
2. **LDA**: further compress to `n_classes - 1` discriminative dimensions.
3. These vectors are encoded into 10240-bit HVs using the FPE codebook.

At **inference** time, only the stored projector matrices are applied — no
training, no gradients.

### 5.6 PyTorch (optional, transplant only)

Used only when harvesting embeddings from user-provided pretrained models
(e.g. BERT, CLIP).  Once embeddings are projected into HV space, PyTorch is
no longer needed.

### 5.7 What is NOT used

- No convolutional neural networks (CNNs) in the inference path
- No recurrent neural networks (RNNs/LSTMs)
- No full Transformers
- No autograd / backpropagation in the core cognitive loop
- No GPU required
- No end-to-end gradient-based learning of symbolic knowledge

---

## 6. The Societal Graph (V26)

### What it is

The Societal Hypervector Knowledge Representation (SHVKR) treats concepts as
**living agents** in a social network:

- `LivingHyperVector`: a concept with a 10240-bit body plus metadata:
  `activation` ∈ [0,1], `_bonds` (dict concept_id → Bond strength),
  `_cluster_id`, `domain_path`.
- `Bond`: a directed weighted connection between two `LivingHyperVector`s.
  Strength decays over time (configurable `bond_decay`), grows when
  re-activated.
- `SocietyManager`: the graph container.  Operations:
  - **Auto-bond**: connect concepts whose HVs are similar above threshold
  - **Leiden cluster**: greedy community detection → assign cluster IDs
  - **Spread activation**: visiting a concept activates its neighbours
  - **Topological health**: N concepts, N bonds, percolation check
- `SocietalContextRouter`: routes a query HV to the top-K nearest concepts,
  activates them, returns a context dict for the cognitive loop.

### Where it is used

1. `NSCKSubstrate.process()`: the perception HV is routed through the societal
   graph; `societal_context` is added to `SubstrateResult`.
2. `NSCKSubstrate.feedback()`: action concepts are auto-registered into the
   society.
3. `CognitiveEngine._compute_and_record_vsa_importance()`: when V27 SoCL is
   enabled, societal centrality scores boost EWC Fisher weights.

### No matrix multiplications

All operations are HV similarity (bitwise Hamming), Python dict lookups, and
float arithmetic on scalar activation/bond values.  The Leiden clustering is
a greedy modularity-maximisation BFS — no matrices.

---

## 7. Societal-Guided EWC / SoCL (V27)

### The forgetting problem

The continual learning benchmark showed **16% catastrophic forgetting**: when
the system learns Task 2, it overwrites concept importances from Task 1.

### Standard EWC (V16)

Elastic Weight Consolidation adds a regularization term to protect important
parameters:
```
L_EWC = (λ/2) · Σᵢ Fᵢ · (θᵢ − θ*ᵢ)²
```
`Fᵢ` = Fisher Information diagonal (squared gradients at convergence).
`θ*ᵢ` = optimal Task 1 parameters (concept importance scalars).

### SoCL boost (V27)

The societal graph tells us *which concepts are cross-task hubs* — something
the Fisher information alone cannot know.  SoCL multiplies the Fisher diagonal
by a centrality boost factor:
```
F_boosted[c] = F[c] × (1 + α × centrality[c])

centrality[c] = w_d · degree/max_degree
              + w_a · activation
              + w_c · cluster_size/N

Default: w_d=0.4, w_a=0.4, w_c=0.2, α=0.5
```

Concepts that are highly bonded, recently activated, and members of large
clusters get up to 50% stronger EWC protection.

**No matrix multiplications**: centrality is a weighted sum of 3 scalars per
concept — O(N) total, runs in <0.5 ms for N=100.

---

## 8. Efficiency Analysis

### Bitwise operations vs float operations

| Operation | Type | Cost per call | Used in |
|-----------|------|--------------|---------|
| XOR two 10240-bit HVs | bitwise | 0.37 µs (Rust) / 1.35 µs (Py) | everywhere |
| Bundle two HVs | bitwise | 0.81 µs (Rust) / 57.81 µs (Py) | bundling |
| Similarity (Hamming) | bitwise | 0.22 µs (Rust) / 7.20 µs (Py) | retrieval |
| SNN input projection | float matmul | ~0.1 ms (256×64) | each perception |
| GWT attention score | float dot | ~1 µs (64-d) | each decision |
| RuleNeuralScorer | float matmul | ~5 µs (6→16→1) | optional, per rule |
| RandomProjector (fit) | float matmul | ~3 s (10k×768×10240) | **once** at transplant |
| SVD (fit) | float SVD | ~2 s (10k×768) | **once** at transplant |
| PCA+LDA (fit) | float | ~0.5 s (1797×64) | **once** at training |

### Why bitwise dominates at inference

A 10240-bit vector = 160 × 64-bit integers.  The Rust XOR of two such vectors
is 160 POPCNT instructions — latency **0.37 µs**.  A 10 000-concept ANN query
does 10 000 such operations = **3.7 ms** (measured: 4.94 ms at 10k including
Python overhead).

For comparison: a single matrix multiply `[768 × 10240]` takes ~3 ms.  If
NSCK used float vectors instead of binary HVs, every similarity query would
cost 3 ms instead of 0.22 µs — a **13 600× slowdown**.

### Memory efficiency

| Format | Memory per concept |
|--------|--------------------|
| 10240-bit HV (Rust uint64 × 160) | 1 280 bytes = **1.25 KB** |
| 768-float BERT embedding (float32) | 3 072 bytes = 3 KB |
| 4096-float GPT-3 embedding | 16 384 bytes = 16 KB |

NSCK uses **2.4× less memory** than BERT and **12.8× less** than GPT-3
embeddings per concept.

---

## 9. Benchmark Numbers

All measured on CPU (no GPU), Python 3.12, NumPy 1.26.

### Rust backend (hypervec_rs, Rust 1.93):

| Operation | Latency |
|-----------|---------|
| XOR | 0.37 µs |
| Bundle | 0.81 µs |
| Similarity | 0.22 µs |
| ANN query (1k concepts) | 0.31 ms |
| ANN query (10k concepts) | 4.94 ms |
| Speedup vs Python (avg) | **51.7×** |

### Python fallback (HyperVectorPy, NumPy):

| Operation | Latency |
|-----------|---------|
| XOR | ~2.1 µs |
| Bundle | ~64.5 µs |
| Similarity | ~10.7 µs |

### Cognitive engine (decision loop):

| Task | Latency |
|------|---------|
| `decide()` (Rust backend) | ~2–4 ms/turn |
| `decide()` (Python backend) | ~10–30 ms/turn |
| SNN perception (20 ms sim, snn_size=256) | ~43 ms (Python) / ~1.85 ms (Rust) |
| Societal routing (top-5, 100 concepts) | ~0.65 ms |
| SoCL centrality boost (100 concepts) | ~0.5 ms (one-time) |

### Accuracy:

| Task | Metric |
|------|--------|
| NSCKHDVisionClassifier (MNIST digits, clean) | **93.3%** |
| Rotation robustness (90°/180°/270°) | 84% |
| Text TF-IDF FPE absorption | 58.8% |
| Cross-modal retrieval | **100%** |
| SNN pattern classification (10 patterns) | **100%** |
| NSCK-ES (evaluation score) | **1.000** |
| Lifelong forgetting (vanilla EWC) | 16% |
| Lifelong forgetting (with SoCL) | reduced (most at task boundaries) |

---

## 10. Design Decisions Explained

### Why 10 240 bits and not 4 096 or 65 536?

- **4 096 bits**: collision probability becomes significant (>1%) for
  vocabularies > 1 000 concepts.
- **65 536 bits**: 5× more RAM and latency for marginal accuracy gain.
- **10 240 bits**: empirically the sweet spot.  Packs into 160 × 64-bit
  integers for POPCNT-optimized Hamming distance.  Used by the HDC community
  (Kanerva, 2009; Frady et al., 2022) as a standard reference dimension.

### Why binary (0/1) and not real-valued hypervectors?

NSCK uses **Binary Sparse Distributed Representation (BSR)** not FHRR
(Fourier HRR) or MAP (Multiply-Add-Permute).  Binary is:
- **Fast**: bitwise XOR / AND / POPCNT on integer registers.
- **Memory-efficient**: 1 bit per dimension vs 32 bits for float.
- **Noise-tolerant**: majority-vote bundling naturally handles noise.
- **Rust-acceleratable**: 160 × 64-bit register operations, no SIMD float.

FHRR (`nsck/python/core/vsa/fhrr.py`) is available as an optional
complex-valued variant for gradient-compatible use cases, but is not on the
default inference path.

### Why not use PyTorch for everything?

NSCK is designed to run **without a GPU** and without a large DL framework.
The goals are:
1. **Transparency**: every decision is traceable to a rule or causal edge.
2. **Continual learning**: symbolic knowledge never suffers catastrophic
   forgetting; only neural-style concept importances need EWC.
3. **Embeddable**: must run on a Raspberry Pi, in a game loop, or on a
   satellite with limited power.

PyTorch is imported *lazily* (inside function bodies with `try/except`) only
for optional Hebbian GPU acceleration and when harvesting embeddings from
user-provided pretrained models.

### Why SVD for model transplantation?

When absorbing a pretrained model (e.g. BERT, CLIP), embeddings are in a
768-dimensional float space.  SVD finds the `k=128` directions of maximum
variance and discards the rest.  Then FPE (Fixed-Point Encoding) converts
each 128-dim reduced vector into a 10240-bit HV while preserving relative
distances (Spearman ρ > 0.995 measured).

Random projection is the fallback when SVD fails, but SVD gives significantly
better FPE encoding quality (58.8% vs 23.5% text classification).

### Why a societal graph for continual learning?

Standard EWC knows *how important* each parameter was (Fisher information) but
not *why*.  The societal graph provides structural context: a concept with many
bonds to different task-specific concepts is a cross-task hub.  Boosting its
EWC protection is semantically motivated — it is the "shared vocabulary" that
both tasks depend on and should never be overwritten.

---

## Summary: The Three-Sentence Answer

**How**: NSCK thinks in 10240-bit binary hypervectors using XOR (bind),
majority-vote (bundle), and Hamming similarity (retrieve).  Matrix
multiplications appear only in the SNN synapse model (~327 K FLOPs/call), one
tiny 2-layer perceptron (129 parameters, optional), and one-time model
transplant projections (SVD + random projection, run once and discarded).

**Why**: Binary VSA operations are 50–13 600× faster than float matrix
multiplication for the same representational capacity, require no GPU, no
backpropagation, and produce fully interpretable symbolic knowledge as a
natural by-product.

**Still using 10 240 bits?** Yes. Every hypervector in the entire system —
from raw percepts to emotion states to societal living concepts to EWC
importance scores — uses the same 10240-bit binary format, stored as
160 × uint64 in Rust or a `numpy.int8[10240]` array in Python.
