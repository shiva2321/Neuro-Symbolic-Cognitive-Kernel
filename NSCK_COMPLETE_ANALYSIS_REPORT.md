# NSCK Complete System Analysis Report
**Date:** 2026-03-02  
**Scope:** End-to-end audit of all code, live benchmarks with both Python and Rust backends  
**Method:** Direct code inspection + live execution (no reliance on documentation alone)

---

## Table of Contents
1. [What is NSCK?](#1-what-is-nsck)
2. [Architecture: The Complete Layer Stack](#2-architecture-the-complete-layer-stack)
3. [Hypervector Dimensions — Are We Still at 10240 Bits?](#3-hypervector-dimensions)
4. [Matrix Multiplications and Deep Learning — Honest Audit](#4-matrix-multiplications-and-deep-learning)
5. [Python vs Rust Backend — Live Benchmarks](#5-python-vs-rust-backend-live-benchmarks)
6. [Every Subsystem Explained](#6-every-subsystem-explained)
7. [Identified Bugs and Performance Issues](#7-identified-bugs-and-performance-issues)
8. [Where the System Stands Today](#8-where-the-system-stands-today)
9. [Final Verdict and Recommendations](#9-final-verdict-and-recommendations)

---

## 1. What is NSCK?

NSCK (Neuro-Symbolic Cognitive Kernel) is a **non-neural, algebraic AI framework** built on Vector Symbolic Architectures (VSA). It deliberately avoids gradient-based deep learning. Instead it uses:

- **Hypervectors** — 10,240-bit binary vectors representing concepts, memories, and percepts
- **Algebraic operations** — XOR (bind), majority-vote bundle (superpose), cosine/Hamming similarity (compare)
- **Graph structures** — semantic memory as a weighted graph, spreading activation for reasoning
- **Symbolic logic** — rule learning, causal graphs, Global Workspace Theory (GWT) for decision-making

The system is organized as layers, from raw bits up to a high-level cognitive substrate API.

**Codebase size (verified):**
- 580 Python source files (580 `.py` files, excluding `__pycache__`)
- 13 Rust source files (`rust_vsa/src/`, `rust_snn/src/`)
- ~8,000 lines in the 11 most critical files
- 15+ evaluation/benchmark scripts in `nsck/eval/`
- Archive of ~60+ older prototypes (not deleted, kept for reference)

---

## 2. Architecture: The Complete Layer Stack

```
┌─────────────────────────────────────────────────────┐
│                   NSCKSubstrate API                  │  ← Public entry point
│           ingest() / feedback() / process()          │    nsck/python/core/substrate.py
├─────────────────────────────────────────────────────┤
│                  CognitiveEngine                     │  ← Orchestrator
│   decide() / perceive_and_decide() / learn()         │    cognitive_engine.py (2136 lines)
├──────────────┬──────────────────┬────────────────────┤
│  Global      │  Rule Learner    │  Causal Reasoner   │
│  Workspace   │  (VSA-based)     │  (CausalGraph)     │
│  (GWT)       │                  │                    │
├──────────────┴──────────────────┴────────────────────┤
│                 Memory Systems                       │
│  EpisodicMemory  │  SemanticMemory  │  Procedural    │
│  (LSH + Rust)    │  (Graph + HNSW)  │  Memory        │
├──────────────────────────────────────────────────────┤
│              Perception / Adapters                   │
│  ImageAdapter  │  AudioAdapter  │  TextAdapter       │
│  SNN Shim      │  VideoAdapter  │  NumericAdapter    │
├──────────────────────────────────────────────────────┤
│             Vision Classifier (NSCK-UPMA)            │
│  NSCKHDVisionClassifier: HOG+LBP+Gabor+FFT → LDA    │
│  → level-coded HV prototypes → cosine nearest-       │
│    centroid (93.3% clean, 24.2% adversarial UPMA)   │
├──────────────────────────────────────────────────────┤
│            Language / NLU Layer                      │
│  VSANLUEngine (DistributionalCodebook)               │
│  TextKnowledgeLearner / SemanticBootstrapper         │
│  ConstructionGrammar / FrameSemantics / Coreference  │
├──────────────────────────────────────────────────────┤
│          Societal Knowledge World (V5)               │
│  LivingHyperVector │ ValenceEngine │ SpectralRG       │
│  SocietalHNSW      │ TDA Monitor   │ Percolation      │
├──────────────────────────────────────────────────────┤
│                 VSA Layer (Core)                     │
│  HyperVector: XOR / bundle / similarity / permute   │
│  CleanupMemory │ HyperVectorRegistry │ LSH           │
├──────────────────────────────────────────────────────┤
│           Backends (selected by import)              │
│  hypervec_rs.so (Rust, ChaCha8 RNG, SIMD u64)       │
│  hypervec_py.py (Python fallback, numpy int8)        │
└──────────────────────────────────────────────────────┘
```

---

## 3. Hypervector Dimensions

**Answer: Yes, still exactly 10,240 bits. Everywhere.**

Verified in source code and live execution:

| File | Constant | Value |
|------|----------|-------|
| `nsck/python/core/vsa/hypervec_py.py:5` | `DIMENSION` | `10240` |
| `nsck/rust_vsa/src/lib.rs:14` | `DIMENSION` | `10240` |
| `nsck/python/core/societal/navigation/societal_hnsw.py:33` | `_DIM` | `10240` |
| `nsck/python/core/vsa/hypervec_shim.py:58,60` | hard-coded | `10240` |

**Live verification:**
```
Rust HyperVector.__getstate__() → list of 160 uint64 words = 160 × 64 = 10,240 bits ✓
Python HyperVectorPy.bits → numpy int8 array, length 10,240 ✓
```

**Memory per HV:**
- Rust: 160 × 8 bytes = **1,280 bytes** (packed uint64)
- Python: 10,240 × 1 byte = **10,240 bytes** (one int8 per bit)
- TraceMalloc shows 10,000 Rust HVs = ~0.5 MB peak (Python overhead dominates)

**Why 10,240?**
- Divisible by 64 → fills uint64 words exactly (critical for Rust SIMD efficiency)
- At this dimension: two random HVs have ~50% bit overlap by chance → cosine ≈ 0 (near-orthogonal)
- Theoretical capacity: 10,240 random vectors can be memorized with <1% interference

---

## 4. Matrix Multiplications and Deep Learning

### Short Answer
**NSCK does NOT use deep learning.** There are no PyTorch, TensorFlow, Keras, or JAX dependencies. No gradient descent, no backpropagation, no learned weight matrices.

There ARE limited matrix operations, used for geometry/statistics, not for training.

### Deep Learning Components: None Found
```
grep -r "torch\|tensorflow\|keras\|jax\|gradient\|backward\|optimizer\|nn.Module" 
Result: 0 matches in nsck/python/core/, nsck/rust_vsa/
```

| DL Concept | Present in NSCK? | NSCK Equivalent |
|-----------|-----------------|----------------|
| Weight matrix (trained) | ❌ No | Random 10240-bit HV (fixed at creation) |
| Gradient descent | ❌ No | Single-pass HV bundling |
| Backpropagation | ❌ No | Not needed — algebraic VSA |
| Softmax classifier | ❌ No | Cosine similarity argmax |
| Attention mechanism | ❌ No | Spreading activation on graph |
| Conv/Dense layers | ❌ No | Feature extraction via classical signal processing |
| Loss function | ❌ No | Reward-based rule confidence update |
| Training epochs | ❌ No | Online, one-shot learning |

### Matrix Operations That Do Exist

| Location | Operation | Size | Frequency | Cost |
|----------|-----------|------|-----------|------|
| `SpectralLaplacianRG.build_similarity_graph()` | `(N×10240) @ (10240×N)` matmul | N=50: 25.6M multiply-adds | On-demand diagnostic | ~7 ms |
| `TDAHealthMonitor._select_landmarks()` | `(N×10240) @ 10240` = N dot products | N × 10240 | On-demand health check | <1 ms |
| `transplant_extension.py:172` | `sub_embs @ sub_embs.T` | m×128×m (embedding dims) | One-time seeding | <1 ms |
| Every `cosine_similarity()` call | `dot(a_bipolar, b_bipolar)` | 10,240 multiply-adds | Per pair comparison | 17.7 µs (Python) / 0.22 µs (Rust) |
| `NSCKHDVisionClassifier.fit()` | PCA-128 + LDA projection | 644 features per sample | One-time training | 1559 ms for 90 samples |
| `NSCKHDVisionClassifier.predict()` | LDA cosine nearest-centroid | 128-dim vectors | Per image | 99 ms for 15 samples |

### Cost Analysis vs Deep Learning

```
SpectralRG matmul (50 concepts):  25,600,000 multiply-adds  = 0.026 B FLOPs
ResNet-50 forward pass:        4,100,000,000 multiply-adds  = 4.1 B FLOPs
GPT-2 (smallest) forward pass: 1,500,000,000 multiply-adds  = 1.5 B FLOPs
NSCK per-query cosine sim:            10,240 multiply-adds  = 0.00001 B FLOPs
```

NSCK's most expensive matrix op (SpectralRG coarse-graining) is **160× smaller than a single ResNet-50 forward pass** and runs in 7 ms on CPU. This is not a training operation — it runs once when explicitly invoked for topological analysis.

**The HD Vision Classifier does use LDA projection** (a matrix multiply at fit-time and predict-time), but:
- LDA is classical discriminant analysis, not a neural network
- No gradient descent — it's solved algebraically via eigendecomposition
- Fit cost: 1559 ms for 90 training images (one-time)
- Predict cost: 99 ms for 15 images (primarily feature extraction)

---

## 5. Python vs Rust Backend — Live Benchmarks

### Backend Selection (Verified)
The backend is selected by attempted import of `hypervec_rs`:
```python
# nsck/python/core/vsa/hypervec_shim.py:32-45
try:
    import hypervec_rs as _ext
    _USE_RUST = True
    print("[VSA] Using Rust Accelerator (hypervec_rs) [10-100x Performance]")
except ImportError:
    _USE_RUST = False
    _ext = None
```
The `NSCK_USE_RUST` environment variable is **NOT consulted** (confirmed by grep).

### Core VSA Operations (10,240-bit vectors)

| Operation | Rust (ops/s) | Python (ops/s) | Speedup |
|-----------|-------------|----------------|---------|
| HV create (random) | 466,620 | 20,074 | **23.2×** |
| XOR (bind/unbind) | 3,994,172 | 735,679 | **5.4×** |
| Bundle (2-way) | 1,425,689 | 17,772 | **80.2×** |
| Similarity (Hamming) | 4,600,207 | 138,892 | **33.1×** |
| 100-way bundle | 14,582 | 178 | **82.1×** |
| Linear search 1,000 HVs | 4,472 | 137 | **32.7×** |

### Latency in Microseconds (µs per call)

| Operation | Rust µs | Python µs |
|-----------|---------|-----------|
| XOR | 0.37 | 1.35 |
| Bundle (2-way) | 0.81 | 57.81 |
| Similarity (Hamming) | 0.22 | 7.20 |
| Negate | 0.64 | — |
| Permute | 0.63 | — |
| Bind + Unbind roundtrip | 0.44 | — |

### Parallel / Bulk Operations (Rust-only, using Rayon)

| Operation | Rust Time | Description |
|-----------|-----------|-------------|
| `parallel_similarity_search(query, 1000 HVs, top-10)` | **0.31 ms** | Rayon parallel cosine search |
| `batch_parallel_search(10 queries × 1000 HVs, top-5)` | **1.57 ms** | 10 queries simultaneously |
| `parallel_bundle(100 HVs)` | **0.45 ms** | N-way majority vote, parallel |
| `nearest_neighbors(query, 1000 HVs)` | **0.42 ms** | DashMap + Rayon ANN |
| `nearest_neighbors(query, 10,000 HVs)` | **4.94 ms** | Scales sub-linearly with Rayon |
| `batch_similarity_matrix(50 HVs)` | **0.27 ms** | Lower-triangle 50×50 matrix |
| `spreading_activation_step(20 nodes)` | **2.9 µs** | Graph BFS + activation propagation |
| `weber_fechner_compress(100 values)` | **2.67 µs** | log-scaling for biological plausibility |

**Python equivalent for 10,000 HV search:** 81.34 ms vs Rust 4.94 ms = **16.5× speedup**.

### Memory Tier Benchmarks

| Operation | Rust | Python | Notes |
|-----------|------|--------|-------|
| EpisodicMemory.recall_recent(10) from 200 eps | **0.002 ms** | **0.002 ms** | Both fast (deque slice) |
| EpisodicMemory.recall_similar (k=5, 200 eps) | **0.170 ms** | **0.043 ms** | Python LSH faster here |
| SemanticMemory.query (500 concepts, top-10) | **1.71 ms** | — | Vector search via HNSW |
| SemanticMemory.spread_activation (500 nodes, 3 hops) | **1.42 ms** | — | Graph BFS |

**Observation on EpisodicMemory:** The Python LSH path (0.043 ms) is currently faster than the Rust kNN path (0.170 ms) for small memories because the LSH index lookup is O(1) while Rust kNN does a linear scan through the Rust backend's mirror copy + Python-Rust boundary overhead.

### Cognitive Engine Throughput

| Metric | Value |
|--------|-------|
| NSCKSubstrate init (Rust backend) | ~2,000 ms (one-time) |
| substrate.ingest() per sentence | ~2.5 ms |
| CognitiveEngine.decide() per state | ~0.6 ms |
| NSCKHDVisionClassifier.fit() (90 samples) | 1,559 ms |
| NSCKHDVisionClassifier.predict() (15 samples) | 99 ms (feature extraction dominates) |
| Image feature extraction (28×28) | 10.1 ms, 644 features |

---

## 6. Every Subsystem Explained

### 6.1 VSA Layer (`nsck/python/core/vsa/`)

The mathematical foundation. Three operations on 10,240-bit binary vectors:

**XOR (Binding):** `A ⊗ B = A XOR B`
- Creates a new unique vector encoding "A in relation to B"
- Self-inverse: `(A ⊗ B) ⊗ B = A` (perfect unbinding)
- Verified: `XOR is self-inverse: True` (live test)
- Rust: 0.37 µs | Python: 1.35 µs

**Bundle (Superposition):** `A ⊕ B = majority_vote(A, B, random_tiebreak)`
- Creates a vector similar to both A and B
- `similarity(bundle(A,B), A) ≈ 0.74` (verified: 0.7432)
- `similarity(bundle(A,B), B) ≈ 0.75` (verified: 0.7468)
- Rust: 0.81 µs | Python: 57.81 µs (**71× speedup**)

**Similarity (Hamming-based):** `1 - hamming_distance / 10240`
- `similarity(A, A) = 1.0` (verified: 1.0000)
- `similarity(A, random_B) ≈ 0.5` (verified: 0.4899)
- Rust: 0.22 µs | Python: 7.20 µs (**33× speedup**)

Additional operations: `negate` (XOR with fixed role vector), `permute` (circular rotation for temporal encoding), `weighted_bundle`, `lsh_hash` (deterministic LSH bucketing).

**Random Number Generators:**
- Rust: ChaCha8 (cryptographic stream cipher, extremely fast)
- Python: PCG64 (numpy default)
- **These produce DIFFERENT bit patterns for the same seed** — cross-backend HV comparison is unreliable. Tests correctly mark such comparisons as `xfail`.

### 6.2 Hypervec Shim (`nsck/python/core/vsa/hypervec_shim.py`)

The shim is a compatibility layer that:
1. Tries to import `hypervec_rs` (Rust .so)
2. If successful: patches missing methods onto the Rust class (`cosine_similarity`, `bits` property, `from_bits`, `permute`, etc.)
3. If failed: uses `HyperVectorPy` (numpy implementation) as fallback

**Critical performance issue in the shim (verified):** The `bits` property installed on Rust HVs unpacks 160 uint64 words bit-by-bit in a Python for-loop:
```python
for word_idx, word in enumerate(state):  # 160 iterations
    for bit_pos in range(64):             # 64 iterations each = 10,240 total
        if word & (1 << bit_pos):
            arr[word_idx * 64 + bit_pos] = 1
```
This makes `.bits` cost **1,193 µs/call** and `cosine_similarity()` cost **3,293 µs/call** — vs native Rust `similarity()` at **0.22 µs**. This is a **15,000× overhead** for cosine similarity on Rust HVs.

The shim also exports: `HyperVectorRegistry`, `EpisodicMemoryConcurrent`, `SemanticMemoryConcurrent`, `CognitiveWorkerPool`, `AsyncCognitiveRuntime`, `PersistentStorage`, `ActivationAccumulator`, parallel functions (`parallel_bundle`, `parallel_similarity_search`, `batch_parallel_similarity_search`, `run_semantic_search_async`).

### 6.3 Semantic Memory (`nsck/python/core/memory/semantic_memory.py`, 907 lines)

A concept knowledge graph with VSA similarity search:

**Architecture:**
- `networkx.DiGraph` for symbolic graph (concepts as nodes, typed relations as edges)
- HNSW index for approximate nearest-neighbor HV search (optional, `enable_hnsw_index`)
- Rust backend via `hypervec_rs.SemanticMemoryConcurrent` (when available)
- Hot cache (LRU, configurable size) for frequently accessed concepts

**Relations:** `is_a`, `has_property`, `relates_to`, `precedes`, `follows`, `implies`, `conditional_on`, `not_relates_to`

**Key methods:**
- `add_concept(name, properties)` — register concept + create HV
- `add_relation(concept1, relation_type, concept2)` — add typed edge
- `query(hv, k=5)` — find k nearest concepts by HV similarity
- `spread_activation(start_concepts, steps, decay)` — BFS activation propagation
- `infer_transitive(relation_type)` — close over `is_a` chains
- `build_prototypes(relation_type)` — bundle member HVs into prototype

**Performance (verified, 500 concepts, 2000 relations):**
- `query()` top-10: **1.71 ms/query** (Rust backend)
- `spread_activation()` 3 hops: **1.42 ms** (Rust backend)

### 6.4 Episodic Memory (`nsck/python/core/memory/episodic_memory.py`, 522 lines)

A temporal event store with two retrieval paths:

**Architecture:**
- Python `deque` (primary, authoritative) — holds `LiveEpisode` objects with full state
- Rust `EpisodicMemoryConcurrent` mirror — for parallel kNN when available
- Multi-table LSH index (4 tables × 10-bit hash → ~1024 buckets/table)

**LiveEpisode fields:** `timestamp`, `task_tag`, `situation_hv`, `state`, `action`, `outcome`, `reward`, `emotion`, `tom_beliefs`, `impact_score`

**Performance (verified, 200 episodes):**
- `recall_recent(10)`: **0.002 ms** (deque slice, O(1))
- `recall_similar(query, k=5)`: **0.170 ms** (Rust kNN)
- Python LSH path: **0.043 ms** (currently faster than Rust for small memories due to Python/Rust boundary overhead)

### 6.5 CognitiveEngine (`nsck/python/core/reasoning/cognitive_engine.py`, 2136 lines)

The largest file and the orchestrating brain. The `decide()` cycle:

```
state_dict
    → ModalityAdapter.encode() → PerceptPacket
    → GroundingVerifier.extract_predicates() → {predicate set}
    → RuleLearner.match_rules() → rule-based action proposals
    → Planner.plan() → plan-based proposals  
    → EpisodicMemory.recall_similar() → memory-based proposals
    → GlobalWorkspace.compete(coalitions) → winning coalition
    → return (action, confidence, explanation)
```

**Global Workspace Theory (GWT):** Coalitions (rule, memory, planner) compete for "broadcast" — the most salient/relevant wins. This implements a basic architecture of consciousness (Bernard Baars' GWT).

**Decision outcomes (verified, live test):**
- 4 diverse text inputs all got confidence = 0.500 (random because no training data)
- After `record_outcome()` calls, rules accumulate and confidence improves
- Actions: ACTION_UP/DOWN/LEFT/RIGHT/WAIT (default action space)

### 6.6 HD Vision Classifier (`nsck/python/core/vision/hd_classifier.py`, 731 lines)

Dual-architecture image classifier:

**Classification path (primary, high accuracy):**
1. `_extract_enhanced_features(img)` → 644-dimensional feature vector:
   - HOG (Histogram of Oriented Gradients) — shape/texture
   - LBP (Local Binary Pattern) — fine texture
   - HSV color histogram — color
   - Gabor filter bank (128 features, neuroscience-inspired)
   - FFT radial power (16 features, spatial frequency)
   - Euler characteristic (3 features, topology)
2. PCA-128 (dimensionality reduction, sklearn)
3. LDA projection (class discriminant, sklearn)
4. Cosine nearest-centroid in LDA space

**NSCK cognitive path (for memory binding):**
1. Same features + LDA
2. Level-coding codebook (64 levels, adjacent levels share 97.5% bits)
3. HV prototype per class for spreading activation / cross-modal binding

**Performance (verified, 90 training samples, 15 test):**
- Fit: 1,559 ms (dominated by 644 feature extraction × 90 samples = 10 ms × 90)
- Predict: 99 ms for 15 samples
- Accuracy: 100% on synthetic separable data
- Literature: 93.3% clean, 24.2% adversarial UPMA (documented)
- Class HVs are well-separated: `sim(class_0, class_1) = 0.451` ✓

### 6.7 Language Layer (`nsck/python/core/language/`)

15+ files covering NLU, NLG, semantics, grammar:

**VSANLUEngine:** Uses `DistributionalCodebook` — maps words to HVs via co-occurrence statistics. Intent classification via cosine similarity to prototype HVs.
- Init: 28 ms
- Classify: <0.1 ms per sentence
- **Current accuracy: 0/5** on basic test sentences (observed). The distributional codebook built from `BUILTIN_CORPUS` alone is too small. With large corpus (HuggingFace), accuracy improves significantly.

**TextKnowledgeLearner:** Reads text, extracts (subject, predicate, object) triples, stores in SemanticMemory as typed relations.

**SemanticBootstrapper:** 4-tier fallback: prebuilt HVs → bridge → HuggingFace corpus → built-in corpus.

**Level-coding breakthrough (V24):** Adjacent levels in the 64-level codebook share 97.5% bits → HV cosine approximates Euclidean distance. Gives 58.8% text classification accuracy vs 23.5% with random FPE.

### 6.8 Societal Knowledge World (`nsck/python/core/societal/`)

V5 feature: concepts as living agents in a self-organizing world:

**LivingHyperVector:** Wraps a raw HV with `stability` (0→1 with age), `valence` (±1 emotional charge), `activation` (decays 5%/tick), `bonds` (weighted links to other concepts), `domain_affinities`, `stability_class` (volatile/active/stable/crystallized).

**ValenceEngine:** Bond formation/breaking based on: 60% HV cosine similarity + 20% valence compatibility + 20% domain overlap. Threshold: 0.30 to form, 0.10 to break.

**SpectralLaplacianRG:** The one real matrix multiply in the codebase:
1. Build (N×10240) bipolar matrix
2. `normalized @ normalized.T` → N×N cosine similarity matrix
3. Normalized Laplacian: `L = I - D^{-½} A D^{-½}`
4. Eigendecompose: `eigh(L)` → spectral gap (Fiedler value)
5. k-means on top-k eigenvectors → supernodes

**Performance (verified, 50 concepts):**
- Similarity matrix matmul: **6.99 ms** (25.6M multiply-adds)
- Eigendecompose 50×50: **0.26 ms**
- Total spectral RG: **7.34 ms**

**SocietalHNSW:** 3-layer HNSW index (domain anchors → neighborhood anchors → all concepts) with `hnswlib` or pure-Python greedy NSW fallback.

**Identified performance bug:** `_hv_cosine_sim()` in `living_hypervector.py` calls `hv.cosine_similarity()`, which on Rust HVs invokes the shim's `.bits` property (1,193 µs/call) instead of the native `hv.similarity()` (0.22 µs/call). This makes the societal world query **15,000× slower than necessary** (66 ms for 20 concepts instead of <0.01 ms).

### 6.9 SNN Layer (Spiking Neural Network)

Located in `nsck/python/core/perception/` and `nsck/rust_snn/`.

**What it does:** Leaky Integrate-and-Fire (LIF) neurons for biological-style perception. Converts raw sensory arrays to spike trains. The spike patterns are then mapped to concept IDs via a `ConceptMapper` (cleanup memory lookup in SemanticMemory).

**When used:** `perceive_and_decide(sensory_array)` — the SNN path. `decide(state_dict)` bypasses the SNN.

**Status:** Disabled in fast tests (`cfg.enable_snn = False`). When enabled, `perceive_and_decide()` runs SNN every 5th call (STDP update) and uses fast inference otherwise.

### 6.10 Knowledge Transplant (`nsck/python/core/transplant/`)

Bridge between pre-trained embeddings and NSCK's VSA world:

```
Pre-trained embedding (e.g. sentence-transformers)
    → SVDFactoredProjector or RandomProjector
    → Project to 10240-dim space
    → FPE (Fixed-Point Encoding) to binary HV
    → Store in SemanticMemory as concept HV
```

**Performance:** SVDFactoredProjector gives 12.8% UPMA (Universal Pre-trained Model Absorption); the LDA discriminative path gives 18.9% UPMA.

### 6.11 Persistence (`nsck/python/core/integration/persistence.py`)

SQLite-backed brain store via `rusqlite` (bundled in Rust). Stores episodes, concepts, rules, and weights. Rust backend exposes `PersistentStorage` class.

---

## 7. Identified Bugs and Performance Issues

### Bug #1: `_hv_cosine_sim` Uses Slow Path for Rust HVs (CRITICAL PERFORMANCE)
**File:** `nsck/python/core/societal/living_hypervector.py:38-52`  
**Impact:** Societal world query is ~15,000× slower than necessary  
**Measured:** 66 ms for 20 concepts (should be <0.01 ms)  
**Root cause:** `_hv_cosine_sim` calls `hv_a.cosine_similarity()`, which on Rust HVs invokes the shim's `.bits` property (1,193 µs/call Python bit-unpacking loop), then does cosine. Should use `hv_a.similarity(hv_b)` (0.22 µs native Rust).  
**Fix:**
```python
def _hv_cosine_sim(hv_a, hv_b) -> float:
    # Use native Rust similarity (Hamming-based, fast) when available
    try:
        return float(hv_a.similarity(hv_b))
    except AttributeError:
        pass
    # ... fallback
```

### Bug #2: `NSCKConfig` Not Accepted by `SemanticMemory.__init__`
**File:** `nsck/python/core/memory/semantic_memory.py:216`  
**Impact:** Cannot pass `NSCKConfig()` directly; must pass `None` or a compatible dict-like object  
**Root cause:** `self.relation_weights.update(relation_weights)` called with the config object

### Bug #3: NLU Accuracy Near Random (0/5 test sentences)
**File:** `nsck/python/core/language/vsa_nlu.py`  
**Impact:** Intent classification falls back to `farewell`/`negation`/`question` for unrelated inputs  
**Root cause:** `DistributionalCodebook.build_default()` uses `BUILTIN_CORPUS` (too small), all intents end up with similar HV prototypes — cosine similarity between all prototypes is near 0.5 (random)  
**Fix:** Enable `enable_hf_corpus=True` or supply a pre-built codebook

### Bug #4: `EpisodicMemory.recall_similar` (Rust Slower Than Python)
**File:** `nsck/python/core/memory/episodic_memory.py:325`  
**Impact:** For small memories (<500 episodes), Rust kNN (0.170 ms) is 4× slower than Python LSH (0.043 ms) due to Python↔Rust boundary overhead  
**This is architectural, not a bug** — Rust wins at scale (>1000 episodes)

### Issue #5: `.bits` Property is O(10240) Python Loop
**File:** `nsck/python/core/vsa/hypervec_shim.py:56-66`  
**Impact:** Any code path that calls `.bits` on Rust HVs pays 1,193 µs  
**Affects:** `cosine_similarity()`, `_hv_cosine_sim()`, `negate()` (Python fallback), `permute()` (Python fallback)  
**Fix:** Use `numpy.packbits`/`unpackbits` or vectorized bit extraction:
```python
def _bits_property(self):
    state = self.__getstate__()  # list of 160 u64 ints
    arr = np.array(state, dtype=np.uint64)
    return np.unpackbits(arr.view(np.uint8), bitorder='little')[:10240].astype(np.int8)
```

---

## 8. Where the System Stands Today

### What Works Well

| Capability | Status | Evidence |
|-----------|--------|---------|
| Core VSA operations | ✅ Excellent | XOR, bundle, similarity all correct; self-inverse, orthogonality verified |
| Rust backend speedup | ✅ Excellent | 5–82× speedup depending on op; bundle especially (82×) |
| Rust parallel ANN | ✅ Good | 0.31 ms for 1,000 HVs; 4.94 ms for 10,000 HVs |
| HD Vision Classifier | ✅ Good | 100% on separable synthetic; 93.3% documented clean accuracy |
| Semantic Memory | ✅ Good | Correct nearest-neighbor + spreading activation; HNSW available |
| Episodic Memory | ✅ Good | LSH retrieval fast; Rust mirror works for large scale |
| CognitiveEngine decide() | ✅ Good | 0.6 ms/decision; GWT architecture integrated |
| Rust spreading activation | ✅ Fast | 2.9 µs/step |
| Weber-Fechner compression | ✅ Works | Correct log scaling: 0→0, 10→2.40, 99→4.61 |
| Persistent Storage | ✅ Available | SQLite via rusqlite; PersistentStorage exposed |

### What Has Issues

| Capability | Status | Issue |
|-----------|--------|-------|
| Societal world query | ⚠️ Bug | 15,000× slower than necessary (`.bits` property overhead) |
| NLU intent classification | ⚠️ Weak | Near-random without large corpus; 0/5 on basic sentences |
| Image feature extraction | ⚠️ Slow | 10.1 ms per 28×28 image (CPU-only, no optimization) |
| HDClassifier fit | ⚠️ Slow | 1,559 ms for 90 samples (feature extraction bottleneck) |
| EpisodicMemory Rust recall | ⚠️ Suboptimal | Python LSH 4× faster than Rust kNN for small memories |

### Code Quality Observations

- **Well-structured layering:** Each subsystem is in its own module with clear responsibilities
- **Dual-backend pattern consistently applied:** Every major component checks for Rust and falls back to Python
- **Extensive test infrastructure:** 580 Python files including many tests in `nsck/tests/`
- **Config explosion:** `NSCKConfig` has 50+ feature flags — some are checked inconsistently (e.g., `enable_semantic_bootstrap` exists but is never checked in `TextKnowledgeLearner`)
- **Archive folder:** ~60 older prototype files kept in `nsck/archive/` — these are not dead code traps (they're separated) but increase codebase navigation complexity
- **Societal world is ambitious but has the critical performance bug** — fixing `.bits` would make it viable

---

## 9. Final Verdict and Recommendations

### System Character

NSCK is a **mathematically principled, non-neural cognitive architecture** that achieves:
- **Instant learning** (one bundle = one memory write, no iterations)
- **No catastrophic forgetting** (VSA memories superpose gracefully)
- **Interpretability** (every operation has a clear algebraic meaning)
- **Determinism** (same inputs → same outputs, except for stochastic tiebreaking in bundle)
- **Extreme efficiency for core ops** (0.22 µs similarity, 0.81 µs bundle in Rust)

### Compared to Deep Learning

| Dimension | Deep Learning | NSCK |
|----------|--------------|------|
| Training time | Hours to weeks (GPUs) | Milliseconds (CPU) |
| Memory per concept | Embedded in weight matrices | 1,280 bytes per concept (Rust HV) |
| Catastrophic forgetting | Major problem | By design avoided (VSA superposition) |
| Interpretability | Black box | Every operation algebraically interpretable |
| Accuracy (vision) | >99% (ResNet on MNIST) | 93.3% clean (HD classifier) |
| Adversarial robustness | Poor | 24.2% UPMA |
| Gradient required | Yes | No |
| GPU required | Typically yes | No — pure CPU |

### Priority Recommendations

1. **Fix the `.bits` property** in `hypervec_shim.py` — use `numpy.unpackbits` instead of the Python loop. This will make the societal world viable and fix `cosine_similarity()` on Rust HVs. **Single biggest win available.**

2. **Fix `_hv_cosine_sim`** in `living_hypervector.py` to call `hv_a.similarity(hv_b)` instead of `hv_a.cosine_similarity(hv_b)` when dealing with Rust HVs.

3. **NLU corpus:** Enable `enable_hf_corpus=True` or ship a pre-built `DistributionalCodebook` with the package. The current built-in corpus is too small for useful intent classification.

4. **Feature extraction caching:** Image feature extraction (10.1 ms/image) dominates `NSCKHDVisionClassifier` performance. Adding an LRU cache for repeated images would help production use.

5. **Test the societal world at scale:** After fixing the `.bits` bug, run the spectral RG and HNSW on 1,000+ concepts to validate the V5 societal architecture at real scale.

---

## 10. Fixes Applied in This Analysis

### Fix 1: `.bits` Property — 75× Speedup
**File:** `nsck/python/core/vsa/hypervec_shim.py`  
**Change:** Replaced 10,240-iteration Python bit-unpacking loop with `numpy.unpackbits`:
```python
# Before (O(10240) Python loop — 1,193 µs/call)
for word_idx, word in enumerate(state):
    for bit_pos in range(64):
        if word & (1 << bit_pos):
            arr[word_idx * 64 + bit_pos] = 1

# After (vectorised numpy — 16 µs/call — 75× faster)
u64_arr = np.array(state, dtype=np.uint64)
unpacked = np.unpackbits(u64_arr.view(np.uint8), bitorder="little")
return unpacked[:10240].astype(np.int8)
```
**Correctness:** Round-trip verified (`pack(unpack(x)) == x`). All 1309 unit tests pass.

### Fix 2: `_hv_cosine_sim` — 12,000× Speedup
**File:** `nsck/python/core/societal/living_hypervector.py`  
**Change:** Replaced `hv_a.cosine_similarity(hv_b)` (which went through `.bits`) with `hv_a.similarity(hv_b)` (native Rust Hamming similarity):
```python
# Before (3,423 µs via .bits overhead)
return float(hv_a.cosine_similarity(hv_b))

# After (0.28 µs via native similarity())
return float(hv_a.similarity(hv_b))
```
**Impact:** Societal world query: 66.87 ms → 0.013 ms (**5,149× speedup**).  
**Societal full evaluation:** 26/26 checks pass in **0.297 seconds**.

---

## Appendix: Verified Live Measurements Summary

All numbers below are from live execution in this environment (Python 3.12, Linux x86_64, no GPU).

```
=== VSA Core (Rust backend) ===
HV dimension:              10,240 bits (160 × uint64)
XOR:                          0.37 µs/call
Bundle (2-way):               0.81 µs/call  
Similarity:                   0.22 µs/call
Negate:                       0.64 µs/call
Permute:                      0.63 µs/call
Bind + Unbind roundtrip:      0.44 µs/pair
100-way bundle:               0.07 ms/call
Parallel ANN (1,000 HVs):    0.31 ms/query
Parallel ANN (10,000 HVs):   4.94 ms/query

=== VSA Core (Python fallback) ===
HV create:                   49.8 µs/call  (23× slower than Rust)
XOR:                          1.35 µs/call
Bundle (2-way):              57.81 µs/call  (71× slower than Rust)
Similarity:                   7.20 µs/call  (33× slower than Rust)
Linear search (1,000 HVs):   7.41 ms/query (24× slower than Rust)
Linear search (10,000 HVs): 81.34 ms/query (16× slower than Rust)

=== VSA Capability Benchmark (official, Rust) ===
causal_enrich_ops/s:       1,230,645
causal_chain_ops/s:          402,285
perceptual_enrich_ops/s:   1,113,015
semantic_enrich_ops/s:     1,560,479
semantic_bulk_10_ops/s:      109,046
glass_box_decisions/s:       169,142
crossmodal_link_ops/s:       981,251

=== Memory Systems ===
SemanticMemory query (500 concepts):       1.71 ms/query
SemanticMemory spread (500 nodes, 3 hops): 1.42 ms/call
EpisodicMemory recall_recent(10):          0.002 ms/call
EpisodicMemory recall_similar(k=5):        0.170 ms/call (Rust kNN)

=== Societal Knowledge World (after fix) ===
World query (20 concepts, top-5):          0.013 ms/query  (was: 66.87 ms → 5149× fix)
Societal full eval (26 checks):            0.297 s total, 26/26 PASS (100%)

=== Application Layer ===
NSCKSubstrate init:               2,057 ms (one-time)
substrate.ingest() per sentence:     2.5 ms
CognitiveEngine.decide():            0.6 ms
HDVisionClassifier.fit (90 samples): 1,559 ms
HDVisionClassifier.predict (15):      99 ms
Image feature extraction (28×28):    10.1 ms, 644 features

=== Rust Parallel Ops ===
parallel_similarity_search (1k):  0.31 ms
batch_parallel_search (10q×1k):   1.57 ms
parallel_bundle (100 HVs):        0.45 ms
HyperVectorRegistry (1k):         0.42 ms
HyperVectorRegistry (10k):        4.94 ms (Rayon parallel)
batch_similarity_matrix (50):      0.27 ms
spreading_activation (20 nodes):  2.9 µs
weber_fechner_compress (100 val): 2.67 µs

=== Algebraic Correctness ===
XOR is self-inverse:              True ✓
similarity(A,A):                  1.0000 ✓
similarity(A,random):             0.4899 ✓ (expected ~0.5)
bundle(A,B) sim to A:             0.7432 ✓ (expected ~0.75)
bundle(A,B) sim to B:             0.7468 ✓ (expected ~0.75)
negate(negate(A)) == A:           True ✓
permute(A,1) sim to A:            0.4980 ✓ (expected ~0.5, orthogonal)
LSH hash deterministic:           True ✓
.bits round-trip (new impl):      True ✓

=== Test Suite ===
Unit tests passed:  1309
Skipped (SNN .so): 33 (not built)
Expected failures: 1 (cross-backend RNG, correct)
FAILURES:          0
```
