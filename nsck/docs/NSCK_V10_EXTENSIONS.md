# NSCK V10 — Intelligence Extensions

V10 builds on the V9 modality-agnostic substrate (see `NSCK_V9_SUBSTRATE.md`) by adding eight new modules that enrich the cognitive engine with richer VSA operations, neural rule scoring, attention-guided arbitration, and a formal safety layer.

---

## Table of Contents

1. [Overview](#1-overview)
2. [VSA Embedding Bridge](#2-vsa-embedding-bridge)
3. [FHRR — Fourier Holographic Reduced Representations](#3-fhrr)
4. [Rust Concurrent Memory Shim](#4-rust-concurrent-memory-shim)
5. [N-gram NLU](#5-n-gram-nlu)
6. [Attention-GWT Bridge](#6-attention-gwt-bridge)
7. [Rule Neural Scorer](#7-rule-neural-scorer)
8. [Safety Verifier](#8-safety-verifier)
9. [NSCK REST API](#9-nsck-rest-api)
10. [Integration Wiring](#10-integration-wiring)

---

## 1. Overview

| # | Module file | Key class(es) | Role |
|---|---|---|---|
| V10.1 | `vsa/vsa_embedding_bridge.py` | `EmbeddingVSABridge` | Dense embedding ↔ HyperVector projection |
| V10.2 | `vsa/fhrr.py` | `FHRRVector`, `FHRRMemory` | Complex-phasor VSA (invertible bind, gradient-friendly) |
| V10.3 | `vsa/rust_concurrent_shim.py` | `SemanticMemoryConcurrent`, `EpisodicMemoryConcurrent`, … | Rust-parallel memory with Python fallbacks |
| V10.4 | `language/ngram_nlu.py` | `NgramNLU` | Naive Bayes intent + entity extraction |
| V10.5 | `reasoning/attention_gwt_bridge.py` | `MultiHeadAttentionGWT`, `GWTAttentionBridge` | Attention-based coalition salience re-weighting |
| V10.6 | `learning/rule_neural_scorer.py` | `RuleNeuralScorer`, `RuleFeaturizer` | Online perceptron rule ranking |
| V10.7 | `cognitive/safety_verifier.py` | `SafetyRuleVerifier`, `SafetyGateVerifier`, `SafetyProperty` | Declarative safety property checking |
| V10.8 | `api/nsck_api.py` | `NSCKApiServer` | FastAPI / stdlib HTTP REST wrapper |

---

## 2. VSA Embedding Bridge

**File:** `python/core/vsa/vsa_embedding_bridge.py`

Bridges any dense floating-point embedding (e.g. from sentence-transformers or a
custom neural encoder) into the binary HyperVector space used by NSCK's VSA engine.

### How it works

A random projection matrix **P** ∈ ℝ^(dim_in × hv_dim) is drawn once at construction
time.  To project a dense vector **e** ∈ ℝ^dim_in:

```
bits = sign(e @ P)     → binary HyperVector ∈ {0,1}^hv_dim
```

The reverse projection (for interpretability) computes:

```
reconstructed = bits @ P^T            (pseudo-inverse)
reconstructed = reconstructed / ‖reconstructed‖
```

This preserves directional structure: similar embeddings project to HVs with higher
Hamming similarity, enabling VSA binding and similarity search on neural representations.

### Usage

```python
from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge

bridge = EmbeddingVSABridge(dim_in=384, hv_dim=10240)

# From a numpy embedding
import numpy as np
emb = np.random.randn(384).astype(np.float32)
hv = bridge.embed_to_hv(emb)

# Text encoding (sentence-transformers if installed, n-gram fallback otherwise)
hv2 = bridge.encode_text("the cat sat on the mat")
print(hv.similarity(hv2))

# Reverse projection
approx_emb = bridge.hv_to_embed(hv)
```

---

## 3. FHRR

**File:** `python/core/vsa/fhrr.py`

Fourier Holographic Reduced Representations (FHRR) encode information in the
*phases* of unit-magnitude complex numbers (phasors).  Unlike binary HVs, FHRR
supports:

- **Exact inverse binding** — `v.bind(k).unbind(k) ≈ v` (not possible with binary HV XOR).
- **Gradient-compatible similarity** — real/imag decomposition enables back-propagation.
- **Scalar encoding** — deterministic, quasi-orthogonal phasors for numeric values.

### Operations

| Operation | Formula |
|---|---|
| Bind | `(a ⊗ b)_i = a_i · b_i` (complex multiply) |
| Unbind | `(a ⊘ b)_i = a_i · conj(b_i)` |
| Bundle | `normalise(a + b + c + …)_i` |
| Similarity | `cos(‖a‖, ‖b‖)` (magnitude cosine) |

### Usage

```python
from python.core.vsa.fhrr import FHRRVector, FHRRMemory

dog = FHRRVector.encode_symbol("dog")
mammal = FHRRVector.encode_symbol("mammal")
bound = dog.bind(mammal)          # encodes "dog is-a mammal"

mem = FHRRMemory(dim=1024)
mem.store("dog_isa_mammal", bound)
retrieved = mem.retrieve("dog_isa_mammal")
name, sim = mem.cleanup(retrieved, [("dog_isa_mammal", bound)])
print(name, sim)  # dog_isa_mammal 1.0
```

---

## 4. Rust Concurrent Memory Shim

**File:** `python/core/vsa/rust_concurrent_shim.py`

Transparent shim that selects between the compiled Rust extension
(`hypervec_rs.SemanticMemoryConcurrent`, etc.) and pure-Python fallbacks.

### Exported symbols

| Symbol | Rust | Python fallback |
|---|---|---|
| `SemanticMemoryConcurrent` | Rayon-parallel k-NN search | Sequential dict-based search |
| `EpisodicMemoryConcurrent` | Lock-free episode buffer | Deque-based episode list |
| `Episode` | Rust struct | Python dataclass |
| `HyperVectorRegistry` | HashMap-backed | Dict-backed |
| `PersistentStorage` | SQLite-backed | In-memory dict |
| `parallel_bundle` | Rayon bundle | Sequential bundle |
| `batch_parallel_similarity_search` | Rayon k-NN | Sequential k-NN |
| `batch_similarity_matrix` | Rayon matrix | Nested loop |

### Status inspection

```python
from python.core.vsa.rust_concurrent_shim import get_status
print(get_status())
# {'use_rust': True, 'available_classes': ['SemanticMemoryConcurrent', ...]}
```

---

## 5. N-gram NLU

**File:** `python/core/language/ngram_nlu.py`

A dependency-free probabilistic NLU layer.  No external NLP library is required.

### Model

Laplace-smoothed Naive Bayes over unigram + bigram (default) features.  Default
intent labels: `question`, `command`, `statement`, `greeting`, `farewell`.

### Usage

```python
from python.core.language.ngram_nlu import NgramNLU

nlu = NgramNLU(n=2)
# Train on labeled examples (optional — heuristics work without training)
nlu.train(["hello there", "hi"], ["greeting", "greeting"])

intent, confidence = nlu.extract_intent("What is the capital of France?")
print(intent, confidence)  # question  0.89

entities = nlu.extract_entities("The cat sat on the mat in London")
print(entities)  # [('cat', 'NOUN'), ('mat', 'NOUN'), ('London', 'PROPN')]
```

---

## 6. Attention-GWT Bridge

**File:** `python/core/reasoning/attention_gwt_bridge.py`

Implements multi-head attention over the set of competing GWT coalitions.  Each
coalition is projected to a key vector; the current situation HV is the query.
Attention scores are used to scale `base_salience` values before the final GWT
winner-take-all competition.

### Architecture

```
n_heads × AttentionHead (key_dim=64)
    each head: Wq (key_dim × key_dim), Wk (key_dim × key_dim)
    score = softmax(q·k^T / √key_dim)
average head scores → per-coalition attention weight
salience_new = salience_old × attention_weight
```

### Usage

```python
from python.core.reasoning.attention_gwt_bridge import GWTAttentionBridge

bridge = GWTAttentionBridge(n_heads=4, key_dim=64)
reranked = bridge.rerank(coalitions, situation_hv)
# reranked coalitions have adjusted base_salience values
```

---

## 7. Rule Neural Scorer

**File:** `python/core/learning/rule_neural_scorer.py`

A two-layer MLP (6 → 16 → 1, sigmoid) that learns to score symbolic rules.
Trained online: after each `learn()` cycle, `update(rule, reward)` performs one
gradient step.

### Feature vector (6 dimensions)

| Index | Feature | Description |
|---|---|---|
| 0 | `confidence` | Rule confidence in [0, 1] |
| 1 | `support` | `support_count / 100` capped at 1 |
| 2 | `fire_ratio` | `fire_count / 10000` capped at 1 |
| 3 | `complexity` | `len(conditions) / 10` capped at 1 |
| 4 | `trend` | Linear slope of recent confidence history |
| 5 | `has_task` | 1.0 if task_tag is not None/global |

### Integration with CognitiveEngine

```python
from python.core.learning.rule_neural_scorer import RuleNeuralScorer

engine.rule_scorer = RuleNeuralScorer()

# Scorer is consulted automatically in decide():
#   scored = engine.rule_scorer.rank_rules(applicable_rules)
#   rule = scored[0]
```

Weight persistence:

```python
engine.rule_scorer.save("checkpoints/rule_scorer")
engine.rule_scorer.load("checkpoints/rule_scorer")
```

---

## 8. Safety Verifier

**File:** `python/core/cognitive/safety_verifier.py`

Provides a declarative safety layer.  Each `SafetyProperty` is a named Python
expression evaluated in a restricted namespace (no arbitrary builtins — `__builtins__`
is replaced with `{}`).

### Default properties

| Name | Severity | Formula |
|---|---|---|
| `min_confidence` | warning | `confidence > 0.3` |
| `min_support` | warning | `support >= 2` |
| `no_runaway` | critical | `fire_count < 10000` |
| `no_code_injection` | critical | `action not in FORBIDDEN_ACTIONS` |

`FORBIDDEN_ACTIONS = ["__import__", "exec", "eval", "os.system", "subprocess"]`

### Adding custom properties

```python
from python.core.cognitive.safety_verifier import SafetyRuleVerifier, SafetyProperty

verifier = SafetyRuleVerifier()
verifier.add_property(SafetyProperty(
    "max_conditions",
    "len(conditions) < 20",
    severity="warning",
))
report = verifier.verify_rule(my_rule)
print(report)
# {'safe': True, 'violations': [], 'score': 1.0, 'critical_violations': 0}
```

### Gating decisions

`SafetyGateVerifier.gate_decision(action, confidence, active_rules)` blocks the
action (returns `(False, reason)`) if any active rule has a critical violation, or
if confidence < 0.3 for a non-explore action.

---

## 9. NSCK REST API

**File:** `api/nsck_api.py`

Wraps `CognitiveEngine` as an HTTP service.  Uses FastAPI + uvicorn if available;
falls back to Python's built-in `http.server`.

### Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/decide` | `{state: dict, task_tag: str}` | `{action, confidence, explanation, active_predicates}` |
| POST | `/learn` | `{state, action, reward, task_tag, outcome?}` | `{status: "ok"}` |
| POST | `/sleep` | `{task_tag?: str}` | `{status: "sleep_complete"}` |
| GET | `/status` | — | `{status, tasks, task_count, decisions, config}` |

Error responses from `/decide` include `{error: str, action: "explore", confidence: 0.0}`.

### Quick start

```python
from api.nsck_api import NSCKApiServer

server = NSCKApiServer()
server.run(host="127.0.0.1", port=8000)
```

Or use programmatically (no HTTP server needed):

```python
server = NSCKApiServer()
resp = server.handle_decide({"x": 1, "y": 2}, task_tag="demo")
print(resp["action"])
```

---

## 10. Integration Wiring

All V10 modules are optional enhancements — the cognitive engine runs correctly
without them.  To activate them:

```python
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.config import NSCKConfig
from python.core.learning.rule_neural_scorer import RuleNeuralScorer
from python.core.reasoning.attention_gwt_bridge import GWTAttentionBridge

cfg = NSCKConfig.research()
engine = CognitiveEngine(cfg)

# Attach neural rule scorer
engine.rule_scorer = RuleNeuralScorer()

# The attention bridge and safety verifier are consulted automatically
# when wired into the decide() pipeline via the config flags.
```

`NSCKConfig` flags relevant to V10:

| Flag | Default | Effect |
|---|---|---|
| `enable_auto_transfer` | `True` | V9 auto-transfer on register_task() |
| `enable_sleep` | `True` | Offline consolidation via sleep() |
| `enable_homeostasis` | `True` | Memory homeostasis after sleep() |

See `python/core/integration/config.py` for the full list of flags.
