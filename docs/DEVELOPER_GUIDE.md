# NSCK Developer Guide

> Everything you need to modify, extend, and contribute to the Neuro-Symbolic Cognitive Kernel.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Repository Layout](#2-repository-layout)
3. [Module Anatomy](#3-module-anatomy)
4. [The VSA Contract](#4-the-vsa-contract)
5. [Adding a New Module](#5-adding-a-new-module)
6. [Data Flow Walk-Through](#6-data-flow-walk-through)
7. [Configuration & Tuning](#7-configuration--tuning)
8. [Testing](#8-testing)
9. [Coding Standards](#9-coding-standards)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Debugging Tips](#11-debugging-tips)
12. [Performance Considerations](#12-performance-considerations)
13. [Extension Points](#13-extension-points)
14. [Git Workflow](#14-git-workflow)

---

## 1. Environment Setup

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Required by `pyproject.toml` |
| pip | ≥ 23.0 | For dependency resolution |
| numpy | ≥ 1.24.0 | Core numerical dependency |
| scipy | ≥ 1.10.0 | Distance calculations |
| scikit-learn | ≥ 1.3.0 | Used by world model |
| rustworkx | ≥ 0.14.0 | Graph library for causal reasoning |
| torch | ≥ 2.0.0 | World model numeric ensemble (CPU only) |
| pydantic | ≥ 2.0.0 | Config / structured output |
| rich | ≥ 13.0 | Terminal formatting |

### Quick Setup

```bash
# 1. Clone
git clone https://github.com/shiva2321/Node_network.git
cd Node_network
git checkout NSCK_V2

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
PYTHONPATH=nsck-demo/python python nsck_capability_test.py
# Expected: capability suite completes without errors
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `NSCK_DEVICE` | `cpu` | Device backend (`cpu` or `cuda`) |
| `NSCK_LR` | `1e-3` | Learning rate |
| `NSCK_MODEL_PATH` | `snn_task_aware.pth` | SNN model checkpoint path |
| `NSCK_DB_PATH` | `nsck_brain.db` | SQLite persistence DB |
| `PYTHONPATH` | — | Must include `nsck-demo/python` for imports |

### Optional Dependencies

```bash
# For SNN integration
pip install snntorch>=0.9.0

# For web dashboard
pip install flask>=2.3.0 flask-socketio>=5.3.0

# For enhanced NLU (not required for VSA-only mode)
pip install sentence-transformers>=2.2.0
```

---

## 2. Repository Layout

```
Node_network/
├── README.md                          # Project overview & Quick Start
├── requirements.txt                   # pip dependencies
├── pytest.ini                         # Pytest configuration
├── nsck_capability_test.py            # Main capability suite (see output for counts)
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── VSA_THEORY.md                  # Mathematical foundations
│   ├── MODULE_REFERENCE.md            # Complete API reference
│   ├── TESTING.md                     # Test methodology & evidence
│   └── DEVELOPER_GUIDE.md            # This file
│
└── nsck-demo/                         # Core package
    ├── pyproject.toml                 # Package metadata
    ├── conftest.py                    # Pytest fixtures
    │
    ├── python/                        # All source modules
    │   ├── __init__.py
    │   ├── config.py                  # NSCKConfig dataclass
    │   ├── hypervec_py.py             # Pure-Python VSA primitives
    │   ├── hypervec_shim.py           # HyperVector API facade
    │   ├── universal_input.py         # Grounding layer
    │   ├── cognitive_engine.py        # Main orchestrator
    │   ├── episodic_memory.py         # LSH episode store
    │   ├── semantic_memory.py         # Concept graph + VSA
    │   ├── causal_reasoning.py        # Delta-P + chain reasoning
    │   ├── emotion_system.py          # Drive-based affect
    │   ├── theory_of_mind.py          # Agent mental models
    │   ├── world_model.py             # VSA + numeric ensemble
    │   ├── global_workspace.py        # GWT competition
    │   ├── module_registry.py         # Module SDK discovery/registration
    │   ├── brain_fusion.py            # Cross-task transfer
    │   ├── analogy.py                 # Structure mapping
    │   ├── curiosity.py               # Novelty detection
    │   ├── planner.py                 # STRIPS BFS planner
    │   ├── rule_learner.py            # Frequency-based ILP
    │   ├── self_model.py              # Performance metacognition
    │   ├── nlg.py                     # Natural language generation
    │   ├── language_module.py         # POS tagger + parser
    │   ├── grounding_verifier.py      # Symbol↔sensor audit
    │   ├── persistence.py             # SQLite BrainStore
    │   └── unified_dashboard.py        # Unified scientific dashboard (Flask)
    │
    ├── rust_vsa/                      # Rust VSA backend (optional)
    │   └── src/lib.rs
    │
    ├── tests/                         # Additional test files (500+ test functions)
    ├── nsck_sdk/                       # External module SDK + examples
    │   └── ...
    │
    └── web/                           # Dashboard frontend
        └── ...
```

---

## 3. Module Anatomy

Every NSCK module follows a consistent pattern:

```python
"""
Module docstring with purpose, design decisions, and references to VSA_THEORY.md.
"""

import numpy as np
import hypervec_shim as hv
from config import NSCKConfig

# Constants (module-level)
DEFAULT_THRESHOLD = 0.55

class ModuleName:
    """One-line summary.
    
    Detailed description explaining:
    - What cognitive function this provides
    - How it uses VSA operations
    - What other modules it depends on
    """
    
    def __init__(self, config: NSCKConfig = None, **kwargs):
        """Constructor always accepts NSCKConfig."""
        self.config = config or NSCKConfig()
        # Internal state initialisation
        
    def primary_operation(self, input_hv: hv.HyperVector) -> Any:
        """Main cognitive operation.
        
        Args:
            input_hv: Grounded HyperVector from UniversalInput
            
        Returns:
            Result type depends on module
        """
        pass
```

### Key Patterns

1. **Config injection**: Every module accepts `NSCKConfig` (defaults to global singleton).
2. **HyperVector as lingua franca**: Modules communicate via 10 240-bit binary HVs.
3. **No external network calls**: All processing is local — no APIs, no web requests.
4. **Stateful but serialisable**: Internal state can be persisted via `BrainStore`.
5. **Module independence**: Each module can be instantiated and tested in isolation.

---

## 4. The VSA Contract

All modules must respect the **VSA contract** — the algebraic rules that make hypervector operations meaningful.

### HyperVector Basics

```python
import hypervec_shim as hv

# Create
v = hv.HyperVector(10240)              # Random 10,240-bit vector
v = hv.HyperVector(10240, seed=42)     # Deterministic from seed

# Operations
c = hv.HyperVector.xor(a, b)          # Bind (≈ variable binding)
s = hv.HyperVector.bundle([a, b, c])  # Superposition (≈ set union)
p = a.permute(1)                       # Sequence marker (shift bits)

# Similarity
sim = a.cosine_similarity(b)           # Normalised Hamming: 0.5 = random, 1.0 = identical
```

### Rules You Must Not Break

| Rule | Why |
|---|---|
| **DIMENSION = 10 240** always | Changing this breaks all existing codebook entries, episode stores, and LSH indexes |
| **XOR is binding**, not addition | XOR is self-inverse: `(a ⊕ b) ⊕ b = a`. This property powers unbinding. |
| **Bundle is majority vote for odd n** | For even n, use `_deterministic_bundle()` from `universal_input.py` to avoid non-determinism |
| **Permute is cyclic left-shift** | Permutation power encodes position: `π^k(v)` marks element at index `k` |
| **0.5 is random baseline** | Similarity < 0.5 means *anti-correlated*, not "far apart" |
| **Threshold for meaningful similarity: 0.55** | Below this, treat as noise |

### Segment Concatenation

NSCK uses a **non-standard** encoding for text: segment-based concatenation instead of pure bundling. The 10 240 bits are split into four segments:

| Segment | Bits | Weight | Content |
|---|---|---|---|
| Keyword | 0–5119 | 50% | Content-word codebook HVs |
| N-gram | 5120–6655 | 15% | Character trigram HVs |
| Word-order | 6656–8191 | 15% | Positional permutation HVs |
| Phrase-structure | 8192–10239 | 20% | Compositional parse tree HVs |

**Important**: If you add a new text component, you must update the segment allocations in `universal_input.py`, recalculate the math in `VSA_THEORY.md` §9, and re-run all text-related tests.

---

## 5. Adding a New Module

### Step-by-step

#### 1. Create the module file

```bash
touch nsck-demo/python/my_module.py
```

#### 2. Implement the module

```python
"""
NSCK My Module
=============
Purpose: [What cognitive function does this provide?]
VSA Operations Used: [bind / bundle / permute / similarity]
Dependencies: [What other modules does this need?]
"""
import numpy as np
import hypervec_shim as hv
from config import NSCKConfig

class MyModule:
    """One-line description of function."""
    
    def __init__(self, config: NSCKConfig = None):
        self.config = config or NSCKConfig()
        # Initialise internal state
        self._internal_store = {}
    
    def process(self, input_hv: hv.HyperVector) -> dict:
        """Process a grounded input.
        
        Args:
            input_hv: 10,240-bit HyperVector from UniversalInput
            
        Returns:
            Dict with processing results
        """
        # Your cognitive computation here
        return {"result": "computed"}
    
    def get_stats(self) -> dict:
        """Return internal diagnostics (for testing/debugging)."""
        return {"store_size": len(self._internal_store)}
```

#### 3. Register in CognitiveEngine

Edit `cognitive_engine.py`:

```python
# In imports section
from my_module import MyModule

# In CognitiveEngine.__init__()
self.my_module = MyModule(config=self.config)

# In decide() or appropriate integration point
my_result = self.my_module.process(situation_hv)
```

#### 4. Add tests

Add a test group in `nsck_capability_test.py`:

```python
# ── Test N: My Module ────────────────────────────────────────────
test_group_start("My Module", test_num := test_num + 1)

try:
    from my_module import MyModule
    mod = MyModule()
    
    # Test N.a — basic operation
    result = mod.process(some_hv)
    assert result is not None, "process returned None"
    record("my_module: basic operation", True, f"result={result}")
    
    # Test N.b — edge case
    edge_result = mod.process(zero_hv)
    assert edge_result["result"] != "", "empty result on edge case"
    record("my_module: edge case", True, f"edge={edge_result}")
    
except Exception as e:
    record("my_module: failed", False, str(e))
```

#### 5. Document

- Add to `MODULE_REFERENCE.md` — every public method with signature and description
- Add to `ARCHITECTURE.md` — where it fits in the layer diagram
- Update `TESTING.md` — new test group with expected values

#### 6. Verify

```bash
PYTHONPATH=nsck-demo/python python nsck_capability_test.py
# Must show N+85/N+85 tests passed with 0 failures
```

---

## 6. Data Flow Walk-Through

Here's a complete trace of a single cognitive cycle — follow along in the source code:

### Phase 1: Perception

```
Raw sensor input (numpy array or dict or text)
    │
    ▼
UniversalInput.encode(data, domain="snake")     [universal_input.py]
    │
    ├── If text:  _encode_text_composite()
    │       ├── _encode_text_keywords()        → 5120 bits
    │       ├── _encode_text_ngrams()          → 1536 bits
    │       ├── _encode_text_word_order()      → 1536 bits
    │       └── _encode_text_phrase()          → 2048 bits
    │       └── concatenate → 10240 bits
    │
    ├── If scalar: _encode_scalar_thermometer() → 10240 bits
    ├── If dict:   _encode_dict_role_filler()   → 10240 bits
    └── If list:   _encode_list_permutation()   → 10240 bits
    │
    ▼
situation_hv: HyperVector[10240]
```

### Phase 2: Multi-Module Processing

```
situation_hv
    │
    ├──→ EpisodicMemory.recall_similar(hv, k=5)     → past episodes
    ├──→ SemanticMemory.query_similar(hv, k=5)       → concepts + relations
    ├──→ CausalReasoner.forward_chain(predicates)    → predicted effects
    ├──→ EmotionSystem.update(hv, reward, outcome)   → affect state
    ├──→ CuriosityModule.novelty(hv)                 → novelty score
    ├──→ SelfModel.predict_success(task)             → confidence
    ├──→ TheoryOfMind.predict_action(agent, hv)      → agent prediction
    └──→ RuleLearner.applicable_rules(predicates)    → matching rules
    │
    ▼
List[Proposal(module, action, content, salience)]
```

### Phase 3: Global Workspace Competition

```
proposals
    │
    ▼
GlobalWorkspace.compete(proposals)                   [global_workspace.py]
    │
    ├── Sort by salience (highest wins ignition)
    ├── If danger_registered and salience > 0.75: force danger veto
    └── Winner = highest salience coalition
    │
    ▼
winning_action: str
```

### Phase 4: Mental Rehearsal

```
winning_action
    │
    ▼
WorldModel.imagine(situation_hv, action_hv)          [world_model.py]
    │
    ├── VSA transition memory: k-NN on (state ⊕ action)
    ├── Numeric ensemble: trust-weighted blend
    └── Expected reward + next_state_hv
    │
    ├── If imagined_reward < threshold → try next-best action (max 3 loops)
    └── If all imagined actions bad → explore (curiosity fallback)
    │
    ▼
final_action: str
```

### Phase 5: Learning

```
After environment returns real reward:
    │
    ├──→ EpisodicMemory.store(episode)
    ├──→ RuleLearner.record(predicates, action, outcome)
    ├──→ CausalDiscovery.update(event_a, event_b, co_occur?)
    ├──→ SelfModel.record(task, predicted, actual)
    ├──→ WorldModel.train(state_hv, action_hv, reward, next_state_hv)
    └──→ BrainFusion.align_concepts()
```

---

## 7. Configuration & Tuning

### NSCKConfig Fields

| Field | Type | Default | Effect |
|---|---|---|---|
| `device` | str | `"cpu"` | Computation backend |
| `learning_rate` | float | `1e-3` | SNN weight updates |
| `beta` | float | `0.5` | LIF neuron decay constant |
| `sleep_epochs` | int | `5` | Replay consolidation epochs |
| `replay_batch_size` | int | `32` | Batch size for experience replay |
| `save_interval` | float | `60.0` | Persistence save interval (seconds) |
| `vsa_strength` | float | `5.0` | VSA influence in hybrid decision |
| `confidence_threshold` | float | `0.6` | Entropy threshold for VSA rescue |
| `novelty_threshold` | float | `0.5` | Curiosity explore/exploit boundary |
| `grid_size` | int | `10` | Game grid dimensions |
| `adversarial_rate` | float | `0.05` | Adversarial noise injection rate |
| `episode_capacity` | int | `10000` | Max episodes in memory |
| `memory_capacity` | int | `2500` | Recent-memory ring buffer size |
| `min_rule_support` | int | `5` | Minimum observation count for rule induction |
| `min_rule_confidence` | float | `0.7` | Confidence threshold for rule induction |
| `min_success_rate` | float | `0.6` | Success rate filter |
| `persistence_db` | str | `"nsck_brain.db"` | SQLite database path |

### Hardcoded Constants (in source)

| Constant | Location | Value | Purpose |
|---|---|---|---|
| `DIMENSION` | `universal_input.py` | 10240 | VSA vector dimensionality |
| `DEFAULT_THERMOMETER_BINS` | `universal_input.py` | 100 | Scalar quantisation resolution |
| `MAX_CODEBOOK_SIZE` | `universal_input.py` | 10000 | LRU cache limit for categorical HVs |
| `TRUST_THRESHOLD` | `world_model.py` | 0.55 | VSA memory trust boundary |
| `DANGER_VETO_THRESHOLD` | `global_workspace.py` | 0.75 | Safety override salience |
| `GWT_ATTENTION_THRESHOLD` | `global_workspace.py` | 0.5 | Minimum salience for ignition |
| `ANALOGY_THRESHOLD` | `analogy.py` | 0.52 | Minimum similarity for concept transfer |
| `HV_ALIGNMENT_THRESHOLD` | `brain_fusion.py` | 0.75 | Cross-task concept alignment |
| `CTX_ALIGNMENT_THRESHOLD` | `brain_fusion.py` | 0.65 | Context alignment relaxed threshold |
| `LSH_HASH_BITS` | `episodic_memory.py` | 16 | LSH bucket hash width |
| `LSH_NUM_TABLES` | `episodic_memory.py` | 8 | Number of LSH hash tables |

### Tuning Guidelines

| Goal | What to change | Direction |
|---|---|---|
| Better text discrimination | Increase keyword segment weight (currently 50%) | ↑ keyword%, ↓ others |
| Better handling of word order | Increase word-order segment weight | ↑ word-order%, ↓ keyword% |
| Faster episode recall | Add more LSH tables | ↑ `LSH_NUM_TABLES` (cost: memory) |
| More precise episode recall | Increase LSH bits | ↑ `LSH_HASH_BITS` (cost: sparser buckets) |
| Explore more | Lower curiosity threshold | ↓ `novelty_threshold` |
| Stronger safety veto | Lower danger threshold | ↓ `DANGER_VETO_THRESHOLD` |
| Faster rule induction | Lower support requirement | ↓ `min_rule_support` (risk: spurious rules) |
| Better causal discovery | Add more training data | ↑ observation count per pair |

---

## 8. Testing

### Test File Structure

`nsck_capability_test.py` is structured as:

```python
# 1. Module loading verification (18/18 modules)
# 2. Corpus preparation (94 items: text/scalar/dict/sequence)
# 3. Test groups 1–20, each with:
#    - test_group_start(name, number)
#    - Individual tests with record(name, passed, details)
# 4. Final report: capabilities, design constraints, addressed limitations
```

### Running Specific Test Groups

The tests are monolithic (not parametrised). To run a subset, comment out other groups or add early `sys.exit()`.

For faster iteration during development:

```python
# Add at the end of any test group to stop early
if test_num >= 5:
    print_final_report()
    sys.exit(0)
```

### Adding Tests

When adding tests, follow the `record()` pattern:

```python
record(
    name="module_name: capability being tested",    # test label
    passed=True,                                    # bool
    details=f"measured_value={measured}"             # string with measured values
)
```

Always print measured values in the details — this makes debugging failures much easier than a bare `True/False`.

### Test Isolation

Each test group creates its own module instances. This means:

- No state leaks between test groups
- Each group can be understood independently
- Order of test groups shouldn't matter (though some share the same corpus)

---

## 9. Coding Standards

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Module files | `snake_case.py` | `causal_reasoning.py` |
| Classes | `PascalCase` | `CausalReasoner` |
| Methods | `snake_case` | `forward_chain()` |
| Constants | `UPPER_SNAKE_CASE` | `DIMENSION` |
| Private methods | `_leading_underscore` | `_encode_text_keywords()` |
| HV variables | suffix `_hv` | `situation_hv`, `action_hv` |
| Configuration | `lower_snake_case` | `min_rule_support` |
| Test functions | `test_group_start()` + `record()` | See test section |

### Docstrings

Every public class and method must have a docstring:

```python
def forward_chain(self, initial_predicates: Set[str], max_depth: int = 3) -> List[List[str]]:
    """Derive consequences from initial predicates via causal links.
    
    Uses breadth-first search through the CausalGraph, applying each
    causal link whose antecedent is in the current predicate set.
    
    Args:
        initial_predicates: Starting set of true predicates
        max_depth: Maximum chain length (prevents infinite loops)
        
    Returns:
        List of causal chains, each a list of predicate strings
        
    Example:
        >>> reasoner.forward_chain({"REL_ABOVE"}, max_depth=2)
        [["REL_ABOVE", "DANGER_ABOVE"]]
    """
```

### Type Annotations

All public methods must have type annotations:

```python
# Good
def recall_similar(self, query: hv.HyperVector, k: int = 5) -> List[Episode]:

# Bad
def recall_similar(self, query, k=5):
```

### Import Order

```python
# 1. Standard library
import os
import time
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import rustworkx as rx

# 3. NSCK modules
import hypervec_shim as hv
from config import NSCKConfig
from universal_input import UniversalInput
```

---

## 10. Common Pitfalls

### 1. Non-deterministic bundling

**Problem**: `HyperVector.bundle()` uses random tiebreaking for even-length input lists. This means calling `bundle([a, b])` twice with the same inputs can produce different results.

**Solution**: Use `_deterministic_bundle()` from `universal_input.py`:

```python
from universal_input import _deterministic_bundle
combined = _deterministic_bundle([hv_a, hv_b])  # Always same result
```

### 2. Similarity baseline confusion

**Problem**: New developers expect similarity = 0.0 for unrelated vectors. In binary VSA, unrelated vectors have similarity ≈ 0.5 (expected Hamming overlap of random bits).

**Solution**: Remember the scale:
- **0.50** = random / unrelated
- **0.55+** = weak signal
- **0.70+** = strong match
- **1.00** = identical

### 3. Forgetting PYTHONPATH

**Problem**: `ModuleNotFoundError: No module named 'hypervec_shim'` when running scripts.

**Solution**: Always set `PYTHONPATH`:
```bash
PYTHONPATH=nsck-demo/python python your_script.py
```

### 4. Mutating HyperVectors in place

**Problem**: HyperVector operations return new objects. Do not expect in-place mutation.

```python
# WRONG — a is unchanged
a.permute(1)

# RIGHT — capture the result
a_shifted = a.permute(1)
```

### 5. Changing DIMENSION without migration

**Problem**: Changing `DIMENSION` in code breaks all stored HVs, LSH indices, and codebook entries.

**Solution**: If you must change dimension, you need a migration script that re-grounds everything. Don't change it casually.

### 6. Testing with real randomness

**Problem**: Tests fail intermittently because HyperVector constructors use random seeds.

**Solution**: Always use deterministic seeds in tests:
```python
hv_a = hv.HyperVector(10240, seed=42)
```

---

## 11. Debugging Tips

### Similarity Debugging

When a similarity check fails, print the intermediate stages:

```python
# Ground both inputs
hv_a = ui.encode("The cat sat on the mat")
hv_b = ui.encode("A cat sitting on a mat")

# Check overall similarity
print(f"Overall: {hv_a.cosine_similarity(hv_b):.4f}")

# Check segment-level similarity (text only)
# Keyword segment: bits 0–5119
kw_a = hv_a.bits[:5120]
kw_b = hv_b.bits[:5120]
kw_sim = 1.0 - np.count_nonzero(np.bitwise_xor(kw_a, kw_b)) / 5120
print(f"Keyword seg: {kw_sim:.4f}")
```

### Module-Level Tracing

The `CognitiveState.trace` dict captures decision metadata:

```python
state = engine.decide(raw_input, task_tag="snake")
print(state.trace)  # Shows which module won, what proposals were made, etc.
```

### Episode Memory Inspection

```python
mem = EpisodicMemory(config)
# Store some episodes...

stats = mem.get_stats()
print(f"Total episodes: {stats['recent_count']}")
print(f"LSH tables: {stats.get('lsh_tables', 'N/A')}")

# Check what a query returns
results = mem.recall_similar(query_hv, k=10)
for ep in results:
    print(f"  sim={ep.similarity:.3f} action={ep.action} reward={ep.reward}")
```

### Causal Graph Inspection

```python
from causal_reasoning import CausalDiscovery

discovery = CausalDiscovery()
# Feed observations...

graph = discovery.get_graph()
for link in graph.links:
    print(f"  {link.cause} → {link.effect} (delta_p={link.strength:.3f})")
```

---

## 12. Performance Considerations

### Memory Budget

| Component | Memory | Notes |
|---|---|---|
| One HyperVector | 1.25 KB | 10240 bits = 1280 bytes |
| Codebook (10K entries) | ~12.5 MB | LRU-evicted beyond limit |
| Episode store (10K) | ~15 MB | HV + metadata per episode |
| LSH index (8 tables) | ~2 MB | Hash buckets + pointers |
| World model (MLP) | ~33 KB | Only learnable weights in system |

### CPU Hotspots

1. **Text grounding** (~5 ms): 4-component architecture with n-gram loops
2. **Episode storage**: LSH index update is $O(T)$ per table where $T$ = number of tables
3. **Similarity search**: Hamming distance over 10240 bits is the inner loop — numpy vectorises this well
4. **Bundle operation**: Majority vote over $n$ vectors is $O(n \times D)$

### Optimisation Opportunities

| Opportunity | Effort | Speedup |
|---|---|---|
| Compile to Rust VSA backend | Medium | 5–10× on HV operations |
| Batch grounding (multiple inputs per call) | Low | Amortises Python overhead |
| Pre-compute codebook HVs at startup | Low | Eliminates SHA-256 on repeated categories |
| SIMD Hamming distance | Medium | 2–4× on similarity |

---

## 13. Extension Points

### Adding a New Sensor Modality

1. Add encoder method in `universal_input.py`:
   ```python
   def _encode_lidar(self, scan: np.ndarray) -> hv.HyperVector:
       """Encode 360-degree lidar scan into HV."""
       # Quantise ranges → thermometer HVs → bind with angle roles
   ```

2. Register in `encode()` dispatcher:
   ```python
   if isinstance(data, np.ndarray) and domain == "lidar":
       return self._encode_lidar(data)
   ```

### Adding a New Game Environment

1. Create verifier in `grounding_verifier.py`:
   ```python
   def create_chess_verifier() -> GroundingVerifier:
       """Create a grounding verifier for chess states."""
   ```

2. Add causal graph in `causal_reasoning.py`:
   ```python
   def create_chess_causal_graph() -> CausalGraph:
       """Chess-specific causal relations (takes, checks, etc.)."""
   ```

3. Register task in `CognitiveEngine`.

### Adding New Emotion Dimensions

Edit `emotion_system.py`:

```python
# In EmotionSystem.__init__()
self.drives["surprise"] = 0.5
self._keyword_emotions["astonishing"] = ("surprise", 0.9)
```

### Adding New Causal Reasoning Modes

Extend `CausalReasoner`:

```python
def abductive_reason(self, observation: str) -> List[str]:
    """Find most probable causes for an observation (abduction)."""
    # Backward chain + rank by Delta-P strength
```

---

## 14. Git Workflow

### Branches

| Branch | Purpose |
|---|---|
| `main` | Stable release |
| `NSCK_V2` | Active development (current) |
| `feature/*` | New module or capability |
| `fix/*` | Bug fixes |
| `docs/*` | Documentation changes |

### Commit Messages

Follow the format:

```
[module] Short description

- Detail 1
- Detail 2
```

Examples:

```
[universal_input] Add phrase-structure encoding component

- 4th segment: compositional parse tree via role-filler binding
- Updated segment weights to 50/15/15/20
- Added heuristic POS tagger for NP/VP/PP chunking

[tests] Add enhanced text grounding tests (#67-#69)

- Similar texts closer than unrelated (0.956 vs 0.564)
- Single word fallback works
- Word order sensitivity verified
```

### Pre-Commit Checklist

- [ ] `PYTHONPATH=nsck-demo/python python nsck_capability_test.py` → 85/85 (or more)
- [ ] No new `ModuleNotFoundError` in imports
- [ ] All new public methods have docstrings and type annotations
- [ ] Updated relevant docs (MODULE_REFERENCE, TESTING, ARCHITECTURE)
- [ ] No hardcoded file paths or absolute paths

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) for system design, [VSA_THEORY.md](VSA_THEORY.md) for mathematical foundations, [MODULE_REFERENCE.md](MODULE_REFERENCE.md) for API reference, [TESTING.md](TESTING.md) for test methodology.*
