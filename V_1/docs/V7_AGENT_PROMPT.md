# NGCN v7 AI Agent Implementation Prompt

## Mission
Implement NGCN v7: A Data-Oriented Neuro-Symbolic cognitive architecture using Rustworkx, sparse matrices, and local LLMs.

---

## Critical Constraints

### 1. Data-Oriented Design (MANDATORY)
```
❌ WRONG (v6 style):
class Node:
    def __init__(self):
        self.energy = 0.0  # Object attribute
        
✅ CORRECT (v7 style):
class CognitiveState:
    activations = np.zeros(N, dtype=np.float32)  # Contiguous array
```

### 2. Integer Indexing (MANDATORY)
Rustworkx uses integers only. Maintain perfect sync:
```python
idx = graph.add_node("dog")  # Returns int
registry.register("dog", idx)
activations[idx] = 0.5  # Use idx for arrays
```

### 3. Deletion Safety (CRITICAL)
When removing nodes, IMMEDIATELY zero state arrays:
```python
def remove(label):
    idx = registry.get_index(label)
    graph.remove_node(idx)
    activations[idx] = 0.0  # MUST DO THIS
    thresholds[idx] = DEFAULT
    registry.unregister(idx)
```

---

## File Implementation Order

### Phase 1: Create Package Structure
```
d:/NGCN/ncgn_v7/
├── __init__.py
├── config.py
├── topology.py      # IMPLEMENT FIRST
├── state.py         # IMPLEMENT SECOND
├── engine.py        # IMPLEMENT THIRD
├── learner.py
├── embeddings.py
├── reasoner.py
├── brain.py         # IMPLEMENT LAST
└── schemas/
    ├── __init__.py
    └── cognitive.py
```

### Phase 2: Tests
```
d:/NGCN/tests/test_v7/
├── __init__.py
├── test_topology.py
├── test_state.py
├── test_engine.py
└── test_brain.py
```

---

## Module Specifications

### topology.py
**Classes**: `IndexRegistry`, `GraphTopology`
**Dependencies**: `rustworkx`, `threading`, `numpy`

Key methods:
- `GraphTopology.add_concept(label) -> int`
- `GraphTopology.remove_concept(label) -> int`
- `GraphTopology.add_connection(src, tgt, weight)`
- `GraphTopology.get_adjacency_data() -> (sources, targets, weights)`
- Property: `is_dirty` for matrix cache invalidation

### state.py
**Classes**: `CognitiveState`
**Dependencies**: `numpy`, `scipy.sparse`

Arrays (float32):
- `activations[N]` - membrane potential
- `thresholds[N]` - firing threshold
- `refractory_counters[N]` - cooldown

Methods:
- `ensure_capacity(idx)` - double when needed
- `zero_index(idx)` - clear deleted node
- `synchronize_matrix(topology)` - rebuild CSR if dirty

### engine.py
**Classes**: `PropagationEngine`
**Dependencies**: `numpy`, `scipy.sparse`

Core equation (vectorized):
```
A_new = sigmoid((A * (1-decay) + W.dot(A) * flow + I_ext) / (1 + beta * sum(A)))
A_new = A_new * (refractory == 0)  # Mask
```

Methods:
- `inject(idx, energy)` - queue external input
- `propagate(steps) -> Dict[label, energy]` - run dynamics
- `get_top_k_active(k) -> Dict` - highest activations

### learner.py
**Classes**: `HebbianLearner`

3-Factor rule: `ΔW = η × modulator × pre × post`

Methods:
- `apply_reward(modulator: float) -> int` - update weights
- `propose_new_edges(threshold) -> List[(src, tgt, strength)]`

### embeddings.py
**Classes**: `SemanticLayer`
**Dependencies**: `sentence-transformers`

Methods:
- `encode(texts) -> np.ndarray`
- `compute_semantic_weight(a, b) -> float` (cosine mapped to 0-1)
- `find_nearest(query, candidates, k) -> List[(label, sim)]`

### reasoner.py
**Classes**: `LLMReasoner`
**Dependencies**: `llama-cpp-python`, `instructor`, `pydantic`

Methods:
- `extract_knowledge(input, context) -> KnowledgeGraphUpdate`
- `answer_query(question, context) -> QueryResponse`

### schemas/cognitive.py
```python
class ConceptNode(BaseModel):
    label: str
    properties: Dict[str, Any] = {}

class RelationEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    weight: float = Field(ge=0.0, le=1.0)

class KnowledgeGraphUpdate(BaseModel):
    reasoning: str  # FIRST - Chain of Thought
    new_nodes: List[ConceptNode] = []
    new_edges: List[RelationEdge] = []

class QueryResponse(BaseModel):
    reasoning: str  # FIRST
    answer: str
    confidence: float = 0.5
```

### brain.py
**Classes**: `Brain`
Coordinates all components.

Methods:
- `add_concept(label, energy) -> int`
- `connect(src, tgt, weight=None)` - uses semantic if None
- `think(steps) -> Dict` - System 1 propagation
- `process_input(text) -> Dict` - full cognitive cycle
- `learn(reward) -> int` - apply 3-Factor

---

## Testing Requirements

Each test file must verify:

### test_topology.py
1. Add/remove nodes maintain index sync
2. Deleted indices get reused
3. Thread safety of registry
4. Edge weights preserved

### test_state.py
1. Array resizing works
2. `zero_index` clears all arrays
3. CSR matrix matches topology

### test_engine.py
1. Energy decays without input
2. Propagation spreads activation
3. Refractory period prevents firing
4. Seizure damping activates at threshold

### test_brain.py
1. Full cycle: input → propagate → output
2. Semantic entry points work
3. Learning modifies weights
4. (Optional) LLM integration

---

## Dos and Don'ts Summary

| Do | Don't |
|----|-------|
| Use `np.float32` | Use `np.float64` |
| Pre-allocate with capacity buffer | Resize every add |
| Cache CSR, check `is_dirty` | Rebuild matrix every tick |
| Lock registry during writes | Concurrent writes |
| Zero arrays on node delete | Leave stale data |
| Use Pydantic validators | Trust raw LLM output |

---

## Run After Implementation

```powershell
cd d:\NGCN
pytest tests/test_v7/ -v
python -c "
from ncgn_v7.brain import Brain
b = Brain()
b.add_concept('test', 0.5)
print(b.think(5))
"
```
