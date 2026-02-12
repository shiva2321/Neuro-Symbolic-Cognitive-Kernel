# Module Reference

> Complete API reference for all 22 modules in the NSCK cognitive architecture.  
> Every public class, method, parameter, and return type is documented.

---

## Table of Contents

| # | Module | Purpose | Lines |
|---|---|---|---|
| 1 | [`hypervec_py`](#1-hypervec_pypy) | VSA primitives | 84 |
| 2 | [`hypervec_shim`](#2-hypervec_shimpy) | Rust/Python compat layer | 134 |
| 3 | [`config`](#3-configpy) | Central configuration | 73 |
| 4 | [`universal_input`](#4-universal_inputpy) | Multi-type grounding → HV | 714 |
| 5 | [`language_module`](#5-language_modulepy) | Rule-based NLU pipeline | 456 |
| 6 | [`cognitive_engine`](#6-cognitive_enginepy) | Main orchestrator | 1 156 |
| 7 | [`episodic_memory`](#7-episodic_memorypy) | LSH-indexed episode storage | 356 |
| 8 | [`semantic_memory`](#8-semantic_memorypy) | Knowledge graph | 129 |
| 9 | [`causal_reasoning`](#9-causal_reasoningpy) | Delta-P discovery + graphs | 886 |
| 10 | [`world_model`](#10-world_modelpy) | VSA + numeric dynamics | 506 |
| 11 | [`emotion_system`](#11-emotion_systempy) | Circumplex affect model | 417 |
| 12 | [`theory_of_mind`](#12-theory_of_mindpy) | Sally-Anne + recursive belief | 312 |
| 13 | [`planner`](#13-plannerpy) | STRIPS BFS planner | 243 |
| 14 | [`multimodal_processor`](#14-multimodal_processorpy) | Classical CV/DSP | 756 |
| 15 | [`analogy`](#15-analogypy) | Cross-domain transfer | 517 |
| 16 | [`rule_learner`](#16-rule_learnerpy) | ILP-style induction | 418 |
| 17 | [`nlg`](#17-nlgpy) | Template + Markov NLG | 404 |
| 18 | [`curiosity`](#18-curiositypy) | Novelty + count-based | 335 |
| 19 | [`global_workspace`](#19-global_workspacepy) | GWT competition | 309 |
| 20 | [`brain_fusion`](#20-brain_fusionpy) | Multi-task knowledge | 480 |
| 21 | [`self_model`](#21-self_modelpy) | Metacognition | 255 |
| 22 | [`persistence`](#22-persistencepy) | SQLite storage | 424 |
| 23 | [`grounding_verifier`](#23-grounding_verifierpy) | Predicate verification | 414 |

All files are located in `nsck-demo/python/`.

---

## 1. `hypervec_py.py`

> Core VSA implementation: 10 240-bit binary hypervectors.

### Constants

```python
DIMENSION = 10240   # Number of bits per hypervector
```

### Class: `HyperVectorPy`

The fundamental data type. Every HV in the system is an instance of this class.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, seed=None)` | Create a random HV; if `seed` given, deterministic via `numpy.random.default_rng(seed)` |
| `from_bits` | `@classmethod (cls, bits: np.ndarray) → HyperVectorPy` | Wrap an existing int8 numpy array as an HV |
| `zero` | `@staticmethod () → HyperVectorPy` | Create the zero vector (all 0s) |
| `xor` | `(self, other) → HyperVectorPy` | Bitwise XOR (binding). Self-inverse: `a.xor(b).xor(b) == a` |
| `bundle` | `(self, other) → HyperVectorPy` | Majority-vote for 2 vectors. **Non-deterministic** (random tiebreak). See `_deterministic_bundle()` in `universal_input.py` |
| `similarity` | `(self, other) → float` | Normalised Hamming similarity: `1 - hamming_distance / 10240`. Range [0, 1] |
| `permute` | `(self, shift: int) → HyperVectorPy` | Circular bit rotation. `shift > 0` rotates left |
| `permute_inverse` | `(self, shift: int) → HyperVectorPy` | Inverse rotation: `permute(-shift)` |

**Storage format:** `self.bits` — `numpy.int8` array of length 10 240, values ∈ {0, 1}.

**Alias:** `HyperVector = HyperVectorPy` (exported for compatibility).

---

## 2. `hypervec_shim.py`

> Compatibility layer: tries Rust extension first, falls back to Python.

### Behaviour

```python
import hypervec_shim as hvs
hv = hvs.HyperVector(seed=42)     # Uses Rust if available, Python otherwise
```

### Methods Added by Shim

These methods are monkey-patched onto whichever backend is active:

| Method | Signature | Description |
|---|---|---|
| `weighted_bundle` | `(self, other, weight: float, seed=None) → HV` | Bundle with asymmetric weighting. `weight=1.0` → all self, `weight=0.0` → all other |
| `lsh_hash` | `(self, seed: int, n_bits: int) → int` | Locality-sensitive hash for bucket assignment. Returns integer with `n_bits` bits |
| `permute` | `(self, shift: int) → HV` | Circular bit rotation (added if Rust backend lacks it) |
| `permute_inverse` | `(self, shift: int) → HV` | Inverse rotation |

### Variable: `_USE_RUST`

`True` if the compiled Rust extension loaded successfully, `False` for Python fallback.

---

## 3. `config.py`

> Central configuration dataclass.

### Class: `NSCKConfig`

```python
@dataclass
class NSCKConfig:
    device: str = "cpu"                # "cpu" or "cuda" (for optional torch)
    learning_rate: float = 1e-3        # World model SGD learning rate
    beta: float = 0.5                  # Reserved (LIF neuron decay)
    sleep_epochs: int = 5              # Dreaming consolidation epochs
    replay_batch_size: int = 32        # Episodes per replay batch
    save_interval: float = 60.0        # Persistence flush interval (seconds)
    vsa_strength: float = 5.0          # Reserved
    confidence_threshold: float = 0.6  # Entropy threshold for VSA rescue
    novelty_threshold: float = 0.5     # Curiosity exploration threshold
    grid_size: int = 10                # Game grid dimensions
    adversarial_rate: float = 0.05     # Adversarial injection rate
    enable_snn: bool = True            # Enable SNN proposals (reserved)
    enable_vsa: bool = True            # Enable VSA subsystem
    enable_sleep: bool = True          # Enable dreaming/consolidation
    model_path: str = "snn_task_aware.pth"  # Reserved
    persistence_db: str = "nsck_brain.db"   # SQLite database path
    episode_capacity: int = 10000      # Max total episodes stored
    memory_capacity: int = 2500        # Recent memory capacity
    min_rule_support: int = 5          # Min observations before rule induction
    min_rule_confidence: float = 0.7   # Min success rate for rule promotion
    min_success_rate: float = 0.6      # Global success rate threshold
```

| Factory | Signature | Description |
|---|---|---|
| `from_env` | `@classmethod () → NSCKConfig` | Create config from environment variables with defaults |

**Global instance:** `DEFAULT_CONFIG = NSCKConfig()`

---

## 4. `universal_input.py`

> Maps any Python data type to a 10 240-bit HV in a similarity-preserving manner.

### Module-Level Functions

| Function | Signature | Description |
|---|---|---|
| `_stable_seed` | `(label: str) → int` | Deterministic 64-bit seed from SHA-256 hash of label string |
| `_deterministic_bundle` | `(hvs: list) → HyperVector` | Majority-vote bundle with first-vector tiebreaking. Critical for reproducibility |
| `_role_hv` | `(role_name: str) → HyperVector` | Get a deterministic role HV for syntactic functions (SUBJECT, PREDICATE, OBJECT, PP) |
| `_pos_tag_simple` | `(word: str) → str` | Heuristic POS tagger → DET/PREP/ADJ/VERB/NOUN/AUX/CONJ/ADV/PRON/UNK |
| `_chunk_phrases` | `(words: list) → dict` | Bottom-up chunker → `{subject_np, verb, object_np, prep_phrases}` |
| `_parse_np` | `(words, tags, start) → tuple` | Parse a Noun Phrase starting at position `start` |
| `_encode_np` | `(np_dict: dict) → HyperVector` | Encode NP using role-filler binding (det + adj + head×3 weight) |
| `_build_phrase_structure_hv` | `(words: list) → HyperVector` | Full pipeline: POS tag → chunk → role-bind → bundle |

### Constants

```python
DIMENSION = 10240
DEFAULT_THERMOMETER_BINS = 100    # Number of quantisation bins for scalars
MAX_CODEBOOK_SIZE = 10_000        # LRU eviction threshold for category codebook
_STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", ...}  # ~30 words
```

### Class: `UniversalInput`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, n_bins=100, max_codebook=10000)` | Initialise grounding with thermometer bins and LRU codebook |
| `ground` | `(self, data, domain="default", min_val=0.0, max_val=1.0) → HV` | Auto-detect type and dispatch: str→`ground_text`, float/int→`ground_scalar`, dict→`ground_dict`, list→`ground_sequence` |
| `ground_scalar` | `(self, value, min_val=0.0, max_val=1.0, domain="default") → HV` | Thermometer encoding: quantise → 7-bin window → deterministic bundle. Preserves scalar similarity |
| `ground_category` | `(self, label, domain="default") → HV` | Deterministic random HV from label hash. LRU eviction at `max_codebook` |
| `ground_text` | `(self, text, domain="default", ngram_size=3) → HV` | 4-component segment concatenation: keyword (50%) + n-gram (15%) + word-order (15%) + phrase-structure (20%) |
| `ground_dict` | `(self, data, domain="default") → HV` | Role-filler binding: each (key, value) bound with XOR, all pairs bundled |
| `ground_sequence` | `(self, data, domain="default", min_val=0.0, max_val=1.0) → HV` | Positional permutation: element HVs permuted by index, then bundled |
| `get_stats` | `(self) → dict` | Return grounding statistics: counts per type, codebook sizes |

---

## 5. `language_module.py`

> Natural language understanding (NLU) and generation (NLG) interface.  
> Always operates in MOCK mode (no LLM loaded).

### Class: `LanguageModule`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, model_path=DEFAULT_MODEL_PATH)` | Attempt to load LLM; falls back to mock mode (always in practice) |
| `understand` | `(self, text: str) → dict` | Parse text → `{intent, entities, relation, deps, frames, clauses, coref, grounded_hv}` |
| `generate` | `(self, intent_data: dict, context_hv=None) → str` | Convert intent dict to natural language response |

### NLU Pipeline (`_mock_understand`)

The rule-based NLU pipeline processes text through 6 stages:

```
Input: "The cat chased the mouse because it was hungry"
   │
   ├── 1. Tokenise → ["the", "cat", "chased", "the", "mouse", ...]
   ├── 2. POS Tag  → [DET, NOUN, VERB, DET, NOUN, CONJ, PRON, AUX, ADJ]
   ├── 3. Chunk    → NP[the cat] VP[chased] NP[the mouse] ...
   ├── 4. Clause Segmentation → main clause + subordinate (because ...)
   ├── 5. Semantic Frame Extraction:
   │      Frame: {
   │        agent: "cat",
   │        patient: "mouse",
   │        action: "chased",
   │        cause: "it was hungry"
   │      }
   ├── 6. Coreference Hints → "it" → "mouse" (most recent NP)
   │
   └── Output: {
         intent: "DESCRIBE_RELATION",
         entities: ["cat", "chased", "mouse"],
         relation: "cat — chased → mouse",
         frames: [{agent:"cat", patient:"mouse", action:"chased", cause:"hungry"}],
         clauses: [{type:"main", text:"the cat chased the mouse"},
                   {type:"subordinate", marker:"because", text:"it was hungry"}],
         coref: {"it": "mouse"},
         grounded_hv: <HyperVector>
       }
```

### Frame Roles Extracted

| Role | Detection Method | Example |
|---|---|---|
| Agent | First NP before main verb | "The **cat** chased..." |
| Patient | First NP after main verb | "...chased **the mouse**" |
| Instrument | NP after "with" / "using" | "cut **with a knife**" |
| Location | NP after "in" / "on" / "at" / "near" / "inside" | "sat **on the mat**" |
| Source | NP after "from" | "came **from the garden**" |
| Destination | NP after "to" / "toward" / "towards" | "walked **to the park**" |
| Time | NP after "before" / "after" / "during" / "when" | "ran **before dawn**" |
| Cause | Clause after "because" / "since" | "fell **because it was slippery**" |
| Condition | Clause after "if" / "unless" / "when" / "although" | "will go **if it rains**" |

---

## 6. `cognitive_engine.py`

> Central orchestrator: initialises all modules, runs the decide→learn loop.

### Dataclasses

```python
@dataclass
class Proposal:
    module: str       # e.g. "RULES", "PLANNER", "CURIOSITY"
    action: str       # e.g. "ACTION_UP"
    content: str      # human-readable explanation
    salience: float   # proposal strength [0, 1]

@dataclass
class CognitiveState:
    task_tag: str
    situation_hv: HyperVector
    active_predicates: List[str]
    chosen_action: str = "ACTION_STAY"
    confidence: float = 0.5
    self_confidence: float = 0.5
    exploration_mode: bool = False
    explanation: Optional[Explanation] = None
    imagined_reward: float = 0.0
    emotion: str = "neutral"
    trace: Dict[str, Any] = field(default_factory=dict)
```

### Class: `CognitiveEngine`

#### Core Cycle

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, config=None, persistence_path=None)` | Initialise all 19+ sub-modules, register default tasks (snake, maze, pong) |
| `decide` | `(self, state, task_tag, metacognition_result=None) → CognitiveState` | One cognitive cycle: perception → coalition formation → GWT competition → mental rehearsal → action selection |
| `learn` | `(self, state, action, reward, task_tag, outcome="neutral", next_state=None, image=None)` | Post-action learning: update homeostasis, emotions, episodic memory, rules, causal discovery, world model, self-model, brain fusion |

#### Knowledge & Transfer

| Method | Signature | Description |
|---|---|---|
| `transfer` | `(self, source_task, target_task, state, active_predicates) → Optional[str]` | Zero-shot action via analogy: lift predicates → find mapping → transfer rules |
| `dream` | `(self, num_samples=10, task_tag="snake") → List[dict]` | Generative dreaming: replay episodes through world model for consolidation |
| `generate_hypothetical_lessons` | `(self, num_anchors=5, task_tag="snake") → List[dict]` | Generate synthetic extreme-outcome scenarios for training |
| `imagine_rollout` | `(self, initial_hv, action_sequence, task_tag, gamma=0.9) → float` | Simulate action sequence, return discounted cumulative reward |

#### Explanation

| Method | Signature | Description |
|---|---|---|
| `explain` | `(self, query_type="action") → str` | Explain the most recent decision |
| `why_not` | `(self, rejected_action) → str` | Explain why a specific action was not chosen |
| `counterfactual` | `(self, alternative_action) → str` | "What if I had done X instead?" |

#### Management

| Method | Signature | Description |
|---|---|---|
| `register_task` | `(self, task_tag, verifier, causal_graph=None)` | Register a new task with its grounding verifier and causal graph |
| `get_concept_hv` | `(self, concept: str) → HV` | Get or create a deterministic HV for any string concept |
| `get_allowed_actions` | `(self, task_tag) → List[str]` | Get valid actions for a task |
| `get_stats` | `(self) → dict` | Comprehensive statistics across all modules |
| `get_workspace_telemetry` | `(self) → dict` | Real-time GWT competition state for dashboards |
| `get_causal_telemetry` | `(self, task_tag) → dict` | Causal graph state for dashboards |
| `set_mission_goal` | `(self, goal_type, threshold)` | Set high-level mission goal |
| `export_traces` | `(self, filename)` | Export all decision traces to JSON file |
| `process_dialogue` | `(self, user_text, teach_mode=False) → str` | Process natural language input through dialogue manager |

#### Factory

```python
def create_cognitive_engine(persistence_path=None) → CognitiveEngine
```

---

## 7. `episodic_memory.py`

> Three-tier episodic memory with LSH indexing.

### Dataclass: `LiveEpisode`

```python
@dataclass
class LiveEpisode:
    timestamp: float
    task_tag: str
    situation_hv: HyperVector
    state: Dict[str, Any]
    action: str
    outcome: str              # "success", "failure", "neutral"
    reward: float
    emotion: str = "neutral"
    tom_beliefs: Optional[Dict] = None
    image: Optional[np.ndarray] = None
    impact_score: float = 0.0
```

### Class: `EpisodicMemory`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, store=None, recent_capacity=1000, total_capacity=10000, consolidation_threshold=500)` | Initialise with BrainStore backend and three-tier architecture |
| `reset` | `(self)` | Clear all episodes and reinitialise |
| `record` | `(self, episode: LiveEpisode)` | Store episode, update LSH index, trigger consolidation if needed |
| `recall_similar` | `(self, query_hv, task_tag, k=5) → List[LiveEpisode]` | LSH-accelerated similarity search. Returns top-k most similar episodes |
| `recall_recent` | `(self, task_tag, n=10) → List[LiveEpisode]` | Most recent episodes for a task |
| `retrieve_salient` | `(self, task_tag, limit=100) → List[LiveEpisode]` | High-impact episodes sorted by |reward| + novelty |
| `recall_by_outcome` | `(self, task_tag, outcome="success", n=10) → List[LiveEpisode]` | Filter by outcome type |
| `recall_by_reward` | `(self, task_tag, min_reward=0.0, n=10) → List[LiveEpisode]` | Filter by minimum reward, sorted descending |
| `sample` | `(self, task_tag, n=32) → List[LiveEpisode]` | Random sample for dreaming/replay |
| `create_situation_hv` | `(self, state, task_tag, active_predicates) → HV` | Bundle active predicate HVs into a situation vector |
| `get_statistics` | `(self, task_tag) → dict` | Memory stats: counts, rates, avg reward, LSH bucket count |

---

## 8. `semantic_memory.py`

> Graph-based concept store with spreading activation.

### Class: `SemanticMemory`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Initialise NetworkX DiGraph + HV concept map |
| `reset` | `(self)` | Clear all concepts and relations |
| `add_concept` | `(self, concept_name, properties, hv_override=None)` | Add concept node with property dict and optional custom HV |
| `add_relation` | `(self, concept1, relation, concept2)` | Add directed edge with relation label |
| `query` | `(self, query_hv, k=5) → List[Tuple[str, float]]` | Find k concepts closest to query HV by Hamming similarity |
| `spread_activation` | `(self, start_concepts, steps=3, decay=0.7) → Dict[str, float]` | BFS-like activation spreading with multiplicative decay |
| `extract_schema` | `(self, concept) → dict` | Extract structural relations: is_a, parts, causes, caused_by |

---

## 9. `causal_reasoning.py`

> Causal discovery (Delta-P), causal graph, counterfactual reasoning.

### Enums & Dataclasses

```python
class CausalRelation(Enum):
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    REQUIRES = "requires"

@dataclass
class CausalLink:
    cause: str
    effect: str
    relation: CausalRelation
    strength: float = 1.0       # ΔP value [0, 1]
    context: Optional[str] = None
    evidence_count: int = 0

@dataclass
class CausalChain:
    links: List[CausalLink]
    start: str
    end: str
    total_strength: float = 1.0  # Product of link strengths
    
    def describe(self) → str     # "A → B → C (strength: 0.72)"

@dataclass
class CounterfactualResult:
    query: str
    original_outcome: str
    counterfactual_outcome: str
    affected_states: List[str]
    confidence: float
    explanation: str
```

### Class: `CausalDiscovery`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Initialise co-occurrence counting structures |
| `reset` | `(self)` | Clear all causal statistics |
| `observe` | `(self, context, causes: List[str], effects: List[str])` | Record one timestep of cause-effect observations |
| `induce_graph` | `(self, context, min_confidence=0.5, min_evidence=5) → CausalGraph` | Compute ΔP for all pairs, create graph from significant links |
| `incremental_update` | `(self, context, graph, min_confidence=0.3, min_evidence=2)` | Add new links to existing graph with lower thresholds |
| `get_hypotheses` | `(self, context, max_confidence=0.5, min_evidence=2) → List[Tuple]` | Return low-confidence pairs worth testing (curiosity fuel) |
| `update_strengths` | `(self, context, graph)` | Refresh ΔP values for existing graph links |

### Class: `CausalGraph`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Initialise with forward/backward link indices |
| `add_link` | `(self, link: CausalLink)` | Add a causal relationship |
| `add_causes` | `(self, cause, effect, strength=1.0, context=None)` | Shorthand for CAUSES relation |
| `add_prevents` | `(self, preventer, prevented, strength=1.0, context=None)` | Shorthand for PREVENTS relation |
| `forward_chain` | `(self, start, max_depth=5, context=None) → List[CausalChain]` | DFS: find all effects reachable from cause |
| `backward_chain` | `(self, end, max_depth=5, context=None) → List[CausalChain]` | DFS: find all causes leading to effect |
| `find_path` | `(self, start, end, max_depth=5, context=None) → Optional[CausalChain]` | Find connecting path between two nodes |
| `get_immediate_effects` | `(self, cause) → List[str]` | Direct effects only (depth 1) |
| `get_immediate_causes` | `(self, effect) → List[str]` | Direct causes only (depth 1) |
| `prune_redundant` | `(self, context=None)` | Transitive reduction: remove A→C if A→B→C exists |

### Class: `CausalReasoner`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, graph, simulator=None)` | Attach to a CausalGraph with optional simulator function |
| `predict_effects` | `(self, state, action, task_tag) → List[str]` | Forward-chain from action to predict outcomes |
| `explain_why` | `(self, effect, state, task_tag, recent_actions=None) → List[CausalChain]` | Backward-chain to explain why an effect occurred |
| `counterfactual` | `(self, actual_action, alternative_action, state, task_tag) → CounterfactualResult` | Full "what if" analysis |
| `why_not` | `(self, rejected_action, chosen_action, state, task_tag) → str` | Explain why an action was not taken |

### Factory Functions

```python
create_snake_causal_graph() → CausalGraph   # Pre-populated for snake game
create_pong_causal_graph() → CausalGraph     # Pre-populated for pong
create_maze_causal_graph() → CausalGraph     # Pre-populated for maze
```

---

## 10. `world_model.py`

> Hybrid dynamics predictor: VSA symbolic memory (primary) + numeric ensemble (secondary).

### Class: `VSATransitionMemory`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, hv_dim=10240, capacity=2000)` | Ring buffer for (state⊗action) → (next_state, reward) |
| `size` | `@property → int` | Current number of stored transitions |
| `store` | `(self, state_bits, action_bits, next_state_bits, reward)` | Record one transition |
| `predict` | `(self, state_bits, action_bits, k=3) → Optional[Tuple[ndarray, float, float]]` | k-NN weighted vote → (predicted_next_state, predicted_reward, max_similarity) |

### Class: `WorldModel`

| Constant | Value | Meaning |
|---|---|---|
| `VSA_TRUST_THRESHOLD` | 0.55 | Minimum similarity to trust VSA prediction alone |

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, hv_dim=10240)` | Init VSA memory + 3-predictor numeric ensemble |
| `is_ready` | `(self, task_tag) → bool` | True if > 100 training steps recorded |
| `update` | `(self, state, action_hv, next_hv, reward)` | Record transition in both VSA memory and numeric ensemble |
| `imagine` | `(self, state, action_hv) → Tuple[ndarray, float]` | Predict next state + reward. Uses VSA if sim ≥ 0.55, blends with numeric otherwise |
| `get_uncertainty` | `(self, state, action_hv) → float` | Ensemble disagreement (std of reward predictions across 3 predictors) |
| `sample_hypothetical_trajectories` | `(self, initial_hv, action_hvs, horizon=5, num_paths=10) → List[List[dict]]` | Monte Carlo rollouts for planning |

### Numeric Ensemble

Three predictors, each:
- Random projection: 10 240 → 128 dims (fixed)
- Architecture: 256 → 128 → 128 → 64 (MLP with ReLU) OR linear fallback (numpy)
- Loss: MSE(state) + 10 × MSE(reward)
- Training: Adam (torch) or SGD (numpy), lr = 0.001

---

## 11. `emotion_system.py`

> Circumplex model (Russell, 1980) + Plutchik's 8 basic emotions.

### Emotion Prototypes

| Emotion | Valence | Arousal |
|---|---|---|
| joy | +0.8 | +0.6 |
| trust | +0.6 | +0.3 |
| surprise | +0.1 | +0.9 |
| anticipation | +0.3 | +0.6 |
| fear | −0.8 | +0.9 |
| sadness | −0.7 | −0.3 |
| anger | −0.6 | +0.8 |
| disgust | −0.5 | +0.3 |
| neutral | 0.0 | 0.0 |

### Class: `EmotionSystem`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Initialise with neutral state, prototype blending, keyword lexicon |
| `reset` | `(self)` | Clear emotional state and history |
| `update_from_drives` | `(self, drives: dict, reward: float)` | Map reward → valence shift (±0.2), drives → arousal. Decay: ×0.95/step |
| `get_emotion_blend` | `(self) → Dict[str, float]` | Current emotion mix (emotion → weight, sums to 1.0) via inverse-distance to prototypes |
| `get_mood` | `(self, window=10) → dict` | Slow-moving average: `{avg_valence, avg_arousal, dominant_emotion, stability}` |
| `get_emotional_trajectory` | `(self) → List[dict]` | Full history (max 200 entries): `{name, valence, arousal, intensity}` per step |
| `recognize_emotion_from_text` | `(self, text) → str` | Keyword-based emotion detection with negation ("not happy" → sadness) and intensity modifiers ("very", "extremely") |
| `get_emotion_hypervector` | `(self) → HV` | VSA vector for current dominant emotion |
| `get_emotion_vector` | `(self, emotion_name) → HV` | VSA vector for any named emotion |
| `get_emotion_info` | `(self) → dict` | Telemetry: `{name, valence, arousal, intensity, blend}` |

---

## 12. `theory_of_mind.py`

> Sally-Anne false belief detection + level-2 recursive belief reasoning.

### Class: `MentalStateModel`

Represents one agent's inferred mental state:

```python
class MentalStateModel:
    agent_id: str
    beliefs: Dict[str, Any]     # What the agent believes about the world
    desires: List[str]          # What the agent wants
    intentions: List[str]       # What the agent plans to do
    position: Optional[str]     # Where the agent is (for observability)
```

### Class: `TheoryOfMind`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Initialise multi-agent mental state tracker |
| `get_or_create_model` | `(self, agent_id) → MentalStateModel` | Get or create a model for an agent |
| `update_agent_perspective` | `(self, agent_id, agent_loc, observable_world)` | Update beliefs based on what the agent can see from their location |
| `predict_action` | `(self, agent_id) → str` | BDI-based action prediction from beliefs + desires |
| `detect_false_belief` | `(self, agent_id, reality: dict) → List[str]` | Sally-Anne Test: compare beliefs vs reality, return mismatching keys |
| `recursive_belief` | `(self, observer, target, key, depth=2) → dict` | "I think A thinks B thinks..." (up to depth levels) |
| `update_observer_model_of_target` | `(self, observer, target, key, value)` | Record that observer knows target believes key=value |
| `get_agent_summary` | `(self, agent_id) → dict` | Human-readable mental model summary |
| `simulate_belief` | `(self, observer, target, key, observation_history=None) → dict` | Simulate target's belief evolution by replaying their observation history |
| `predict_action_from_simulation` | `(self, agent_id, world_state) → str` | Predict action using simulated subjective world model |

---

## 13. `planner.py`

> STRIPS-style BFS planner with learned operator support.

### Dataclasses

```python
@dataclass
class PlanStep:
    action: str
    state_snapshot: FrozenSet[str]

@dataclass  
class PlanNode:
    cost: int
    state: FrozenSet[str]
    history: List[str]
```

### Class: `STRIPSPlanner`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, operators=None)` | Init with optional pre-defined operators |
| `set_operators` | `(self, rules: list)` | Update operators from learned rules |
| `learn_operators_from_graph` | `(self, graph, context=None)` | Extract STRIPS operators (preconditions, add-effects, delete-effects) from CausalGraph |
| `plan` | `(self, initial_state: set, goal: set, max_depth=10) → Optional[List[str]]` | BFS search for goal-reaching action sequence |
| `simulate_sequence` | `(self, state: frozenset, actions: list) → frozenset` | Apply action sequence to state |
| `plan_hierarchical` | `(self, initial_state: set, goal: set, max_depth=50) → Optional[List[str]]` | Hierarchical planning with goal decomposition |
| `set_reasoner` | `(self, reasoner)` | Attach CausalReasoner for transition prediction |

---

## 14. `multimodal_processor.py`

> Classical computer vision and digital signal processing → HV.

### Dataclasses

```python
@dataclass
class MultimodalInput:
    text: Optional[str] = None
    image: Optional[np.ndarray] = None       # H×W or H×W×C
    audio: Optional[np.ndarray] = None       # 1D float array
    video: Optional[List[np.ndarray]] = None # List of frames
    structured: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedInput:
    fused_hv: HyperVector
    modality_results: List[ModalityResult]
    extracted_concepts: List[str]
    context_cues: Dict[str, Any]
    confidence: float = 1.0
```

### Class: `MultimodalProcessor`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, context_engine=None, semantic_memory=None)` | Init with optional context engine and semantic memory |
| `process` | `(self, inp: MultimodalInput) → ProcessedInput` | Full multimodal processing pipeline |
| `supported_modalities` | `@property → List[str]` | `["text", "image", "audio", "video", "structured"]` |

### Image Pipeline

```
Input: numpy array H×W or H×W×C
   │
   ├── HOG-lite: Sobel gradients → 4×4 grid × 8 orientations → 128-dim histogram
   ├── Color histogram: 8 bins per channel (or grayscale)
   ├── Edge density: fraction of pixels with gradient magnitude > threshold
   ├── LBP texture: 8-neighbour binary pattern → 4-quadrant histograms
   └── Spatial quadrants: mean/std per quadrant
   │
   └── Quantise → string → SHA-256 → seed → HV
```

### Audio Pipeline

```
Input: 1D numpy float array (raw samples)
   │
   ├── MFCC: FFT → mel filterbank (26 filters) → log → DCT → 13 coefficients
   ├── Spectral centroid: Σ(f × |X(f)|) / Σ|X(f)|
   ├── Spectral rolloff: frequency below which 85% of energy
   ├── Zero-crossing rate: sign changes per sample
   └── Energy bands: 4 frequency bands (0-500, 500-2k, 2k-8k, 8k+)
   │
   └── Quantise → string → SHA-256 → seed → HV
```

### Video Pipeline

```
Input: List of numpy frame arrays
   │
   ├── Subsample to 8 frames
   ├── Per-frame: run image pipeline → frame HV
   ├── Optical flow: block matching (8×8 blocks, ±4px search)
   ├── Motion direction histogram: 8 direction bins
   └── Temporal bundle: all frame HVs + motion HV
```

> **Known limitation:** SHA-256 hashing of quantised features destroys continuous similarity. Two similar images may produce unrelated HVs. See [Architecture — Known Issues](ARCHITECTURE.md#known-limitation-multimodal-hashing).

---

## 15. `analogy.py`

> Cross-domain knowledge transfer via abstract concept alignment.

### Default Abstract Concepts

| Concept | Snake Grounding | Maze Grounding | Pong Grounding |
|---|---|---|---|
| AGENT | SNAKE_HEAD | MAZE_PLAYER | PADDLE |
| TARGET | SNAKE_FOOD | MAZE_EXIT | BALL |
| DANGER | SNAKE_BODY, WALL | MAZE_WALL | MISS |
| TARGET_ABOVE | REL_ABOVE | REL_ABOVE | BALL_ABOVE |
| TARGET_BELOW | REL_BELOW | REL_BELOW | BALL_BELOW |
| MOVE_UP | ACTION_UP | ACTION_UP | ACTION_UP |
| MOVE_DOWN | ACTION_DOWN | ACTION_DOWN | ACTION_DOWN |

### Class: `AnalogyEngine`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Init with default abstract concepts for snake/maze/pong |
| `register_abstract` | `(self, name, description, groundings: dict)` | Register new abstract concept with domain-specific groundings |
| `lift_to_abstract` | `(self, local_concept, domain) → Optional[str]` | Map domain-specific concept to abstract level |
| `ground_to_domain` | `(self, abstract_concept, target_domain) → Optional[str]` | Map abstract concept to domain-specific form |
| `find_analogy` | `(self, source_domain, target_domain) → Analogy` | Build full analogical mapping between two domains |
| `transfer_rule` | `(self, rule_condition, rule_action, source_domain, target_domain) → Tuple[Set, str]` | Transfer a learned rule to a new domain |
| `zero_shot_action` | `(self, state, known_domain, new_domain, learned_rules, active_predicates) → Optional[str]` | Get action for unknown domain using transferred knowledge |
| `auto_discover_abstractions` | `(self, domain_a, domain_b, concept_hvs_a, concept_hvs_b, threshold=0.52) → List[ConceptMapping]` | Automatic cross-domain alignment via HV similarity + name similarity bonus |
| `get_all_abstractions` | `(self) → dict` | Summary of all abstract concepts and their groundings |
| `get_transfer_explanation` | `(self, source_domain, target_domain) → str` | Human-readable transfer explanation |

---

## 16. `rule_learner.py`

> Frequency-based ILP-style symbolic rule induction.

### Class: `RuleLearner`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, verifier, store=None, min_support=5, min_success_rate=0.7, max_rules_per_task=50)` | Init with grounding verifier and thresholds |
| `observe` | `(self, state, action, reward, task_tag, outcome="neutral")` | Record observation. Exact match + 60% overlap approximate matching |
| `induce_rules` | `(self, task_tag=None) → List[Rule]` | Promote candidates exceeding support and success rate thresholds |
| `validate_rules` | `(self, task_tag, validation_episodes) → Dict[int, float]` | Check rules against held-out data → accuracy per rule ID |
| `prune_rules` | `(self, task_tag, keep_top_n=None)` | Remove low-performing rules with tenure-aware thresholds |
| `get_rules` | `(self, task_tag) → List[Rule]` | Get all learned rules for a task |
| `get_applicable_rules` | `(self, active_preds, task_tag) → List[Tuple[Rule, float]]` | Find rules matching current predicates, sorted by score |
| `load_from_store` | `(self, task_tag=None)` | Load persisted rules from BrainStore |

### Rule Induction Formula

```
For each candidate (predicates, action):
    success_rate = successes / support
    IF support ≥ min_support AND success_rate ≥ min_success_rate:
        promote to Rule
        
Approximate matching (during observe()):
    IF overlap(observed_predicates, candidate_predicates) ≥ 0.6:
        candidate.support += 0.5
        IF reward > 0: candidate.successes += 0.5
```

---

## 17. `nlg.py`

> Template slot-filling + bigram Markov chain text generation.

### Class: `NLGEngine`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, templates=None)` | Init with 23 default templates + empty Markov chain |
| `generate` | `(self, category, slots: dict) → str` | Pick best template for category, fill {slot} placeholders |
| `narrate_rule` | `(self, condition, consequence, support=0) → str` | "When [condition], [consequence] (observed N times)" |
| `narrate_episode` | `(self, domain, action, outcome, reward) → str` | "In [domain], took [action] → [outcome] (reward: N)" |
| `narrate_emotion` | `(self, emotion, valence=0, arousal=0.5, intensity=0.5) → str` | "Currently feeling [emotion] (valence: V, arousal: A)" |
| `narrate_causal` | `(self, cause, effect, chain=None, counterfactual=None) → str` | "Because [cause], [effect] happened" |
| `narrate_analogy` | `(self, source_domain, target_domain, ...) → str` | "[source concept] in [source] is like [target concept] in [target]" |
| `narrate_reflection` | `(self, domain, trend, confidence, ...) → str` | Self-reflection text |
| `narrate_tom` | `(self, agent, belief, believed, reality) → str` | Theory of Mind narration |
| `narrate_prediction` | `(self, state, prediction, action, reason) → str` | Prediction explanation |
| `summarise_session` | `(self, stats: dict) → str` | Session summary from stats dict |
| `learn_corpus` | `(self, sentences: list)` | Train bigram language model incrementally |
| `generate_novel` | `(self, seed="", max_words=20, ...) → str` | Generate novel text via Markov chain |
| `is_generative` | `@property → bool` | Whether Markov model has been trained |
| `get_history` | `(self, n=10) → List[str]` | Last n generated texts |
| `add_template` | `(self, pattern, category, required_slots, priority=1.0)` | Register custom template at runtime |

---

## 18. `curiosity.py`

> Novelty detection + count-based exploration + learning progress tracking.

### Dataclass: `ExplorationDecision`

```python
@dataclass
class ExplorationDecision:
    should_explore: bool
    novelty_score: float        # [0, 1]: 0 = familiar, 1 = completely novel
    learning_progress: float    # Recent success rate - past success rate
    reason: str                 # Human-readable explanation
    testable_hypotheses: List[Tuple[str, str]]
```

### Class: `CuriosityModule`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, novelty_threshold=0.7, progress_window=100, stagnation_threshold=0.01)` | Init with VSA prototype memory and visit counting |
| `compute_novelty` | `(self, situation_hv, task_tag) → float` | `1 - max_similarity(situation, known_prototypes)`. Max 100 prototypes, FIFO eviction |
| `compute_learning_progress` | `(self, task_tag) → float` | Success rate of recent half minus past half of sliding window |
| `update_prototype` | `(self, situation_hv, task_tag)` | Add situation to prototypes if novel enough |
| `record_outcome` | `(self, task_tag, success: bool)` | Record success/failure for learning progress |
| `record_visit` | `(self, situation_hv, task_tag)` | Increment LSH-based visit counter |
| `get_visit_count` | `(self, situation_hv, task_tag) → int` | How many times this state was visited |
| `set_subgoals` | `(self, task_tag, subgoals: List[HV])` | Set active subgoals to bias exploration |
| `should_explore` | `(self, situation_hv, task_tag, confidence, hypotheses=None, active_predicates=None) → ExplorationDecision` | Multi-factor decision: novelty + confidence + stagnation + visit count + causal curiosity + subgoal proximity |
| `get_exploration_action` | `(self, available_actions, action_probs, explore_rate=0.3) → str` | Softmax-temperature exploration bias for action selection |
| `get_statistics` | `(self, task_tag) → dict` | Prototype count, learning progress, unique states, success rate |

### Exploration Decision Logic

```
should_explore = True IF:
    (high_novelty AND low_confidence)
    OR (learning_is_stagnant)
    OR (state_rarely_visited AND moderate_novelty)
    
Bonus: +0.1 novelty if testable causal hypotheses are active
Bonus: +0.1 novelty if near an active subgoal
```

---

## 19. `global_workspace.py`

> Global Workspace Theory (Baars, 1988) competition with mental rehearsal veto.

### Abstract Base Class: `WorkspaceModule`

```python
class WorkspaceModule(ABC):
    @abstractmethod
    def receive_broadcast(self, content: Any): ...
```

### Dataclass: `Coalition`

```python
@dataclass
class Coalition:
    source: str                     # "RULES", "PLANNER", "CURIOSITY", etc.
    content: Any                    # Action + metadata
    base_salience: float            # [0, 1]
    relevance: float = 0.0          # Context-dependent boost
    affect_match: float = 0.0       # Emotional congruence
    sender_confidence: float = 0.5  # Module self-assessment
    
    @property
    def activation(self) → float:
        return self.base_salience + self.relevance + self.affect_match + self.sender_confidence * 0.5
```

### Class: `GlobalWorkspace`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, attention_threshold=0.5)` | Init workspace with threshold and empty danger registry |
| `register_module` | `(self, name, module: WorkspaceModule)` | Register module for content broadcasts |
| `compete` | `(self, proposals: List[Coalition]) → Optional[Coalition]` | Rank by activation, threshold check, broadcast winner |
| `broadcast` | `(self, content)` | Send content to all registered modules |
| `register_danger` | `(self, danger_hv)` | Add state HV to danger registry (LRU, max 200) |
| `compete_with_rehearsal` | `(self, proposals, current_state_hv, world_model, get_action_hv_fn, n_cycles=None) → Optional[Coalition]` | Full competition with mental rehearsal: imagine → check danger similarity → veto if ≥ 0.75 |
| `get_status` | `(self) → dict` | Workspace state: winner, history length, danger count, recent vetoes |
| `get_recent_vetoes` | `(self, n=5) → List[dict]` | Recent veto events with timestamps and details |

---

## 20. `brain_fusion.py`

> Two-layer knowledge fusion: task-specific + global.

### Enum: `ConceptType`

```python
class ConceptType(Enum):
    ACTION, RELATION, OBJECT, STATE, GOAL, UNKNOWN
```

### Class: `TaskBrain`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, task_tag)` | Init per-task codebook and rule set |
| `add_concept` | `(self, name, hv, concept_type=None)` | Add concept with inferred type from naming convention |
| `add_rule` | `(self, condition: frozenset, consequence, strength=1.0, priority=0) → Rule` | Add a directional inference rule |
| `get_concept_context` | `(self, concept_name, top_k=5) → Optional[HV]` | Context signature: bundle of co-occurring concept HVs |

### Class: `FusedBrain`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Init global + task-specific two-layer system |
| `resolve_rules` | `(self, facts: set, task_tag, top_k=5) → List[QueryResult]` | **Logic channel**: 1-step rule inference from active facts. Score: `strength × layer_boost × priority_boost`, saturated via `x/(x+1)` |
| `query` | `(self, query_hv, task_tag=None, top_k=5) → List[QueryResult]` | **Similarity channel**: fuzzy lookup by HV Hamming similarity |
| `forward_chain_multi` | `(self, facts: set, max_steps=5) → Tuple[Set, List[Set]]` | Iterated rule firing to fixed point (max 5 steps). Only non-ACTION rules fire |

### Class: `BrainFusion`

| Constant | Value | Purpose |
|---|---|---|
| `HV_THRESHOLD` | 0.75 | Minimum HV similarity for concept alignment |
| `CTX_THRESHOLD` | 0.65 | Minimum context similarity for promotion |
| `STRICT_HV_THRESHOLD` | 0.92 | Strict mode HV threshold (safety-critical) |
| `STRICT_CTX_THRESHOLD` | 0.85 | Strict mode context threshold |

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Init with empty brain registry |
| `register_brain` | `(self, brain: TaskBrain)` | Register a task brain for fusion |
| `align_concepts` | `(self, brain_a, brain_b) → List[Tuple]` | Find matching concepts across brains with type consistency gate |
| `fuse` | `(self, strategy="tagged_conservative") → FusedBrain` | Fuse all brains: global primitives → promoted concepts → task-specific |

---

## 21. `self_model.py`

> Metacognition: calibrated confidence estimation, capability tracking, trend detection.

### Class: `SelfModel`

| Constant | Value |
|---|---|
| `RECENT_WINDOW_SIZE` | 20 |

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Init with identity HV, per-task stats, context stats, capability map |
| `reset` | `(self)` | Clear all metrics |
| `predict_success` | `(self, task_tag, state=None) → float` | Blended prediction: base_rate × (1−context_weight) + context_rate × context_weight. Context weight grows to 0.6 with evidence |
| `update` | `(self, task_tag, predicted_confidence, actual_success, action="UNKNOWN", reward=0.0, state=None)` | Record outcome: update stats, calibration error, capability scores, context performance |
| `update_confidence` | `(self, task_tag, confidence)` | Set current confidence level |
| `get_confidence` | `(self, task_tag) → float` | Get current confidence |
| `get_identity` | `(self) → HV` | Deterministic identity HV from hash("SELF_NSCK_V1") |
| `get_capability` | `(self, action) → float` | Proficiency for a specific action |
| `get_calibration_error` | `(self, task_tag) → float` | Expected Calibration Error: mean |predicted − actual| |
| `get_stats` | `(self, task_tag) → dict` | Raw task statistics |
| `get_context_performance` | `(self, task_tag) → dict` | Context-specific performance breakdown |
| `get_improvement_trend` | `(self, task_tag) → Optional[str]` | "improving" / "declining" / "stable" based on window comparison |

---

## 22. `persistence.py`

> SQLite persistence with WAL mode and write-behind buffering.

### Dataclasses

```python
@dataclass
class Rule:
    id: Optional[int]
    condition: frozenset       # Set of predicate strings
    consequence: str           # Action string
    priority: int = 1
    source: str = "bootstrap"  # "bootstrap", "learned", "transferred"
    task_tag: str = "global"
    scope: str = "task_local"
    support_count: int = 0
    success_rate: float = 0.0
    created_at: float = 0.0

@dataclass
class Concept:
    id: Optional[int]
    name: str
    hv_bytes: bytes            # Serialised HV bits
    concept_type: str
    task_tag: str = "global"
    created_at: float = 0.0
    access_count: int = 0

@dataclass
class Episode:
    id: Optional[int]
    timestamp: float
    task_tag: str
    situation_hv_bytes: bytes
    state_sketch: Dict[str, Any]
    action: str
    outcome: str
    reward: float
    impact_score: float = 0.0
```

### Class: `BrainStore`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, db_path="nsck_brain.db")` | Open SQLite with WAL mode, create tables, init write-behind buffer |
| `save_rule` | `(self, rule: Rule) → int` | Upsert rule, return ID |
| `load_rules` | `(self, task_tag=None, scope=None) → List[Rule]` | Load rules with optional filters |
| `delete_rule` | `(self, rule_id: int)` | Delete by ID |
| `save_concept` | `(self, concept: Concept) → int` | Upsert concept, return ID |
| `load_concept` | `(self, name) → Optional[Concept]` | Load by name |
| `load_concepts` | `(self, task_tag=None) → List[Concept]` | Load all concepts, optional task filter |
| `record_episode` | `(self, episode: Episode)` | Buffer episode for batch persistence |
| `flush_episodes` | `(self)` | Write buffered episodes to database |
| `load_recent_episodes` | `(self, task_tag, limit=100) → List[Episode]` | Most recent episodes |
| `count_episodes` | `(self, task_tag=None) → int` | Count episodes |
| `sample_random_episodes` | `(self, limit=32) → List[Episode]` | Random sample for dreaming |
| `prune_old_episodes` | `(self, task_tag, keep_count=10000)` | Remove oldest beyond capacity |
| `close` | `(self)` | Flush and close database |

---

## 23. `grounding_verifier.py`

> Domain-specific predicate verification and action effect checking.

### Class: `GroundingVerifier`

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self)` | Init with empty predicate and action registries |
| `register_predicate` | `(self, name, check: Callable[[Dict], bool], context=None)` | Define what a predicate means (e.g., "REL_ABOVE" → food_y < agent_y) |
| `register_action` | `(self, name, effect_check: Callable[[Dict, Dict], bool], context=None)` | Define expected action effects |
| `verify_predicate` | `(self, name, state, context=None) → bool` | Check if predicate holds in given state |
| `verify_action` | `(self, name, before_state, after_state, context=None) → bool` | Verify action had expected effect |
| `verify_rule` | `(self, rule_condition: frozenset, episodes, context=None) → float` | Check rule accuracy against episodes [0.0–1.0] |
| `get_active_predicates` | `(self, state, context=None) → List[str]` | Get all predicates true in current state |

### Factory Functions

```python
create_snake_verifier() → GroundingVerifier
    # Registers: REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT
    #            DANGER_UP, DANGER_DOWN, DANGER_LEFT, DANGER_RIGHT
    #            TARGET_NEAR, DANGER_NEAR
    # Actions:   ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

create_pong_verifier() → GroundingVerifier  
    # Registers: BALL_ABOVE, BALL_BELOW, BALL_ALIGNED, BALL_APPROACHING
    # Actions:   ACTION_UP, ACTION_DOWN, ACTION_STAY

create_maze_verifier() → GroundingVerifier
    # Registers: REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT
    #            WALL_AT_UP, WALL_AT_DOWN, WALL_AT_LEFT, WALL_AT_RIGHT
    # Actions:   ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
```

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) for system design, [VSA_THEORY.md](VSA_THEORY.md) for mathematical foundations.*
