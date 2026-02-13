# NSCK Substrate Architecture Audit
**Date:** February 12, 2026  
**Purpose:** Document current module coupling patterns and migration path to plugin-based substrate architecture

## Executive Summary

The NSCK_V2 has the **foundational infrastructure** for a plugin-based cognitive substrate but doesn't consistently use it. This audit identifies:
- 34 module dependencies in CognitiveEngine
- Existing `WorkspaceModule` abstract interface (underutilized)
- GlobalWorkspace message bus with competitive selection (LIDA-Lite implementation)
- Missing: standardized propose/update/telemetry interface, VSA cleanup, dynamic rule confidence, brain versioning

## Current Architecture Analysis

### 1. Module Coupling in CognitiveEngine

**Location:** `nsck-demo/python/cognitive_engine.py`

**Direct Instantiation (Tight Coupling):**
```python
# Lines 24-52: Import statements for 24+ modules
# Lines 150-228: Direct instantiation in __init__

self.semantic_memory = SemanticMemory()
self.self_model = SelfModel()
self.active_agent = ActiveAgent(w=10, h=10, horizon=3)
self.language = LanguageModule()
self.dialogue = DialogueManager(self, self.language)
# ... 29 more modules ...
```

**Count:** ~34 total dependencies (24 direct + 10 optional/conditional)

**Pattern:** Modules are stored as instance attributes and called directly, bypassing GlobalWorkspace registration system.

### 2. GlobalWorkspace Plugin Infrastructure

**Location:** `nsck-demo/python/global_workspace.py`

**Existing Capabilities:**
- ✅ `WorkspaceModule(ABC)` abstract base class (line 22)
- ✅ `register_module(name, module)` method (line 82)
- ✅ `Coalition` dataclass for proposals (lines 28-40)
- ✅ Competitive selection via `compete()` (lines 86-107)
- ✅ Broadcast channel to all registered modules (lines 109-117)
- ✅ Mental rehearsal with veto mechanism (Phase 8, lines 166-273)

**Current Interface:**
```python
class WorkspaceModule(ABC):
    @abstractmethod
    def receive_broadcast(self, content: Any):
        """Receive content broadcast from the global workspace."""
        pass
```

**Problem:** Only 1 required method. No standardized way for modules to:
- Propose actions (`propose()`)
- Learn from feedback (`update()`)
- Report status (`get_telemetry()`)

### 3. Module Implementation Survey

| Module | Inherits WorkspaceModule? | Registered with GW? | Direct Reference in CE? |
|--------|---------------------------|---------------------|------------------------|
| EpisodicMemory | ❌ No | ❌ No | ✅ Yes (self.episodic) |
| RuleLearner | ❌ No | ❌ No | ✅ Yes (self.rule_learner) |
| CausalReasoner | ❌ No | ❌ No | ✅ Yes (self.causal_reasoners) |
| AnalogyEngine | ❌ No | ❌ No | ✅ Yes (self.analogy) |
| EmotionSystem | ❌ No | ❌ No | ✅ Yes (self.emotion) |
| TheoryOfMind | ❌ No | ❌ No | ✅ Yes (self.tom) |
| SemanticMemory | ❌ No | ❌ No | ✅ Yes (self.semantic_memory) |
| CuriosityModule | ❌ No | ❌ No | ✅ Yes (self.curiosity) |
| SelfModel | ❌ No | ❌ No | ✅ Yes (self.self_model) |
| LanguageModule | ❌ No | ❌ No | ✅ Yes (self.language) |

**Result:** 0 out of 10 core cognitive modules use the plugin pattern.

### 4. VSA Operations & Cleanup

**Location:** `nsck-demo/python/hypervec_py.py`

**Current Operations:**
- `xor()` - Binding (lines 23-26)
- `bundle()` - Superposition with random tiebreak (lines 28-48)
- `similarity()` - Hamming distance (lines 50-56)
- `permute()` - Circular rotation (lines 58-66)

**Cleanup Mechanism:** ❌ **NONE**

**Issue:** As concept count grows, Hamming distance noise accumulates. No associative memory to "snap" noisy vectors back to known clean atomic vectors.

**Risk:** False positives in similarity matching when scaling to 10,000+ concepts.

### 5. Rule Induction Brittleness

**Location:** `nsck-demo/python/rule_learner.py`

**Current Configuration:**
```python
def __init__(self, verifier, store, min_support: int = 5, ...):
    self.min_support = min_support  # Line 46
```

**Mitigation Already Present:**
- Approximate matching (60% overlap threshold, line 67)
- Tenure system for rule protection (lines 60-65)
- Partial credit for overlapping patterns (lines 175-186)

**Gap:** Rules don't form until exactly 5 observations. No "low confidence" rule seeding for 1-3 observations.

### 6. Text Knowledge Learner

**Location:** `nsck-demo/python/text_knowledge_learner.py`

**Architecture:** Hybrid (VSA + Regex)
- ✅ LinguaCortex for semantic folding (line 106)
- ✅ Hypervector encoding via VSA
- ❌ 10 hardcoded regex patterns for relations (lines 367-393):
  ```python
  patterns = [
      (r'(\w+)\s+is\s+a\s+(\w+)', 'is_a'),
      (r'(\w+)\s+has\s+(\w+)', 'has_property'),
      (r'(\w+)\s+causes\s+(\w+)', 'causes'),
      ...
  ]
  ```

**Issue:** Relations are "discovered" only if they match pre-defined linguistic patterns. Not true emergent learning.

### 7. Universal Input Layer

**Location:** `nsck-demo/python/universal_input.py`

**Grounding Approach:** Heuristic POS tagging
- Closed-class word lists (25 determiners, 30 prepositions, line 102-128)
- Suffix-based verb/adjective detection (lines 139-187)
- Phrase chunking: S → NP VP (PP)* (lines 189-248)

**Semantic Roles:** ❌ Only extracts Subject/Verb/Object, not Agent/Patient/Instrument

### 8. Persistence & Brain Packaging

**Location:** `nsck-demo/python/persistence.py`

**Current Features:**
- ✅ SQLite with WAL mode (line 71)
- ✅ Tables: rules, concepts, episodes (lines 76-125)
- ✅ CRUD operations

**Missing:**
- ❌ Version tracking
- ❌ Brain export/import as .nsck package
- ❌ Checkpoint/restore functionality
- ❌ Merge strategies for brain fusion

### 9. Traceability

**Location:** `nsck-demo/python/unified_dashboard.py`, `launch_dashboard.py`

**Current:** Unified dashboard with structured logs and decision trace endpoints.

**Implemented:**
- Winning coalition details
- Competing proposals ranked
- Decision trace export via `/api/decision_traces`

**Remaining Gaps:**
- Causal chain attribution in traces (if/when available from GlobalWorkspace)

## Migration Path to Substrate Architecture

### Phase A: Interface Standardization ✅ **PRIORITY**
1. Extend `WorkspaceModule` with `propose()`, `update()`, `get_telemetry()`
2. Create `MODULE_INTERFACE_SPEC.md` documentation
3. Build reference implementation template

### Phase B: Core Module Refactoring
1. Refactor 10 core modules to inherit from `WorkspaceModule`
2. Update `CognitiveEngine` to use `register_module()` pattern
3. Remove direct attribute storage where possible

### Phase C: Robustness Enhancements
1. Add `CleanupMemory` class to VSA operations
2. Implement dynamic rule confidence (0.3-1.0 scale)
3. Replace regex relation extraction with co-occurrence semantic folding
4. Add functional role extraction to UniversalInput

### Phase D: Developer SDK
1. Create `nsck_sdk/` package
2. Build 3 reference external modules
3. Write `MODULE_DEV_GUIDE.md`
4. Implement `ModuleRegistry` for plugin discovery

### Phase E: Brain Packaging
1. Add versioning table to BrainStore
2. Implement checkpoint/export system
3. Create `.nsck` file format spec
4. Build brain merge strategies

### Phase F: Validation
1. Stress test with noisy rule learning
2. Integration test with external plugin
3. Developer validation exercise

## Success Criteria

The substrate transformation is complete when:
1. ✅ External developer can create 20-line module without modifying CognitiveEngine
2. ✅ Module competes in GlobalWorkspace and influences decisions
3. ✅ Trained brain can be exported, shared, and imported
4. ✅ Decision traces show complete explanations of action selection
5. ✅ System learns from 1-2 observations (low confidence) without brittleness

## Estimated Effort

- **Phase A:** 2-3 hours (foundation)
- **Phase B:** 4-6 hours (refactoring)
- **Phase C:** 6-8 hours (robustness)
- **Phase D:** 4-5 hours (SDK)
- **Phase E:** 3-4 hours (packaging)
- **Phase F:** 2-3 hours (testing)

**Total:** 21-29 hours of focused development

## References

- `global_workspace.py` - Lines 22-27 (WorkspaceModule), 82-117 (registration/broadcast)
- `cognitive_engine.py` - Lines 150-228 (module instantiation)
- `rule_learner.py` - Line 46 (min_support threshold)
- `hypervec_py.py` - Lines 28-48 (bundle operation, no cleanup)
- `text_knowledge_learner.py` - Lines 367-393 (regex patterns)
- `universal_input.py` - Lines 139-248 (POS tagging, phrase parsing)
- `persistence.py` - Full file (no versioning)

---
**Last Updated:** February 12, 2026
