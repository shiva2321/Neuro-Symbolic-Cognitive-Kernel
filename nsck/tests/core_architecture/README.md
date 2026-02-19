# NSCK Core Architecture - Comprehensive Testing Suite

## Overview

This testing suite comprehensively validates the **NSCK (Neural-Symbolic Cognitive Kernel)** core architecture across seven critical dimensions:

1. **Novelty Assessment** - Validates novel architectural patterns
2. **Input Understanding** - Tests semantic comprehension
3. **Learning Capability** - Proves genuine learning from data
4. **Reasoning & Decision-Making** - Complex inference and goal-oriented behavior
5. **Explainability** - Glass-box traceability and confidence calibration
6. **Efficiency** - Performance benchmarks and scalability
7. **Extensibility** - Building new domains and integrations

## Quick Start

### Run All Tests
```bash
cd /workspaces/Node_network/nsck
python tests/core_architecture/run_comprehensive_tests.py
```

### Run Individual Phase
```bash
pytest tests/core_architecture/test_novelty.py -v -s
pytest tests/core_architecture/test_understanding.py -v -s
pytest tests/core_architecture/test_learning.py -v -s
pytest tests/core_architecture/test_reasoning.py -v -s
pytest tests/core_architecture/test_explainability.py -v -s
pytest tests/core_architecture/test_efficiency.py -v -s
pytest tests/core_architecture/test_extensibility.py -v -s
```

### Run Specific Test
```bash
pytest tests/core_architecture/test_novelty.py::TestVSAFoundation::test_hypervector_dimensions -v -s
```

## Test Structure

```
tests/core_architecture/
├── __init__.py
├── run_comprehensive_tests.py    # Test runner & report generator
├── test_novelty.py               # Phase 1: 6 tests
├── test_understanding.py         # Phase 2: 6 tests
├── test_learning.py              # Phase 3: 6 tests
├── test_reasoning.py             # Phase 4: 6 tests
├── test_explainability.py        # Phase 5: 6 tests
├── test_efficiency.py            # Phase 6: 6 tests
├── test_extensibility.py         # Phase 7: 6 tests
└── data/                         # Test data files
    ├── animal_facts.txt
    ├── geography_facts.txt
    └── physics_facts.txt
```

## Validation Criteria

### Phase 1: Novelty (6 tests)
- ✅ **NOV-1**: VSA algebra (XOR, majority-vote, permutation) implemented and composable
- ✅ **NOV-2**: Every inference produces complete ThoughtTrace
- ✅ **NOV-3**: No LLM/transformer dependency - pure symbolic reasoning
- ✅ **NOV-4**: Deterministic symbol ↔ HV mapping (invertible)
- ✅ **NOV-5**: 11-stage pipeline flows correctly with measurable latency
- ✅ **NOV-6**: Semantic + Episodic + Causal memories mutually informative

### Phase 2: Understanding (6 tests)
- ✅ **UND-1**: Semantic similarity of texts correlates in HV space
- ✅ **UND-2**: Entity extraction >80% F1 on named entities
- ✅ **UND-3**: Context sensitivity - same word, different contexts → different HVs
- ✅ **UND-4**: Negation handled - NOT(X) ≠ X
- ✅ **UND-5**: Relationship types distinguished (is_a vs subset vs approximates)
- ✅ **UND-6**: Pronoun resolution to antecedents (sim >0.75)

### Phase 3: Learning (6 tests)
- ✅ **LRN-1**: Concepts accumulate and are queryable
- ✅ **LRN-2**: Relations extracted and stored in causal graph
- ✅ **LRN-3**: Single example teaches a concept
- ✅ **LRN-4**: Old knowledge retained after new learning
- ✅ **LRN-5**: Learned rules generalize to unseen instances
- ✅ **LRN-6**: Transfer learning aids related domain learning

### Phase 4: Reasoning (8 tests)
- ✅ **RSN-1**: Causal chains through forward chaining
- ✅ **RSN-2**: Counterfactual reasoning with backtracking
- ✅ **RSN-3**: Analogy structural alignment and transfer
- ✅ **RSN-4**: Multi-step goal decomposition
- ✅ **RSN-5**: Conflict resolution via confidence/meta-reasoning
- ✅ **RSN-6**: Spreading activation finds distant concepts
- ✅ **RSN-7**: GlobalWorkspace coalition competition
- ✅ **RSN-8**: Risk-aware decision making

### Phase 5: Explainability (6 tests)
- ✅ **EXP-1**: Complete thought traces for all outputs
- ✅ **EXP-2**: Source attribution - facts trace to training data
- ✅ **EXP-3**: Confidence scores on assertions (0-1 range)
- ✅ **EXP-4**: Reasoning chains human-readable
- ✅ **EXP-5**: Emotional responses trace to input features
- ✅ **EXP-6**: Uncertainty quantification (marked/scored)

### Phase 6: Efficiency (6 tests)
- ✅ **EFF-1**: Inference speed <500ms for typical queries
- ✅ **EFF-2**: Memory footprint <2GB for 100K concepts
- ✅ **EFF-3**: Encoding speed <10ms per sentence
- ✅ **EFF-4**: Spreading activation <50ms (3-step decay)
- ✅ **EFF-5**: Semantic search <100ms (over 1K concepts)
- ✅ **EFF-6**: Scalability - logarithmic or linear latency growth

### Phase 7: Extensibility (6 tests)
- ✅ **EXT-1**: Custom TaskBrain for new domains
- ✅ **EXT-2**: BrainFusion integration with concept alignment
- ✅ **EXT-3**: New reasoning module composition
- ✅ **EXT-4**: Persistence backend swapping
- ✅ **EXT-5**: Custom perception modules (image → HV)
- ✅ **EXT-6**: Reproducibility - same input → same output

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Novelty | 5+ novel patterns | 🔄 Testing |
| Understanding | >80% semantic correspondence | 🔄 Testing |
| Learning | >90% knowledge retention | 🔄 Testing |
| Reasoning | >70% multi-step accuracy | 🔄 Testing |
| Explainability | 100% output traceability | 🔄 Testing |
| Efficiency | <500ms per query | 🔄 Testing |
| Extensibility | 3+ domains integrated | 🔄 Testing |

## Test Narratives

Each phase includes narrative tests that demonstrate capabilities through realistic examples:

- **Novelty Narratives**: Show VSA advantages vs. transformers
- **Understanding Narratives**: Semantic comprehension examples
- **Learning Narratives**: Incremental learning and generalization
- **Reasoning Narratives**: Causal and analogical inference
- **Explainability Narratives**: Full traceability examples
- **Efficiency Narratives**: Benchmark results
- **Extensibility Narratives**: Domain creation walkthrough

## Generated Reports

After running the comprehensive test suite, the following reports are generated:

### Test Results Report
- CSV format: test_id | aspect | passed | latency_ms | notes
- Summary statistics
- Pass/fail breakdown by phase

### Architecture Validation Report
- Which architectural claims were validated
- Which need additional work
- Novelty assessment
- Capability verification

### Performance Profile
- Latency vs. KB size graphs
- Memory vs. number of concepts
- Throughput metrics
- Scaling behavior

### Improvement Recommendations
- Prioritized by impact
- Specific implementation guidance
- Research directions
- Community contribution opportunities

## Case Studies

The test suite includes 5+ worked examples showing:
1. **Case: Animal Learning** - Learning from few examples
2. **Case: Causal Reasoning** - Multi-step inference chains
3. **Case: Pronoun Resolution** - Context-aware understanding
4. **Case: Game AI** - Domain-specific TaskBrain
5. **Case: Cross-Domain Analogy** - Transfer learning

## Known Limitations & Future Work

### Current Scope
- Text-only inputs (narrative tests for multimodal)
- Single-threaded execution
- In-memory knowledge bases
- No persistent storage tests

### Future Extensions
- Image encoding (multimodal)
- Distributed reasoning
- Long-term episodic consolidation
- Collaborative multi-agent reasoning

## Contributing to Tests

To add new tests:

1. **Create test method** in appropriate phase file
2. **Follow naming**: `test_<aspect_name>`
3. **Include docstring** explaining what's tested
4. **Add narrative** tests demonstrating capability
5. **Update this README** with new test count
6. **Run full suite** to verify no regressions

## Running Test Reports

```bash
# Generate markdown + JSON reports
python /workspaces/Node_network/nsck/tests/core_architecture/run_comprehensive_tests.py

# Reports appear in: /workspaces/Node_network/test_results/
```

## Architecture Under Test

NSCK is an 11-stage cognitive pipeline:

```
Input → Perception → Understanding → Memory & Reasoning → 
Metacognition → Output → ThoughtTrace
```

**Key Components**:
- **VSA Layer**: 10,240-bit HV algebra
- **Memories**: Semantic (graph) + Episodic (LSH) + Causal
- **Reasoning**: GlobalWorkspace, CausalReasoner, AnalogyEngine, Planner
- **Cognition**: EmotionSystem, SelfModel, Theory of Mind
- **Language**: LinguaCortex (NLU) + NLG + Dialogue
- **Integration**: BrainFusion (multi-domain merging)

---

**Status**: 🚀 Comprehensive testing framework ready
**Next**: Run full test suite and analyze results
**Timeline**: 2-4 hours for full execution + analysis
