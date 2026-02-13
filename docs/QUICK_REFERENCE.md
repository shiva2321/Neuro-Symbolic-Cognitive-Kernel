# NSCK Quick Reference Guide

**Quick lookup for key formulas, metrics, and performance numbers**

---

## VSA Operations (10,240-bit Hypervectors)

### Core Formulas

```
XOR Binding:    BIND(A, B) = A ⊕ B
Unbinding:      BIND(A, B) ⊕ B = A  (perfect recovery)
Bundling:       BUNDLE(A, B)[i] = majority_vote(A[i], B[i])
Similarity:     sim(A, B) = 1 - hamming_distance(A, B) / 10,240
Permutation:    ρᵏ(A)[i] = A[(i + k) mod 10,240]
```

### Performance (CPU-only)

| Operation | Time | Throughput |
|-----------|------|------------|
| XOR | 0.002 ms | 504K ops/sec |
| Similarity | 0.008 ms | 132K ops/sec |
| Permutation | 0.009 ms | 114K ops/sec |
| Memory/HV | 1.25 KB | 32× vs float32 |

### Random Vector Properties

```
Expected similarity: 0.5000
Measured similarity: 0.5001 ± 0.0051
Theoretical σ: 1/(2√10,240) = 0.00494
Status: MATCHES THEORY ✅
```

---

## Causal Discovery (Delta-P)

### Formula
```
ΔP(C→E) = P(E|C) - P(E|¬C)

Interpretation:
  +1.0 = Strong positive causation
   0.0 = Independence
  -1.0 = Preventive causation
  
Threshold: |ΔP| > 0.3 → Accept causal link
```

### Example Results
```
SWITCH_ON → LIGHT_ON: ΔP = 1.00 ✅ (strong causation)
CLAP → BIRD_CHIRPS: ΔP = 0.00 ✅ (spurious, rejected)
```

---

## Multi-Task Learning (Gradient Surgery)

### Formula
```
For conflicting gradients (cos(g_i, g_j) < 0):
  g_i_proj = g_i - (g_i·g_j / ||g_j||²) · g_j
  
Result: Eliminates negative transfer
```

---

## Continual Learning (EWC)

### Formula
```
Loss_total = Loss_new + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²

where:
  Fᵢ = Fisher Information (weight importance)
  θ* = Optimal weights from previous task
  λ = Regularization strength (typical: 400)
```

### Performance
```
Without EWC: 47% forgetting (catastrophic)
With EWC: 8% forgetting (acceptable)
Improvement: 5.9× reduction in forgetting
```

---

## Global Workspace Competition

### Formula
```
Activation = base_salience 
           + relevance 
           + affect_match 
           + 0.5 · sender_confidence
           + mission_focus_bonus

Winner = argmax(Activation) if max > threshold (1.5)
```

### Example
```
Coalition A: 0.7 + 0.8 + 0.3 + 0.45 + 0.2 = 2.45 → WINS
Coalition B: 0.6 + 0.5 + 0.0 + 0.35 + 0.0 = 1.45
```

---

## Transfer Learning

### Best Results
```
Catcher → Balancer: +345% performance gain
Balancer → Catcher: +75% gain
Learning speedup: 4.5× faster to 90% performance
Zero-shot (Maze): 68% of trained performance
```

### Transfer Efficiency Formula
```
Efficiency = (Score_transfer - Score_baseline) / Score_baseline

Example: (103.3 - 23.2) / 23.2 = 3.45 = +345%
```

---

## System Performance

### Decision Cycle Breakdown
```
Total: 2.51 ms (398 decisions/sec)

Components:
  Perception: 0.32 ms (13%)
  Proposals: 0.81 ms (32%)
  Competition: 0.41 ms (16%)
  Learning: 0.64 ms (26%)
  Other: 0.33 ms (13%)
```

### Memory Usage
```
Component               Memory
─────────────────────────────────
VSA Core (10K HVs)      12.5 MB
Episodic Memory         14.2 MB
Semantic Memory          8.3 MB
Neural Networks         18.5 MB
Other                   16.5 MB
─────────────────────────────────
TOTAL                   70.0 MB

vs. LLM (7B): 14,000 MB (200× larger)
```

### Real-Time Capability
```
Decision cycle: 2.51 ms
30 FPS budget: 33.3 ms
Usage: 7.5% of budget
Headroom: 92.5% available ✅
```

---

## Memory Systems

### Episodic Memory (LSH)
```
Storage: 0.12 ms/episode
Retrieval: 0.15 ms (k=5 from 10K)
Throughput: 6,667 queries/sec
False positive rate: <0.01%
```

### Semantic Memory (Graph)
```
Nodes: 5,000 concepts
Edges: 12,000 relations
Single hop: 0.08 ms
Spreading (3 hops): 2.4 ms
```

### Rule Base
```
Rules: 500 total
Match (10 predicates): 0.05 ms
Complexity: O(R·P) = 500×10 = 5K ops
```

---

## Test Results

### Overall Statistics
```
Total Tests: 500+ functions
Core Tests: 288/307 passing (94%)
Test Files: 80+
Execution: 0.34s (capability_proofs)
```

### Key Test Results
```
✅ Context prediction: 0.75 vs 0.50 (+50% improvement)
✅ Sally-Anne test: PASSED (false belief)
✅ Causal discovery: ΔP=1.0 (perfect causation)
✅ Emotion blend: Sum=1.0 (correctly weighted)
✅ VSA similarity: 0.500 ± 0.005 (theory match)
✅ World model: 89% direction accuracy
```

---

## Comparison with Baselines

### NSCK vs. Pure Neural (DQN)

| Metric | NSCK | DQN |
|--------|------|-----|
| Sample Efficiency | High | Low |
| Transfer Learning | +345% | <10% |
| Zero-Shot | 68% | ~0% |
| Memory | 70 MB | 500+ MB |
| CPU-only | Yes ✅ | Slow ❌ |
| Interpretable | Yes ✅ | No ❌ |
| Forgetting | 8% (EWC) | 47% |

### VSA vs. Float Embeddings

| Metric | Binary VSA | Float32 |
|--------|------------|---------|
| Memory | 1.25 KB | 40 KB |
| Compression | 32× | 1× |
| Binding | 0.002 ms | 0.15 ms |
| GPU | No | Beneficial |
| Interpretable | Yes | No |

---

## Key Papers & References

1. **Kanerva (2009)** - "Hyperdimensional Computing" - VSA foundations
2. **Cheng & Novick (1992)** - "Covariation in natural causal induction" - Delta-P
3. **Gentner (1983)** - "Structure-mapping" - Analogical transfer
4. **Kirkpatrick et al. (2017)** - "Overcoming catastrophic forgetting" - EWC
5. **Yu et al. (2020)** - "Gradient Surgery for Multi-Task Learning" - PCGrad
6. **Baars (1988)** - *A Cognitive Theory of Consciousness* - Global Workspace
7. **Pearl (2009)** - *Causality* - Causal reasoning framework
8. **Johnson & Lindenstrauss (1984)** - Random projection theorem

---

## Symbol Glossary

| Symbol | Meaning |
|--------|---------|
| D | Hypervector dimension (10,240) |
| ⊗ | XOR binding |
| ⊕ | Bundling (majority vote) |
| ρᵏ | Permutation by k positions |
| ΔP | Delta-P causal strength |
| λ | EWC regularization |
| Fᵢ | Fisher Information |
| sim(A,B) | Similarity (0-1) |
| d_H | Hamming distance |

---

## Quick Commands

### Run Tests
```bash
# Core capability tests
python -m pytest nsck-demo/tests/test_capability_proofs.py -v

# All core tests (no torch needed)
python -m pytest nsck-demo/tests/test_global_workspace.py \
                 nsck-demo/tests/test_causal_discovery.py \
                 nsck-demo/tests/test_homeostasis.py \
                 nsck-demo/tests/test_logic_bridge.py -v
```

### Launch Dashboard
```bash
python launch_dashboard.py
# Open http://localhost:5000
```

### Run Benchmarks
```bash
cd nsck-demo/python
python benchmark.py --game snake --episodes 100
```

---

## File Locations

### Core Modules
```
nsck-demo/python/
  ├── hypervec_py.py          # VSA operations
  ├── cognitive_engine.py     # Central orchestrator
  ├── global_workspace.py     # Competition & mental rehearsal
  ├── analogy.py              # Transfer learning
  ├── rule_learner.py         # Symbolic rule induction
  ├── causal_reasoning.py     # Delta-P & counterfactuals
  ├── episodic_memory.py      # Experience storage
  ├── semantic_memory.py      # Concept graph
  └── emotion_system.py       # Emotion processing
```

### Documentation
```
docs/
  ├── FORMULAS_AND_PROOFS.md       # All math (34KB)
  ├── PHASE_HISTORY.md             # Development timeline (24KB)
  ├── BENCHMARK_RESULTS.md         # Performance data (20KB)
  ├── ARCHITECTURE.md              # System design (with diagrams)
  ├── VSA_THEORY.md                # Mathematical foundations (80+ pages)
  └── TESTING.md                   # Test logs & analysis
```

---

**Last Updated:** 2026-02-13  
**Version:** Phase 8 Complete  
**For Complete Documentation:** See [README.md](../README.md) Documentation section
