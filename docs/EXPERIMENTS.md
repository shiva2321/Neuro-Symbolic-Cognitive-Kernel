# Experiment Guide

**Last Updated:** January 16, 2026

## Quick Start

### Run All Experiments via Launcher

```bash
python launcher.py
```

This provides an interactive menu to run any experiment.

---

## Available Experiments

### 1. Pavlov's Dog (Classical Conditioning) ⭐

**File:** `pavlov_experiment.py`  
**Difficulty:** Easy  
**Training Time:** 15 trials (~2 seconds)  
**Success Rate:** 100%

**Description:**
Demonstrates classical conditioning where a neutral stimulus (bell) becomes associated with an unconditioned stimulus (food) through dopamine-modulated learning.

**Network:**
- Input nodes: Bell (0), Food (1)
- Output node: Salivate (2)
- Connections: Bell→Salivate (0.1), Food→Salivate (2.0)

**Training Protocol:**
1. **Baseline:** Ring bell → No salivation ✓
2. **Training:** Bell + Food + Dopamine (15 trials)
3. **Test:** Ring bell alone → Salivation! ✓

**Expected Results:**
```
Initial Bell→Salivate weight: 0.1000
Final Bell→Salivate weight:   2.0000
Learning: ✓ SUCCESS
```

**Run Command:**
```bash
python pavlov_experiment.py
```

---

### 2. Sequence Learning (Temporal Patterns) ⭐⭐

**File:** `sequence_experiment.py`  
**Difficulty:** Medium  
**Training Time:** 50-500 epochs  
**Success Rate:** 100% (with sufficient training)

**Description:**
Learn to predict the next element in a repeating sequence (A→B→C→A). Tests temporal credit assignment and predictive coding.

**Network:**
- 3 input nodes (A, B, C)
- 3 output nodes (predicting next A, B, C)
- 9 connections (3×3 fully connected)

**Training Protocol:**
1. Present sequence: A→B→C→A→B→C...
2. Reward correct predictions with dopamine
3. Use teacher forcing to guide learning

**Expected Results:**
```
Epoch 1-50:   ~60-80% accuracy
Epoch 50-100: ~90-95% accuracy
Epoch 100+:   100% accuracy

Final weights:
  Correct paths (A→B, B→C, C→A): 2.0000
  Wrong paths:                    0.1000 (pruned)
```

**Run Command:**
```bash
python sequence_experiment.py

# Or with custom epochs:
python launcher.py  # Choose option 2, specify epochs
```

---

### 3. XOR Problem (Non-Linear Classification) ⭐⭐⭐

**File:** `xor_experiment.py`  
**Difficulty:** Hard  
**Training Time:** 400-800 epochs  
**Success Rate:** Partial (50-75%)

**Description:**
Solve the classic XOR problem using neuromorphic learning. Requires selective firing patterns to separate non-linearly separable classes.

**Network:**
- 2 input nodes (X0, X1)
- 2 hidden nodes (OR gate, AND gate)
- 1 output node
- Inhibitory connections from AND gate to output

**Training Protocol:**
1. Train on all 4 XOR cases: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
2. Use teacher forcing on output
3. Apply homeostatic plasticity to prevent saturation

**Known Issue:**
⚠️ Hidden neurons converge to similar weights, losing the differential needed for XOR. Accuracy plateaus at 50-75%.

**Potential Solutions:**
- Stronger inhibition
- Lateral competition between hidden nodes
- Different initialization
- More hidden neurons

**Run Command:**
```bash
python xor_experiment.py

# Or with custom epochs:
python launcher.py  # Choose option 3, specify epochs
```

---

### 4. Unified Learner (Multi-Task Learning) ⭐⭐⭐⭐

**File:** `unified_learner.py`  
**Difficulty:** Advanced  
**Training Time:** ~5 minutes  
**Success Rate:** 100% (Pavlov + Sequence), Partial (XOR)

**Description:**
ONE network learns ALL THREE tasks sequentially. Tests continuous learning and catastrophic forgetting resistance.

**Network:**
- 30 total nodes across 3 task-specific regions
- No overlap between task regions
- Shared learning mechanisms

**Training Protocol:**
1. Learn Pavlov (15 trials)
2. Learn Sequence (50 epochs)
3. Learn XOR (400 epochs)
4. Test retention of all tasks

**Expected Results:**
```
Stage 1: Pavlov learned      ✅ 100%
Stage 2: Sequence learned    ✅ 100%
         Pavlov retained     ✅ 100%
Stage 3: XOR partial         ⚠️  50-75%
         Pavlov retained     ✅ 100%
         Sequence retained   ✅ 100%
```

**Key Finding:** ✅ **NO CATASTROPHIC FORGETTING!**

Separate task regions prevent interference.

**Run Command:**
```bash
python unified_learner.py

# Or via launcher:
python launcher.py  # Choose option 6
```

---

## Test Suite

### Unit Tests

**Location:** `tests/`

1. **test_teacher_forcing.py**: Validates teacher forcing mechanism
2. **test_timing.py**: Performance benchmarks
3. **test_all_fixes.py**: Regression tests for bug fixes

**Run All Tests:**
```bash
cd tests
pytest
```

---

## Visualization & Dashboards

### Learning Dashboard (Pavlov)

**File:** `learning_dashboard.py`

Real-time visualization of Pavlov experiment showing:
- Weight evolution over time
- Neuron firing states
- Phase transitions (baseline → training → test)
- Visual progress bars

**Run Command:**
```bash
python learning_dashboard.py
```

---

## Demos

### Simple Semantic Demo

**File:** `simple_demo.py`

Demonstrates the semantic assistant Q&A system:
- Load knowledge base (RDF triples)
- Ask natural language questions
- Get answers from knowledge graph

**Run Command:**
```bash
python simple_demo.py
```

**Example Interaction:**
```
Q: Who invented the compiler?
A: HOPPER

Q: Who created Python?
A: ROSSUM

Q: What flies?
A: BIRDS
```

---

## Performance Testing

### Stress Test

**File:** `stress_test.py`

Tests large-scale Flash Colony performance:
- Create 100K+ neuron networks
- Measure memory usage
- Benchmark simulation speed
- Test persistence/recovery

**Run Command:**
```bash
python stress_test.py
```

### Persistence Test

**File:** `persistence_test.py`

Validates binary file I/O:
- Write/read neuron states
- Verify synapse persistence
- Test crash recovery

**Run Command:**
```bash
python persistence_test.py
```

---

## Customizing Experiments

### Modify Training Parameters

All experiments support parameter tuning:

```python
from sequence_experiment import SequenceLearningExperiment

exp = SequenceLearningExperiment()
exp.run_training(
    max_epochs=500,           # More training
    learning_rate=0.02,       # Faster learning
    dopamine_amount=1.0       # Stronger rewards
)
```

### Create New Experiments

Template structure:

```python
from network import NeuromorphicNetwork

# 1. Build network
net = NeuromorphicNetwork()
net.add_node(0, "input")
net.add_node(1, "output")
net.connect(0, 1, weight=0.1)

# 2. Training loop
for epoch in range(100):
    # Set inputs
    net.set_input(0, value=1.0)
    
    # Step with dopamine
    dopamine = 1.0 if correct else 0.0
    net.step(dopamine, learning_rate=0.01)
    
    # Check outputs
    fired = net.is_firing(1)

# 3. Test retention
net.reset()  # Clear traces
# ... test protocol ...
```

---

## Troubleshooting

### Experiment Not Learning

1. **Check dopamine timing**: Ensure dopamine arrives when both pre and post neurons fire
2. **Verify connectivity**: Use `print_weights()` to inspect connections
3. **Increase epochs**: Some tasks need 300-500+ epochs
4. **Adjust learning rate**: Try 0.005 - 0.05 range

### Weight Saturation

If weights hit ceiling (2.0) too quickly:
1. Enable homeostatic plasticity (already on in BioNode)
2. Lower learning rate
3. Reduce dopamine amount
4. Add more inhibitory connections

### Catastrophic Forgetting

To prevent forgetting:
1. Use separate node regions for different tasks
2. Call `network.reset()` between training phases
3. Implement sparse, non-overlapping connectivity
4. Consider consolidation/replay mechanisms

---

## Contributing New Experiments

When adding new experiments:

1. **Follow naming convention**: `<task>_experiment.py`
2. **Include docstring**: Explain what it demonstrates
3. **Add to launcher.py**: Make it discoverable
4. **Document expected results**: Include success criteria
5. **Add tests**: Create unit test in `tests/`
6. **Update this guide**: Add entry with difficulty rating

---

## Benchmark Results

*Tested on: Intel i7-10700K, 32GB RAM, Windows 11, Python 3.12*

| Experiment | Network Size | Training Time | Memory Usage | Success Rate |
|------------|-------------|---------------|--------------|--------------|
| Pavlov     | 3 nodes     | 2 seconds     | <1 MB        | 100%         |
| Sequence   | 6 nodes     | 5-30 seconds  | <1 MB        | 100%         |
| XOR        | 5 nodes     | 10-60 seconds | <1 MB        | 50-75%       |
| Unified    | 30 nodes    | 5 minutes     | <5 MB        | 100%/100%/75%|
| Stress Test| 100K nodes  | 1-2 minutes   | 40 MB (mmap) | N/A          |

---

## Further Reading

- **ARCHITECTURE.md**: Deep dive into system design
- **SEMANTIC_PARSER_FIX.md**: Details on natural language processing
- **PHASE_2_BINARY_SYNAPSES.md**: Binary persistence implementation
- **HOMEOSTASIS_IMPLEMENTATION.md**: Synaptic scaling mechanism

