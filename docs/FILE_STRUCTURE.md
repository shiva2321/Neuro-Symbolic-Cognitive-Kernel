# Project File Structure

**Last Updated:** January 16, 2026

## Directory Layout

```
Node_network/
│
├── README.md                           # Main project documentation
├── requirements.txt                    # Dependencies (none required!)
├── launcher.py                         # Interactive experiment launcher
├── knowledge.txt                       # Sample knowledge base for semantic system
│
├── Core Neuromorphic Library
│   ├── bionode.py                     # BioNode: LIF neuron with homeostasis
│   ├── synapse.py                     # Synapse: connection with eligibility traces
│   ├── network.py                     # NeuromorphicNetwork: graph manager
│   ├── flash_colony.py                # FlashColony: binary neuromorphic engine
│   ├── flash_manager.py               # FlashManager: memory-mapped I/O
│   ├── flash_dynamic.py               # Dynamic synapse allocation
│   ├── block_manager.py               # Block allocator for synapses
│   ├── ncgn_anatomy.py                # NcgnAnatomy: brain region management
│   └── neuro_kernel.py                # Advanced kernel operations
│
├── Experiments
│   ├── pavlov_experiment.py           # Pavlov's dog (classical conditioning)
│   ├── sequence_experiment.py         # Sequence learning (A→B→C→A)
│   ├── xor_experiment.py              # XOR problem (non-linear classification)
│   └── unified_learner.py             # Multi-task continuous learning
│
├── Semantic System
│   ├── semantic_brain.py              # SemanticBrain: RDF triple store
│   ├── context_driver.py              # SemanticBrain with NLP parsing
│   ├── semantic_assistant.py          # Interactive Q&A assistant
│   └── simple_demo.py                 # Simple semantic demo
│
├── Testing & Benchmarking
│   ├── stress_test.py                 # Large-scale performance testing
│   ├── persistence_test.py            # Binary I/O validation
│   ├── brain_surgeon.py               # Brain inspection/debugging tool
│   └── main_brain_driver.py           # Advanced brain management
│
├── Visualization
│   └── learning_dashboard.py          # Real-time training visualization
│
├── Unit Tests
│   └── tests/
│       ├── test_teacher_forcing.py    # Teacher forcing validation
│       ├── test_timing.py             # Performance benchmarks
│       └── test_all_fixes.py          # Regression tests
│
├── Documentation
│   └── docs/
│       ├── ARCHITECTURE.md            # System architecture overview
│       ├── EXPERIMENTS.md             # Experiment guide & tutorials
│       ├── IMPLEMENTATION_NOTES.md    # Technical implementation details
│       └── archive/                   # Historical documentation
│           ├── NCGN_ARCHITECTURE.md
│           ├── FLASH_ARCHITECTURE.md
│           ├── HOMEOSTASIS_IMPLEMENTATION.md
│           ├── PHASE_2_BINARY_SYNAPSES.md
│           ├── SEMANTIC_PARSER_FIX.md
│           ├── RETENTION_TEST_RESULTS.md
│           └── originals/
│
└── Generated Files (gitignore)
    ├── *.dat                          # Binary brain files
    ├── __pycache__/                   # Python bytecode
    └── .venv/                         # Virtual environment
```

---

## File Categories

### Core Library (Must Keep)

**Neuromorphic Engine:**
- `bionode.py` - Object-based neuron implementation
- `synapse.py` - Connection between neurons
- `network.py` - Network graph manager
- `flash_colony.py` - Binary memory-mapped implementation
- `flash_manager.py` - File I/O and persistence
- `block_manager.py` - Memory allocation
- `ncgn_anatomy.py` - High-level brain architecture
- `neuro_kernel.py` - Advanced operations
- `flash_dynamic.py` - Dynamic synapse management

**Semantic System:**
- `semantic_brain.py` - Knowledge graph storage
- `context_driver.py` - Natural language processing
- `semantic_assistant.py` - Q&A interface

### Experiments (Demonstrations)

**Core Experiments:**
- `pavlov_experiment.py` - Best introductory example
- `sequence_experiment.py` - Shows temporal learning
- `xor_experiment.py` - Non-linear classification challenge
- `unified_learner.py` - Multi-task learning demo

**Demo Scripts:**
- `simple_demo.py` - Semantic assistant demo
- `launcher.py` - Menu-driven experiment launcher

### Tools & Utilities

**Testing:**
- `stress_test.py` - Performance benchmarking
- `persistence_test.py` - File I/O validation
- `tests/` - Unit test suite (pytest)

**Debugging:**
- `brain_surgeon.py` - Brain inspection tool
- `main_brain_driver.py` - Advanced brain operations

**Visualization:**
- `learning_dashboard.py` - Real-time training display

### Documentation

**Current (Active):**
- `README.md` - Main project documentation
- `docs/ARCHITECTURE.md` - System design
- `docs/EXPERIMENTS.md` - Experiment guide
- `docs/IMPLEMENTATION_NOTES.md` - Technical details

**Archive (Historical):**
- `docs/archive/` - Old documentation for reference

---

## Key Design Decisions

### Why Two Implementations?

**BioNode Network (Object-Based):**
- Easy to understand and debug
- Full introspection capabilities
- Perfect for learning and experimentation
- Used by: pavlov, sequence, xor, unified experiments

**Flash Colony (Binary):**
- 99% memory savings
- Crash-proof persistence
- Scales to millions of neurons
- Used by: stress tests, production deployments

### Why Sparse Connectivity?

Biological brains are ~1% connected. Sparse graphs:
- Scale better (O(edges) not O(nodes²))
- More energy efficient
- Enable event-driven computation
- Match biological reality

### Why No Dependencies?

- **Portability**: Runs anywhere Python runs
- **Transparency**: Easy to understand every line
- **Educational**: See how everything works
- **Deployment**: No version conflicts or bloat

---

## File Dependencies

### Import Graph

```
Experiments
    ↓
network.py, bionode.py, synapse.py
    ↓
(Pure Python - no external deps)

Flash Colony
    ↓
flash_colony.py, flash_manager.py
    ↓
block_manager.py
    ↓
(Only stdlib: struct, mmap)

Semantic System
    ↓
context_driver.py, semantic_brain.py
    ↓
flash_colony.py
    ↓
(Only stdlib)
```

### No Circular Dependencies

All modules are carefully layered:
1. Base classes (synapse, bionode)
2. Network manager (network)
3. Binary layer (flash_*)
4. High-level (anatomy, semantic)
5. Applications (experiments, demos)

---

## Generated Files

### Binary Brain Files (.dat)

Created automatically by experiments:
- `main_brain.dat` - Default Flash Colony brain
- `semantic_brain.dat` - Semantic assistant knowledge
- `test_*.dat` - Test brain files
- `demo_brain.dat` - Demo scripts

**Note:** These are memory-mapped and may be locked while programs run.

### Python Bytecode (__pycache__)

Automatically created by Python interpreter. Safe to delete.

### Virtual Environment (.venv)

Optional Python virtual environment. Not required since there are no dependencies.

---

## Cleanup Commands

### Remove Generated Files

```bash
# Windows PowerShell
Remove-Item *.dat
Remove-Item -Recurse __pycache__

# Linux/Mac
rm *.dat
rm -r __pycache__
```

### Fresh Start

```bash
# Remove all generated files
Remove-Item *.dat, __pycache__ -Recurse -Force
```

---

## Git Configuration

Recommended `.gitignore`:

```
# Binary brain files
*.dat

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

## Recommended Reading Order

For newcomers to the codebase:

1. **README.md** - Project overview
2. **docs/ARCHITECTURE.md** - System design
3. **synapse.py** - Simple, understand eligibility traces
4. **bionode.py** - LIF neuron implementation
5. **network.py** - Graph manager
6. **pavlov_experiment.py** - Complete example
7. **docs/EXPERIMENTS.md** - Try other experiments
8. **flash_colony.py** - Binary implementation
9. **docs/IMPLEMENTATION_NOTES.md** - Deep technical details

---

## Size Estimates

| Component | Lines of Code | Approx Size |
|-----------|---------------|-------------|
| Core library | ~2,000 | 80 KB |
| Experiments | ~1,500 | 60 KB |
| Semantic system | ~1,000 | 40 KB |
| Tests | ~500 | 20 KB |
| Documentation | ~3,000 | 200 KB |
| **Total** | **~8,000** | **~400 KB** |

Remarkably compact for a full neuromorphic computing system!

---

## Maintenance Notes

### What to Update When...

**Adding a new experiment:**
1. Create `<name>_experiment.py`
2. Add to `launcher.py` menu
3. Update `docs/EXPERIMENTS.md`
4. Add unit test in `tests/`

**Changing core learning rules:**
1. Modify `bionode.py` or `synapse.py`
2. Run full test suite: `pytest tests/`
3. Update `docs/IMPLEMENTATION_NOTES.md`
4. Verify all experiments still work

**Modifying binary format:**
1. Update `flash_manager.py` structures
2. Increment version number in header
3. Update `docs/IMPLEMENTATION_NOTES.md`
4. Add migration code if needed

---

**Last Updated:** January 16, 2026

