# Contributing to NCGN

**Guidelines for extending the neuromorphic brain architecture**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Creating New Experiments](#creating-new-experiments)
4. [Adding Core Modules](#adding-core-modules)
5. [Writing Tests](#writing-tests)
6. [Code Style](#code-style)
7. [Reporting Issues](#reporting-issues)

---

## Getting Started

### Fork & Clone

```bash
# Clone your fork
git clone https://github.com/yourusername/Node_network.git
cd Node_network

# Verify setup
python --version  # Should be 3.11+
python -m unittest discover -s tests -p "test_*.py"
```

### Verify Tests Pass

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Should see: OK (250+ tests)
```

---

## Project Structure

```
Node_network/
├── core/                    # ⭐ Neural engine - modify carefully
│   ├── neuron.py           # Neuron model
│   ├── synapse.py          # Synaptic transmission
│   ├── plasticity.py       # Learning rules
│   ├── event_queue.py      # Event management
│   ├── dispatcher.py       # Spike routing
│   ├── orchestrator.py     # Main simulation loop
│   ├── metrics.py          # Observability
│   ├── control_layer.py    # System 1.5
│   ├── region.py           # Regions
│   ├── system2.py          # Reasoning
│   ├── meta_learner.py     # Meta-learning
│   └── self_model.py       # Prediction
│
├── experiments/            # ✅ Add experiments here
│   ├── base_experiment.py  # Template
│   ├── pavlov_experiment.py
│   ├── sequence_experiment.py
│   ├── phase3_demo.py
│   └── ...
│
├── tests/                  # ✅ Add tests here
│   ├── test_neuron.py
│   ├── test_control_layer.py
│   └── ...
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── EXPERIMENTS.md
│   └── archive/
│
└── README.md, GETTING_STARTED.md, etc.
```

---

## Creating New Experiments

### Easiest: Extend BaseExperiment

Create a new file in `experiments/`:

```python
# experiments/my_learning_task.py

from experiments.base_experiment import ExperimentBase
from core.network import NeuromorphicNetwork

class MyLearningTask(ExperimentBase):
    """
    Brief description of your task.
    
    What it learns:
    - Point 1
    - Point 2
    
    Expected outcome:
    - Success condition
    """
    
    def __init__(self, name="my_learning_task"):
        super().__init__(name)
    
    def build_network(self):
        """Create neurons and synapses for the task."""
        # Create input neurons (one per stimulus feature)
        self.network.add_node(0, "input", threshold=0.5)
        self.network.add_node(1, "input", threshold=0.5)
        
        # Create hidden neurons
        self.network.add_node(10, "hidden", threshold=1.0)
        self.network.add_node(11, "hidden", threshold=1.0)
        
        # Create output neurons
        self.network.add_node(20, "output", threshold=1.0)
        
        # Connect input to hidden
        self.network.connect(0, 10, weight=0.2)
        self.network.connect(0, 11, weight=0.15)
        self.network.connect(1, 10, weight=0.15)
        self.network.connect(1, 11, weight=0.2)
        
        # Connect hidden to output
        self.network.connect(10, 20, weight=0.3)
        self.network.connect(11, 20, weight=0.25)
    
    def create_stimuli(self):
        """Define training stimuli and expected outputs."""
        # List of (input_dict, expected_output_dict) pairs
        self.stimuli = [
            ({0: 1.0, 1: 0.0}, {20: 1.0}),  # Input A -> Output fires
            ({0: 0.0, 1: 1.0}, {20: 0.0}),  # Input B -> Output silent
            ({0: 1.0, 1: 1.0}, {20: 1.0}),  # Input A+B -> Output fires
        ]
    
    def run_training(self, duration=100):
        """Execute the learning phase."""
        success_count = 0
        
        for epoch in range(duration):
            epoch_success = 0
            
            for stimulus_dict, expected_dict in self.stimuli:
                # Present stimulus
                for node_id, value in stimulus_dict.items():
                    self.network.set_input(node_id, value)
                
                # Run network
                self.network.step(global_dopamine=0.0, learning_rate=0.01)
                
                # Check if output matches expectation
                for output_id, expected_fires in expected_dict.items():
                    actual_fires = self.network.nodes[output_id].is_firing
                    if actual_fires == expected_fires:
                        reward = 1.0
                        epoch_success += 1
                    else:
                        reward = 0.0
                    
                    # Apply dopamine signal
                    self.network.step(global_dopamine=reward, learning_rate=0.01)
            
            success_count += epoch_success
            if epoch % 10 == 0 and self.verbose:
                print(f"Epoch {epoch}: {epoch_success}/{len(self.stimuli)} correct")
        
        print(f"Training complete. Success rate: {success_count/len(self.stimuli)/duration:.1%}")


if __name__ == "__main__":
    task = MyLearningTask()
    results = task.execute(train_duration=100)
    print(f"Results: {results}")
```

### Register Experiment

Add to `experiments/__init__.py`:

```python
from .my_learning_task import MyLearningTask

__all__ = [
    'MyLearningTask',
    # ... other experiments
]
```

### Run It

```bash
python -m experiments.my_learning_task
```

---

## Adding Core Modules

### When to Add

Add a core module when:
1. It implements a new cognitive tier
2. It's used by multiple experiments
3. It solves a general problem (not specific to one task)

**Don't add** if it's task-specific → put in `experiments/` instead.

### Template

Create `core/my_module.py`:

```python
"""
Brief description of module.

Philosophy:
- What problem does this solve?
- How does it fit in the architecture?
- What invariants must hold?

Invariants:
- Invariant 1: ...
- Invariant 2: ...
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MyComponent:
    """Component description."""
    name: str
    value: float
    # ...
    
    def compute(self) -> float:
        """Brief description of computation."""
        # Implementation
        return result


class MyModule:
    """Main class description."""
    
    def __init__(self, network, param1: float = 0.1):
        """
        Initialize module.
        
        Args:
            network: NeuromorphicNetwork instance
            param1: Parameter description
        """
        self.network = network
        self.param1 = param1
    
    def method1(self, arg1: str) -> float:
        """
        Brief description.
        
        Args:
            arg1: Description
            
        Returns:
            Result description and range
        """
        # Implementation
        result = self.compute()
        return result
    
    def get_info(self) -> Dict:
        """Return module state for inspection."""
        return {
            "param1": self.param1,
            # ...
        }
```

### Export from `core/__init__.py`

Add to the imports and `__all__`:

```python
from .my_module import MyModule, MyComponent

__all__ = [
    # ... existing exports
    'MyModule',
    'MyComponent',
]
```

### Document It

Add section to `docs/API_REFERENCE.md`:

```markdown
### `core.my_module.MyModule`

Brief description.

```python
from core.my_module import MyModule

module = MyModule(
    network,
    param1: float = 0.1
)
```

**Key Methods:**

```python
# Method 1
result = module.method1(arg1: str) -> float

# Method 2
info = module.get_info() -> Dict
```
```

---

## Writing Tests

### Test Structure

Create `tests/test_my_module.py`:

```python
"""
Unit tests for MyModule.

Each test is independent and can run in any order.
"""

import unittest
from core.network import NeuromorphicNetwork
from core.my_module import MyModule, MyComponent


class TestMyModule(unittest.TestCase):
    """Test suite for MyModule."""
    
    def setUp(self):
        """Create fixtures before each test."""
        self.network = NeuromorphicNetwork()
        self.network.add_node(0, "input")
        self.network.add_node(1, "output")
        self.network.connect(0, 1, weight=0.5)
        
        self.module = MyModule(self.network)
    
    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertEqual(self.module.param1, 0.1)
    
    def test_method1_basic(self):
        """Test basic operation of method1."""
        result = self.module.method1("test")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_method1_edge_cases(self):
        """Test edge cases."""
        # Empty input
        result = self.module.method1("")
        self.assertIsInstance(result, float)
        
        # Long input
        result = self.module.method1("x" * 1000)
        self.assertIsInstance(result, float)
    
    def test_state_persistence(self):
        """Test that state is preserved across operations."""
        result1 = self.module.method1("a")
        info = self.module.get_info()
        self.assertIn("param1", info)
        self.assertEqual(info["param1"], 0.1)
    
    def test_invariant_output_range(self):
        """Test that output always satisfies invariants."""
        for i in range(100):
            result = self.module.method1(f"test_{i}")
            # Invariant: output in [0, 1]
            self.assertGreaterEqual(result, 0.0, f"Failed at iteration {i}")
            self.assertLessEqual(result, 1.0, f"Failed at iteration {i}")


class TestMyComponent(unittest.TestCase):
    """Test suite for MyComponent."""
    
    def test_component_creation(self):
        """Test component initialization."""
        comp = MyComponent(name="test", value=0.5)
        self.assertEqual(comp.name, "test")
        self.assertEqual(comp.value, 0.5)
    
    def test_component_compute(self):
        """Test component computation."""
        comp = MyComponent(name="test", value=0.5)
        result = comp.compute()
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
```

### Run Tests

```bash
# Run this test file
python -m unittest tests.test_my_module -v

# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Checklist

- [ ] Basic initialization works
- [ ] Methods return expected types
- [ ] Edge cases handled (empty, null, extreme values)
- [ ] Invariants verified
- [ ] State is preserved correctly
- [ ] Integration with network works
- [ ] No side effects on other components

---

## Code Style

### Style Guide

```python
# 1. Use type hints
def compute(self, value: float, iterations: int) -> Dict[str, float]:
    """Compute something."""
    pass

# 2. Use docstrings
def method(self, arg: str) -> float:
    """
    Brief one-liner.
    
    Longer explanation if needed. Multiple sentences
    describing what this does and why.
    
    Args:
        arg: Description of arg
        
    Returns:
        Description of return value
    """
    pass

# 3. Constants in UPPER_CASE
DEFAULT_THRESHOLD = 1.0
MAX_FIRING_RATE = 100.0

# 4. Private methods start with _
def _internal_computation(self):
    """Private method (not part of public API)."""
    pass

# 5. Docstring examples
def example_method(self):
    """
    Example usage.
    
    Example:
        >>> obj = MyClass()
        >>> result = obj.example_method()
        >>> print(result)
        42
    """
    pass
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `class NoveltySensor` |
| Functions | snake_case | `def compute_novelty()` |
| Constants | UPPER_CASE | `MAX_THRESHOLD = 1.0` |
| Private | _snake_case | `def _internal_method()` |
| Booleans | is_/has_/should_ | `is_firing`, `has_synapses` |

### Documentation Requirements

Every public class/function needs:
1. **Docstring** with brief description
2. **Args section** describing parameters
3. **Returns section** describing output
4. **Example** (for commonly-used APIs)

---

## Reporting Issues

### Bug Reports

Include:
1. Minimal reproducible example
2. Expected vs actual behavior
3. Python version and OS
4. Full error traceback

```
Title: [BUG] Control layer not gating learning

Description:
When novelty is 0.2 (below threshold of 0.5), learning should be OFF.
Instead, learning is ON.

To reproduce:
```python
control = ControlLayer(net, novelty_threshold=0.5)
novelty = control.detect_novelty({1: 0.1, 2: 0.2})  # 0.2
should_learn = control.should_learn(novelty=0.2, confidence=1.0, reward=0.0)
assert not should_learn  # Fails! should_learn is True
```

Expected: should_learn = False
Actual: should_learn = True

Environment: Python 3.11, Windows 11
```

### Feature Requests

Include:
1. Clear description of desired behavior
2. Use case/motivation
3. Proposed implementation (optional)

```
Title: [FEATURE] Add Hebbian learning rule

Description:
Current system only supports STDP. Would like Hebbian learning:
ΔW = learning_rate * pre * post

This would allow experiments without dopamine signal.

Use case:
Unsupervised learning in early visual cortex simulation.
```

---

## Review Process

1. Create feature branch
2. Make changes and commit with clear messages
3. Add tests for all new code
4. Verify: `python -m unittest discover -s tests -p "test_*.py"`
5. Update docs
6. Submit pull request

---

## Areas for Contribution

### High Priority
- [ ] Performance optimization (faster spike routing)
- [ ] New experiments (novel learning tasks)
- [ ] Better visualization tools
- [ ] GPU acceleration (if using numpy in future)

### Medium Priority
- [ ] Additional learning rules (BCM, Oja's rule)
- [ ] Hardware integration (Loihi, TrueNorth)
- [ ] Noise models (realistic spiking)
- [ ] Documentation improvements

### Lower Priority
- [ ] Distributed simulation (multi-node)
- [ ] Advanced visualization (3D network)
- [ ] Evolutionary search for architectures

---

## Questions?

- Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions
- Read [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for API details
- Look at existing experiments for patterns
- Review tests to understand expected behavior

---

**Thank you for contributing! 🧠⚡**

Last Updated: January 17, 2026
