# Contributing to NCGN

Thank you for your interest in contributing to the Neuromorphic Cognitive Graph Network! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Standards
- ✓ Use welcoming and inclusive language
- ✓ Be respectful of differing viewpoints
- ✓ Accept constructive criticism gracefully
- ✓ Focus on what is best for the community
- ✗ No harassment, trolling, or derogatory comments
- ✗ No publishing others' private information

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub (click "Fork" button)

# Clone your fork
git clone https://github.com/YOUR_USERNAME/Node_network.git
cd Node_network

# Add upstream remote
git remote add upstream https://github.com/shiva2321/Node_network.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check .
flake8 .

# Run the demo
python main.py
```

## Development Workflow

### Branch Strategy

```bash
# Always start from main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

**Examples**:
- `feature/add-move-schema`
- `fix/kwta-heap-bug`
- `docs/improve-architecture-guide`
- `refactor/system1-phases`
- `test/dialogue-state-machine`

### Making Changes

```bash
# Make your changes
# Edit files...

# Check what changed
git status
git diff

# Stage changes
git add <file1> <file2>

# Commit with clear message
git commit -m "Brief description of change

Detailed explanation if needed:
- What changed
- Why it changed
- How it was tested"

# Push to your fork
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# ✓ Good
def calculate_surprise(predicted: Dict[str, float], 
                      observed: Dict[str, float],
                      confidence: Dict[str, float]) -> float:
    """
    Calculate RMS surprise between predicted and observed states.
    
    Args:
        predicted: Expected energy values per node
        observed: Actual energy values per node
        confidence: Confidence scores per node
        
    Returns:
        float: RMS surprise value [0, ∞)
        
    Example:
        >>> pred = {"meat": 0.9}
        >>> obs = {"metal": 1.0}
        >>> conf = {"meat": 0.9}
        >>> calculate_surprise(pred, obs, conf)
        0.81
    """
    squared_diffs = []
    for node_id, pred_energy in predicted.items():
        obs_energy = observed.get(node_id, 0.0)
        conf_value = confidence.get(node_id, 1.0)
        diff = (pred_energy - obs_energy) * conf_value
        squared_diffs.append(diff ** 2)
    
    return math.sqrt(sum(squared_diffs))


# ✗ Bad
def calc_s(p,o,c):  # No docstring, unclear names
    s=0
    for k in p:  # Single-letter variable
        s+=(p[k]-o.get(k,0)*c.get(k,1))**2  # Complex one-liner
    return s**0.5
```

### Code Style Rules

| Rule | Example |
|------|---------|
| **Indentation** | 4 spaces (no tabs) |
| **Line Length** | Max 88 characters (Black default) |
| **Naming** | `snake_case` for functions/variables<br>`PascalCase` for classes<br>`UPPER_CASE` for constants |
| **Type Hints** | Required for public functions |
| **Docstrings** | Google style, all public functions |
| **Imports** | Grouped: stdlib, third-party, local |

### Type Hints

```python
from typing import Dict, List, Set, Optional, Tuple

# Function type hints
def add_node(self, 
             node_id: str, 
             energy: float = 0.0,
             threshold: float = 0.75) -> None:
    """Add a new concept node."""
    ...

# Return types
def get_active_nodes(self) -> Set[str]:
    """Return set of active node IDs."""
    return self._active_nodes

# Optional types
def get_node(self, node_id: str) -> Optional[ConceptNode]:
    """Get node by ID, or None if not found."""
    return self._nodes.get(node_id)
```

### Docstring Format

```python
def method_name(param1: str, param2: int) -> bool:
    """
    One-line summary of what the method does.
    
    Longer description with more details about the method's behavior,
    edge cases, and any important notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When param2 is negative
        KeyError: When param1 doesn't exist
        
    Example:
        >>> obj.method_name("test", 42)
        True
        
    Note:
        Any important notes or warnings.
    """
    ...
```

## Testing Guidelines

### Writing Tests

Tests go in `/tests` directory with naming pattern `test_*.py`.

```python
# tests/test_memory.py
import pytest
from core.memory import GraphMemory, ConceptNode

class TestGraphMemory:
    """Test suite for GraphMemory."""
    
    def setup_method(self):
        """Run before each test method."""
        self.memory = GraphMemory()
    
    def test_add_node(self):
        """Test adding a new node."""
        # Arrange
        node_id = "test_node"
        
        # Act
        self.memory.add_node(node_id, energy=0.5)
        
        # Assert
        assert node_id in self.memory.nodes
        assert self.memory.get_node(node_id).energy == 0.5
    
    def test_add_duplicate_node_raises_error(self):
        """Test that adding duplicate node raises error."""
        self.memory.add_node("node1")
        
        with pytest.raises(ValueError):
            self.memory.add_node("node1")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty string
        with pytest.raises(ValueError):
            self.memory.add_node("")
        
        # Energy out of bounds
        with pytest.raises(ValueError):
            self.memory.add_node("node", energy=1.5)
```

### Test Organization

```
tests/
├── test_physics.py          # System 1 physics tests
├── test_system2.py          # System 2 logic tests  
├── test_memory.py           # Graph memory tests
├── test_conversation.py     # Dialogue tests
├── capability_tests/
│   ├── test_capabilities.py # End-to-end tests
│   └── test_scenarios.py    # Scenario tests
└── fixtures/
    └── test_data.json       # Test data
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_memory.py -v

# Specific test
pytest tests/test_memory.py::TestGraphMemory::test_add_node -v

# With coverage
pytest tests/ --cov=core --cov=cortex --cov-report=html

# View coverage
open htmlcov/index.html
```

### Test Coverage Goals

- **Core modules**: Minimum 80% coverage
- **Cortex modules**: Minimum 70% coverage
- **UI modules**: Minimum 50% coverage

## Documentation

### Code Documentation

```python
# Module-level docstring (top of file)
"""
Module for managing graph memory structures.

This module provides the core data structures for storing concepts,
associations, and schemas in a graph-based memory system.
"""

# Class documentation
class ConceptNode:
    """
    Represents a single concept in the knowledge graph.
    
    A ConceptNode is analogous to a neuron, with energy representing
    its activation level. When energy exceeds the threshold, the node
    "fires" and propagates signals to connected nodes.
    
    Attributes:
        energy: Current activation level [0.0, 1.0]
        threshold: Firing threshold (default 0.75)
        novelty_score: How "new" this concept is [0.0, 1.0]
        refractory_timer: Ticks until can fire again
        
    Example:
        >>> node = ConceptNode("dog", energy=0.5)
        >>> node.energy > node.threshold
        False
    """
```

### Documentation Files

When adding features, update relevant documentation:

- **README.md**: High-level overview, quick start
- **ARCHITECTURE.md**: Technical architecture details
- **WORKFLOW.md**: Process flows and diagrams
- **DOCUMENTATION.md**: Complete technical spec
- **docs/*.md**: Specific guides (User, Technical, etc.)

### Writing Good Documentation

```markdown
# ✓ Good Documentation

## Feature Name

**Purpose**: Clear statement of what this does

**When to use**: Explain use cases

**Example**:
```python
# Concrete, runnable example
memory = GraphMemory()
memory.add_node("dog")
```

**Parameters**:
- `param1` (type): What it does
- `param2` (type): What it does

**Returns**: What you get back

**Notes**: Important caveats or tips


# ✗ Bad Documentation

## Thing

It does stuff.
```

## Pull Request Process

### Before Submitting

1. ✓ **Test your changes**
   ```bash
   pytest tests/ -v
   ```

2. ✓ **Check code style**
   ```bash
   black .
   flake8 .
   ```

3. ✓ **Update documentation** if needed

4. ✓ **Add tests** for new features

5. ✓ **Rebase on main**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Creating Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

2. **Open PR on GitHub**
   - Go to your fork on GitHub
   - Click "Pull Request"
   - Select base: `main` ← compare: `your-branch`

3. **Fill out PR template**

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Changes Made
- Bullet list of changes
- Be specific about what changed

## Testing
- [ ] All tests pass
- [ ] Added new tests for new features
- [ ] Manual testing done

## Screenshots (if applicable)
Attach screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### PR Review Process

1. **Automated checks** run (tests, linting)
2. **Maintainer reviews** code
3. **Feedback addressed** if needed
4. **Approved and merged**

### Responding to Feedback

```bash
# Make requested changes
# Edit files...

# Commit changes
git add .
git commit -m "Address review feedback

- Fixed naming in system1.py
- Added docstring to calculate_surprise
- Updated tests"

# Push to update PR
git push origin feature/your-feature
```

## Areas for Contribution

### 🧠 Core System

**Difficulty: Advanced**

Ideas:
- New learning algorithms (Hebbian, STDP)
- Memory consolidation mechanisms
- Attention variants (soft attention, transformer-style)
- Parallel execution optimizations

Example:
```python
# Add new phase to System1Engine
def _phase_9_memory_consolidation(self):
    """
    Consolidate working memory to long-term storage.
    Transfer high-activation patterns to persistent memory.
    """
    ...
```

### 📚 Schemas & Knowledge

**Difficulty: Beginner**

Ideas:
- New action schemas (fly, swim, think, etc.)
- Domain-specific knowledge (biology, physics, math)
- Ontology hierarchies (animal kingdom, object types)

Example:
```json
// core/schemas/fly.json
{
  "id": "schema_fly",
  "action": "fly",
  "confidence": 0.90,
  "roles": {
    "agent": "flying_object"
  },
  "constraints": {
    "agent": ["can_fly", "has_wings"]
  }
}
```

### 🎨 Visualization

**Difficulty: Intermediate**

Ideas:
- 3D graph visualization
- Animation improvements
- Energy flow visualization
- Surprise spike indicators
- Time-series plots

Example:
```python
# ui/graph_visualizer.py
def draw_energy_flow(self, source: str, target: str, 
                     signal: float):
    """
    Animate energy flowing from source to target.
    Shows spike traveling along edge.
    """
    ...
```

### 🧪 Training & Evaluation

**Difficulty: Intermediate**

Ideas:
- New curricula (language, logic, reasoning)
- Evaluation metrics (accuracy, speed, memory usage)
- Benchmark datasets
- Transfer learning experiments

Example:
```python
# training/curricula/logic.json
{
  "name": "Basic Logic",
  "episodes": [
    {
      "input": ["A", "implies", "B"],
      "target": ["if_A_then_B"],
      "label": "Implication"
    }
  ]
}
```

### 📖 Documentation

**Difficulty: Beginner**

Ideas:
- More examples
- Tutorial videos (links/transcripts)
- FAQ section
- Troubleshooting guides
- Translation to other languages

### 🐛 Bug Fixes

**Difficulty: Varies**

Check [GitHub Issues](https://github.com/shiva2321/Node_network/issues) for:
- Bug reports
- Performance issues
- Edge cases
- Platform-specific problems

## Development Tips

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in code
def tick(self):
    print(f"[DEBUG] Tick {self.current_tick}")
    print(f"[DEBUG] Active nodes: {self.memory.get_active_nodes()}")
    ...
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Run your code
engine.tick()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

### Testing Locally

```bash
# Quick smoke test
python main.py

# Full test suite
pytest tests/ -v

# Test specific scenario
python -c "
from core.memory import GraphMemory
from core.system1 import System1Engine

memory = GraphMemory()
engine = System1Engine(memory)

# Your test code here
"
```

## Questions?

- 📖 Read the docs: [docs/](docs/)
- 💬 Ask in [GitHub Discussions](https://github.com/shiva2321/Node_network/discussions)
- 🐛 Report bugs: [GitHub Issues](https://github.com/shiva2321/Node_network/issues)
- 📧 Email: shiva2321@github.com

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to NCGN! 🧠✨
