# NSCK Module SDK

**Build custom cognitive modules for NSCK without modifying the core engine.**

## Quick Start

```python
# 1. Study the reference modules (from repo root)
python nsck-demo/nsck_sdk/security_monitor.py    # Safety & vetoes
python nsck-demo/nsck_sdk/custom_planner.py      # Multi-step planning
python nsck-demo/nsck_sdk/domain_expert.py       # Pattern recognition

# 2. Read the developer guide
# See MODULE_DEV_GUIDE.md for step-by-step tutorial

# 3. Build your module (minimal example)
import numpy as np
from typing import Optional, Dict, Any
from global_workspace import WorkspaceModule, Coalition

class MyModule(WorkspaceModule):
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        # Your logic here
        return Coalition(
            source="MyModule",
            content={"action": "jump"},
            base_salience=0.7,
            relevance=0.8,
            sender_confidence=0.9
        )
    
    def update(self, feedback_hv, reward, info):
        pass  # Learn from outcomes
    
    def receive_broadcast(self, content):
        pass  # React to others
    
    def get_telemetry(self):
        return {"status": "ok"}

# 4. Register with ModuleRegistry
from python.module_registry import ModuleRegistry

registry = ModuleRegistry()
registry.register(MyModule)
# Or auto-discover: registry.discover_modules("./my_modules")

# 5. Integrate with CognitiveEngine
from python.cognitive_engine import CognitiveEngine

for module_class in registry.get_modules():
    instance = registry.instantiate(module_class, brain_store=brain)
    engine.register_module(instance)
```

## What's Included

### Reference Modules

**SecurityMonitor** (`security_monitor.py`)
- Monitors for dangerous states
- Issues vetoes (salience=0.0)
- Tracks violations by rule
- **Use for**: Safety-critical systems, constraint enforcement

**CustomPlanner** (`custom_planner.py`)
- Goal-directed planning
- Cost-based action selection
- Multi-step plan execution
- **Use for**: Navigation, task planning, optimization

**DomainExpert** (`domain_expert.py`)
- Pattern-action knowledge base
- Empirical success tracking
- Confidence adaptation
- **Use for**: Expert systems, heuristics, pattern recognition

### ModuleRegistry

Automatic module discovery and validation:
```python
registry = ModuleRegistry()

# Manual registration
registry.register(MyModule, metadata={"author": "Me", "version": "1.0"})

# Auto-discovery from directory
registry.discover_modules("./plugins")

# Instantiation with dependency injection
instance = registry.instantiate(MyModule, brain_store=brain, threshold=0.8)
```

### Developer Guide

`MODULE_DEV_GUIDE.md` - 600+ line comprehensive tutorial:
- WorkspaceModule interface specification
- Building your first module (step-by-step)
- Coalition proposal patterns
- Testing strategies
- Advanced topics (CleanupMemory, episodic memory)
- Best practices & troubleshooting

## Architecture

NSCK uses **Global Workspace Theory** where modules compete for consciousness:

1. **Every cycle**, all modules call `propose(state_hv)`
2. Modules return `Coalition` objects with:
   - `base_salience` (0-1): How important/loud
   - `relevance` (0-1): Relevance to current goal
   - `sender_confidence` (0-1): Module's confidence
3. **Winning coalition** has highest activation (salience + relevance + confidence*0.5)
4. Winner's `content` is broadcast to all modules via `receive_broadcast()`
5. If action taken, winner's `update()` is called with reward

**Modules don't call each other** - they only see:
- Global state (10,240-bit hypervector)
- Broadcasts from winners
- Their own internal state

## Coalition API

```python
Coalition(
    source="ModuleName",              # Your module's name
    content={                         # Arbitrary dict (avoid large objects)
        "type": "action",
        "action_hv": action_vector,
        "description": "move_forward"
    },
    base_salience=0.7,                # How important (0-1)
    relevance=0.8,                    # Relevance to goal (0-1)
    sender_confidence=0.9             # Your confidence (0-1)
)

# Activation = base_salience + relevance + sender_confidence * 0.5
#            = 0.7 + 0.8 + 0.9*0.5 = 1.95
```

## Common Patterns

### Pattern 1: Action Proposal
```python
return Coalition(
    source="Planner",
    content={"action": "move_north"},
    base_salience=0.7,
    relevance=0.9,      # High relevance to goal
    sender_confidence=0.8
)
```

### Pattern 2: Safety Veto
```python
return Coalition(
    source="SafetyMonitor",
    content={"type": "veto", "reason": "Danger detected"},
    base_salience=0.0,  # Zero = veto signal
    relevance=0.0,
    sender_confidence=0.95
)
```

### Pattern 3: Information Broadcast
```python
return Coalition(
    source="Perception",
    content={"type": "percept", "object": "ball", "location": [5, 3]},
    base_salience=0.5,
    relevance=0.6,
    sender_confidence=0.8
)
```

## Hypervectors

NSCK uses **Vector Symbolic Architecture** (VSA):
- All states/concepts are 10,240-bit binary vectors
- Encoded deterministically: `HyperVector(seed).bits`
- Similarity via Hamming distance: `1.0 - (bitwise_xor_count / 10240)`
- Binding via XOR: `A ⊗ B = A.xor(B)`
- Bundling via majority vote: `A + B = (A & B) | (A ^ B & random)`

**Example**:
```python
import hashlib
import hypervec_shim as hv

def encode(label: str) -> np.ndarray:
    """Deterministic encoding."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "little")
    return hv.HyperVector(seed).bits

def similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Compute Hamming similarity."""
    diff = np.bitwise_xor(hv1, hv2)
    return 1.0 - (np.sum(diff) / len(hv1))

# Usage
obstacle_hv = encode("obstacle")
state_similarity = similarity(state_hv, obstacle_hv)
if state_similarity > 0.7:
    # Danger!
```

## Testing

```python
import pytest
from my_module import MyModule
import numpy as np

def test_propose_returns_coalition():
    module = MyModule()
    state_hv = np.random.randint(0, 2, 10240, dtype=np.int8)
    
    coalition = module.propose(state_hv)
    
    assert coalition is not None
    assert coalition.source == "MyModule"
    assert 0.0 <= coalition.base_salience <= 1.0

def test_update_improves_confidence():
    module = MyModule()
    module.confidence = 0.5
    
    feedback_hv = np.zeros(10240, dtype=np.int8)
    module.update(feedback_hv, reward=10.0, info={})
    
    assert module.confidence > 0.5

def test_telemetry_has_expected_keys():
    module = MyModule()
    telemetry = module.get_telemetry()
    
    assert "proposal_count" in telemetry
```

## Requirements

- Python 3.8+
- NumPy
- NSCK (parent directory)

No external ML dependencies (PyTorch, TensorFlow, etc.) - pure symbolic!

## Resources

- **Developer Guide**: `MODULE_DEV_GUIDE.md` - Start here!
- **Module Interface Spec**: `../docs/MODULE_INTERFACE_SPEC.md`
- **VSA Theory**: `../docs/VSA_THEORY.md`
- **System Architecture**: `../docs/ARCHITECTURE.md`
- **Example Modules**: `security_monitor.py`, `custom_planner.py`, `domain_expert.py`

## FAQ

**Q: Do I need to modify global_workspace.py?**  
A: No! Just implement WorkspaceModule interface and register.

**Q: How do I communicate with other modules?**  
A: Via `receive_broadcast()` when they win, or via shared BrainStore.

**Q: Can my module call other modules directly?**  
A: No - modules are loosely coupled. Use broadcasts or shared state.

**Q: How do I debug my module?**  
A: Use `get_telemetry()` to expose internal state, check dashboard.

**Q: What if my module always loses competition?**  
A: Increase base_salience or relevance, or make pattern matching more selective.

**Q: Can I use ML models in my module?**  
A: Yes, but NSCK is designed for symbolic reasoning. Keep models small.

**Q: How do I distribute my module?**  
A: Single .py file with `__author__` and `__version__` metadata. Users drop it in a plugins directory.

## Contributing

Built a cool module? Share it with the community!
- Add it to this SDK directory
- Include demo in `if __name__ == "__main__"` block
- Add tests to `tests/test_my_module.py`
- Update this README

## License

Same as NSCK parent project.

---

**Happy Module Building!** 🚀

For questions, see `MODULE_DEV_GUIDE.md` or open an issue.
