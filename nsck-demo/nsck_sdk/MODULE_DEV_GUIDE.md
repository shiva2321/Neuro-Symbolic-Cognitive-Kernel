"""
NSCK Module Development Guide
==============================

A step-by-step guide to building custom cognitive modules for NSCK.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [WorkspaceModule Interface](#workspacemodule-interface)
4. [Building Your First Module](#building-your-first-module)
5. [Coalition Proposal Patterns](#coalition-proposal-patterns)
6. [Example Modules](#example-modules)
7. [Testing Your Module](#testing-your-module)
8. [Integration with CognitiveEngine](#integration-with-cognitiveengine)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

---

## Introduction

NSCK (Neuro-Symbolic Cognitive Kernel) uses a **Global Workspace Architecture** where
multiple cognitive modules compete for consciousness. Each module:

- **Proposes** actions/ideas based on current state
- **Competes** with other modules via Coalition activation
- **Learns** from feedback/rewards
- **Broadcasts** its winning ideas to the workspace

Modules are **loosely coupled** - they don't call each other directly. Instead, they:
1. Receive the global state (10,240-bit hypervector)
2. Propose coalitions with salience/relevance/confidence
3. The highest-activation coalition wins
4. Winners broadcast to all modules

This guide shows you how to build custom modules that integrate seamlessly.

---

## Quick Start

### Installation

```bash
# Clone NSCK repository
git clone <repo_url>
cd nsck-demo

# Install dependencies
pip install numpy

# Test the SDK examples
python nsck_sdk/security_monitor.py
python nsck_sdk/custom_planner.py
python nsck_sdk/domain_expert.py
```

### Minimal Module Example

```python
import numpy as np
from typing import Optional, Dict, Any
from python.core.reasoning.global_workspace import WorkspaceModule, Coalition
import python.core.vsa.hypervec_shim as hv

class MyFirstModule(WorkspaceModule):
    \"\"\"A simple module that detects when the agent is tired.\"\"\"
    
    def __init__(self):
        self.proposal_count = 0
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        \"\"\"Propose 'rest' action if tired.\"\"\"
        self.proposal_count += 1
        
        # Encode "tired" concept
        tired_hv = hv.HyperVector(12345).bits
        
        # Check similarity
        diff = np.bitwise_xor(state_hv, tired_hv)
        similarity = 1.0 - (np.sum(diff) / len(state_hv))
        
        if similarity > 0.7:
            return Coalition(
                source="MyFirstModule",
                content={"type": "action", "action": "rest"},
                base_salience=0.8,
                relevance=0.6,
                sender_confidence=similarity
            )
        return None
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        \"\"\"Learn from feedback (optional for simple modules).\"\"\"
        pass
    
    def receive_broadcast(self, content: Any):
        \"\"\"React to winning coalitions from other modules.\"\"\"
        pass
    
    def get_telemetry(self) -> Dict[str, Any]:
        \"\"\"Return diagnostic information.\"\"\"
        return {"proposal_count": self.proposal_count}
```

---

## WorkspaceModule Interface

All modules must inherit from `WorkspaceModule` and implement 4 methods:

### 1. `propose(state_hv) -> Optional[Coalition]`

**When called**: Every cognitive cycle (10-100 Hz)

**Purpose**: Evaluate current state and propose action/idea

**Parameters**:
- `state_hv`: Current workspace state (10,240-bit np.ndarray of int8)

**Returns**:
- `Coalition` object if module has something to propose
- `None` if module has nothing to contribute

**Example**:
```python
def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
    # Pattern matching
    similarity = self._check_pattern(state_hv)
    
    if similarity > self.threshold:
        return Coalition(
            source="MyModule",
            content={"action": "jump"},
            base_salience=0.7,        # How loud/important (0-1)
            relevance=0.5,            # Relevance to current goal (0-1)
            sender_confidence=0.9     # Module's confidence (0-1)
        )
    return None
```

### 2. `update(feedback_hv, reward, info)`

**When called**: After module's coalition wins and executes

**Purpose**: Learn from outcomes to improve future proposals

**Parameters**:
- `feedback_hv`: Actual resulting state (10,240-bit np.ndarray)
- `reward`: Scalar reward (-inf to +inf, typically -10 to +10)
- `info`: Dictionary with additional information

**Use cases**:
- Adjust pattern confidence based on success
- Update cost estimates
- Refine heuristics
- Track win/loss statistics

**Example**:
```python
def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
    if reward > 0:
        self.success_count += 1
        self.confidence = min(1.0, self.confidence + 0.1)
    elif reward < 0:
        self.failure_count += 1
        self.confidence = max(0.1, self.confidence - 0.05)
```

### 3. `receive_broadcast(content)`

**When called**: Every time ANY module wins

**Purpose**: React to global events/decisions

**Parameters**:
- `content`: Whatever the winning coalition put in its `content` field

**Use cases**:
- Update beliefs based on other modules' decisions
- Coordinate with other modules
- Track what actions are being taken
- Adjust strategies based on environment changes

**Example**:
```python
def receive_broadcast(self, content: Any):
    if isinstance(content, dict) and content.get("type") == "danger_alert":
        self.danger_level = content.get("severity", 0.5)
        print(f"⚠️  {self.__class__.__name__} received danger alert!")
```

### 4. `get_telemetry() -> Dict[str, Any]`

**When called**: For debugging/dashboards (not every cycle)

**Purpose**: Return diagnostic information

**Returns**: Dictionary with any useful metrics

**Best practices**:
- Include counters (proposals, wins, updates)
- Include rates (success_rate, win_rate)
- Include last_action/last_pattern for debugging
- Keep it lightweight (avoid expensive computation)

**Example**:
```python
def get_telemetry(self) -> Dict[str, Any]:
    return {
        "total_proposals": self.total_proposals,
        "total_wins": self.total_wins,
        "win_rate": self.total_wins / self.total_proposals if self.total_proposals > 0 else 0.0,
        "avg_confidence": np.mean(self.confidence_history) if self.confidence_history else 0.0,
        "last_action": self.last_action
    }
```

---

## Building Your First Module

Let's build a **CollisionDetector** module that vetoes dangerous actions.

### Step 1: Set up the skeleton

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from typing import Optional, Dict, Any
from global_workspace import WorkspaceModule, Coalition
import hypervec_shim as hv
import hashlib

class CollisionDetector(WorkspaceModule):
    \"\"\"Detects potential collisions and vetoes risky actions.\"\"\"
    
    def __init__(self, look_ahead_distance: float = 2.0):
        self.look_ahead_distance = look_ahead_distance
        self.vetoes_issued = 0
```

### Step 2: Implement propose()

```python
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        \"\"\"Check if current action will cause collision.\"\"\"
        
        # Encode "obstacle_ahead" pattern
        obstacle_hv = self._encode("obstacle_ahead").bits
        
        # Compute similarity
        diff = np.bitwise_xor(state_hv, obstacle_hv)
        similarity = 1.0 - (np.sum(diff) / len(state_hv))
        
        # If obstacle detected, issue veto (low salience = veto)
        if similarity > 0.7:
            self.vetoes_issued += 1
            return Coalition(
                source="CollisionDetector",
                content={"type": "veto", "reason": "Obstacle ahead"},
                base_salience=0.0,  # Zero salience = strong veto
                relevance=0.0,
                sender_confidence=similarity
            )
        
        return None  # Safe - no proposal
    
    def _encode(self, label: str) -> hv.HyperVector:
        \"\"\"Deterministic encoding from string.\"\"\"
        h = hashlib.sha256(label.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        return hv.HyperVector(seed)
```

### Step 3: Implement update() (optional)

```python
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        \"\"\"Learn from near-misses.\"\"\"
        if reward < -5:
            # Collision happened despite veto - increase sensitivity
            self.look_ahead_distance *= 1.1
        elif reward > 0:
            # Safe action - our veto might be too sensitive
            pass  # Keep current sensitivity
```

### Step 4: Implement receive_broadcast()

```python
    def receive_broadcast(self, content: Any):
        \"\"\"Track what actions are being taken.\"\"\"
        if isinstance(content, dict) and content.get("type") == "movement":
            # Could track movement patterns here
            pass
```

### Step 5: Implement get_telemetry()

```python
    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "vetoes_issued": self.vetoes_issued,
            "look_ahead_distance": self.look_ahead_distance
        }
```

### Step 6: Test it

```python
if __name__ == "__main__":
    detector = CollisionDetector()
    
    # Test safe state
    safe_state = detector._encode("clear_path").bits
    result = detector.propose(safe_state)
    print(f"Safe state: {result}")  # Should be None
    
    # Test dangerous state
    danger_state = detector._encode("obstacle_ahead").bits
    result = detector.propose(danger_state)
    print(f"Danger state: {result}")  # Should return veto Coalition
    print(f"Telemetry: {detector.get_telemetry()}")
```

---

## Coalition Proposal Patterns

### Pattern 1: Action Proposal

**Use case**: Module wants to execute an action

```python
return Coalition(
    source="MyModule",
    content={
        "type": "action",
        "action_hv": action_vector,
        "description": "move_forward"
    },
    base_salience=0.7,      # How important
    relevance=0.8,          # Relevance to goal
    sender_confidence=0.9   # Confidence in success
)
```

**Activation**: 0.7 + 0.8 + 0.9*0.5 = 1.95

### Pattern 2: Veto (Negative Proposal)

**Use case**: Module wants to prevent a dangerous action

```python
return Coalition(
    source="SafetyModule",
    content={"type": "veto", "reason": "Too dangerous"},
    base_salience=0.0,      # Minimum salience = veto signal
    relevance=0.0,
    sender_confidence=0.95  # Confidence in danger detection
)
```

**Activation**: 0.0 + 0.0 + 0.95*0.5 = 0.475 (low but non-zero)

### Pattern 3: Information Broadcast

**Use case**: Module wants to share information without acting

```python
return Coalition(
    source="PerceptionModule",
    content={
        "type": "percept",
        "object": "red_ball",
        "location": [5.0, 3.2]
    },
    base_salience=0.5,      # Moderate importance
    relevance=0.6,
    sender_confidence=0.8
)
```

### Pattern 4: Goal Pursuit

**Use case**: Module has multi-step plan towards goal

```python
return Coalition(
    source="PlannerModule",
    content={
        "type": "subgoal",
        "action_hv": next_action,
        "progress": 0.75,  # 75% towards goal
        "remaining_steps": 3
    },
    base_salience=0.6,
    relevance=0.9,          # High relevance to goal
    sender_confidence=1.0 / (remaining_steps + 1)  # Less confident with more steps
)
```

---

## Example Modules

The SDK includes 3 reference implementations:

### 1. SecurityMonitor

**Purpose**: Anomaly detection and safety vetoes

**Key features**:
- Pattern-based safety rules
- Severity-weighted vetos
- Violation tracking

**When to use this pattern**:
- Safety-critical applications
- Constraint enforcement
- Rule-based vetoes

**See**: [nsck_sdk/security_monitor.py](nsck_sdk/security_monitor.py)

### 2. CustomPlanner

**Purpose**: Goal-directed multi-step planning

**Key features**:
- State-action simulation
- Cost-based bidding
- Plan refinement from feedback

**When to use this pattern**:
- Navigation tasks
- Multi-step goals
- Resource optimization

**See**: [nsck_sdk/custom_planner.py](nsck_sdk/custom_planner.py)

### 3. DomainExpert

**Purpose**: Domain-specific pattern-action knowledge

**Key features**:
- Pattern library with confidences
- Empirical success tracking
- Confidence adaptation

**When to use this pattern**:
- Expert systems
- Domain-specific heuristics
- Pattern recognition tasks

**See**: [nsck_sdk/domain_expert.py](nsck_sdk/domain_expert.py)

---

## Testing Your Module

### Unit Testing

```python
import pytest
import numpy as np
from my_module import MyModule

def test_propose_returns_coalition_when_pattern_matches():
    module = MyModule()
    
    # Create test state
    state_hv = np.random.randint(0, 2, 10240, dtype=np.int8)
    
    # Should return coalition
    coalition = module.propose(state_hv)
    
    assert coalition is not None
    assert coalition.source == "MyModule"
    assert "action" in coalition.content

def test_update_increases_confidence_on_success():
    module = MyModule()
    module.confidence = 0.5
    
    # Simulate successful outcome
    feedback_hv = np.zeros(10240, dtype=np.int8)
    module.update(feedback_hv, reward=10.0, info={})
    
    assert module.confidence > 0.5

def test_telemetry_includes_expected_metrics():
    module = MyModule()
    telemetry = module.get_telemetry()
    
    assert "proposal_count" in telemetry
    assert "win_rate" in telemetry
```

### Integration Testing with ModuleRegistry

```python
from python.core.integration.module_registry import ModuleRegistry
from my_module import MyModule

def test_module_registration():
    registry = ModuleRegistry()
    registry.register(MyModule)
    
    modules = registry.get_modules()
    assert MyModule in modules

def test_module_instantiation():
    registry = ModuleRegistry()
    registry.register(MyModule)
    
    instance = registry.instantiate(MyModule, config={"threshold": 0.8})
    
    assert instance.threshold == 0.8
```

---

## Integration with CognitiveEngine

### Manual Registration

```python
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.persistence import BrainStore
from my_module import MyModule

# Initialize engine
brain = BrainStore("brain.db")
engine = CognitiveEngine(brain_store=brain)

# Register your module
my_module = MyModule(threshold=0.7)
engine.register_module(my_module)

# Run cycle
state_hv = engine.get_current_state()
engine.step()
```

### Automatic Discovery

```python
from python.core.integration.module_registry import ModuleRegistry
from python.core.reasoning.cognitive_engine import CognitiveEngine

# Discover modules from directory
registry = ModuleRegistry()
registry.discover_modules("./my_custom_modules")

# Register all discovered modules
for module_class in registry.get_modules():
    instance = registry.instantiate(module_class, brain_store=brain)
    engine.register_module(instance)

print(f"Registered {len(registry.get_modules())} custom modules")
```

---

## Advanced Topics

### 1. Using CleanupMemory for VSA Denoising

```python
from python.core.vsa.hypervec_py import CleanupMemory

class AdvancedModule(WorkspaceModule):
    def __init__(self, brain_store):
        self.cleanup = CleanupMemory(max_size=5000)
        
        # Load known concepts from brain
        concepts = brain_store.load_concepts()
        for concept in concepts:
            if concept.vector is not None:
                hv = hv.HyperVector.from_bits(concept.vector)
                self.cleanup.register(concept.name, hv)
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        # Denoise state before processing
        state_obj = hv.HyperVector.from_bits(state_hv)
        clean_state, label = self.cleanup.cleanup(state_obj, threshold=0.6)
        
        if clean_state is not None:
            # Recognized state! Use label for decision
            return self._handle_known_state(label)
        else:
            # Unknown/noisy state
            return None
```

### 2. Learning from Episodic Memory

```python
class LearningModule(WorkspaceModule):
    def __init__(self, brain_store):
        self.brain = brain_store
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        # Store episode for future learning
        episode = {
            "state": info.get("initial_state"),
            "action": info.get("action"),
            "result": feedback_hv,
            "reward": reward,
            "timestamp": time.time()
        }
        
        # If very good or very bad outcome, store as episode
        if abs(reward) > 5.0:
            self.brain.store_episode(
                content_hv=feedback_hv,
                emotion_hv=self._encode_emotion(reward),
                importance=abs(reward) / 10.0
            )
```

### 3. Semantic Roles in Action Proposals

```python
from python.core.language.universal_input import UniversalInput

class LanguageModule(WorkspaceModule):
    def __init__(self):
        self.input_encoder = UniversalInput()
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        # Parse state as language
        sentence = "The robot pushes the box"
        
        # Encode with semantic roles
        encoded = self.input_encoder.encode(sentence)
        
        # Extract action role
        action_role = encoded["semantic_roles"]["action"]
        
        return Coalition(
            source="LanguageModule",
            content={
                "type": "language_action",
                "action_hv": action_role,
                "description": "push"
            },
            base_salience=0.7,
            relevance=0.8,
            sender_confidence=0.9
        )
```

---

## Best Practices

### 1. Proposal Strategy

**DO**:
- ✅ Return `None` when module has nothing to contribute
- ✅ Use appropriate salience (0.0 for veto, 0.5-1.0 for action)
- ✅ Include descriptive content for debugging
- ✅ Scale confidence by pattern match quality

**DON'T**:
- ❌ Propose every cycle (be selective!)
- ❌ Always return maximum salience (competitive balance)
- ❌ Forget to handle `None` returns
- ❌ Put large objects in `content` (use references/indices)

### 2. Learning Strategy

**DO**:
- ✅ Update confidence/parameters gradually (small step sizes)
- ✅ Track success/failure counts for telemetry
- ✅ Use info dict to pass context from proposal to update
- ✅ Handle positive and negative rewards differently

**DON'T**:
- ❌ Make drastic changes from single feedback
- ❌ Assume update() is called after every propose()
- ❌ Ignore reward magnitude (reward=0.1 vs reward=10 matters)
- ❌ Overfits to recent outcomes (use exponential moving average)

### 3. Hypervector Encoding

**DO**:
- ✅ Use deterministic seeding for reproducibility
- ✅ Cache frequently-used encodings
- ✅ Use CleanupMemory for denoising
- ✅ Normalize state before similarity comparison

**DON'T**:
- ❌ Create random vectors (use seeded HyperVector(seed))
- ❌ Compare raw bits directly (use Hamming similarity)
- ❌ Bind too many vectors without cleanup (noise accumulates)
- ❌ Forget that XOR is self-inverse (A ⊗ A = identity)

### 4. Module Architecture

**DO**:
- ✅ Keep modules focused (single responsibility)
- ✅ Use `__init__` parameters for configuration
- ✅ Provide telemetry for debugging
- ✅ Document expected state patterns

**DON'T**:
- ❌ Access other modules directly (use broadcasts)
- ❌ Maintain duplicate state (trust the BrainStore)
- ❌ Block in propose() (must return quickly)
- ❌ Assume module will win (many may compete)

---

## Troubleshooting

### Module never wins

**Symptoms**: Your module proposes but never wins competition

**Causes**:
1. Salience too low (other modules louder)
2. Pattern never matches (check similarity threshold)
3. Returning `None` too often

**Solutions**:
```python
# Add debugging in propose()
coalition = self._create_coalition()
print(f"[MyModule] Proposing with activation={coalition.activation}")
return coalition

# Check telemetry
telemetry = module.get_telemetry()
print(f"Proposal rate: {telemetry['proposals'] / telemetry['cycles']}")
```

### Module wins too often

**Symptoms**: Module dominates, prevents others from winning

**Causes**:
1. Salience too high
2. Pattern matches too often
3. Not respecting goal relevance

**Solutions**:
```python
# Scale by goal relevance
goal_similarity = self._compute_goal_match(state_hv)
salience = base_salience * goal_similarity

# Be more selective
if similarity > self.threshold + 0.2:  # Stricter threshold
    return coalition
```

### Update() never called

**Symptoms**: update() method not receiving feedback

**Causes**:
1. Module's coalitions never win
2. CognitiveEngine not configured to call update()
3. Module returning `None` (no coalition to win)

**Solutions**:
```python
# Track wins in receive_broadcast()
def receive_broadcast(self, content: Any):
    if isinstance(content, dict) and content.get("source") == "MyModule":
        self.win_count += 1
```

---

## Next Steps

1. **Read the example modules**: Start with [security_monitor.py](nsck_sdk/security_monitor.py)
2. **Build a simple module**: Use the CollisionDetector example as template
3. **Test your module**: Run standalone before integrating
4. **Register with engine**: Use ModuleRegistry for discovery
5. **Tune parameters**: Adjust thresholds/salience based on telemetry
6. **Share your module**: Contribute back to the community!

## Additional Resources

- [MODULE_INTERFACE_SPEC.md](../docs/MODULE_INTERFACE_SPEC.md) - Formal specification
- [VSA_THEORY.md](../docs/VSA_THEORY.md) - Hypervector algebra
- [TESTING.md](../docs/TESTING.md) - Testing best practices
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - System architecture

---

**Happy Coding!** 🚀

For questions or contributions, see [DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md).
