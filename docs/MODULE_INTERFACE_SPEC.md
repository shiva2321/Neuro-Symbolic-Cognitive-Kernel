# NSCK Module Interface Specification
**Version:** 2.0 (Substrate Architecture)  
**Date:** February 12, 2026  
**Status:** ✅ Active Standard

## Overview

This document defines the **standardized interface** that all NSCK cognitive modules must implement to participate in the Global Workspace architecture. This interface enables:

- **Decoupled Architecture:** Modules interact only through the GlobalWorkspace message bus
- **Competitive Selection:** Multiple modules propose actions; best proposal wins via activation scoring
- **Observability:** Standardized telemetry for debugging and dashboard visualization
- **Extensibility:** External developers can create plugins without modifying core engine

This spec is implemented by the Module SDK (`nsck-demo/nsck_sdk/`) and the discovery tooling in `nsck-demo/python/module_registry.py`.

## Core Concepts

### The Global Workspace Theory (GWT)

NSCK implements a computational model inspired by Bernard Baars' Global Workspace Theory:

1. **Specialized Modules** operate in parallel, each solving specific problems (planning, memory, emotion, etc.)
2. **Competitive Selection** determines which module's proposal gains "consciousness" (broadcast rights)
3. **Global Broadcast** shares the winning proposal with all modules for coordination
4. **Learning Loop** provides feedback after action execution for continuous improvement

### The WorkspaceModule Interface

**Location:** `nsck-demo/python/global_workspace.py`

All cognitive modules must inherit from `WorkspaceModule(ABC)` and implement 4 required methods:

```python
from global_workspace import WorkspaceModule, Coalition
import numpy as np
from typing import Optional, Dict, Any

class MyCustomModule(WorkspaceModule):
    def receive_broadcast(self, content: Any):
        """React to winning coalition's broadcast"""
        pass
    
    def propose(self, state_hv: Optional[np.ndarray]) -> Optional[Coalition]:
        """Generate action proposal for current state"""
        pass
    
    def update(self, feedback_hv: Optional[np.ndarray], reward: float, info: Dict[str, Any]):
        """Learn from action outcome"""
        pass
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Report current status"""
        pass
```

### Module Discovery (ModuleRegistry)

External modules can be auto-discovered and instantiated with dependency injection:

```python
from python.module_registry import ModuleRegistry

registry = ModuleRegistry()
registry.discover_modules("./plugins")

for module_class in registry.get_modules():
    instance = registry.instantiate(module_class, brain_store=brain)
    engine.register_module(instance)
```

Reference implementations live in `nsck-demo/nsck_sdk/`:
- `security_monitor.py` (safety vetoes)
- `custom_planner.py` (goal-directed planning)
- `domain_expert.py` (pattern-action expertise)

## Method Specifications

### 1. `receive_broadcast(content: Any)`

**Purpose:** Receive notifications when another module wins the Global Workspace competition.

**When Called:** After every `GlobalWorkspace.compete()` cycle, sent to ALL registered modules.

**Parameters:**
- `content`: Arbitrary data from the winning Coalition (typically Dict with 'action', 'reasoning', etc.)

**Responsibilities:**
- Update internal state based on winner's content
- Adjust future proposals based on observed decisions
- Enable inter-module coordination (e.g., Planner notifies Emotion system of goal changes)

**Best Practices:**
- Keep processing lightweight (called on every decision)
- Don't raise exceptions (would disrupt other modules)
- Log important state changes for debugging

**Example:**
```python
def receive_broadcast(self, content: Any):
    if isinstance(content, dict) and 'action' in content:
        # Update internal model with observed action
        self.last_observed_action = content['action']
        if content.get('winner') == 'RuleLearner':
            self.rule_based_decision_count += 1
```

---

### 2. `propose(state_hv: Optional[np.ndarray]) -> Optional[Coalition]`

**Purpose:** Generate a proposal (Coalition) for the current decision cycle.

**When Called:** Every decision cycle by `CognitiveEngine` (typically 10-30 Hz depending on environment).

**Parameters:**
- `state_hv`: Current environment/agent state encoded as 10,240-bit hypervector (NumPy array), or `None` if encoding unavailable

**Returns:**
- `Coalition` object if module has a suggestion
- `None` if module has nothing to contribute this cycle (common for dormant modules)

**Coalition Structure:**
```python
@dataclass
class Coalition:
    source: str              # Module name (e.g., "RuleLearner")
    content: Any             # Proposal details (typically Dict)
    base_salience: float     # Intrinsic loudness [0.0, 1.0]
    relevance: float         # Context match [0.0, 1.0]
    affect_match: float      # Drive/emotion alignment [0.0, 1.0]
    sender_confidence: float # Module's certainty [0.0, 1.0]
    
    @property
    def activation(self) -> float:
        # Final score = salience + relevance + affect + (confidence * 0.5)
        return self.base_salience + self.relevance + self.affect_match + (self.sender_confidence * 0.5)
```

**Scoring Guidelines:**
- **base_salience:** How "loud" is this proposal? Default: 0.5. Urgent/critical: 0.8-1.0. Background: 0.2-0.4.
- **relevance:** How well does it match current goal/context? Use hypervector similarity if available.
- **affect_match:** Does it align with current drives? (e.g., exploration vs. exploitation, fear vs. curiosity)
- **sender_confidence:** Module's certainty in proposal. Low-confidence rules: 0.3-0.5. Verified: 0.7-1.0.

**Best Practices:**
- Return `None` frequently if no strong suggestion (reduces noise)
- High activation = more competitive, but don't "cheat" with inflated scores
- Include reasoning in `content` for explainability
- Use `state_hv` for similarity matching with episodic memory

**Example:**
```python
def propose(self, state_hv: Optional[np.ndarray]) -> Optional[Coalition]:
    # Check if we have a matching rule
    if state_hv is None:
        return None
    
    best_rule = self.find_best_rule(state_hv)
    if best_rule is None:
        return None  # No applicable rule
    
    # Calculate relevance via similarity
    relevance = self.calculate_similarity(state_hv, best_rule.condition_hv)
    
    return Coalition(
        source="RuleLearner",
        content={
            "action": best_rule.action,
            "rule_id": best_rule.id,
            "reasoning": f"Rule #{best_rule.id}: {best_rule.template}"
        },
        base_salience=0.6,
        relevance=relevance,
        affect_match=0.0,
        sender_confidence=best_rule.confidence
    )
```

---

### 3. `update(feedback_hv: Optional[np.ndarray], reward: float, info: Dict[str, Any])`

**Purpose:** Learn from action outcomes to improve future proposals.

**When Called:** After every action execution, sent to ALL modules (not just winner).

**Parameters:**
- `feedback_hv`: Resulting state after action, encoded as hypervector (or `None`)
- `reward`: Scalar reward signal. Convention: [-1.0, 1.0] range
  - Positive: Desired outcome (goal progress, exploration)
  - Negative: Undesired (collision, failure, danger)
  - Zero: Neutral (no change)
- `info`: Dictionary with context:
  - `'winner'`: Name of module that won (str)
  - `'action_taken'`: Actual action executed (str or int)
  - `'success'`: Boolean outcome
  - Environment-specific keys (e.g., `'food_eaten'`, `'collision'`)

**Responsibilities:**
- Update internal models (rules, memory, predictions)
- Adjust confidence scores based on outcomes
- Store episodes for future reasoning
- Even non-winning modules can learn by observing

**Best Practices:**
- Always learn, even if you didn't win (observational learning)
- Update confidence dynamically (+reward = increase, -reward = decrease)
- Store (state, action, outcome) tuples for pattern recognition
- Prune low-performing rules/patterns periodically

**Example:**
```python
def update(self, feedback_hv: Optional[np.ndarray], reward: float, info: Dict[str, Any]):
    action = info.get('action_taken')
    winner = info.get('winner')
    
    # Record episode for all modules (observational learning)
    if self.last_state_hv is not None and feedback_hv is not None:
        episode = Episode(
            state=self.last_state_hv,
            action=action,
            next_state=feedback_hv,
            reward=reward,
            source=winner
        )
        self.episodic_memory.store(episode)
    
    # If we won, update our confidence
    if winner == "RuleLearner" and self.last_proposed_rule is not None:
        if reward > 0:
            self.last_proposed_rule.confidence = min(1.0, self.last_proposed_rule.confidence + 0.05)
        else:
            self.last_proposed_rule.confidence = max(0.1, self.last_proposed_rule.confidence - 0.1)
    
    # Store current state for next update
    self.last_state_hv = feedback_hv
```

---

### 4. `get_telemetry() -> Dict[str, Any]`

**Purpose:** Report current module status for monitoring, debugging, and dashboard visualization.

**When Called:** On-demand by dashboards, loggers, or diagnostic tools (typically 1-10 Hz).

**Returns:** Dictionary with JSON-serializable values (str, int, float, bool, list, dict).

**Standard Keys (Recommended):**
- `'active'` (bool): Is module currently participating in proposals?
- `'proposals_count'` (int): Total proposals submitted since initialization
- `'wins_count'` (int): Number of times this module won competition
- `'win_rate'` (float): wins_count / proposals_count (if proposals > 0)
- `'confidence'` (float): Current confidence level [0.0, 1.0]
- `'memory_size'` (int): Size of internal knowledge base (rules, episodes, concepts)

**Custom Keys (Module-Specific):**
Add domain-specific metrics relevant to your module's function.

**Best Practices:**
- Keep computation lightweight (no heavy processing)
- Return snapshot of current state (don't modify)
- Use consistent key names across modules where applicable
- Include timestamps if reporting historical data

**Example:**
```python
def get_telemetry(self) -> Dict[str, Any]:
    return {
        'active': True,
        'proposals_count': self.total_proposals,
        'wins_count': self.wins,
        'win_rate': self.wins / max(1, self.total_proposals),
        'confidence': self.average_rule_confidence(),
        'memory_size': len(self.rules),
        'rules_induced': len([r for r in self.rules if r.source == 'induction']),
        'rules_tenured': len([r for r in self.rules if r.tenure]),
        'last_rule_time': self.last_induction_timestamp
    }
```

## Registration Process

### Step 1: Implement the Interface

Create your module class inheriting from `WorkspaceModule`:

```python
from global_workspace import WorkspaceModule, Coalition
import numpy as np

class SecurityMonitor(WorkspaceModule):
    def __init__(self):
        self.threat_level = 0.0
        self.alerts = []
    
    def receive_broadcast(self, content: Any):
        # Monitor for dangerous actions
        if isinstance(content, dict):
            action = content.get('action')
            if action in ['approach_danger', 'ignore_warning']:
                self.threat_level += 0.2
    
    def propose(self, state_hv: Optional[np.ndarray]) -> Optional[Coalition]:
        if self.threat_level > 0.5:
            return Coalition(
                source="SecurityMonitor",
                content={"action": "retreat", "reason": "High threat detected"},
                base_salience=0.9,  # High urgency
                relevance=self.threat_level,
                sender_confidence=0.8
            )
        return None
    
    def update(self, feedback_hv: Optional[np.ndarray], reward: float, info: Dict[str, Any]):
        # Decay threat over time if rewards are positive
        if reward > 0:
            self.threat_level = max(0.0, self.threat_level - 0.1)
    
    def get_telemetry(self) -> Dict[str, Any]:
        return {
            'active': True,
            'threat_level': self.threat_level,
            'alerts_count': len(self.alerts)
        }
```

### Step 2: Register with GlobalWorkspace

In `CognitiveEngine` or your initialization code:

```python
# Instantiate module
security_monitor = SecurityMonitor()

# Register with GlobalWorkspace
self.global_workspace.register_module("SecurityMonitor", security_monitor)

# Store reference for proposal collection (optional)
self.security_monitor = security_monitor
```

### Step 3: Participate in Competition Cycle

The `CognitiveEngine` calls modules during decision cycles:

```python
# Collect proposals from all registered modules
proposals = []
for module_name, module in self.global_workspace.modules.items():
    coalition = module.propose(current_state_hv)
    if coalition is not None:
        proposals.append(coalition)

# Run competition
winner = self.global_workspace.compete(proposals)

# Execute action and provide feedback
feedback_hv, reward, info = self.execute(winner.content['action'])

# Update all modules
for module in self.global_workspace.modules.values():
    module.update(feedback_hv, reward, info)
```

## Module Design Patterns

### Pattern 1: Rule-Based Module

**Use Case:** Discrete symbolic rules (if-then logic)

**Key Features:**
- Maintains rule database with confidence scores
- Matches state to rule conditions
- Proposes action from best-matching rule
- Updates confidence based on outcome

**Example:** `RuleLearner`, `CausalReasoner`

---

### Pattern 2: Memory-Based Module

**Use Case:** Episodic recall and case-based reasoning

**Key Features:**
- Stores (state, action, outcome) episodes
- Retrieves similar past experiences via vector similarity
- Proposes action that worked in similar context
- Updates memory with new episodes

**Example:** `EpisodicMemory`, `AnalogyEngine`

---

### Pattern 3: Model-Based Module

**Use Case:** Predictive world model simulation

**Key Features:**
- Maintains forward model (state → action → next_state)
- Simulates outcomes before proposing
- High confidence when prediction accuracy is good
- Updates model parameters from feedback

**Example:** `WorldModel`, `STRIPSPlanner`

---

### Pattern 4: Reactive Module

**Use Case:** Fast heuristic responses (reflexes)

**Key Features:**
- Pattern recognition on state features
- Immediate proposals without heavy computation
- High salience for urgent situations
- Simple win/loss tracking for confidence

**Example:** `EmotionSystem`, `HomeostaticMonitor`

---

### Pattern 5: Meta-Cognitive Module

**Use Case:** Monitor and modulate other modules

**Key Features:**
- Observes win rates and patterns across modules
- Proposes parameter changes or mode switches
- Low proposal frequency, high impact when triggered
- Analyzes telemetry from all modules

**Example:** `Metacognition`, `SelfModifier`, `ConsciousnessMonitor`

## Interface Evolution

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Phase 3.1 (2024) | Initial `WorkspaceModule` with `receive_broadcast()` only |
| 2.0 | Feb 12, 2026 | Added `propose()`, `update()`, `get_telemetry()` for substrate architecture |

### Deprecation Policy

- **Breaking Changes:** Major version bump (e.g., 2.0 → 3.0)
- **Backward Compatibility:** Modules implementing 1.0 will trigger warnings but still function
- **Migration Period:** 6 months for core modules, 12 months for external plugins

### Future Considerations

Potential additions for Version 3.0:
- `initialize(config: Dict[str, Any])` for dynamic configuration
- `shutdown()` for graceful cleanup
- `checkpoint() / restore(data)` for state serialization
- Async variants (`async def propose()`) for long-running reasoning

## Testing Your Module

### Unit Test Template

```python
import pytest
from your_module import YourModule
from global_workspace import Coalition
import numpy as np

def test_module_implements_interface():
    """Verify all required methods exist"""
    module = YourModule()
    assert hasattr(module, 'receive_broadcast')
    assert hasattr(module, 'propose')
    assert hasattr(module, 'update')
    assert hasattr(module, 'get_telemetry')

def test_propose_returns_coalition_or_none():
    """Propose should return Coalition or None"""
    module = YourModule()
    state = np.random.randint(0, 2, 10240, dtype=np.uint8)
    result = module.propose(state)
    assert result is None or isinstance(result, Coalition)

def test_update_does_not_crash():
    """Update should handle various feedback"""
    module = YourModule()
    state = np.random.randint(0, 2, 10240, dtype=np.uint8)
    module.update(state, reward=1.0, info={'winner': 'TestModule'})
    module.update(None, reward=-1.0, info={})  # Handle None state
    module.update(state, reward=0.0, info={'action_taken': 'move_up'})

def test_telemetry_returns_dict():
    """Telemetry should return valid dictionary"""
    module = YourModule()
    telemetry = module.get_telemetry()
    assert isinstance(telemetry, dict)
    assert 'active' in telemetry
    assert isinstance(telemetry['active'], bool)
```

### Integration Test

See `tests/test_module_plugin.py` for full external module integration test (Task 12).

## FAQ

**Q: Can my module store persistent state?**  
A: Yes! Modules are stateful objects. Use `__init__` to load from BrainStore, `update()` to modify, and save periodically.

**Q: What if I don't want to propose every cycle?**  
A: Return `None` from `propose()`. This is common and encouraged—only propose when you have something meaningful.

**Q: Can I call other modules directly?**  
A: No. Inter-module communication must go through `receive_broadcast()`. This maintains decoupling and prevents circular dependencies.

**Q: How do I debug which module is winning?**  
A: Use `get_telemetry()` to track `wins_count`. Enable logging with `logging.getLogger('global_workspace').setLevel(logging.DEBUG)`.

**Q: Can I create modules in other languages (Rust, C++)?**  
A: Not yet in v2.0. Future versions may support IPC-based modules. For now, use Python with optional Rust extensions via PyO3.

**Q: What happens if my module crashes?**  
A: GlobalWorkspace wraps broadcasts in try-except. Your crash won't kill other modules, but you'll be silently skipped. Monitor logs.

## References

- **Implementation:** `nsck-demo/python/global_workspace.py`
- **Examples:** `nsck-demo/python/rule_learner.py`, `episodic_memory.py`
- **Theory:** Baars, B. J. (1988). *A Cognitive Theory of Consciousness*
- **Architecture:** `docs/ARCHITECTURE.md`
- **Substrate Audit:** `docs/SUBSTRATE_AUDIT.md`

---
**Maintained by:** NSCK Core Team  
**Last Updated:** February 12, 2026  
**License:** MIT
