# NSCK Transparency Guarantee

**Version:** Post Q-Learning Integration  
**Date:** February 19, 2026  
**Status:** ✅ FULLY TRANSPARENT (Glass Box System)

---

## Executive Summary

**NSCK remains a completely transparent system** even after adding Q-learning and reward-modulated Hebbian learning. Every decision can be traced end-to-end, explained in natural language, and inspected at the data level.

Unlike deep neural networks that are "black boxes," NSCK is a **glass box** where all reasoning is explicit, symbolic, and interpretable.

---

## Transparency Features

### 1. **Decision Tracing** ✅
Every decision records:
- **Winner Coalition**: Which module made the decision (RULES, EXPLORATION, Q_LEARNING, MEMORY, PLANNER)
- **Competing Proposals**: What alternatives were considered
- **Activation Scores**: Why the winner won (salience, relevance, confidence)
- **Timestamp & Context**: When and in what situation

```python
decision = engine.decide(state, task_tag="maze_navigation")
print(decision.trace)
# → {'winner': 'Q_LEARNING', 'proposals': 3, 'reason': 'Winner: Q_LEARNING (Activation: 0.85)'}
```

### 2. **Natural Language Explanations** ✅
Built-in methods for human-readable explanations:

```python
# Why did you do X?
engine.explain()
# → "I chose move right because the Q-value indicated this leads to the goal"

# Why didn't you do Y?
engine.why_not("ACTION_STAY")
# → "I rejected stay still because exploration suggested movement"

# What if you had done Z?
engine.counterfactual("ACTION_LEFT")
# → "If I had moved left, I would have hit a wall"
```

### 3. **Q-Learning Transparency** ✅
All Q-values are **explicit and inspectable**:

```python
# Inspect all learned state-action values
for (state_key, action), q_value in engine.q_values.items():
    print(f"{state_key}, {action}: Q={q_value:.3f}")

# Check exploration parameters
print(f"Epsilon: {engine.epsilon}")  # Current exploration rate
print(f"Learning rate: {engine.learning_rate}")  # α parameter
print(f"Discount factor: {engine.discount_factor}")  # γ parameter

# See state visit counts
print(f"State visits: {engine.state_visits}")
```

**Q-Learning Formula is Explicit:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
All components (Q-values, rewards, TD errors) are stored and accessible.

### 4. **Episodic Memory Inspection** ✅
Every experience is recorded:

```python
# Access recent episodes by task
recent = engine.episodic_memory.recent["maze_navigation"]
for episode in recent:
    print(f"Action: {episode.action}, Reward: {episode.reward}")

# Full episode history is stored
```

### 5. **Rule Learning Transparency** ✅
Learned rules are symbolic and human-readable:

```python
# Inspect learned rules
for task, rules in engine.rule_learner.learned_rules.items():
    for rule in rules:
        print(f"IF {' AND '.join(rule.antecedent)}")
        print(f"THEN {rule.consequence}")
        print(f"Confidence: {rule.confidence}, Support: {rule.support}")

# See rule candidates being evaluated
candidates = engine.rule_learner.candidates
```

### 6. **Semantic Memory Access** ✅
Concepts stored explicitly as hypervectors:

```python
# Inspect all stored concepts
concepts = engine.semantic_memory.concept_hvs
print(f"Concepts: {list(concepts.keys())}")

# Query similar concepts
similar = engine.semantic_memory.retrieve("apple", top_k=5)
```

### 7. **Global Workspace Logs** ✅
All coalition competitions are recorded:

```python
# Trace history of decisions
for trace in engine.trace_history:
    print(f"[{trace['task']}] Winner: {trace['winner']}")
    print(f"Competing: {trace['proposals']}")
```

### 8. **Statistics Dashboard** ✅
Comprehensive telemetry:

```python
print(engine.stats)
# → {
#     'decisions': 156,
#     'gwt_broadcasts': 156,
#     'rules_induced': 12,
#     'total_reward': 45.2,
#     'reward_count': 50,
#     ...
# }
```

### 9. **State Inspection** ✅
Current cognitive state is fully accessible:

```python
state = engine.current_state
print(f"Task: {state.task_tag}")
print(f"Action: {state.chosen_action}")
print(f"Confidence: {state.confidence}")
print(f"Active Predicates: {state.active_predicates}")
print(f"Explanation: {state.explanation.details}")
```

### 10. **Confidence Tracking** ✅
Self-model maintains confidence estimates:

```python
# Per-task confidence
confidence = engine.self_model.get_confidence("maze_navigation")

# Updated based on outcomes
engine.record_outcome(reward=1.0, task_tag="maze_navigation")
# → Confidence increases
```

---

## Why This Matters

### Comparison with Black-Box Systems

| Feature | NSCK (Glass Box) | Deep Neural Network (Black Box) |
|---------|------------------|----------------------------------|
| **Decision Tracing** | ✅ Explicit coalition winners | ❌ Activations across millions of weights |
| **Explanations** | ✅ Natural language "why" | ❌ Saliency maps at best |
| **Value Inspection** | ✅ Q(s,a) stored explicitly | ❌ Distributed across network |
| **Rule Inspection** | ✅ IF-THEN rules readable | ❌ Implicit in weights |
| **Memory Access** | ✅ Episodes stored symbolically | ❌ Embedded in parameters |
| **Debugging** | ✅ Trace exact reasoning path | ❌ Adversarial examples, failures |
| **Auditability** | ✅ Full decision history | ❌ Limited interpretability |

### Key Advantages

1. **Safety**: Can verify reasoning before deployment
2. **Debugging**: Trace exact failure points
3. **Trust**: Stakeholders can audit decisions
4. **Compliance**: Explain decisions for regulations (GDPR, etc.)
5. **Learning**: Understand what the system knows
6. **Transfer**: Copy specific rules/knowledge between agents

---

## Q-Learning Maintains Transparency

### How Q-Learning Stays Interpretable

**Traditional RL (Opaque):**
```
Deep Q-Network (DQN)
State → [Conv Layer → FC Layer → FC Layer → Q-values]
         ↑ Millions of parameters, uninterpretable
```

**NSCK Q-Learning (Transparent):**
```
State → Hash Function → (state_key, action) → Q-value lookup
        ↑ Symbolic          ↑ Explicit table     ↑ Single float

Q-values stored as: {("maze:(0,0)", "ACTION_DOWN"): 0.394}
```

### Advantages:
- **Exact values**: See `Q(s,a) = 0.394` for any state-action pair
- **Credit assignment**: Trace how rewards propagate through Q-values
- **Exploration tracking**: See epsilon decay from 0.3 → 0.1
- **State aggregation**: Human-readable state keys like `"maze:(0,0)"`

### Debugging Q-Learning

When Q-learning fails, you can:
1. **Inspect Q-values**: See which actions have highest value
2. **Check visit counts**: Identify under-explored states
3. **Examine TD errors**: See if learning is converging
4. **Review reward history**: Verify reward shaping is correct

```python
# Example debugging session
state_key = "maze:(1,2)"
for action in ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]:
    q = engine.q_values.get((state_key, action), 0.0)
    visits = engine.state_visits.get(state_key, 0)
    print(f"{action}: Q={q:.3f}, visits={visits}")
# → ACTION_DOWN: Q=1.853, visits=12  ← Clearly learned!
```

---

## Testing Transparency

Run the transparency test:

```bash
python python/core/tests/test_transparency.py
```

Expected output:
```
✅ RESULT: THIS IS A GLASS BOX SYSTEM

Every decision can be:
  • Traced to its source (which module/coalition won)
  • Explained in natural language (why X? why not Y?)
  • Inspected at the data level (Q-values, rules, memory)
  • Audited via statistics and history logs

Unlike neural networks, there are NO black-box components.
All reasoning is symbolic, explicit, and interpretable.
```

---

## Proof by Test

The transparency test ([test_transparency.py](../python/core/tests/test_transparency.py)) demonstrates:

1. ✅ **Decision tracing** - traces coalition competition
2. ✅ **Natural language explanations** - `explain()`, `why_not()`
3. ✅ **Q-value inspection** - all Q(s,a) pairs accessible
4. ✅ **Episodic memory** - recent experiences stored
5. ✅ **Rule inspection** - IF-THEN rules readable
6. ✅ **Semantic memory** - concept hypervectors accessible
7. ✅ **GWT logs** - coalition competition history
8. ✅ **Statistics** - comprehensive telemetry
9. ✅ **State access** - full cognitive state inspection
10. ✅ **Confidence tracking** - per-task confidence values

---

## Conclusion

**NSCK remains fully transparent** even with Q-learning integrated.

- No hidden layers
- No distributed representations
- No black-box optimization
- All reasoning is **symbolic, explicit, and traceable**

This is a **glass box system** where every decision can be explained end-to-end.

---

## Related Documentation

- [Architecture](ARCHITECTURE.md) — System design with Mermaid diagrams
- [Formulas](FORMULAS.md) — Mathematical foundations
- [Workflows](WORKFLOWS.md) — Decision loop and data flow
- [Module Reference](MODULE_REFERENCE.md) — API reference
- [Repository Map](REPOMAP.md) — Complete codebase map

---

**Last Updated:** February 19, 2026  
**Test Coverage:** All transparency features verified  
**Status:** ✅ Production-ready glass box system

---

## V13 Glass-Box Additions (February 2026)

The following V13 modules extend the glass-box API with explicit uncertainty quantification and auditability:

### 11. KLE Uncertainty in GlobalWorkspace
```python
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition
gw = GlobalWorkspace()
gw.compete([Coalition("A", "act", 0.8), Coalition("B", "act2", 0.4)])
kle = gw.get_kle_uncertainty()   # entropy of competition activations
status = gw.get_status()         # includes "kle_uncertainty" key
```
- `kle_uncertainty` is the Shannon entropy of coalition activation probabilities
- Higher entropy = more uncertainty (more equally competing proposals)
- Fully inspectable via `get_status()["kle_uncertainty"]`

### 12. CognitiveState Uncertainty Fields
```python
from python.core.reasoning.cognitive_engine import CognitiveState
cs = CognitiveState(task_tag="t")
cs.kle_uncertainty   # float or None — competition entropy
cs.uncertainty_bounds  # (lower, upper) conformal bounds or None
```

### 13. CausalRuleAuditor per-rule causal_score
```python
from python.core.reasoning.causal_rule_auditor import CausalRuleAuditor
auditor = CausalRuleAuditor()
auditor.add_causal_edge("rain", "wet", strength=0.9)
rule = auditor.audit_rule("rain", "wet", confidence=0.8)
rule.causal_score    # float [0,1] — how causally grounded the rule is
rule.audit_trace     # List[str] — glass-box explanation
rule.combined_score  # weighted combination
```

### 14. ConformalWrapper calibrated prediction sets
```python
from python.core.learning.conformal_wrapper import ConformalWrapper
wrapper = ConformalWrapper(alpha=0.1)
wrapper.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
lower, upper = wrapper.uncertainty_bound(0.2)
result = wrapper.predict_set(0.15)   # {included, q_hat, coverage=0.9}
```

### 15. SubstrateResult V13 fields (NSCKSubstrate)
```python
result = substrate.ingest("input", "task")
result.kle_uncertainty    # float or None
result.uncertainty_bounds # (lower, upper) or None
result.encoding_stats     # {"source_type": ..., "stats": ..., ...}
result.procedural_hit     # bool — True if served from ProceduralMemory
```

---

**Last Updated:** February 2026  
**V13 glass-box features:** KLE uncertainty, causal audit traces, conformal bounds, encoding stats
