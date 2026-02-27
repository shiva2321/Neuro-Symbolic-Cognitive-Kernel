# NSCK V13 — Transparency & Explainability Guarantee

**Version:** 13 (V13)  
**Date:** February 2026  
**Status:** ✅ FULLY TRANSPARENT — Glass-Box System

---

## 1. Executive Summary

NSCK is a **glass-box cognitive architecture**. Every decision records its complete
reasoning chain, every internal data structure is inspectable, and every component
produces human-readable output. This stands in direct contrast to black-box neural
networks, where decisions are buried in millions of opaque weights.

In V13, transparency is extended with **uncertainty quantification** (KLE entropy,
conformal prediction bounds), **causal audit traces**, **encoding statistics**, and
**procedural memory hit indicators** — so you always know *what* the system decided,
*why* it decided it, and *how confident* it is.

---

## 2. Core Transparency Features

These capabilities are always available and form the foundation of NSCK's glass-box
guarantee.

### 2.1 Decision Tracing

Every `CognitiveState` contains a `trace` dict recording the full competition:

- **winner** — which coalition won (e.g. `RULES`, `Q_LEARNING`, `MEMORY`)
- **proposals** — all competing proposals and their activation scores
- **confidence** — the winning coalition's confidence value
- **reason** — human-readable summary of why the winner was selected

### 2.2 Natural Language Explanations

`ExplanationGenerator` produces human-readable summaries. Three entry points:

| Method | Question it answers |
|--------|---------------------|
| `engine.explain_action()` | *"Why did you do X?"* |
| `engine.why_not(action)` | *"Why didn't you do Y?"* |
| `engine.counterfactual(action)` | *"What would have happened if you did Z?"* |

### 2.3 Q-Value Inspection

All Q-values are explicit floats in a lookup table — no hidden weights:

```python
for (state_key, action), q in engine.q_table.items():
    print(f"Q({state_key}, {action}) = {q:.3f}")
```

The update rule is the standard Bellman equation with fully visible parameters
(α, γ, ε).

### 2.4 Rule Inspection

All rules are symbolic and human-readable. Each `Rule` object exposes:

- `condition` — antecedent predicates
- `action` — consequent action
- `confidence` — current confidence score
- `fire_count` — number of times this rule has fired
- `confidence_history` — list of confidence values over time

### 2.5 Semantic Memory Queries

The knowledge graph is fully navigable:

- `get_concept(name)` — retrieve a concept's hypervector and relations
- `add_relation(src, rel, tgt)` — add a typed edge
- `spread_activation(start, depth)` — traverse the graph with activation decay

### 2.6 Episodic Memory Recall

Episodes are stored as `(state, action, reward)` tuples. Retrieval is by
similarity — you can query what the system remembers and why it remembers it.

### 2.7 Causal Graph Inspection

`CausalGraph` stores directed edges with **ΔP** (delta-probability) strengths.
Confounder detection uses mutual information (MI). Every edge is enumerable and
its strength is a plain float.

### 2.8 Self-Model State

`SelfModel` tracks:

- **identity HV** — the system's hypervector self-representation
- **confidence** — per-task confidence estimates
- **role** — current role context

### 2.9 Metacognitive Monitoring

`MetacognitiveEngine` monitors reasoning quality in real time. When confidence
drops or anomalies are detected, it can trigger **System 2** deliberation,
overriding the fast System 1 path.

### 2.10 Coalition Competition Log

`GlobalWorkspace` logs every competition round: all proposals, their activation
scores, and the broadcast winner. The full history is accessible via
`engine.trace_history`.

---

## 3. V13 Glass-Box Additions

V13 introduces five new transparency surfaces.

### 3.1 KLE Uncertainty (Feature 11)

Shannon entropy of coalition activation probabilities:

```python
kle = global_workspace.get_kle_uncertainty()
```

- Higher entropy → more competing proposals → less certainty.
- Surfaced in `SubstrateResult.kle_uncertainty`.

### 3.2 Conformal Prediction Bounds (Feature 12)

`ConformalWrapper` provides calibrated `[lower, upper]` uncertainty bounds with
**guaranteed coverage** at level `1 − α`:

```python
wrapper = ConformalWrapper(alpha=0.1)
wrapper.calibrate(calibration_scores)
lower, upper = wrapper.uncertainty_bound(point_estimate)
```

- Surfaced in `SubstrateResult.uncertainty_bounds`.

### 3.3 Causal Rule Audit Traces (Feature 13)

`CausalRuleAuditor.audit_rule()` returns an `audit_trace` — a list of
human-readable strings explaining step-by-step how `causal_score` was computed:

```python
result = auditor.audit_rule(cause, effect, confidence=0.8)
for step in result.audit_trace:
    print(step)
```

### 3.4 Encoding Statistics (Feature 14)

`SubstrateResult.encoding_stats` is a dict showing:

- `source_type` — what produced the encoding (e.g. `"hv"`, `"snn"`)
- `n_dims` — dimensionality of the encoding vector
- `mean`, `std`, `min`, `max`, `l2_norm` — summary statistics

### 3.5 Procedural Memory Transparency (Feature 15)

`SubstrateResult.procedural_hit` is a boolean. When `True`, the system served a
cached skill from `ProceduralMemory` instead of running full deliberation — and
it tells you so explicitly.

---

## 4. How to Inspect a Decision

A complete example using `NSCKSubstrate`:

```python
from python.core.substrate import NSCKSubstrate

sub = NSCKSubstrate()
sub.register_task("demo")

result = sub.process("hello world", task_tag="demo")

# Core decision output
print(result.chosen_action)       # e.g. "ACTION_RIGHT"
print(result.confidence)           # e.g. 0.5
print(result.explanation)          # Explanation object with .summary
print(result.trace)                # dict: proposals, winner, reason
print(result.predicates)           # grounded symbols e.g. {"INTENT_GREETING"}

# V13 transparency fields
print(result.kle_uncertainty)      # e.g. 0.69 (competition entropy)
print(result.encoding_stats)       # encoding metadata dict
print(result.procedural_hit)       # False (not a cached skill)
print(result.uncertainty_bounds)   # e.g. (0.3, 0.7) or None
```

Every field above is a plain Python object — no hidden state, no opaque tensors.

---

## 5. Comparison with Black-Box Systems

| Feature | NSCK | Deep Neural Network |
|---|---|---|
| **Decision traceable** | ✅ Full trace with winner, proposals, scores | ❌ Gradient-based attribution only |
| **Explanations** | ✅ Natural language (`explain`, `why_not`) | ❌ Post-hoc approximations (LIME, SHAP) |
| **Rules inspectable** | ✅ Symbolic IF-THEN rules with confidence | ❌ Weights are opaque |
| **Memory inspectable** | ✅ Concepts, episodes, causal edges | ❌ Activations are dense vectors |
| **Uncertainty quantified** | ✅ KLE entropy + conformal bounds | ⚠️ Calibration varies by architecture |
| **Causal reasoning** | ✅ ΔP strengths + MI confounder detection | ❌ Correlation only |

### Why This Matters

- **Safety** — verify reasoning *before* deployment
- **Debugging** — trace the exact failure point in any decision
- **Trust** — stakeholders can audit every choice
- **Compliance** — satisfy GDPR and AI Act explainability requirements
- **Transfer** — copy specific rules or knowledge between agents

---

## 6. What NSCK Cannot Explain

Transparency has limits. We are honest about them.

### 6.1 Why a Particular Hypervector Was Generated

Hypervectors are initialized from high-dimensional random projections. The system
can explain *what* a hypervector represents (via its bound concept) and *how* it is
used, but not *why those specific bits* were chosen — because they are random by
design.

### 6.2 SNN Internal Dynamics

When the optional spiking neural network (SNN) substrate is used, individual spike
timing patterns are not human-interpretable. The system can report *aggregate*
firing rates and the *output* of SNN processing, but the internal dynamics of
individual neurons remain below the explainability horizon.

### 6.3 Distributional Semantics Weights

Co-occurrence statistics used for distributional semantics are aggregate counts
over a corpus. They inform the system's similarity judgments, but a per-decision
attribution to specific co-occurrence events is not feasible.

---

## Related Documentation

- [Architecture](ARCHITECTURE.md) — System design with Mermaid diagrams
- [Formulas](FORMULAS.md) — Mathematical foundations
- [Workflows](WORKFLOWS.md) — Decision loop and data flow
- [Module Reference](MODULE_REFERENCE.md) — API reference
- [Repository Map](REPOMAP.md) — Complete codebase map

---

**Last Updated:** February 2026  
**Test Coverage:** All transparency features verified  
**Status:** ✅ Production-ready glass-box system
