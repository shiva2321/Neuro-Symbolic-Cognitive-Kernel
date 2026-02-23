# Auditable Causal Reasoning in a Neuro-Symbolic Architecture via Δ-P Discovery and Global Workspace Broadcasting

**Shivam Prajapati**
Bachelor of Computer Science, University of Prince Edward Island
Charlottetown, Prince Edward Island, Canada

*Open-source implementation — February 2026*

---

## Abstract

We present an auditable causal reasoning system that discovers causal relationships from as few as 2–5 observations using the Δ-P contingency formula with adaptive Laplace smoothing, integrates them into a Global Workspace Theory (GWT) coalition competition, and produces fully traceable causal explanations for every decision. The system operates entirely within a Vector Symbolic Architecture (VSA) framework, where all concepts, causes, and effects are represented as 10,240-bit binary hypervectors — enabling bitwise-efficient causal computation without matrix multiplication or gradient descent.

We describe three novel contributions: (1) an online causal discovery module that incrementally learns causal graphs from experience streams using Δ-P statistics; (2) a counterfactual reasoning mechanism based on a simplified do-operator that simulates "what if X had not occurred" by graph surgery; and (3) integration of causal chains into the GWT coalition scoring, so that every decision includes a causal explanation chain as a first-class object — not a post-hoc rationalisation.

Evaluation across four domains (robot navigation, medical triage, financial trading, environmental monitoring) shows 9 causal rules learned from 500 navigation cycles, sub-millisecond causal chain computation, and complete decision auditability. We document honest limitations: Δ-P requires discrete symbolic states, chain strength assumes independence, and ground-truth causal graphs in our evaluation are synthetic.

**Keywords:** Causal Reasoning, Explainable AI, Vector Symbolic Architecture, Global Workspace Theory, Δ-P Statistics, Counterfactual Reasoning, Neuro-Symbolic AI

---

## 1. Introduction

### 1.1 The Problem

AI systems increasingly make consequential decisions — from medical diagnosis to financial trading to autonomous navigation — yet the causal reasoning behind these decisions is typically either absent or reconstructed after the fact. Neural networks learn correlations but not causation. Post-hoc explanation methods like LIME [1] and SHAP [2] approximate the reasoning of a black-box model, but they do not reveal the model's actual causal understanding — because the model has none.

Pearl [3] distinguished three levels of causal reasoning: **association** (observing co-occurrences), **intervention** (predicting effects of actions), and **counterfactual** (reasoning about alternative histories). Most deployed AI systems operate only at the association level. Moving up Pearl's ladder requires explicit causal models.

### 1.2 Our Approach

We present a causal reasoning module embedded within NSCK, a neuro-symbolic cognitive architecture [4]. The module implements:

- **Level 1 (Association):** Δ-P statistics [5] discover cause-effect relationships from observed co-occurrences, with adaptive Laplace smoothing for data-sparse settings.
- **Level 2 (Intervention):** Causal graphs with typed links (CAUSES, PREVENTS, ENABLES, REQUIRES) enable forward prediction of intervention outcomes.
- **Level 3 (Counterfactual):** A do-operator simulates alternative histories by graph surgery — removing causal links and re-propagating effects.

Critically, this causal reasoning is not a standalone module. It is integrated into the Global Workspace Theory (GWT) [6] coalition competition that governs all decisions. Causal chains compete as coalitions alongside rules, memories, and plans. The winning coalition's causal explanation is included in the decision trace — making every decision auditable by construction.

### 1.3 Contributions

1. An online, incremental causal discovery algorithm using Δ-P statistics with adaptive Laplace smoothing that learns from as few as 2 observations.
2. A counterfactual reasoning mechanism using simplified do-operator graph surgery within a VSA framework.
3. Integration of causal explanations into GWT coalition competition, providing ante-hoc (not post-hoc) decision auditability.
4. Honest evaluation across four domains with documented limitations.

---

## 2. Background

### 2.1 Causal Inference — From Hume to Pearl

Hume [7] first formalised causation as regular conjunction: A causes B if B regularly follows A. This purely associative view was formalised statistically by Cheng and Novick [5] as the **Δ-P rule** — the difference in the probability of an effect given the presence versus absence of a putative cause.

Pearl [3] elevated causal reasoning beyond association by introducing the do-operator: P(Y | do(X = x)) represents the probability of Y when X is forcibly set to x, as distinct from P(Y | X = x) which represents passive observation. This distinction is crucial because correlation between X and Y may be due to a common cause rather than a direct causal link.

### 2.2 Δ-P: The Probabilistic Contrast Model

The Δ-P rule computes the causal strength of a putative cause c on an effect e:

$$\Delta P(c \to e) = P(e \mid c) - P(e \mid \neg c)$$

**Interpretation:**
- Δ-P > 0: c is a generative cause of e
- Δ-P < 0: c is a preventive cause of e
- Δ-P ≈ 0: c is not causally related to e

The Δ-P rule was empirically validated by Cheng and Novick [5] as a model of how humans actually make causal judgments from contingency data.

### 2.3 Explainability: Post-Hoc vs. Ante-Hoc

Rudin [8] argued that for high-stakes decisions, the field should build inherently interpretable models rather than explaining black boxes after the fact. NSCK follows this principle: causal reasoning is the actual decision mechanism, not an explanation layer added on top of a neural network.

### 2.4 Global Workspace Theory

Baars [6] proposed that conscious cognition involves multiple specialist modules competing for access to a shared broadcast channel. Dehaene et al. [9] provided neuroimaging evidence. In NSCK, the causal reasoning module is one of several specialists that propose "coalitions" (bundles of information) to the global workspace. The winning coalition is broadcast to all modules, creating a unified decision context.

---

## 3. The Δ-P Causal Discovery Module

### 3.1 Data Representation

All concepts, causes, and effects are represented as 10,240-bit binary hypervectors using Vector Symbolic Architecture (BSC variant) [10]. This means causal relationships are computed over the same representational space used for perception, memory, and language — no translation layer is needed.

A causal observation consists of a triple: (context, causes, effects), where:
- **context** is a string identifying the domain or situation type
- **causes** is a set of symbolic predicates active in this timestep
- **effects** is a set of symbolic predicates observed as outcomes

### 3.2 The Δ-P Formula — Implementation

From the CausalDiscovery module (causal_reasoning.py, 1,233 lines):

For each (cause, effect) pair within a context, the module maintains three count tables:
- N(c): number of timesteps where cause c was present
- N(e): number of timesteps where effect e was observed
- N(c, e): number of timesteps where both c and e co-occurred

The total step count is T. Then:

$$\Delta P(c \to e) = P(e \mid c) - P(e \mid \neg c)$$

where:

$$P(e \mid c) = \frac{N(c, e) + \alpha}{N(c) + 2\alpha}$$

$$P(e \mid \neg c) = \frac{N(e) - N(c, e) + \alpha}{T - N(c) + 2\alpha}$$

**Adaptive Laplace smoothing:**

$$\alpha = \begin{cases} 1.0 & \text{if } T < 10 \\ 0.0 & \text{if } T \geq 10 \end{cases}$$

When observations are sparse (T < 10), Laplace smoothing prevents degenerate probability estimates (division by zero, probabilities of 0 or 1). Once sufficient data has been collected (T ≥ 10), the smoothing is removed to avoid biasing the estimate.

**Property:** With α = 1 and a single observation (T = 1, N(c) = 1, N(c,e) = 1):

$$P(e \mid c) = \frac{1 + 1}{1 + 2} = \frac{2}{3} \approx 0.67$$

This is conservative — the system does not conclude certainty from a single observation.

### 3.3 The Online Learning Algorithm

**Algorithm 1: Incremental Causal Discovery**

```
OBSERVE(context, causes, effects):
    T[context] ← T[context] + 1
    for each c in causes:
        N_c[context][c] ← N_c[context][c] + 1
        for each e in effects:
            N_ce[context][(c, e)] ← N_ce[context][(c, e)] + 1
    for each e in effects:
        N_e[context][e] ← N_e[context][e] + 1

INDUCE_GRAPH(context, min_confidence, min_evidence):
    G ← new CausalGraph()
    for each (c, e) in N_ce[context]:
        if N_ce[context][(c, e)] ≥ min_evidence:
            Δp ← compute_delta_p(c, e, context)
            if |Δp| ≥ min_confidence:
                type ← CAUSES if Δp > 0 else PREVENTS
                G.add_link(c, e, type, strength=|Δp|)
    return G
```

**Complexity:** observe() is O(|causes| × |effects|) per timestep — typically O(1) to O(10) in practice. induce_graph() is O(|pairs|) where |pairs| is the number of co-occurrence entries, bounded by O(|causes| × |effects|) accumulated over time.

### 3.4 Spurious Correlation Rejection

A critical property of Δ-P is its ability to distinguish causation from mere correlation. Consider:

**Example (from the test suite):** A bird chirps whenever the sun rises. A person claps whenever the sun rises. After many observations:
- N(CLAP, BIRD_CHIRPS) is high (they co-occur)
- N(CLAP) ≈ N(SUN_RISES) and N(¬CLAP) includes many steps where SUN_RISES without CLAP

Therefore:
- P(BIRD_CHIRPS | CLAP) ≈ P(BIRD_CHIRPS | ¬CLAP) (bird chirps regardless of clapping)
- Δ-P(CLAP → BIRD_CHIRPS) ≈ 0

The system correctly rejects the spurious correlation between clapping and bird chirping. In contrast, simple co-occurrence counting would flag this as a causal relationship.

### 3.5 CausalGraph Structure

The CausalGraph is a directed graph where each edge is a CausalLink with:
- **source**: cause predicate
- **target**: effect predicate
- **link_type**: CAUSES, PREVENTS, ENABLES, or REQUIRES
- **strength**: |Δ-P| value
- **evidence**: number of supporting observations
- **context**: domain tag

Access patterns:
- `forward[cause]` → list of (effect, CausalLink) — for prediction
- `backward[effect]` → list of (cause, CausalLink) — for diagnosis

**Causal chain strength** for a multi-hop chain c₁ → c₂ → ⋯ → cₙ:

$$\text{strength}(c_1 \to \cdots \to c_n) = \prod_{i=1}^{n-1} \Delta P(c_i \to c_{i+1})$$

This product formula assumes independence between links — a simplification that we acknowledge as a limitation (Section 7).

---

## 4. Counterfactual Simulation

### 4.1 The Do-Operator

Pearl's do-operator [3] formalises the difference between observation and intervention. The standard definition:

$$P(Y \mid do(X = x)) = \sum_z P(Y \mid X = x, Z = z) \cdot P(Z = z)$$

where Z is the set of confounders.

### 4.2 Our Simplified Implementation

NSCK implements a simplified do-operator for discrete symbolic states:

**Algorithm 2: Counterfactual Query**

```
COUNTERFACTUAL(do_X, observe_Y):
    # Step 1: Record original outcome
    original ← trace_chain(X → ... → Y) in current graph
    
    # Step 2: Graph surgery — remove outgoing links from X
    G' ← copy(G)
    remove all links from do_X in G'
    
    # Step 3: Re-propagate activation through modified graph
    counterfactual ← trace_chain(... → Y) in G'
    
    # Step 4: Compare
    return CounterfactualResult(
        query = "What if {do_X} had not occurred?",
        original_outcome = original,
        counterfactual_outcome = counterfactual,
        affected_states = states that changed,
        confidence = chain_strength(original)
    )
```

**What this captures:** If a causal chain A → B → C exists and we simulate "what if A had not occurred," the system removes A's outgoing links and re-propagates. If B has no other causes, B disappears, and so does C. If B has alternative causes, B may persist with reduced strength.

**What this does not capture:** Pearl's full do-calculus handles back-door and front-door adjustments for complex confounding structures. Our implementation handles the simpler case of direct and chain-mediated causation — sufficient for the discrete symbolic domains we target.

---

## 5. Integration with Global Workspace Theory

### 5.1 The Coalition Competition

In NSCK's GWT implementation, each decision cycle involves competition among specialist modules. Each module proposes a **Coalition** — a bundle of information with a proposed action and evidence.

The CausalReasoningModule proposes coalitions when it can construct a causal chain from the current state to a desirable (or undesirable) outcome:

**Coalition activation score:**

$$\alpha(C) = s(C) + r(C) + e(C) + 0.5 \cdot q(C)$$

where:
- s(C) = base_salience — intrinsic importance of the causal chain (proportional to chain length and link strength)
- r(C) = relevance — how well the chain's initial conditions match the current state
- e(C) = affect_match — emotional valence of the predicted outcome
- q(C) = sender_confidence — the causal module's self-assessed reliability

The winner: C* = argmax_C α(C), subject to α(C*) ≥ 0.5 (attention threshold).

### 5.2 SafetyGate Veto

Before the winning coalition is executed, the SafetyGate evaluates it:

1. If the causal chain predicts a negative outcome (e.g., collision, patient harm), the coalition is vetoed.
2. The next-ranked coalition is evaluated.
3. Up to 3 deliberation rounds.

The SafetyGate uses the same causal graph to predict downstream effects:

$$\text{danger}(a) = \max_{e \in \text{effects}(a)} \text{sim}(\text{HV}(e), \mathbf{v}_{\text{danger}})$$

If danger > 0.75, the action is vetoed.

### 5.3 The CognitiveState Trace

Every decision produces a CognitiveState object that includes:

```
CognitiveState:
  action: "chosen_action"
  confidence: 0.82
  explanation:
    type: CAUSAL
    summary: "Action X chosen because causal chain: 
              obstacle_ahead →[0.87]→ collision →[PREVENTS]→ 
              STOP prevents collision"
    causal_chain: [CausalLink(...), CausalLink(...)]
    coalition_scores: {RULES: 0.6, CAUSAL: 0.82, MEMORY: 0.4}
    safety_check: PASSED
```

This trace is the **actual computational path** the system took — not a reconstruction. It includes the specific causal links, their Δ-P strengths, the competing coalition scores, and the safety check result.

---

## 6. Experiments

### 6.1 Domains

We evaluate the causal discovery module across four domains implemented in the benchmark suite:

| Domain | Cycles | Predicates | Causal Ground Truth |
|---|---|---|---|
| Robot Navigation | 500 | obstacle_ahead, goal_visible, path_clear, etc. | 12 known causal links |
| Medical Triage | 300 | high_fever, chest_pain, breathing_difficulty, etc. | 8 known causal links |
| Financial Trading | 400 | price_rising, volume_high, trend_bullish, etc. | 10 known causal links |
| Env. Monitoring | 200 | temperature_high, pollution_elevated, etc. | 6 known causal links |

### 6.2 Causal Discovery Results

*Table 1: Causal rules discovered across domains.*

| Domain | Rules Discovered | Cycles | avg (ms/cycle) | p99 (ms) |
|---|---|---|---|---|
| Robot Navigation | 9 | 500 | 0.90 | 1.38 |
| Medical Triage | 6 | 300 | 0.66 | 0.94 |
| Financial Trading | 8 | 400 | 0.71 | 1.01 |
| Env. Monitoring | 5 | 200 | 0.65 | 0.91 |

From the Robot Navigation scenario (500 cycles), the causal discovery module learned:
- 9 causal rules from the CausalGraph
- 499 episodic memory recalls used as evidence
- 472 STRIPS plans generated using operators derived from the causal graph
- 250 episodes consolidated to warm storage during sleep()

### 6.3 Decision Auditability

Every decision across all four domains included a complete causal explanation chain. The explanation includes:
- The specific causal links (with Δ-P strengths) that led to the decision
- The coalition scores from competing modules
- The safety check result
- The confidence calibration from the self-model

This represents **100% auditability** — no decision is made without a traceable causal justification.

### 6.4 Spurious Correlation Rejection

We tested the spurious correlation rejection capability using the CLAP→BIRD_CHIRPS scenario (Section 3.4). After 100 observations:
- SUN_RISES → BIRD_CHIRPS: Δ-P = 0.85 ✓ (correctly identified as causal)
- CLAP → BIRD_CHIRPS: Δ-P = 0.02 ✗ (correctly rejected as spurious)

### 6.5 Performance

*Table 2: Causal computation latency (measured with Rust backend on x86-64).*

| Operation | Latency |
|---|---|
| observe() — single timestep | < 0.01 ms |
| induce_graph() — 500 observations | ~0.5 ms |
| counterfactual query | ~0.1 ms |
| Causal chain lookup (3-hop) | < 0.01 ms |
| Full decision cycle with causal | 0.10–0.13 ms (p50–p95) |

All causal operations are sub-millisecond, making them practical for real-time decision loops. With the Rust VSA backend active, the full decision cycle (including causal chain evaluation) runs at p50 = 0.10 ms — well under the 1 ms threshold for real-time systems.

---

## 7. Limitations

We document the following limitations honestly:

1. **Δ-P requires discrete symbolic states.** The causal discovery module operates on symbolic predicates (e.g., "obstacle_ahead"), not continuous sensor values. A grounding layer that maps continuous observations to discrete predicates is required — and this grounding layer's quality directly determines causal discovery quality.

2. **Chain strength assumes independence.** The product formula for multi-hop chain strength (Section 3.5) assumes that each link's strength is independent of the others. In reality, causal relationships may interact — A's effect on B may depend on whether C is present. Modelling these interactions would require conditional Δ-P tables, which we have not implemented.

3. **Ground-truth causal graphs are synthetic.** The evaluation domains use hand-crafted causal structures, not real-world observational data. This means our "accuracy" numbers reflect the system's ability to recover known synthetic structures, not its performance on genuine causal discovery in noisy, complex environments.

4. **No hidden confounders.** Δ-P assumes that all relevant causes are observed. Hidden confounders (unobserved variables that cause both the putative cause and the effect) can create spurious causal links that Δ-P cannot detect. Pearl's full do-calculus with graphical models handles this; our simplified implementation does not.

5. **Binary causal types.** Causal links are typed as CAUSES or PREVENTS, but real-world causation includes probabilistic, dose-dependent, and context-sensitive relationships that our binary typing does not capture.

---

## 8. Related Work

| System | Causal Method | Online Learning | Explainable | VSA-Based |
|---|---|---|---|---|
| Granger causality [11] | Temporal precedence | Yes | Partial | No |
| PC Algorithm [12] | Conditional independence | No (batch) | Yes | No |
| Neural causal models [13] | Gradient-based | Yes | No | No |
| **NSCK (this work)** | **Δ-P + GWT** | **Yes** | **Yes** | **Yes** |

Our approach is closest to Cheng's [5] original Δ-P model, extended with: (a) adaptive Laplace smoothing for sparse data, (b) integration into a GWT coalition competition, and (c) counterfactual simulation via graph surgery. To our knowledge, no prior system combines Δ-P causal discovery with GWT broadcasting in a VSA representation.

---

## 9. Future Work

1. **Conditional Δ-P tables:** Model interactions between causes to handle non-independent causal chains.
2. **Hidden confounder detection:** Implement front-door and back-door criteria from Pearl's framework.
3. **Continuous-valued causal discovery:** Extend Δ-P to handle graded causes and effects, not just binary predicates.
4. **Real-world evaluation:** Test on publicly available causal benchmark datasets (e.g., Tübingen cause-effect pairs, CausalDiscovery.org).
5. **Temporal Δ-P:** Model time-lagged causal relationships with configurable temporal windows.

---

## 10. Conclusion

We have presented an auditable causal reasoning system that:
- Discovers causal relationships from as few as 2–5 observations using Δ-P statistics with adaptive Laplace smoothing
- Provides counterfactual reasoning via simplified do-operator graph surgery
- Integrates causal chains into GWT coalition competition for ante-hoc decision auditability
- Operates entirely within a VSA framework (10,240-bit binary hypervectors), requiring no matrix multiplication or gradient descent

The system learned 9 causal rules from 500 navigation cycles, correctly rejected spurious correlations, and produced fully traceable causal explanations for every decision. These results demonstrate that principled causal reasoning can be embedded within a lightweight, interpretable cognitive architecture.

The honest limitations are clear: synthetic ground truth, independence assumptions in chain strength, no hidden confounder handling, and a dependence on discrete symbolic grounding. We present this as a foundation for further research in auditable causal AI.

Code and benchmarks: **https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel**

---

## References

[1] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. KDD 2016*, pp. 1135–1144.

[2] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *Advances in NeurIPS*, vol. 30, 2017.

[3] J. Pearl, *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2000.

[4] S. Prajapati, "NSCK: A unified neuro-symbolic cognitive kernel using binary hypervector representations," *arXiv preprint*, 2026. [Paper 1 in this series]

[5] P. W. Cheng and L. R. Novick, "A probabilistic contrast model of causal induction," *J. Personality and Social Psychology*, vol. 58, no. 4, pp. 545–567, 1990.

[6] B. J. Baars, *A Cognitive Theory of Consciousness*. Cambridge University Press, 1988.

[7] D. Hume, *A Treatise of Human Nature*. John Noon, 1739.

[8] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nature Machine Intelligence*, vol. 1, pp. 206–215, 2019.

[9] S. Dehaene, J.-P. Changeux, and J.-P. Nadal, "Global workspace and metacognition," *Proc. National Academy of Sciences*, vol. 108, no. 7, pp. 3142–3148, 2011.

[10] P. Kanerva, "Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors," *Cognitive Computation*, vol. 1, no. 2, pp. 139–159, 2009.

[11] C. W. J. Granger, "Investigating causal relations by econometric models and cross-spectral methods," *Econometrica*, vol. 37, no. 3, pp. 424–438, 1969.

[12] P. Spirtes, C. Glymour, and R. Scheines, *Causation, Prediction, and Search*, 2nd ed. MIT Press, 2000.

[13] Y. Bengio et al., "A meta-transfer objective for learning to disentangle causal mechanisms," in *Proc. ICLR 2020*.

[14] S. Franklin and F. G. Patterson, "The LIDA architecture: Adding new modes of learning to an intelligent, autonomous, software agent," in *Proc. IDPT*, 2006.

---

*All results are derived from the actual NSCK implementation. Benchmark numbers reflect measured performance on commodity x86-64 hardware. We welcome replication, critique, and contribution.*
