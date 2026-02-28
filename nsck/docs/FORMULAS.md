# NSCK V14 — Theory, Formulas, and Mathematical Foundations

This document covers the complete mathematical and theoretical foundations of
NSCK V14: every formula, derivation, and design rationale — extracted directly
from the source code and the research ideas that motivated each component.

---

## Table of Contents

1. [Design Goals and Theoretical Motivation](#1-design-goals-and-theoretical-motivation)
2. [Vector Symbolic Architecture](#2-vector-symbolic-architecture)
3. [Text Encoding and Semantic Folding](#3-text-encoding-and-semantic-folding)
4. [Spreading Activation](#4-spreading-activation)
5. [Global Workspace Theory](#5-global-workspace-theory)
6. [Causal Discovery (ΔP Statistics)](#6-causal-discovery-δp-statistics)
7. [Q-Learning and Reward Processing](#7-q-learning-and-reward-processing)
8. [Hebbian Learning and Oja's Rule](#8-hebbian-learning-and-ojas-rule)
9. [Spiking Neural Networks](#9-spiking-neural-networks)
10. [Curiosity and Novelty](#10-curiosity-and-novelty)
11. [Emotion System (Affective Computing)](#11-emotion-system-affective-computing)
12. [Self-Model Confidence](#12-self-model-confidence)
13. [Episodic Memory and LSH](#13-episodic-memory-and-lsh)
14. [Rule Learner](#14-rule-learner)
15. [Analogy Engine](#15-analogy-engine)
16. [Response Composition (IDF Scoring)](#16-response-composition-idf-scoring)
17. [Safety and Veto Logic](#17-safety-and-veto-logic)
18. [FHRR Complex Phasor VSA (V10)](#18-fhrr-complex-phasor-vsa-v10)
19. [VSA Embedding Bridge (V10)](#19-vsa-embedding-bridge-v10)
20. [Attention-GWT Bridge (V10)](#20-attention-gwt-bridge-v10)
21. [Rule Neural Scorer (V10)](#21-rule-neural-scorer-v10)
22. [Safety Gate Verifier (V10)](#22-safety-gate-verifier-v10)
23. [KLE Uncertainty (V13)](#23-kle-uncertainty-v13)
24. [Conformal Prediction (V13)](#24-conformal-prediction-v13)
25. [Signal Ingestor (V13)](#25-signal-ingestor-v13)
26. [Universal HV Encoder (V13)](#26-universal-hv-encoder-v13)
27. [Procedural Memory (V13)](#27-procedural-memory-v13)
28. [Concept Drift Detection (V13)](#28-concept-drift-detection-v13)
29. [Cross-Modal Associative Binding (V13)](#29-cross-modal-associative-binding-v13)
30. [Causal Rule Auditor (V13)](#30-causal-rule-auditor-v13)
31. [Pattern Generalizer (V13)](#31-pattern-generalizer-v13)
32. [Bridge Projection (V14)](#32-bridge-projection-v14)
33. [Distillation Quality (V14)](#33-distillation-quality-v14)
34. [References](#references)

---

## 1. Design Goals and Theoretical Motivation

NSCK was designed to answer a specific question:

> *Can a cognitive system be simultaneously intelligent, transparent, and
> biologically plausible — without matrix multiplication or gradient descent?*

The core thesis is: **Vector Symbolic Architecture provides a common
mathematical language for both neural-style association and symbolic
reasoning.** The same 10,240-bit binary HV is used for every representation —
concepts, predicates, sensory inputs, episodes, rules — so there is no
translation layer between neural and symbolic processing.

### Primary design goals

| Goal | Implementation | Status |
|---|---|---|
| Glass-box transparency | ExplanationGenerator + ThoughtTrace | Achieved |
| No black boxes | No neural inference; only VSA + symbolic | Achieved |
| Biological plausibility | GWT, Hebbian, SNN, curiosity | Achieved |
| Domain agnosticism | `register_task()` interface | Achieved |
| Self-contained (no external AI) | All reasoning in-process | Achieved |
| Efficient on commodity hardware | Rust concurrent layer | Achieved |
| Continual learning (no forgetting) | Episode replay + tenure-based rules | Achieved (V13) |
| Counterfactual reasoning | CausalGraph + CounterfactualReasoner | Achieved |
| Uncertainty quantification | KLE entropy + Conformal prediction | Achieved (V13) |
| Multi-modal perception | SignalIngestor + UniversalHVEncoder | Achieved (V13) |

### Theoretical frameworks used

| Framework | Source | NSCK component |
|---|---|---|
| Binary VSA (B-VSA) | Kanerva (1988, 2009) | All knowledge representation |
| FHRR (complex phasor VSA) | Plate (2003) | `vsa/fhrr.py` (V10) |
| Global Workspace Theory (GWT) | Baars (1988), Dehaene (2011) | GlobalWorkspace competition |
| Spreading activation | Collins & Loftus (1975) | SemanticMemory graph traversal |
| Δ-P causal inference | Cheng & Novick (1990) | CausalDiscovery |
| Leaky Integrate-and-Fire | Lapicque (1907) | SNN perception |
| STDP | Bi & Poo (1998) | SNN learning |
| Russell's Circumplex | Russell (1980) | EmotionSystem |
| Plutchik's Wheel | Plutchik (1980) | EmotionSystem |
| Oja's rule | Oja (1982) | HebbianMatrix |
| STRIPS planning | Fikes & Nilsson (1971) | STRIPSPlanner |
| Locality-Sensitive Hashing | Indyk & Motwani (1998) | EpisodicMemory retrieval |
| Conformal prediction | Vovk et al. (2005) | `learning/conformal_wrapper.py` (V13) |
| MaxEnt / information theory | Jaynes (1957) | KLE uncertainty (V13) |
| Free-energy principle | Friston (2010) | Active inference motivation (V13) |

---

## 2. Vector Symbolic Architecture

*Source: `vsa/hypervec_py.py`, `vsa/hypervec_shim.py`, `rust_vsa/src/lib.rs`*

### Dimension

$$d = 10{,}240 \text{ bits}$$

All hypervectors $\mathbf{v} \in \{0, 1\}^{10240}$. This provides a space of $2^{10240}$ possible vectors. The expected number of vectors within Hamming distance $0.45d$ of a random vector is approximately $2^{100}$ — more than enough for any practical knowledge base (Kanerva, 2009).

### Normalised Hamming Similarity

$$\text{sim}(\mathbf{A}, \mathbf{B}) = 1 - \frac{d_H(\mathbf{A}, \mathbf{B})}{d} = 1 - \frac{\sum_{i=1}^{d} A_i \oplus B_i}{d}$$

Properties:
- $\text{sim}(\mathbf{A}, \mathbf{A}) = 1$
- $\text{sim}(\mathbf{A}, \bar{\mathbf{A}}) = 0$ (complement is maximally different)
- For random $\mathbf{A}, \mathbf{B}$: $\mathbb{E}[\text{sim}(\mathbf{A}, \mathbf{B})] = 0.5$
- Standard deviation: $\sigma \approx \frac{1}{2\sqrt{d}} \approx 0.00494$
- Meaningful threshold: $\text{sim} > 0.55 \approx 10\sigma$ above chance

### Cosine Similarity (bipolar representation)

Converting binary $\{0,1\}^d$ to bipolar $\{-1,+1\}^d$:

$$\hat{A}_i = 2A_i - 1, \quad \hat{B}_i = 2B_i - 1$$

$$\text{cos\_sim}(\mathbf{A}, \mathbf{B}) = \frac{\hat{\mathbf{A}} \cdot \hat{\mathbf{B}}}{\|\hat{\mathbf{A}}\| \|\hat{\mathbf{B}}\|} = \frac{\hat{\mathbf{A}} \cdot \hat{\mathbf{B}}}{d}$$

**Normalisation to [0,1]:**

$$\text{sim\_robust}(\mathbf{A}, \mathbf{B}) = \frac{\text{cos\_sim}(\mathbf{A}, \mathbf{B}) + 1}{2}$$

### Binding (XOR / MAP binding)

$$\mathbf{A} \otimes \mathbf{B} = \mathbf{A} \oplus \mathbf{B} \quad \text{(element-wise XOR)}$$

Properties:
- **Self-inverse**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{B} = \mathbf{A}$
- **Commutative**: $\mathbf{A} \otimes \mathbf{B} = \mathbf{B} \otimes \mathbf{A}$
- **Associative**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{C} = \mathbf{A} \otimes (\mathbf{B} \otimes \mathbf{C})$
- **Quasi-orthogonal to operands**: $\text{sim}(\mathbf{A} \otimes \mathbf{B}, \mathbf{A}) \approx 0.5$

**Proof of self-inverse property:**
$$(\mathbf{A} \oplus \mathbf{B}) \oplus \mathbf{B} = \mathbf{A} \oplus (\mathbf{B} \oplus \mathbf{B}) = \mathbf{A} \oplus \mathbf{0} = \mathbf{A}$$

**Role-filler binding:**
$$\text{HV}(\text{concept}) = \text{HV}(\text{name}) \oplus \bigoplus_{\text{prop} \in \text{properties}} \text{HV}(\text{prop}) \oplus \text{HV}(\text{value})$$

### Bundling (Majority Vote / BSC superposition)

For two vectors:

$$(\mathbf{A} + \mathbf{B})_i = \begin{cases} A_i & \text{if } A_i = B_i \\ \text{Bernoulli}(0.5) & \text{if } A_i \neq B_i \end{cases}$$

Properties:
- **Similar to both operands**: $\mathbb{E}[\text{sim}(\mathbf{A} + \mathbf{B}, \mathbf{A})] = 0.75$
- For $n$ vectors bundled: $\mathbb{E}[\text{sim}(\text{bundle}, \mathbf{v}_i)] \approx 0.5 + \frac{0.5}{n}$

### Permutation (Circular Shift)

$$\rho^k(\mathbf{v})_i = v_{(i + k) \bmod d}$$

Properties:
- **Invertible**: $\rho^{-k}(\rho^k(\mathbf{v})) = \mathbf{v}$
- **Orthogonal**: $\text{sim}(\rho^k(\mathbf{v}), \mathbf{v}) \approx 0.5$ for $k \neq 0$

### Negation (V6)

Let $\mathbf{S} \in \{0,1\}^d$ be the fixed negation seed (`0xDEADBEEFCAFEBABE` repeated):

$$\neg\mathbf{A} = \mathbf{A} \oplus \mathbf{S}$$

- **Orthogonality**: $\mathbb{E}[\text{sim}(\mathbf{A}, \neg\mathbf{A})] \approx 0.5$
- **Involution**: $\neg(\neg\mathbf{A}) = \mathbf{A}$

### FPE Bit-Flip Position Encoding (V5)

*Source: `reasoning/spatial_reasoning.py`*

For axis base vector $\mathbf{b}_{\text{axis}}$ and normalised coordinate $x_{\text{norm}} \in [0,1]$:

$$\text{pos\_hv}(x) = \mathbf{b}_{\text{axis}} \oplus \text{flip}(\mathbf{b}_{\text{axis}},\; k), \quad k = \lfloor x_{\text{norm}} \cdot 50 \rfloor$$

Similarity gradient: $\text{sim}(\text{pos\_hv}(x_1), \text{pos\_hv}(x_2)) \approx 1 - |k_1 - k_2| / d$

### CleanupMemory

$$\text{cleanup}(\mathbf{v}) = \argmax_j \text{sim}(\mathbf{v}, \mathbf{p}_j) \quad \text{subject to} \quad \text{sim} > \theta$$

Default threshold $\theta = 0.4$.

---

## 3. Text Encoding and Semantic Folding

*Source: `language/lingua_cortex.py`, `nsck_ai_model/ai_engine.py`*

### Word HV generation (deterministic)

$$\mathbf{w}_{\text{base}} = \text{HyperVector}(\text{hash}(w) \bmod 2^{32})$$

### Positional encoding

$$\mathbf{w}_{(i)} = \rho^i(\mathbf{w}_{\text{base}})$$

Maximum shift: `MAX_POSITION_SHIFT = 64`.

### Sentence HV (superposition)

$$\mathbf{s} = \bigoplus_{i=1}^{n} \mathbf{w}_{(i)}$$

### Context HV (window-based)

For context window of size $c$:

$$\mathbf{c}_i = \bigoplus_{j=\max(0,\,i-c)}^{\min(n,\,i+c)} \mathbf{w}_{(j)}$$

### Distributional Semantics (V3/V7)

*Source: `language/distributional_semantics.py`*

Context HV for word $w$ with co-occurrence radius $r = 5$:

$$\text{ctx\_hv}(w) = \bigoplus_{(w',\, k) \in \text{ctx}(w)} \rho^k(\text{word\_hv}(w'))$$

$$\text{distributional\_sim}(w_1, w_2) = \text{sim}(\text{ctx\_hv}(w_1), \text{ctx\_hv}(w_2))$$

### Pointwise Mutual Information (V4)

*Source: `learning/pmi_learner.py`*

$$\text{PMI}(a, b) = \log_2 \frac{P(a, b)}{P(a) \cdot P(b)}, \quad \text{PPMI}(a, b) = \max(0,\; \text{PMI}(a, b))$$

### KG Stop-Concept Filter (V7)

*Source: `language/text_knowledge_learner.py`*

$$\text{add\_relation}(s, r, o) \iff \text{sim}(\text{HV}(s), \text{HV}(o)) < \tau_{\text{generic}}$$

where $\tau_{\text{generic}} = 0.62$.

---

## 4. Spreading Activation

*Source: `memory/semantic_memory.py`*

Given start concepts $S = \{s_1, \ldots, s_k\}$ with initial activation $a_{s_i} = 1.0$:

$$a_j^{(t+1)} = a_j^{(t)} + \sum_{(i,j) \in E} a_i^{(t)} \cdot \gamma \cdot w_{\text{rel}(i,j)}$$

where:
- $\gamma = 0.7$ (global decay factor)
- $w_{\text{is\_a}} = 0.9$, $w_{\text{has\_property}} = 0.7$, $w_{\text{causes}} = 0.6$, $w_{\text{part\_of}} = 0.5$, $w_{\text{similar\_to}} = 0.4$

**Convergence:** After $T = 3$ (default) steps: $a^{(T)} \leq (0.7 \cdot 0.9)^3 = 0.63^3 \approx 0.25$.

**Complexity:** Frontier cap of top-200 active nodes → $O(200 \cdot \bar{d} \cdot T)$ per call.

---

## 5. Global Workspace Theory

*Source: `reasoning/global_workspace.py`*

### Coalition activation

$$\alpha(C) = s(C) + r(C) + e(C) + 0.5 \cdot q(C)$$

where $s$ = base salience, $r$ = relevance, $e$ = affect match, $q$ = sender confidence.

### Competition rule

$$C^* = \argmax_{C \in \mathcal{C}} \alpha(C)$$

### Mission focus bonus

$$\alpha'(C) = \alpha(C) + 0.2 \quad \text{if } C \text{ is mission-aligned}$$

### Mental rehearsal veto

1. Predict next state: $\mathbf{s}' = \text{WorldModel.predict}(C^*.action, \mathbf{s})$
2. Compute danger similarity: $d = \text{sim}(\mathbf{s}', \mathbf{v}_{\text{danger}})$
3. If $d > \theta_{\text{danger}} = 0.75$: veto $C^*$ and try next coalition

Max rehearsal cycles: 3. Max danger vectors: 200 (LRU eviction).

---

## 6. Causal Discovery (ΔP Statistics)

*Source: `reasoning/causal_reasoning.py`*

### Delta-P formula

$$\Delta P(c \to e) = P(e \mid c) - P(e \mid \neg c)$$

### With Laplace smoothing ($\alpha = 1$)

$$P(e \mid c) = \frac{N(e, c) + \alpha}{N(c) + 2\alpha}, \quad P(e \mid \neg c) = \frac{N(e, \neg c) + \alpha}{N(\neg c) + 2\alpha}$$

### Causal chain strength

$$\text{strength}(c_1 \to c_n) = \prod_{i=1}^{n-1} \Delta P(c_i \to c_{i+1})$$

### Counterfactual: do-operator

To simulate "what if $c$ had not occurred?": remove all causal links from $c$, propagate, and compare. This implements Pearl's do-calculus in simplified discrete form.

---

## 7. Q-Learning and Reward Processing

*Source: `reasoning/cognitive_engine.py`*

### Tabular Q-learning update

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\alpha = 0.1$ (learning rate) and $\gamma = 0.95$ (discount factor).

### Confidence from Q-values

$$\text{confidence}(s, a) = \frac{Q(s, a) - Q_{\min}}{Q_{\max} - Q_{\min}}$$

Cold start: if $|Q| = 0$, confidence defaults to 0.5.

### Exploration (curiosity-driven)

$$\text{explore} = \big[(\text{novelty}(s) > \theta_N) \wedge (\text{confidence}(s) < 0.5)\big] \vee (\text{progress} < \theta_P)$$

---

## 8. Hebbian Learning and Oja's Rule

*Source: `learning/hebbian.py`, `rust_snn/src/hebbian.rs`*

### Basic Hebbian rule

$$\Delta w_{ij} = \eta \cdot x_i \cdot y_j$$

### Oja's rule (normalised Hebbian)

$$\Delta w_{ij} = \eta \left( x_i y_j - y_j^2 w_{ij} \right)$$

The forgetting term $-y_j^2 w_{ij}$ prevents weight explosion.

**Convergence:** At equilibrium $\mathbf{w}^* = \text{PC}_1(\mathbf{x})$ — the first principal component of the input distribution (Oja, 1982).

### VSA-level Hebbian association

$$\text{related}(c) = \text{top-}k\left(\{(c', W_{cc'}) : c' \in \text{concepts}\}\right)$$

---

## 9. Spiking Neural Networks

*Source: `perception/snn_perception.py`, `rust_snn/src/lib.rs`*

### Leaky Integrate-and-Fire (LIF)

Discrete-time update (Euler method):

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m} \left[ -(V(t) - V_{\text{rest}}) + I(t) \right]$$

**Spike condition:**
$$\text{if } V(t) \geq V_{\text{thresh}}: \text{ emit spike}, \quad V \leftarrow V_{\text{reset}}$$

**Biological parameters:**
$\tau_m = 20$ ms, $V_{\text{rest}} = -70$ mV, $V_{\text{thresh}} = -55$ mV, $V_{\text{reset}} = -75$ mV.

### Rate coding

$$\Pr[\text{spike at time } t] = x, \quad x \in [0, 1]$$

### Temporal coding

$$t_{\text{spike}} = T_{\max}(1 - x)$$

### STDP (Spike-Timing-Dependent Plasticity)

$$\Delta w = \begin{cases}
A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t = t_{\text{post}} - t_{\text{pre}} > 0 \\
-A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0
\end{cases}$$

Parameters: $A_+ = 0.01$, $A_- = 0.012$, $\tau_+ = \tau_- = 20$ ms.

---

## 10. Curiosity and Novelty

*Source: `learning/curiosity.py`*

### Novelty score

$$\text{novelty}(s) = 1 - \max_{v \in \text{known}} \text{sim}(\text{encode}(s), v)$$

### Learning progress

$$\text{progress}(t) = \frac{1}{\lfloor T/2 \rfloor} \sum_{\tau=t-T}^{t-T/2} r_\tau - \frac{1}{\lfloor T/2 \rfloor} \sum_{\tau=t-T/2}^{t} r_\tau$$

### Exploration decision

$$\text{explore} = \big[(\text{novelty} > \theta_N) \wedge (\text{confidence} < 0.5)\big] \vee (\text{progress} < \theta_P)$$

Default: $\theta_N = 0.7$, $\theta_P = 0.01$.

---

## 11. Emotion System (Affective Computing)

*Source: `cognitive/emotion_system.py`*

### Russell's Circumplex model

$$V \in [-1, +1] \; \text{(valence)}, \quad A \in [0, 1] \; \text{(arousal)}$$

### Closest prototype (nearest-neighbour)

$$\hat{e} = \argmin_{e \in \text{Plutchik}} \left\| (V, A) - (V_e, A_e) \right\|_2$$

| Emotion | $V_e$ | $A_e$ |
|---|---|---|
| joy | 0.8 | 0.7 |
| trust | 0.5 | 0.2 |
| fear | −0.7 | 0.8 |
| surprise | 0.0 | 0.9 |
| sadness | −0.6 | 0.2 |
| disgust | −0.5 | 0.4 |
| anger | −0.5 | 0.8 |
| anticipation | 0.3 | 0.5 |
| neutral | 0.0 | 0.0 |

### Emotion blending (Gaussian kernel softmax)

$$b_e = \frac{\exp\!\big(-\|(V, A) - (V_e, A_e)\|^2 / 2\sigma^2\big)}{\sum_{e'} \exp\!\big(-\|(V, A) - (V_{e'}, A_{e'})\|^2 / 2\sigma^2\big)}$$

### Drive-to-emotion mapping

$$V \leftarrow V + \Delta V(\text{drives}), \quad A \leftarrow A + \Delta A(\text{drives})$$

---

## 12. Self-Model Confidence

*Source: `cognitive/self_model.py`*

### Running success ratio

$$\hat{p}(T, k) = \frac{\text{successes}(T, k)}{\text{attempts}(T, k)}$$

Context-aware: use $\hat{p}(T, k)$ if $\text{attempts}(T, k) \geq n_{\min}$, otherwise fall back to global $\hat{p}_{\text{global}}(T)$. Cold start default: $\hat{p} = 0.5$.

### Calibration error

$$\text{cal\_error}(T) = |\hat{p}(T, \cdot) - \bar{r}(T)|$$

---

## 13. Episodic Memory and LSH

*Source: `memory/episodic_memory.py`*

### Locality-Sensitive Hashing

1. Generate $m$ random projection vectors $\mathbf{r}_1, \ldots, \mathbf{r}_m \in \{-1, +1\}^d$
2. Hash: $h_j(\mathbf{v}) = \text{sgn}(\mathbf{r}_j \cdot \mathbf{v})$
3. Bucket key: $B(\mathbf{v}) = \text{concat}(h_1(\mathbf{v}), \ldots, h_m(\mathbf{v}))$

**Collision probability:** $\Pr[h(\mathbf{u}) = h(\mathbf{v})] = 1 - \theta / \pi$, where $\theta = \arccos(c)$.

Parameters: $m = 16$ bits, 8 hash tables.

### Two-tier design

- **Hot tier** (in-memory deque, capacity 500): $O(1)$ access
- **Warm tier** (SQLite): persistent, $O(\log n)$ access

**Impact score for consolidation:**

$$\text{impact\_score}(e) = |r_e| + \text{novelty}(e)$$

---

## 14. Rule Learner

*Source: `reasoning/rule_learner.py`*

### Rule confidence

$$\text{confidence}(R) = \frac{\text{successes}(R)}{\text{support}(R)}$$

### Induction threshold

$$\text{support}(R) \geq n_{\min} = 5 \quad \text{AND} \quad \text{confidence}(R) \geq c_{\min} = 0.3$$

### Tenure progression

| State | Condition | Demotion |
|---|---|---|
| new | $\text{support} < n_{\min}$ | N/A |
| bootstrap | $\text{support} \geq n_{\min}$ AND $\text{confidence} \geq c_{\min}$ | confidence falls below threshold |
| tenured | held bootstrap for `tenure_threshold` cycles | requires $2\times$ counter-evidence |

Pruning: rules with success rate $< 0.7$ after $n_{\min}$ observations are removed.

---

## 15. Analogy Engine

*Source: `reasoning/analogy.py`*

### Structural similarity

Given domains $D_1, D_2$ with concept sets $C_1, C_2$:

$$S_{ij} = \text{sim}(\text{HV}(c_i), \text{HV}(c_j)), \quad M = \{(c_i, \argmax_{c_j} S_{ij})\}$$

$$\text{quality}(M) = \frac{1}{|M|} \sum_{(i,j) \in M} S_{ij}$$

### Analogy-based transfer

Map predicates: $P_2 = \{M(p) : p \in P_1\}$, map action: $a_2 = M(a_1)$, register with reduced initial confidence.

---

## 16. Response Composition (IDF Scoring)

*Source: `nsck_ai_model/response_composer.py`*

### IDF-weighted relevance

$$\text{score}(s, q) = \sum_{w \in s \cap q} \text{IDF}(w), \quad \text{IDF}(w) = \log \frac{N}{1 + |\{s \in \text{corpus} : w \in s\}|}$$

### Jaccard deduplication

$$J(s_1, s_2) = \frac{|\text{tokens}(s_1) \cap \text{tokens}(s_2)|}{|\text{tokens}(s_1) \cup \text{tokens}(s_2)|} \geq 0.5$$

---

## 17. Safety and Veto Logic

*Source: `reasoning/global_workspace.py`, `cognitive/metacognition.py`*

### Mental rehearsal veto

$$\text{veto}(C) = \text{sim}(\text{WorldModel.predict}(C.\text{action}, \mathbf{s}), \mathbf{v}_{\text{danger}}) > \theta_D$$

Default: $\theta_D = 0.8$.

**Cascading veto:** If top-1 coalition is vetoed, try top-2, top-3, etc. If all vetoed, fall back to the most conservative default action.

### SafetyGate constraint check

$$\text{violates}(C, s) = \exists\; \text{constraint } \phi : \phi(s) \wedge (C.\text{content} \in \text{forbidden}(\phi))$$

### Metacognitive sleep trigger

$$\text{should\_sleep}() = \bar{r}_{\text{recent}} < \bar{r}_{\text{baseline}} - \sigma_r$$

---

## 18. FHRR Complex Phasor VSA (V10)

*Source: `vsa/fhrr.py`*

Fourier Holographic Reduced Representations (FHRR) represent vectors as
unit-magnitude complex phasors $\mathbf{v} \in \mathbb{C}^d$ where $|v_i| = 1$
for all $i$.

### Phasor generation

Each component is a random phase angle $\theta_i \in [-\pi, \pi]$:

$$v_i = e^{j\theta_i}, \quad \theta_i \sim \text{Uniform}(-\pi, \pi)$$

### Binding (element-wise complex multiplication)

$$(\mathbf{a} \circledast \mathbf{b})_i = a_i \cdot b_i$$

Since both are unit phasors, binding sums their phase angles modulo $2\pi$.

### Unbinding (conjugate multiplication)

$$(\mathbf{a} \oslash \mathbf{b})_i = a_i \cdot \overline{b_i}$$

**Self-inverse property:** $(\mathbf{a} \circledast \mathbf{b}) \oslash \mathbf{b} = \mathbf{a}$ because $b_i \cdot \overline{b_i} = |b_i|^2 = 1$.

### Bundling (normalised sum)

$$\mathbf{c} = \sum_{k=1}^{n} \mathbf{v}_k, \quad \hat{c}_i = \frac{c_i}{|c_i|}$$

The sum is projected back to unit magnitude per component, keeping the resulting vector on the phasor manifold.

### Similarity (cosine of magnitudes)

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\sum_i |a_i| \cdot |b_i|}{\sqrt{\sum_i |a_i|^2} \cdot \sqrt{\sum_i |b_i|^2}}$$

For unit phasors this reduces to 1.0 for identical vectors. A gradient-aware variant decomposes as $[\text{Re}(\mathbf{v}),\; \text{Im}(\mathbf{v})]$ and computes cosine on the concatenated real vector.

### Scalar encoding via golden-ratio phase

$$\theta_i(x) = 2\pi \cdot \text{frac}\!\big(x \cdot \varphi \cdot (i + 1)\big)$$

where $\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618$ (golden ratio) and $\text{frac}(z) = z \bmod 1$. The per-dimension factor $(i+1)$ ensures each dimension samples a different phase, creating a distributed representation of the scalar $x$.

---

## 19. VSA Embedding Bridge (V10)

*Source: `vsa/vsa_embedding_bridge.py`*

Bridges dense real-valued embeddings (e.g., from attention layers) to binary
hypervectors and back.

### Projection matrix

$$\mathbf{P} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{hv}}}, \quad P_{ij} \sim \mathcal{N}(0, 1)$$

### Dense → VSA (embed to HV)

$$\text{projected} = \mathbf{e} \cdot \mathbf{P}, \quad \text{bits}_i = \begin{cases} 1 & \text{if } \text{projected}_i \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

This is a random-hyperplane binarisation that approximately preserves cosine similarity in the original space.

### VSA → Dense (HV to embed)

$$\text{reconstructed} = \frac{\text{bits} \cdot \mathbf{P}^\top}{\|\mathbf{P}^\top\|}$$

Row-wise normalisation restores approximate scale.

### Similarity in embedding space

$$\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \cdot \|\mathbf{e}_2\|}$$

### Text fallback (n-gram hashing)

When no embedding model is available, text is encoded via character-sum and bigram hash indices.

---

## 20. Attention-GWT Bridge (V10)

*Source: `reasoning/attention_gwt_bridge.py`*

Connects transformer-style attention scoring with GWT coalition salience,
allowing attention weights to modulate which coalitions win the global
broadcast.

### Attention score

$$\text{score}(q, k) = \frac{\mathbf{q}_{\text{proj}} \cdot \mathbf{k}_{\text{proj}}}{\sqrt{d_k} + \varepsilon}$$

where $\varepsilon = 10^{-9}$ for numerical stability.

### Softmax normalisation

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_j \exp(x_j - \max(\mathbf{x}))}$$

The $\max$-subtraction trick prevents overflow.

### Multi-head averaging

For $H$ attention heads, scores are averaged then re-normalised:

$$\bar{s}_i = \frac{1}{H} \sum_{h=1}^{H} s_i^{(h)}, \quad \hat{s} = \text{softmax}(\bar{s})$$

### Reranked salience

$$\text{final\_salience}_i = \hat{s}_i \cdot \text{base\_salience}_i$$

The attention weight multiplicatively modulates the GWT base salience.

---

## 21. Rule Neural Scorer (V10)

*Source: `learning/rule_neural_scorer.py`*

A lightweight two-layer perceptron that scores rule quality from hand-crafted
features, adding a learned ranking to the symbolic rule learner.

### Feature extraction (6 features)

| Feature | Formula |
|---|---|
| Confidence | $\text{confidence}(R)$ (raw) |
| Normalised support | $\min(\text{support} / 100,\; 1.0)$ |
| Fire ratio | $\min(\text{fire\_count} / 10000,\; 1.0)$ |
| Complexity | $\min(|\text{conditions}| / 10,\; 1.0)$ |
| Confidence trend | Linear slope of last 3 confidence history values |
| Task tag | $1.0$ if task tag is present, else $0.0$ |

### Network architecture

$$\mathbf{h} = \sigma(\mathbf{x} \cdot \mathbf{W}_1 + \mathbf{b}_1), \quad \text{out} = \sigma(\mathbf{h} \cdot \mathbf{W}_2 + \mathbf{b}_2)$$

Dimensions: $\mathbf{W}_1 \in \mathbb{R}^{6 \times 16}$, $\mathbf{W}_2 \in \mathbb{R}^{16 \times 1}$.

### Sigmoid activation (clipped)

$$\sigma(x) = \frac{1}{1 + e^{-\text{clip}(x,\; -30,\; 30)}}$$

Clipping to $[-30, 30]$ prevents floating-point overflow.

### Backpropagation (MSE loss)

$$\frac{\partial L}{\partial \text{out}} = 2(\text{out} - \text{target})$$

$$\delta_{\text{out}} = \text{out}(1 - \text{out}) \cdot \frac{\partial L}{\partial \text{out}}$$

$$\delta_{\mathbf{h}} = \mathbf{h} \odot (1 - \mathbf{h}) \odot (\delta_{\text{out}} \cdot \mathbf{W}_2^\top)$$

Weight update with learning rate $\eta = 0.01$:

$$\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta \cdot \mathbf{h}^\top \cdot \delta_{\text{out}}, \quad \mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \cdot \mathbf{x}^\top \cdot \delta_{\mathbf{h}}$$

---

## 22. Safety Gate Verifier (V10)

*Source: `cognitive/safety_verifier.py`*

### Default safety properties

| Check | Condition | Severity |
|---|---|---|
| Min confidence | $\text{confidence} > 0.3$ | warning |
| Min support | $\text{support} \geq 2$ | warning |
| No runaway | $\text{fire\_count} < 10{,}000$ | critical |
| No code injection | $\text{action} \notin \{\texttt{\_\_import\_\_}, \texttt{exec}, \texttt{eval}, \texttt{os.system}, \texttt{subprocess}\}$ | critical |

### Safety score

$$\text{score} = 1 - \frac{\text{violations}}{\max(\text{properties},\; 1)}$$

A score of 1.0 means all properties are satisfied; 0.0 means all are violated.

### Confidence gate

Veto action if $\text{confidence} < 0.3$ and $\text{action} \neq \texttt{"explore"}$. Exploration is always allowed to prevent deadlock.

---

## 23. KLE Uncertainty (V13)

*Source: `reasoning/global_workspace.py`*

The Kullback–Leibler Entropy (Shannon entropy of the coalition activation
distribution) quantifies decision uncertainty in the global workspace.

### Coalition activation

$$a_i = s_i + r_i + e_i + 0.5 \cdot q_i$$

(same as §5 coalition activation formula)

### Normalised distribution

$$p_i = \frac{\max(a_i,\; 10^{-9})}{\sum_j \max(a_j,\; 10^{-9})}$$

The $10^{-9}$ floor prevents $\log(0)$.

### Shannon entropy

$$H = -\sum_i p_i \ln(p_i)$$

### Interpretation

| $H$ value | Meaning |
|---|---|
| Low ($H \to 0$) | One dominant coalition — high decision confidence |
| High ($H \to \ln n$) | Uniform competition — maximum uncertainty |

High entropy signals tight competition among coalitions and triggers
deliberative processing (additional rehearsal cycles).

---

## 24. Conformal Prediction (V13)

*Source: `learning/conformal_wrapper.py`*

Provides distribution-free confidence intervals around rule confidence
estimates with finite-sample coverage guarantees.

### Calibration quantile

Given $n$ calibration scores and significance level $\alpha$ (default 0.1):

$$\hat{q} = \text{quantile}\!\left(\text{scores},\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

### Confidence bounds

$$\text{lower} = \max(0,\; 1 - \hat{q})$$

$$\text{upper} = \min(1,\; 1 - s + 0.1 \cdot \hat{q})$$

where $s$ is the nonconformity score of the current prediction. The $0.1$ scaling factor on $\hat{q}$ is a conservative implementation choice that tightens the upper bound relative to standard conformal intervals.

### Coverage guarantee

$$\Pr\!\big(Y \in C(X)\big) \geq 1 - \alpha$$

This holds for **any** data distribution (exchangeability assumption only).
With the default $\alpha = 0.1$, the system achieves at least 90% coverage.

---

## 25. Signal Ingestor (V13)

*Source: `perception/signal_ingestor.py`*

Normalises arbitrary input modalities (text, bytes, numeric arrays,
dictionaries) into a uniform $[0, 1]$ float array for downstream encoding.

### Text normalisation

$$\hat{x}_i = \frac{\text{ord}(c_i)}{128}$$

### Byte normalisation

$$\hat{x}_i = \frac{b_i}{255}$$

### Array min-max scaling

$$\hat{x}_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \quad \text{if } (x_{\max} - x_{\min}) > 10^{-8}$$

Falls back to zero vector if range is degenerate.

### Statistics (per ingested signal)

$$\text{stats} = \big\{\mu,\; \sigma,\; x_{\min},\; x_{\max},\; \|\mathbf{x}\|_2\big\}$$

### Dictionary encoding

Numeric values are summed; string values are hashed. All results are normalised to $[0, 1]$.

---

## 26. Universal HV Encoder (V13)

*Source: `vsa/universal_hv_encoder.py`*

Encodes any normalised float array into a binary hypervector using
FPE (Finite Pulse Encoding) quantisation with learnable Hebbian weights.

### Feature projection (resampling)

$$\text{projected} = \text{interp}(\mathbf{x},\; n_{\text{features}})$$

Linear interpolation resamples input to exactly $n_{\text{features}}$ bins.

### FPE quantisation

$$\text{weighted}_i = w_i \cdot \text{projected}_i$$

$$\text{bin}_i = \text{clip}\!\left(\lfloor \text{weighted}_i \cdot 255 \rfloor,\; 0,\; 255\right)$$

Each bin index selects a codebook HV, XOR-bound with a deterministic role HV:

$$\text{role\_seed}(i) = (i \cdot 1013 + 5003) \bmod 2^{32}$$

$$\text{hv} = \bigoplus_{i=1}^{n_{\text{features}}} \text{codebook}[\text{bin}_i] \oplus \text{role\_hv}(i)$$

### Hebbian weight update

After receiving reward signal $r$:

$$w_i \leftarrow \text{clip}\!\big(w_i + 0.01 \cdot r \cdot |\text{projected}_i|,\; 0.1,\; 5.0\big)$$

### Feature importance (exponential moving average)

$$\text{importance}_i \leftarrow 0.9 \cdot \text{importance}_i + 0.1 \cdot |\text{projected}_i|$$

---

## 27. Procedural Memory (V13)

*Source: `memory/procedural_memory.py`*

Stores learned motor/action skills as (state HV, action, reward) tuples with
similarity-based recall.

### Recall threshold

$$\text{recall if } \text{sim}(\mathbf{s}_{\text{query}}, \mathbf{s}_{\text{stored}}) \geq 0.85$$

### Skill update (near-duplicate)

If $\text{sim} > 0.95$ for an existing skill, update in-place only if $r_{\text{new}} > r_{\text{old}}$.

### LRU eviction

$$|\text{skills}| > \text{max\_skills} \Rightarrow \text{evict oldest } (|\text{skills}| - \text{max\_skills}) \text{ entries}$$

Default: $\text{max\_skills} = 500$.

### Hit rate

$$\text{hit\_rate} = \frac{\text{hits}}{\text{hits} + \text{misses}}$$

---

## 28. Concept Drift Detection (V13)

*Source: `memory/concept_drift_detector.py`*

Monitors hypervector representations over time and triggers alarms when
concept meanings shift beyond a threshold.

### Drift magnitude

$$d = 1 - \text{sim}(\mathbf{v}_{\text{current}}, \mathbf{v}_{\text{reference}})$$

### Alarm condition

$$\text{alarm} = \begin{cases} \text{True} & \text{if } d > \theta_{\text{drift}} \\ \text{False} & \text{otherwise} \end{cases}$$

Default: $\theta_{\text{drift}} = 0.2$.

### LRU eviction (danger vectors)

Maximum 200 stored danger vectors. When exceeded, the oldest vectors are evicted.

---

## 29. Cross-Modal Associative Binding (V13)

*Source: `memory/cross_modal_associative_memory.py`*

Associates hypervectors from different modalities (e.g., vision + language)
using XOR binding.

### Binding

$$\mathbf{b} = \mathbf{v}_a \oplus \mathbf{v}_b$$

Stores the association between modality-$a$ and modality-$b$ representations.

### Recall

$$\hat{\mathbf{v}}_b = \mathbf{v}_a \oplus \mathbf{b} \approx \mathbf{v}_b$$

This works because $\mathbf{v}_a \oplus (\mathbf{v}_a \oplus \mathbf{v}_b) = \mathbf{v}_b$ (XOR self-inverse).

### Similarity-weighted recall strength

$$\text{strength} = \text{sim}(\mathbf{v}_{\text{query}}, \mathbf{v}_{\text{stored}}) \cdot w_{\text{association}}$$

---

## 30. Causal Rule Auditor (V13)

*Source: `reasoning/causal_rule_auditor.py`*

Re-scores learned rules by combining symbolic confidence with causal graph
evidence, filtering out spurious correlations.

### Direct causal score

$$s_{\text{direct}} = \text{CausalGraph.edge\_strength}(A \to B)$$

### Multi-hop path strength

For an indirect path $A \to X \to B$:

$$s_{\text{indirect}} = s_{A \to X} \cdot s_{X \to B}$$

### Best causal score

$$s_{\text{causal}} = \max(s_{\text{direct}},\; \max_{\text{paths}} s_{\text{indirect}})$$

### Combined score

$$\text{combined} = (1 - w) \cdot \text{confidence} + w \cdot s_{\text{causal}}$$

Default causal weight: $w = 0.4$.

Rules with low causal support are down-weighted even if they have high
empirical confidence, reducing the impact of spurious correlations.

---

## 31. Pattern Generalizer (V13)

*Source: `learning/pattern_generalizer.py`*

Discovers abstract patterns by clustering similar hypervectors and building
prototype representations.

### Clustering threshold

Two HVs are assigned to the same cluster if:

$$\text{sim}(\mathbf{v}_1, \mathbf{v}_2) \geq \theta_{\text{cluster}} = 0.7$$

### Prototype construction

$$\mathbf{p}_C = \text{majority\_bundle}\!\left(\{\mathbf{v}_i : v_i \in C\}\right)$$

### Abstraction level

$$\text{level}(C) = \min\!\left(1,\; \frac{|C|}{\text{min\_members} \cdot 4}\right)$$

Default $\text{min\_members} = 3$. The level saturates at 1.0 when the
cluster has $4\times$ the minimum membership, indicating a well-established
abstract concept.

### Transfer score (analogy-based)

$$\text{transfer}(A \to B) = \text{sim}(\mathbf{v}_{\text{source}}, \mathbf{p}_A) \cdot \text{sim}(\mathbf{v}_{\text{target}}, \mathbf{p}_B)$$

---

## 32. Bridge Projection (V14)

*Source: `adapters/rich_text_adapter.py`, `vsa/vsa_embedding_bridge.py`*

The Rich Perception adapters use a random projection matrix to convert
dense real-valued embeddings into binary hypervectors.

### Bridge projection

Let $\mathbf{e} \in \mathbb{R}^{d_{\text{bridge}}}$ be the embedding produced
by a bridge model (e.g. sentence-transformers with $d_{\text{bridge}} = 384$).
Let $\mathbf{P} \in \mathbb{R}^{d \times d_{\text{bridge}}}$ be a fixed random
Gaussian projection matrix ($d = 10{,}240$ bits).

$$\mathbf{b} = \text{sign}\!\left(\mathbf{P}\,\mathbf{e}\right) \in \{0, 1\}^d$$

where sign is taken in the $\{-1, +1\}$ bipolar convention and mapped to
$\{0, 1\}$:

$$b_i = \begin{cases} 1 & \text{if } (\mathbf{P}\,\mathbf{e})_i \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

This is a Johnson–Lindenstrauss–style random projection: the Hamming similarity
of bridge HVs approximates the cosine similarity of the original embeddings
(for large $d$).

---

## 33. Distillation Quality (V14)

*Source: `learning/perception_distiller.py`*

`PerceptionDistiller` tracks whether the internal (pure VSA) encoding
of a percept converges to the bridge encoding. Let $\mathbf{h}_b$ be the
bridge HV and $\mathbf{h}_p$ be the pure/internal HV for the same input.

### Per-observation quality

$$q_t = \text{sim}(\mathbf{h}_b^{(t)},\, \mathbf{h}_p^{(t)})$$

When cosine similarity is available (bipolar vectors):

$$q_t = \frac{\cos\!\bigl(\mathbf{h}_b^{(t)},\, \mathbf{h}_p^{(t)}\bigr) + 1}{2} \in [0,\, 1]$$

When only Hamming similarity is available (binary vectors):

$$q_t = 1 - \frac{d_H\!\bigl(\mathbf{h}_b^{(t)},\, \mathbf{h}_p^{(t)}\bigr)}{d} \in [0,\, 1]$$

### Rolling average quality

Let $W = 100$ (window size). The rolling average over the last $W$
observations is:

$$\bar{q} = \frac{1}{\min(t, W)} \sum_{i=\max(0,\, t-W)}^{t} q_i$$

### Graduation criterion

A modality is **graduated** (bridge dependency can be removed) when:

$$\text{graduated} = \left[\bar{q} \geq \theta \right]$$

where $\theta$ is `distillation_threshold` (default $\theta = 0.80$), and
the rolling window contains at least 10 observations.

---

## References

| Reference | Used in |
|---|---|
| Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. | VSA foundations (§2) |
| Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*, 1(2), 139–159. | Near-orthogonality, binding/bundling (§2) |
| Plate, T.A. (2003). *Holographic Reduced Representations*. CSLI Publications. | FHRR phasor VSA (§18) |
| Plate, T.A. (1994). Distributed representations and nested compositional structure. PhD Thesis, U. of Toronto. | VSA binding algebra (§2) |
| Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press. | Global Workspace Theory (§5) |
| Dehaene, S., Changeux, J-P., & Nadal, J-P. (2011). Global workspace and metacognition. *PNAS*. | GWT implementation (§5) |
| Cheng, P.W. & Novick, L.R. (1990). A probabilistic contrast model of causal induction. *JPSP*, 58(4), 545–567. | ΔP causal discovery (§6) |
| Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs. *J. Physiol. Pathol. Gén.*, 9, 620–635. | LIF model (§9) |
| Bi, G-Q. & Poo, M-M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neuroscience*, 18(24), 10464–10472. | STDP learning (§9) |
| Russell, J.A. (1980). A circumplex model of affect. *JPSP*, 39(6), 1161–1178. | Emotion model (§11) |
| Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row. | 8 basic emotions (§11) |
| Oja, E. (1982). Simplified neuron model as a principal component analyzer. *J. Math. Biology*, 15(3), 267–273. | Oja's rule (§8) |
| Fikes, R.E. & Nilsson, N.J. (1971). STRIPS: A new approach to theorem proving. *AI*, 2(3–4), 189–208. | STRIPS planning |
| Indyk, P. & Motwani, R. (1998). Approximate nearest neighbors. *STOC '98*, 604–613. | LSH (§13) |
| Rosch, E. (1973). Natural categories. *Cognitive Psychology*, 4(3), 328–350. | Prototype theory (§31) |
| Church, K.W. & Hanks, P. (1990). Word association norms. *Computational Linguistics*, 16(1), 22–29. | PMI learning (§3) |
| Grice, H.P. (1975). Logic and conversation. *Syntax and Semantics*, 3, 41–58. | Gricean maxims |
| Brill, E. (1992). A simple rule-based POS tagger. *ANLP '92*, 152–155. | BrillPosTagger |
| Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. | Conformal prediction (§24) |
| Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138. | Active inference motivation (§23, §28) |
| Jaynes, E.T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620–630. | MaxEnt / KLE entropy (§23) |
| Collins, A.M. & Loftus, E.F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82(6), 407–428. | Spreading activation (§4) |

---

## V4 Architecture Formulae

### 1. Procedural Memory — Familiarity Threshold (V4)

Threshold lowered from V3 value of 0.85 to **0.72** to enable faster System-1 activation:

```
familiar(q, s) = similarity(q.context_hv, s.context_hv) ≥ 0.72
```

Where `q` is the query context and `s` is a cached skill. At threshold 0.72, a state is considered "familiar" after fewer exposures, enabling earlier fast-path activation.

### 2. LSH Bucket Key Formula

For a binary hypervector `b ∈ {0,1}^D`, the LSH bucket key is computed using `n` fixed random projections `{p_0, ..., p_{n-1}}` where `p_j ∈ {0,...,D-1}` (sampled with seed `0xDEAD`):

```
lsh_key(b, n) = Σ_{j=0}^{n-1} (b[p_j] & 1) × 2^j
```

This maps D=10,240 dimensions to an n-bit integer key in O(n) time. Default n=16 gives 65,536 buckets. Expected bucket size for N skills: N/65,536.

### 3. Majority-Vote Bundle Formula (V4)

For N binary hypervectors `{v_0, ..., v_{N-1}}`:

```
bundle[i] = 1  if  Σ_k v_k[i] > N/2
           = 0  if  Σ_k v_k[i] < N/2
           = i % 2  (tie-break, deterministic)
```

The Rust `bundle_hvs()` function implements this directly. The Python pairwise `bundle(A, B)` uses random tie-breaking with a content-derived seed — mathematically equivalent in expectation.

### 4. Sentence HV Encoding — Positional Role-Filler (V4)

For a token sequence `[w_0, w_1, ..., w_{L-1}]`:

```
sentence_hv = Bundle_{i=0}^{L-1} [ word_hv(w_i) XOR position_hv(i) ]
```

Where:
- `word_hv(w)` = distributional context HV from DistributionalCodebook
- `position_hv(i)` = `HyperVector(seed = 0x50531 + i)` (fixed, deterministic)
- `Bundle` = iterative pairwise majority-vote bundle

This encodes word order: permuting tokens changes the sentence HV.
