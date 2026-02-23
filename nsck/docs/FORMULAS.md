# Theory, Formulas, Calculations, and Design Goals

This document covers the complete mathematical and theoretical foundations of NSCK: every formula, derivation, proof, and design goal — extracted directly from the source code and the research ideas that motivated each component.

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

---

## 1. Design Goals and Theoretical Motivation

NSCK was designed to answer a specific question:

> *Can a cognitive system be simultaneously intelligent, transparent, and biologically plausible — without matrix multiplication or gradient descent?*

The core thesis is: **Vector Symbolic Architecture provides a common mathematical language for both neural-style association and symbolic reasoning.** The same 10,240-bit binary HV is used for every representation — concepts, predicates, sensory inputs, episodes, rules — so there is no translation layer between neural and symbolic processing.

### Primary design goals

| Goal | Implementation | Status |
|---|---|---|
| Glass-box transparency | ExplanationGenerator + ThoughtTrace | Achieved — every decision is traceable |
| No black boxes | No neural inference; only VSA + symbolic | Achieved |
| Biological plausibility | GWT, Hebbian, SNN, curiosity | Achieved |
| Domain agnosticism | register_task() interface | Achieved |
| Self-contained (no external AI) | All reasoning in-process | Achieved |
| Efficient on commodity hardware | Rust concurrent layer | Achieved — 21–206× speedup |
| Continual learning (no forgetting) | Episode replay + tenure-based rules | Partial |
| Counterfactual reasoning | CausalGraph + CounterfactualReasoner | Achieved |

### Theoretical frameworks used

| Framework | Source | NSCK component |
|---|---|---|
| Binary VSA (B-VSA) | Kanerva (1988, 2009) | All knowledge representation |
| Global Workspace Theory (GWT) | Baars (1988), Dehaene (2011) | GlobalWorkspace competition |
| Spreading activation | Collins & Loftus (1975) | SemanticMemory graph traversal |
| Δ-P causal inference | Cheng & Novick (1990) | CausalDiscovery |
| Leaky Integrate-and-Fire | Lapicque (1907) | SNN perception |
| STDP (Spike-Timing-Dependent Plasticity) | Bi & Poo (1998) | SNN learning |
| Russell's Circumplex | Russell (1980) | EmotionSystem |
| Plutchik's Wheel | Plutchik (1980) | EmotionSystem |
| Oja's rule | Oja (1982) | HebbianMatrix |
| STRIPS planning | Fikes & Nilsson (1971) | STRIPSPlanner |
| Locality-Sensitive Hashing | Indyk & Motwani (1998) | EpisodicMemory retrieval |

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

The last equality holds because $\|\hat{\mathbf{v}}\| = \sqrt{d}$ for all binary vectors.

**Why cosine over Hamming?** After repeated XOR/bundle operations, noise accumulates in proportion to the number of operations. Cosine similarity in bipolar space is more robust to this noise because the dot product is linear in the number of agreeing bits.

**Normalisation to [0,1]:**

$$\text{sim\_robust}(\mathbf{A}, \mathbf{B}) = \frac{\text{cos\_sim}(\mathbf{A}, \mathbf{B}) + 1}{2}$$

### Binding (XOR / MAP binding)

$$\mathbf{A} \otimes \mathbf{B} = \mathbf{A} \oplus \mathbf{B} \quad \text{(element-wise XOR)}$$

Properties:
- **Self-inverse**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{B} = \mathbf{A}$ — unbinding is the same operation
- **Commutative**: $\mathbf{A} \otimes \mathbf{B} = \mathbf{B} \otimes \mathbf{A}$
- **Associative**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{C} = \mathbf{A} \otimes (\mathbf{B} \otimes \mathbf{C})$
- **Quasi-orthogonal to operands**: $\text{sim}(\mathbf{A} \otimes \mathbf{B}, \mathbf{A}) \approx 0.5$

**Role-filler binding** for concepts:
$$\text{HV}(\text{concept}) = \text{HV}(\text{name}) \oplus \bigoplus_{\text{prop} \in \text{properties}} \text{HV}(\text{prop}) \oplus \text{HV}(\text{value})$$

**Proof of self-inverse property:**
$$(\mathbf{A} \oplus \mathbf{B}) \oplus \mathbf{B} = \mathbf{A} \oplus (\mathbf{B} \oplus \mathbf{B}) = \mathbf{A} \oplus \mathbf{0} = \mathbf{A}$$
(using XOR associativity and $x \oplus x = 0$)

### Bundling (Majority Vote / BSC superposition)

For two vectors:

$$(\mathbf{A} + \mathbf{B})_i = \begin{cases} A_i & \text{if } A_i = B_i \\ \text{Bernoulli}(0.5) & \text{if } A_i \neq B_i \end{cases}$$

> **Implementation note:** In `HyperVectorPy`, the tie-breaking mask is seeded deterministically from the input vectors (XOR-weight of the first 64 bits of each), so `bundle(A, B)` always returns the same result for the same pair. The mathematical properties above are unaffected — the tie-bits are still independent of the differing positions.

Properties:
- **Similar to both operands**: $\mathbb{E}[\text{sim}(\mathbf{A} + \mathbf{B}, \mathbf{A})] = 0.75$
  - Matching bits: $A_i = B_i$ → result agrees with $A$ with probability 1
  - Differing bits: probability 0.5 → expected agreement $= 0.5$
  - Total: $\frac{d_{\text{match}}}{d} \cdot 1 + \frac{d-d_{\text{match}}}{d} \cdot 0.5 \approx 0.5 + 0.5 \cdot 0.5 = 0.75$ for random $A, B$

For $n$ vectors bundled: $\mathbb{E}[\text{sim}(\text{bundle}, \mathbf{v}_i)] \approx 0.5 + \frac{0.5}{n}$

### Permutation (Circular Shift)

$$\rho^k(\mathbf{v})_i = v_{(i + k) \bmod d}$$

Properties:
- **Invertible**: $\rho^{-k}(\rho^k(\mathbf{v})) = \mathbf{v}$
- **Orthogonal**: $\text{sim}(\rho^k(\mathbf{v}), \mathbf{v}) \approx 0.5$ for $k \neq 0$
- **Consistent**: same shift applied to both vectors → same similarity as without shift

**Role in text encoding:** $\rho^i(\text{word\_hv})$ encodes word $w$ at position $i$, making word order distinguishable.

### CleanupMemory

Given a set of registered clean prototypes $\{(\ell_j, \mathbf{p}_j)\}$, cleanup finds:

$$\text{cleanup}(\mathbf{v}) = \argmax_j \text{sim}(\mathbf{v}, \mathbf{p}_j) \quad \text{subject to} \quad \text{sim} > \theta$$

where $\theta$ is the cleanup threshold (default 0.4). This implements the **nearest-neighbour decoder** for VSA.

**Why cleanup is necessary:** After $n$ binding operations, noise accumulates. For $k$ operations each adding $\sim 0.01$ noise bits, the total noise is $\sim 0.01k$. Cleanup snaps the result back to the nearest prototype, resetting the noise.

---

## 3. Text Encoding and Semantic Folding

*Source: `language/lingua_cortex.py`, `nsck_ai_model/ai_engine.py`*

### Step 1: Word HV generation (deterministic)

$$\mathbf{w}_{\text{base}} = \text{HyperVector}(\text{hash}(w) \bmod 2^{32})$$

Using a seeded RNG ensures the same word always gets the same HV (deterministic across sessions).

### Step 2: Positional encoding

$$\mathbf{w}_{(i)} = \rho^i(\mathbf{w}_{\text{base}})$$

where $i$ is the word's position in the sentence. The permutation makes $\mathbf{w}_{(i)}$ quasi-orthogonal to $\mathbf{w}_{(j)}$ for $i \neq j$, preserving word-order information.

Maximum shift: `MAX_POSITION_SHIFT = 64` (sufficient for typical sentence lengths; shifts beyond $d/2$ create near-random positions).

### Step 3: Sentence HV (superposition)

$$\mathbf{s} = \bigoplus_{i=1}^{n} \mathbf{w}_{(i)}$$

(majority-vote bundle of all positional word HVs)

### Step 4: Context HV (window-based)

For context window of size $c$:

$$\mathbf{c}_i = \bigoplus_{j=\max(0,i-c)}^{\min(n,i+c)} \mathbf{w}_{(j)}$$

This captures local context without global sequence encoding overhead.

### Property: Associative retrieval

Given a sentence HV $\mathbf{s}$ and a word HV $\mathbf{w}$, we can retrieve position-linked words:

$$\text{probe} = \rho^{-i}(\mathbf{s} \otimes \rho^{-i}(\mathbf{w})) \approx \mathbf{w}_{\text{neighbour}}$$

This is the **distributed associative memory** property of VSA.

---

## 4. Spreading Activation

*Source: `memory/semantic_memory.py`*

### Algorithm

Given start concepts $S = \{s_1, \ldots, s_k\}$ with initial activation $a_{s_i} = 1.0$:

$$a_j^{(t+1)} = a_j^{(t)} + \sum_{(i,j) \in E} a_i^{(t)} \cdot \gamma \cdot w_{\text{rel}(i,j)}$$

where:
- $\gamma = 0.7$ is the global decay factor
- $w_{\text{rel}}$ is the relation-type weight:
  - $w_{\text{is\_a}} = 0.9$, $w_{\text{has\_property}} = 0.7$, $w_{\text{causes}} = 0.6$
  - $w_{\text{part\_of}} = 0.5$, $w_{\text{similar\_to}} = 0.4$

### Convergence

After $T$ steps, activation propagated via a chain of length $T$:

$$a^{(T)} \leq (\gamma \cdot w_{\max})^T = (0.7 \cdot 0.9)^T = 0.63^T$$

For $T = 3$ (default): $0.63^3 \approx 0.25$ — activation decays to ~25% of its initial value after 3 hops. This naturally limits the search radius.

### Complexity bound

Without the frontier cap, spreading has $O(|V| \cdot |E| \cdot T)$ complexity. The cap of top-200 active nodes per step bounds this to $O(200 \cdot \bar{d} \cdot T)$ where $\bar{d}$ is mean out-degree — typically $O(10)$, giving $O(6000)$ per call.

---

## 5. Global Workspace Theory

*Source: `reasoning/global_workspace.py`*

### Coalition activation

$$\alpha(C) = s(C) + r(C) + e(C) + 0.5 \cdot q(C)$$

where:
- $s(C)$ = `base_salience` — intrinsic importance
- $r(C)$ = `relevance` — match with current context and goal
- $e(C)$ = `affect_match` — match with emotional drives
- $q(C)$ = `sender_confidence` — module's self-reported confidence

### Competition rule

$$C^* = \argmax_{C \in \mathcal{C}} \alpha(C)$$

### Mental rehearsal veto

Before committing to $C^*$:

1. Predict next state: $\mathbf{s}' = \text{WorldModel.predict}(C^*.action, \mathbf{s})$
2. Compute danger similarity: $d = \text{sim}(\mathbf{s}', \mathbf{v}_{\text{danger}})$
3. If $d > \theta_{\text{danger}}$: veto $C^*$ and try $C^{(2)} = \argmax_{C \neq C^*} \alpha(C)$

This implements **predictive safety** — the system refuses actions that lead to predicted dangerous states before executing them.

---

## 6. Causal Discovery (ΔP Statistics)

*Source: `reasoning/causal_reasoning.py`*

### Delta-P formula

$$\Delta P(c \to e) = P(e \mid c) - P(e \mid \neg c)$$

Interpretation:
- $\Delta P > 0$: $c$ is a positive cause of $e$
- $\Delta P < 0$: $c$ is a preventive cause of $e$ (inhibitor)
- $\Delta P \approx 0$: $c$ and $e$ are independent (spurious correlation)

### With Laplace smoothing

$$P(e \mid c) = \frac{N(e, c) + \alpha}{N(c) + 2\alpha}, \quad P(e \mid \neg c) = \frac{N(e, \neg c) + \alpha}{N(\neg c) + 2\alpha}$$

where $\alpha = 1$ (Laplace pseudocount). This prevents zero-probability estimates from sparse data.

**Property:** With as few as $n=2$ observations, Laplace smoothing gives a non-degenerate estimate:
$$P(e \mid c)_{\text{Laplace}} \in \left[\frac{\alpha}{1+2\alpha}, \frac{1+\alpha}{1+2\alpha}\right] = [0.33, 0.67]$$

### Causal chain strength

For a chain $c_1 \to c_2 \to \cdots \to c_n$:

$$\text{strength}(c_1 \to c_n) = \prod_{i=1}^{n-1} \Delta P(c_i \to c_{i+1})$$

This is the joint probability assuming independence along the chain.

### Counterfactual: do-operator

To simulate "what if $c$ had not occurred?":

1. Remove all causal links originating from $c$.
2. Propagate activation from the modified graph.
3. Compare resulting state against the original.

This implements Pearl's do-calculus in a simplified form appropriate for discrete symbolic states.

---

## 7. Q-Learning and Reward Processing

*Source: `reasoning/cognitive_engine.py`*

### State-action value (tabular Q-learning)

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where:
- $\alpha = 0.1$ (learning rate, from `NSCKConfig`)
- $\gamma = 0.95$ (discount factor, from `NSCKConfig`)
- $s$ = `_get_state_key(state, task_tag)` — hashable state representation
- $r$ = scalar reward from `record_outcome()`

State key = frozen set of active predicates → `str` → hashable.

### Confidence from Q-values

$$\text{confidence}(s, a) = \frac{Q(s, a) - Q_{\min}}{Q_{\max} - Q_{\min}} \quad \text{(normalised to [0, 1])}$$

Cold start: if $|Q| = 0$, confidence defaults to 0.5.

### Exploration (ε-greedy + curiosity)

Standard ε-greedy is replaced by curiosity-driven exploration:

$$\text{explore} = (\text{novelty}(s) > \theta_N) \wedge (\text{confidence}(s) < 0.5) \vee (\text{learning\_progress} < \theta_P)$$

This is more sample-efficient than uniform ε-greedy because it focuses exploration on genuinely novel states.

---

## 8. Hebbian Learning and Oja's Rule

*Source: `learning/hebbian.py`, `rust_snn/src/hebbian.rs`*

### Basic Hebbian rule

$$\Delta w_{ij} = \eta \cdot x_i \cdot y_j$$

"Neurons that fire together, wire together." Problem: this is unbounded — weights grow without limit.

### Oja's rule (normalised Hebbian)

$$\Delta w_{ij} = \eta \left( x_i y_j - y_j^2 w_{ij} \right)$$

The forgetting term $-y_j^2 w_{ij}$ prevents weight explosion.

**Proof of stability:** At equilibrium, $\Delta w = 0$:
$$x_i y_j = y_j^2 w_{ij} \Rightarrow w_{ij} = \frac{x_i}{y_j}$$

Since $y_j = \sum_i w_{ij} x_i$ (linear neuron), this converges to the first principal component of the input distribution. Oja (1982) proved that this rule converges to:

$$\mathbf{w}^* = \text{PC}_1(\mathbf{x}) \quad (\text{first principal component})$$

In practice, this means the Hebbian matrix converges to capture the most frequent co-occurrence patterns — exactly what we want for associative concept learning.

### VSA-level Hebbian association

After learning, the association matrix $W$ can be used to find related concepts:

$$\text{related}(c) = \text{top-}k\left(\{(c', W_{cc'}) : c' \in \text{concepts}\}\right)$$

---

## 9. Spiking Neural Networks

*Source: `perception/snn_perception.py`, `rust_snn/src/lib.rs`*

### Leaky Integrate-and-Fire (LIF) model

Continuous-time membrane equation:

$$\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + I(t)$$

Discrete-time update (Euler method, timestep $\Delta t$):

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m} \left[ -(V(t) - V_{\text{rest}}) + I(t) \right]$$

**Spike condition:**
$$\text{if } V(t) \geq V_{\text{thresh}}: \text{ emit spike}, V \leftarrow V_{\text{reset}}, \text{ enter refractory period}$$

**Biological parameters (defaults):**
- $\tau_m = 20 \text{ ms}$ (membrane time constant)
- $V_{\text{rest}} = -70 \text{ mV}$, $V_{\text{thresh}} = -55 \text{ mV}$, $V_{\text{reset}} = -75 \text{ mV}$

### Rate coding

Scalar input $x \in [0, 1]$ → spike train:

$$\Pr[\text{spike at time } t] = x$$

(Bernoulli process — Poisson approximation for small $x$)

### Temporal coding

Scalar input $x$ → spike time $t_{\text{spike}} = T_{\max}(1 - x)$:

Early spikes = high input. Late spikes = low input. Rank-order code.

### STDP (Spike-Timing-Dependent Plasticity)

For a pre-synaptic spike at time $t_{\text{pre}}$ and post-synaptic spike at $t_{\text{post}}$:

$$\Delta w = \begin{cases}
A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t = t_{\text{post}} - t_{\text{pre}} > 0 \quad \text{(causal)} \\
-A_- e^{-\Delta t / \tau_-} & \text{if } \Delta t < 0 \quad \text{(anti-causal)}
\end{cases}$$

Parameters: $A_+ = 0.01$, $A_- = 0.012$, $\tau_+ = 20 \text{ ms}$, $\tau_- = 20 \text{ ms}$.

**Biological interpretation:** Causal (pre before post) → strengthen synapse. Anti-causal (post before pre) → weaken synapse. This implements temporal credit assignment.

---

## 10. Curiosity and Novelty

*Source: `learning/curiosity.py`*

### Novelty score

$$\text{novelty}(s) = 1 - \max_{v \in \text{known}} \text{sim}(\text{encode}(s), v)$$

where `known` = set of HVs for previously visited situations.

**Range:** $[0, 1]$. Novel state → novelty ≈ 1.0. Familiar state → novelty ≈ 0.0.

**Comparison with RND (Random Network Distillation):** RND uses a neural network's prediction error as a novelty proxy. NSCK uses direct HV similarity — simpler, no neural network required, and interpretable (you can see which stored prototype is the nearest neighbour).

### Learning progress

$$\text{progress}(t) = \frac{1}{\lfloor T/2 \rfloor} \sum_{\tau=t-T}^{t-T/2} r_\tau - \frac{1}{\lfloor T/2 \rfloor} \sum_{\tau=t-T/2}^{t} r_\tau$$

(improvement rate: difference between success rate in second half vs first half of window $T$)

### Exploration decision

$$\text{explore} = \big[(\text{novelty} > \theta_N) \wedge (\text{confidence} < 0.5)\big] \vee (\text{progress} < \theta_P)$$

Default: $\theta_N = 0.7$, $\theta_P = 0.01$.

---

## 11. Emotion System (Affective Computing)

*Source: `cognitive/emotion_system.py`*

### Russell's Circumplex model

Emotions are points in 2D affect space:

$$\text{emotion}(\text{valence}, \text{arousal})$$

$$V \in [-1, +1] \quad \text{(negative ↔ positive)}$$
$$A \in [0, 1] \quad \text{(calm ↔ excited)}$$

### Closest prototype (nearest-neighbour classification)

$$\hat{e} = \argmin_{e \in \text{Plutchik}} \left\| (V, A) - (V_e, A_e) \right\|_2$$

Plutchik prototypes:
| Emotion | $V_e$ | $A_e$ |
|---|---|---|
| joy | 0.8 | 0.7 |
| trust | 0.5 | 0.2 |
| fear | -0.7 | 0.8 |
| surprise | 0.0 | 0.9 |
| sadness | -0.6 | 0.2 |
| disgust | -0.5 | 0.4 |
| anger | -0.5 | 0.8 |
| anticipation | 0.3 | 0.5 |
| neutral | 0.0 | 0.0 |

### Emotion blending

Instead of hard classification, maintain a weighted blend:

$$b_e = \frac{e^{-\|(\hat{V}, \hat{A}) - (V_e, A_e)\|^2 / 2\sigma^2}}{\sum_{e'} e^{-\|(\hat{V}, \hat{A}) - (V_{e'}, A_{e'})\|^2 / 2\sigma^2}}$$

(softmax over negative squared distances — Gaussian kernel)

### Drive-to-emotion mapping

Homeostatic drives (hunger, threat, curiosity, …) update $(V, A)$ before emotion classification:

$$V \leftarrow V + \Delta V(\text{drives}), \quad A \leftarrow A + \Delta A(\text{drives})$$

Each drive has a registered $(dV, dA)$ vector. Multiple drives are additively combined.

---

## 12. Self-Model Confidence

*Source: `cognitive/self_model.py`*

### Running success ratio

$$\hat{p}(T, k) = \frac{\text{successes}(T, k)}{\text{attempts}(T, k)}$$

where $T$ = task tag, $k$ = context key.

### Context-aware prediction

If $\text{attempts}(T, k) \geq n_{\text{min}}$: use $\hat{p}(T, k)$.

Otherwise (cold start): use global average:

$$\hat{p}_{\text{global}}(T) = \frac{\text{successes}(T, \cdot)}{\text{attempts}(T, \cdot)}$$

And if $\text{attempts}(T, \cdot) = 0$: default to $\hat{p} = 0.5$.

### Calibration error

$$\text{cal\_error}(T) = |\hat{p}(T, \cdot) - \bar{r}(T)|$$

where $\bar{r}(T)$ is the recent rolling mean of actual rewards. Well-calibrated systems have $\text{cal\_error} \approx 0$.

---

## 13. Episodic Memory and LSH

*Source: `memory/episodic_memory.py`*

### Locality-Sensitive Hashing

To enable fast approximate k-NN over binary HVs, we use random projection LSH:

1. Generate $m$ random projection vectors $\mathbf{r}_1, \ldots, \mathbf{r}_m \in \{-1, +1\}^d$.
2. Hash vector $\mathbf{v}$: $h_j(\mathbf{v}) = \text{sgn}(\mathbf{r}_j \cdot \mathbf{v})$
3. Concatenate $m$ bits into a bucket key: $B(\mathbf{v}) = \text{concat}(h_1(\mathbf{v}), \ldots, h_m(\mathbf{v}))$

**Collision probability:** For two vectors with cosine similarity $c$:

$$\Pr[h(\mathbf{u}) = h(\mathbf{v})] = 1 - \frac{\theta}{\pi}, \quad \theta = \arccos(c)$$

For NSCK: $m = 16$ bits per key, `lsh_num_tables = 8` hash tables. This gives:
- True positive rate (for $\text{sim} > 0.7$): $\approx 0.99$
- False positive rate: $\approx 0.01$

### Two-tier design

**Hot tier** (in-memory `deque`, capacity 500): Recent episodes — fast $O(1)$ amortised access.

**Warm tier** (SQLite): All episodes — persists across sessions, $O(\log n)$ access via B-tree index.

**Consolidation** (`sleep()`): Move hot → warm, prune episodes with `impact_score < \theta_{\text{prune}}`.

$$\text{impact\_score}(e) = |r_e| + \text{novelty}(e)$$

This prioritises memorable (high reward) and surprising (high novelty) episodes for retention.

---

## 14. Rule Learner

*Source: `reasoning/rule_learner.py`*

### Rule confidence

$$\text{confidence}(R) = \frac{\text{successes}(R)}{\text{support}(R)}$$

where $\text{support}(R)$ = number of times the condition set was matched.

### Rule induction threshold

A candidate is promoted to a rule when:

$$\text{support}(R) \geq n_{\text{min}} \quad \text{AND} \quad \text{confidence}(R) \geq c_{\text{min}}$$

Default: $n_{\text{min}} = 5$, $c_{\text{min}} = 0.3$.

### Tenure progression

| State | Condition | Demotion |
|---|---|---|
| new | `support < n_min` | N/A |
| bootstrap | `support ≥ n_min AND confidence ≥ c_min` | confidence falls below threshold |
| tenured | held bootstrap state for `tenure_threshold` cycles | requires `2×` counter-evidence |

Tenured rules are stable — prevents learned rules from being wiped by short-term noise.

### Pruning

Rules with `success_rate < min_success_rate = 0.7` after `min_support` observations are pruned. This prevents the rule store from filling with low-quality rules.

---

## 15. Analogy Engine

*Source: `reasoning/analogy.py`*

### Structural similarity between domains

Given domains $D_1$ and $D_2$ with concept sets $C_1, C_2$:

1. Compute similarity matrix: $S_{ij} = \text{sim}(\text{HV}(c_i), \text{HV}(c_j))$ for all $c_i \in C_1, c_j \in C_2$.

2. Find optimal concept mapping via greedy assignment:
   $$M = \{(c_i, \argmax_{c_j} S_{ij})\}$$

3. Overall analogy quality:
   $$\text{quality}(M) = \frac{1}{|M|} \sum_{(i,j) \in M} S_{ij}$$

### Analogy-based transfer

Once a mapping $M: D_1 \to D_2$ is found, transfer rule $R_1 = (\text{IF } P_1 \text{ THEN } a_1)$ to $D_2$:

1. Map predicates: $P_2 = \{M(p) : p \in P_1\}$
2. Map action: $a_2 = M(a_1)$
3. Register $R_2$ in $D_2$ with reduced initial confidence (transfer discount)

---

## 16. Response Composition (IDF Scoring)

*Source: `nsck_ai_model/response_composer.py`*

### IDF-weighted relevance score

For candidate sentence $s$ and query $q$:

$$\text{score}(s, q) = \sum_{w \in s \cap q} \text{IDF}(w)$$

$$\text{IDF}(w) = \log \frac{N}{1 + |\{s \in \text{corpus} : w \in s\}|}$$

Words rare in the corpus but present in both query and candidate get high weight — this rewards relevant, specific responses over generic ones.

### Jaccard deduplication

Two sentences $s_1, s_2$ are considered near-duplicates if:

$$J(s_1, s_2) = \frac{|\text{tokens}(s_1) \cap \text{tokens}(s_2)|}{|\text{tokens}(s_1) \cup \text{tokens}(s_2)|} \geq 0.5$$

After IDF scoring, candidates are greedily deduplicated: add each candidate if its Jaccard similarity to all already-selected sentences is below 0.5.

---

## 17. Safety and Veto Logic

*Source: `reasoning/global_workspace.py`, `cognitive/metacognition.py`*

### Mental rehearsal veto condition

Let $\mathbf{v}_{\text{danger}} \in \mathbb{R}^d$ be the known danger HV:

$$\text{veto}(C) = \text{sim}(\text{WorldModel.predict}(C.\text{action}, \mathbf{s}), \mathbf{v}_{\text{danger}}) > \theta_D$$

Default: $\theta_D = 0.8$.

**Cascading veto:** If the top-1 coalition is vetoed, try top-2, top-3, etc. If all coalitions are vetoed, fall back to the default action (defined per task — usually the most conservative).

### SafetyGate constraint check

Symbolic constraints are checked before the GWT competition:

$$\text{violates}(C, s) = \exists \text{ constraint } \phi: \phi(s) \wedge (\text{C.content} \in \text{forbidden}(\phi))$$

If any registered safety predicate fires, the coalition is removed from the competition entirely.

### Metacognitive performance threshold

$$\text{should\_sleep}() = \bar{r}_{\text{recent}} < \bar{r}_{\text{baseline}} - \sigma_r$$

Where $\bar{r}_{\text{recent}}$ is the rolling mean reward over the last 50 steps and $\sigma_r$ is its standard deviation. Performance degradation triggers offline consolidation.

---

## 18. V4-V7 Additions: Formulas and Proofs

*Sources: `vsa/hypervec_py.py`, `reasoning/spatial_reasoning.py`, `language/distributional_semantics.py`, `language/pragmatics.py`, `learning/pmi_learner.py`*

### 18.1 VSA Negation (V6)

**Definition.** Let $\mathbf{S} \in \{0,1\}^d$ be the fixed negation seed:
$$S = \texttt{0xDEADBEEFCAFEBABE} \text{ (repeated to fill } d \text{ bits)}$$

The negation of a hypervector $\mathbf{A}$ is:
$$\neg\mathbf{A} = \mathbf{A} \oplus \mathbf{S}$$

**Properties:**

**(a) Orthogonality** — negation produces a vector near-orthogonal to the original:
$$\mathbb{E}[\text{sim}(\mathbf{A}, \neg\mathbf{A})] = \frac{d - \text{count}(\mathbf{S})}{d} \approx 0.5$$

Since $\mathbf{S}$ has approximately $d/2$ set bits (random seed fills ~50% of bits), and bit $i$ of $\neg\mathbf{A}$ equals $A_i$ whenever $S_i = 0$ and is flipped whenever $S_i = 1$, half the bits are flipped on average → Hamming distance $\approx d/2$ → similarity $\approx 0.5$.

**Verified empirically:**
```
sim(hv, negate(hv)) = 0.5025  (10240-bit HV, seed 0xDEADBEEFCAFEBABE)
```

**(b) Involution** — double negation recovers the original (negate is self-inverse):
$$\neg(\neg\mathbf{A}) = (\mathbf{A} \oplus \mathbf{S}) \oplus \mathbf{S} = \mathbf{A} \oplus (\mathbf{S} \oplus \mathbf{S}) = \mathbf{A} \oplus \mathbf{0} = \mathbf{A}$$

**Verified empirically:**
```
sim(hv, negate(negate(hv))) = 1.0000
```

**(c) Application** — negation encodes the absence of a property:
$$\text{HV}(\text{"not } p\text{"}) = \neg \text{HV}(p) = \text{HV}(p) \oplus \mathbf{S}$$

Queries for $p$ and for $\neg p$ will both score $\sim 0.5$ against each other, properly distinguishing presence from absence.

---

### 18.2 FPE Bit-Flip Position Encoding for Spatial Reasoning (V5)

*Source: `reasoning/spatial_reasoning.py`*

Classical VSA permute-based position encoding fails for spatial reasoning because `permute(v, k)` has $\text{sim}(\text{permute}(v, k), v) \approx 0.5$ for **all** $k \neq 0$ — it does not encode proximity.

**FPE Bit-Flip** encodes proximity by flipping bits proportional to distance:

Let $\mathbf{b}_{\text{axis}} \in \{0,1\}^d$ be a random base hypervector for axis $a$.  
Let $x_{\text{norm}} \in [0, 1]$ be the normalized coordinate along axis $a$.

$$\text{pos\_hv}(x) = \mathbf{b}_{\text{axis}} \oplus \text{flip}(\mathbf{b}_{\text{axis}},\ k)$$

where $\text{flip}(\mathbf{v}, k)$ flips exactly $k = \lfloor x_{\text{norm}} \cdot \alpha \rfloor$ randomly selected bits, with $\alpha = \texttt{\_AXIS\_FLIP\_BITS} = 50$.

**Similarity gradient:**
$$\text{sim}(\text{pos\_hv}(x_1), \text{pos\_hv}(x_2)) \approx 1 - \frac{|k_1 - k_2|}{d}$$

This is **monotone**: nearby positions produce higher similarity than distant ones.

**Negative coordinates:** offset by $\texttt{\_NEGATIVE\_STEP\_OFFSET} = 100{,}000$ before normalization to avoid the $x=0$ discontinuity:
$$x_{\text{effective}} = x + 100{,}000$$

**Minimum query threshold:** $\texttt{\_MIN\_QUERY\_SIMILARITY} = 0.4$ — below this, no relation is asserted.

---

### 18.3 Distributional Semantics: Context HV Construction (V3/V7)

*Source: `language/distributional_semantics.py`*

NSCK builds distributional word vectors without a neural network, using VSA context hypervectors:

**Step 1: Collect co-occurrence window**  
For each word $w$ in the corpus, collect its context window of radius $r$ (default $r=5$):
$$\text{ctx}(w) = \{(w', k) : w' \text{ appears at offset } k \text{ from } w,\ |k| \leq r\}$$

**Step 2: Build context HV**
$$\text{ctx\_hv}(w) = \bigoplus_{(w', k) \in \text{ctx}(w)} \rho^k(\text{word\_hv}(w'))$$

where $\rho^k$ is circular permutation by $k$ positions (encodes offset direction and distance).

**Step 3: Similarity**
$$\text{distributional\_sim}(w_1, w_2) = \text{sim}(\text{ctx\_hv}(w_1), \text{ctx\_hv}(w_2))$$

**V7 pre-training:** At initialization (when `enable_distributional_semantics=True`), `DistributionalCodebook` is trained on `BUILTIN_CORPUS` — 200 carefully curated sentences covering science, geography, biology, and everyday language. This provides above-random similarity for well-known co-occurring pairs.

**Expected outcomes:**
- `brain` ↔ `memory` (co-occur in neuroscience text): sim ≈ 0.52–0.65
- `cat` ↔ `dog` (rare co-occurrence): sim ≈ 0.50 (near random)
- Pairs with no shared context: sim ≈ 0.50

**Why 0.50 is the random baseline:** Each `word_hv` is a random 10240-bit vector. Two random vectors differ in exactly $d/2 = 5120$ bits in expectation → Hamming similarity = 0.50.

---

### 18.4 Pointwise Mutual Information (V4)

*Source: `learning/pmi_learner.py`*

**PMI formula:**
$$\text{PMI}(a, b) = \log_2 \frac{P(a, b)}{P(a) \cdot P(b)}$$

where:
- $P(a, b)$ = joint probability (fraction of windows containing both $a$ and $b$)
- $P(a)$ = marginal probability of $a$

**PPMI** (Positive PMI — clamp negative values):
$$\text{PPMI}(a, b) = \max(0, \text{PMI}(a, b))$$

Used in `PMILearner` to identify which word pairs co-occur more than chance.  
PPMI = 0 means no above-chance co-occurrence.  
PPMI > 0 means the words are semantically associated.

**Example:**
- "dog" + "bark": 50 co-occurrences out of 10,000 windows  
  → $P(a,b) = 0.005$, $P(a) = 0.01$, $P(b) = 0.01$  
  → $\text{PMI} = \log_2(0.005 / 0.0001) = \log_2(50) \approx 5.64$ (strong association)

---

### 18.5 KG Stop-Concept Filter (V7)

*Source: `language/text_knowledge_learner.py`*

The `_STOP_CONCEPTS` frozenset (53 function words) prevents noise from entering the knowledge graph:
```python
_STOP_CONCEPTS = frozenset({"the", "a", "an", "is", "are", "was", "were",
    "it", "its", "they", "their", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "by", "as", "or",
    "and", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "need"})
```

**Generic relation threshold:**
$$\text{add\_relation}(s, r, o) \iff \text{sim}(\text{HV}(s), \text{HV}(o)) < \tau_{\text{generic}}$$

where $\tau_{\text{generic}} = \texttt{\_GENERIC\_RELATION\_THRESHOLD} = 0.62$.

**Rationale:** If two concepts have similarity > 0.62, they are likely too generic (e.g., binding "is" to itself multiple times) and adding the relation would create noise rather than knowledge.

---

## References

| Reference | Used in |
|---|---|
| Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. | VSA foundations, dimension selection |
| Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159. | Near-orthogonality bounds, binding/bundling |
| Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press. | Global Workspace Theory |
| Dehaene, S., Changeux, J-P., & Nadal, J-P. (2011). Global workspace and metacognition. *Proc. Natl. Acad. Sci.* | GWT implementation |
| Cheng, P.W. & Novick, L.R. (1990). A probabilistic contrast model of causal induction. *Journal of Personality and Social Psychology*, 58(4), 545–567. | ΔP causal discovery |
| Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs. *J. Physiol. Pathol. Général*, 9, 620–635. | Leaky Integrate-and-Fire model |
| Bi, G-Q. & Poo, M-M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464–10472. | STDP learning rule |
| Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178. | Circumplex emotion model |
| Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row. | 8 basic emotions |
| Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267–273. | Oja's rule for Hebbian learning |
| Fikes, R.E. & Nilsson, N.J. (1971). STRIPS: A new approach to the application of theorem proving. *Artificial Intelligence*, 2(3–4), 189–208. | STRIPS planning |
| Indyk, P. & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *STOC '98*, 604–613. | Locality-Sensitive Hashing |
| Rosch, E. (1973). Natural categories. *Cognitive Psychology*, 4(3), 328–350. | Prototype theory (build_prototypes) |
| Church, K.W. & Hanks, P. (1990). Word association norms, mutual information and lexicography. *Computational Linguistics*, 16(1), 22–29. | PMI learning rule |
| Grice, H.P. (1975). Logic and conversation. *Syntax and Semantics*, 3, 41–58. | Gricean maxims (pragmatics.py) |
| Brill, E. (1992). A simple rule-based part of speech tagger. *ANLP '92*, 152–155. | BrillPosTagger (pos_tagger.py) |
| Plate, T.A. (1994). Distributed representations and nested compositional structure. PhD Thesis, University of Toronto. | VSA binding algebra |
