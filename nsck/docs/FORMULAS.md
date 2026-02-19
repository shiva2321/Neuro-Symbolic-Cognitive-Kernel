# Mathematical Foundations

Every formula below is extracted directly from the NSCK codebase. Each section references the exact source file under `nsck/python/core/`.

---

## Table of Contents

- [1. Vector Symbolic Architecture (VSA)](#1-vector-symbolic-architecture-vsa)
- [2. Text Encoding](#2-text-encoding)
- [3. Causal Reasoning](#3-causal-reasoning)
- [4. Global Workspace (LIDA-Lite)](#4-global-workspace-lida-lite)
- [5. Q-Learning](#5-q-learning)
- [6. Hebbian Learning](#6-hebbian-learning)
- [7. Spiking Neural Network](#7-spiking-neural-network)
- [8. Emotion System](#8-emotion-system)
- [9. Episodic Memory](#9-episodic-memory)
- [10. Self-Model](#10-self-model)
- [11. Curiosity Module](#11-curiosity-module)
- [12. Rule Learner](#12-rule-learner)
- [13. Analogy Engine](#13-analogy-engine)

---

## 1. Vector Symbolic Architecture (VSA)

*Source: `vsa/hypervec_py.py`*

### Dimension

$$d = 10{,}240 \text{ bits}$$

All hypervectors $\mathbf{v} \in \{0, 1\}^{10240}$. This provides a space of $2^{10240}$ possible vectors, with approximately $2^{100}$ near-orthogonal vectors available (Kanerva, 2009).

### Similarity (Normalised Hamming Distance)

$$\text{sim}(\mathbf{A}, \mathbf{B}) = 1 - \frac{d_H(\mathbf{A}, \mathbf{B})}{d} = 1 - \frac{\sum_{i=1}^{d} A_i \oplus B_i}{d}$$

Properties:
- $\text{sim}(\mathbf{A}, \mathbf{A}) = 1$
- $\text{sim}(\mathbf{A}, \bar{\mathbf{A}}) = 0$
- For random $\mathbf{A}, \mathbf{B}$: $\mathbb{E}[\text{sim}(\mathbf{A}, \mathbf{B})] = 0.5$
- Standard deviation: $\sigma \approx \frac{1}{2\sqrt{d}} \approx 0.0049$
- Meaningful similarity threshold: $> 0.55$ ($\approx 10\sigma$ above chance)

### Cosine Similarity

$$\text{cos\_sim}(\mathbf{A}, \mathbf{B}) = \frac{(2\mathbf{A} - \mathbf{1}) \cdot (2\mathbf{B} - \mathbf{1})}{d}$$

Maps binary vectors to $\{-1, +1\}^d$ before computing dot product. Range: $[-1, +1]$.

### Binding (XOR)

$$\mathbf{A} \otimes \mathbf{B} = \mathbf{A} \oplus \mathbf{B} \quad \text{(element-wise XOR)}$$

Properties:
- **Self-inverse**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{B} = \mathbf{A}$
- **Commutative**: $\mathbf{A} \otimes \mathbf{B} = \mathbf{B} \otimes \mathbf{A}$
- **Associative**: $(\mathbf{A} \otimes \mathbf{B}) \otimes \mathbf{C} = \mathbf{A} \otimes (\mathbf{B} \otimes \mathbf{C})$
- **Dissimilar to operands**: $\text{sim}(\mathbf{A} \otimes \mathbf{B}, \mathbf{A}) \approx 0.5$

Used for role-filler binding: $\text{Capital} \otimes \text{Paris}$ creates a compound representation retrievable by applying either component.

### Bundling (Majority Vote)

For two vectors:

$$(\mathbf{A} \oplus \mathbf{B})_i = \begin{cases} A_i & \text{if } A_i = B_i \\ \text{Bernoulli}(0.5) & \text{if } A_i \neq B_i \end{cases}$$

Properties:
- **Similar to both operands**: $\text{sim}(\mathbf{A} \oplus \mathbf{B}, \mathbf{A}) \approx 0.75$
- **Commutative**: $\mathbf{A} \oplus \mathbf{B} = \mathbf{B} \oplus \mathbf{A}$ (in expectation)
- Iterated bundling of $n$ vectors: $\text{sim} \approx 0.5 + \frac{0.5}{n}$ to each constituent

### Permutation (Circular Shift)

$$\rho^k(\mathbf{v})_i = v_{(i + k) \bmod d}$$

Properties:
- **Invertible**: $\rho^{-k}(\rho^k(\mathbf{v})) = \mathbf{v}$
- **Dissimilar for $k > 0$**: $\text{sim}(\rho^k(\mathbf{v}), \mathbf{v}) \approx 0.5$ for $k \geq 1$
- Used for positional/temporal encoding

### Weighted Bundle

$$\text{weighted\_bundle}(\mathbf{A}, \mathbf{B}, w) = \text{bundle}^{n_A}(\mathbf{A}) \oplus \text{bundle}^{n_B}(\mathbf{B})$$

where $n_A = \lfloor w \cdot 7 \rceil, \; n_B = 7 - n_A$, and $\text{bundle}^k$ means $k$-fold iterated bundling. This biases the result toward $\mathbf{A}$ when $w > 0.5$.

---

## 2. Text Encoding

### Sentence Encoding

*Source: `language/text_knowledge_learner.py`*

$$\mathbf{S} = \bigoplus_{w \in \text{words}(s)} \mathbf{HV}\big(\text{hash}(w) \bmod 2^{32}\big)$$

Each word maps to a deterministic HV via its Python hash modulo $2^{32}$ (matching the `HyperVector` constructor's uint32 seed). Function words (130 common English words) are filtered before encoding.

### Context Vector (Semantic Folding)

*Source: `language/text_knowledge_learner.py`*

$$\mathbf{ctx}(c) = \mathbf{HV}(c) \oplus \bigoplus_{\substack{w \in \text{window}(c,\, W) \\ w \neq c}} \rho^{|pos(w) - pos(c)|}\big(\mathbf{HV}(w)\big)$$

where $W = 7$ (the `folding_window_size`). Context words within a 7-word window are permuted by their absolute distance from the concept before bundling. This captures distributional semantics: words in similar contexts produce similar context vectors.

### Relation Detection

Two concepts $a, b$ are considered related when:

$$\text{sim}\big(\mathbf{ctx}(a), \mathbf{ctx}(b)\big) > \tau_r = 0.55$$

### 4-Component Text Grounding (UniversalInput)

*Source: `language/universal_input.py`*

The `UniversalInput` module produces a composite HV from four weighted components:

| Component | Weight | Method |
|-----------|--------|--------|
| Keyword | 50% | Exact word → HV lookup with TF weighting |
| Char N-gram | 15% | Character 3-grams → HV, bundled |
| Word-order | 15% | $\rho^{pos}(\mathbf{HV}(w))$ — positional permutation |
| Phrase-structure | 20% | NP/VP/PP chunks bound with role HVs: $\text{Subject} \otimes \text{NP}$ |

$$\mathbf{HV}_{\text{text}} = \text{weighted\_bundle}\big(\mathbf{KW}_{50\%},\; \mathbf{NG}_{15\%},\; \mathbf{WO}_{15\%},\; \mathbf{PS}_{20\%}\big)$$

---

## 3. Causal Reasoning

*Source: `reasoning/causal_reasoning.py`*

### Delta-P (Causal Strength)

$$\Delta P = P(E \mid C) - P(E \mid \neg C)$$

With adaptive Laplace smoothing:

$$P(E \mid C) = \frac{n_{CE} + \alpha}{n_C + 2\alpha}, \quad P(E \mid \neg C) = \frac{n_E - n_{CE} + \alpha}{(T - n_C) + 2\alpha}$$

$$\alpha = \begin{cases} 1 & \text{if } T < 10 \text{ (sparse data)} \\ 0 & \text{if } T \geq 10 \text{ (sufficient data)} \end{cases}$$

where:
- $T$ = total observation steps
- $n_C$ = count of cause $C$ observed
- $n_E$ = count of effect $E$ observed
- $n_{CE}$ = count of $C$ and $E$ co-occurring

### Link Creation Criteria

$$\text{add link } C \rightarrow E \quad \text{iff} \quad \Delta P > 0.5 \;\wedge\; n_C \geq 5$$

### Forward Chaining

Given active concept $c$, retrieve all effects:

$$\text{effects}(c) = \{e : (c \rightarrow e) \in G_{\text{causal}}\}$$

Chained recursively up to depth 3 with strength propagation:

$$\text{strength}(c \rightarrow e_n) = \prod_{i=1}^{n} \Delta P_i$$

---

## 4. Global Workspace (LIDA-Lite)

*Source: `reasoning/global_workspace.py`*

### Coalition Activation

$$A(c) = S_{\text{base}}(c) + R(c) + M_{\text{affect}}(c) + 0.5 \cdot C_{\text{sender}}(c) + B_{\text{mission}}(c)$$

| Term | Range | Meaning |
|------|-------|---------|
| $S_{\text{base}}$ | [0, 1] | Intrinsic salience of the proposal |
| $R$ | [0, 1] | Relevance to current context/goal |
| $M_{\text{affect}}$ | [0, 1] | Drive-goal match (e.g., "Food" ↔ "Hunger") |
| $C_{\text{sender}}$ | [0, 1] | Module's self-reported confidence (weighted 0.5×) |
| $B_{\text{mission}}$ | 0 or +0.2 | Bias toward focused/mission-aligned module |

### Competition

$$\text{winner} = \arg\max_{c \in \text{proposals}} A(c), \quad \text{subject to } A(\text{winner}) \geq \theta$$

where $\theta = 0.5$ is the attention threshold.

### Danger Veto (Mental Rehearsal)

Before committing to the winning action, simulate through the WorldModel:

$$\text{veto if } \max_{\mathbf{d} \in \text{dangers}} \text{sim}(\hat{\mathbf{s}}_{\text{predicted}}, \mathbf{d}) \geq 0.75$$

On veto: $S_{\text{base}} \leftarrow 0.5 \cdot S_{\text{base}}$, remove from candidates, try next-best (up to 3 cycles). If all vetoed → emergency `ACTION_STAY`.

---

## 5. Q-Learning

*Source: `reasoning/cognitive_engine.py`*

### TD(0) Update

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 0.1 | Learning rate |
| $\gamma$ | 0.9 | Discount factor |

### Epsilon-Greedy Policy

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_{a} Q(s, a) & \text{otherwise} \end{cases}$$

Epsilon decays: $\epsilon = \max(0.1, \; \epsilon_0 \cdot 0.995^t)$, starting from $\epsilon_0 = 0.3$.

### Q-Value Coalition

Q-learning proposals enter GWT competition as a coalition with:
- $\text{salience} = 0.8$ (base)
- $\text{confidence} = 1 - \epsilon$ (decreasing exploration → increasing confidence)
- Source tag: `Q_LEARNING`

---

## 6. Hebbian Learning

*Source: `learning/hebbian.py`*

### Oja's Rule (with Reward Modulation)

$$\Delta W_{ij} = \alpha \cdot \text{pre}_i \cdot \text{post}_j \cdot (\text{reward} - \text{baseline})$$

### Weight Normalization (Oja)

$$W_{ij} \leftarrow W_{ij} + \alpha \cdot \text{post}_j \cdot \big(\text{pre}_i - \text{post}_j \cdot W_{ij}\big)$$

### Eligibility Traces

$$e_{ij}(t) = \lambda \cdot e_{ij}(t-1) + \text{pre}_i(t) \cdot \text{post}_j(t)$$

$$\Delta W_{ij} = \alpha \cdot e_{ij} \cdot (\text{reward} - \text{baseline})$$

where $\lambda = 0.9$ (trace decay rate) and baseline tracks EMA of rewards.

---

## 7. Spiking Neural Network

*Source: `perception/snn_perception.py`*

### Leaky Integrate-and-Fire (LIF) Neuron

$$\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R \cdot I(t)$$

Discretized (Euler method):

$$V(t + \Delta t) = V(t) + \frac{\Delta t}{\tau_m} \big[ -(V(t) - V_{\text{rest}}) + I(t) \big]$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\tau_m$ | 20 ms | Membrane time constant |
| $V_{\text{th}}$ | 1.0 | Spike threshold |
| $V_{\text{reset}}$ | 0.0 | Post-spike reset voltage |
| $V_{\text{rest}}$ | 0.0 | Resting potential |

When $V \geq V_{\text{th}}$: emit spike, $V \leftarrow V_{\text{reset}}$.

### STDP (Spike-Timing-Dependent Plasticity)

$$\Delta W_{ij} = \begin{cases} A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \text{ (pre before post)} \\ -A_- \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \text{ (post before pre)} \end{cases}$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $A_+$ | 0.01 | LTP amplitude |
| $A_-$ | 0.012 | LTD amplitude ($A_- > A_+$ → competition) |
| $\tau_+$ | 20 ms | LTP time constant |
| $\tau_-$ | 20 ms | LTD time constant |

### Rate Coding (VSA-SNN Bridge)

*Source: `perception/vsa_snn_bridge.py`*

$$\mathbf{HV}_{\text{spike}} = \bigoplus_{i:\, \text{rate}_i > \theta} \mathbf{n}_i$$

where $\text{rate}_i = \frac{\text{spike\_count}_i}{T}$ and $\mathbf{n}_i$ is the fixed HyperVector for neuron $i$.

### Temporal Coding

$$\mathbf{HV}_{\text{temporal}} = \bigoplus_{i:\, t_i \leq T} \rho^{t_i}(\mathbf{n}_i)$$

Each neuron's HV is permuted by its first-spike time, encoding temporal structure.

---

## 8. Emotion System

*Source: `cognitive/emotion_system.py`*

### Plutchik Prototypes in Russell's Circumplex

| Emotion | Valence $(v_e)$ | Arousal $(a_e)$ |
|---------|---------|---------|
| Joy | +0.8 | +0.7 |
| Trust | +0.5 | +0.2 |
| Fear | −0.7 | +0.8 |
| Surprise | 0.0 | +0.9 |
| Sadness | −0.6 | +0.2 |
| Disgust | −0.5 | +0.4 |
| Anger | −0.5 | +0.8 |
| Anticipation | +0.3 | +0.5 |
| Neutral | 0.0 | 0.0 |

### Emotion Blending (Inverse-Distance Weighting)

$$w_e = \frac{1}{\sqrt{(v - v_e)^2 + (a - a_e)^2} + \epsilon}, \quad \epsilon = 0.01$$

$$\text{blend}(e) = \frac{w_e}{\sum_{e'} w_{e'}}, \quad \text{for } e \text{ where } \text{blend}(e) \geq 0.05$$

Weights below 5% are pruned and the distribution is renormalized.

### Valence Update

$$v_t = 0.95 \cdot v_{t-1} + \Delta_{\text{reward}} + \Delta_{\text{drives}}$$

$$\Delta_{\text{reward}} = \begin{cases} +0.2 & r > 0.1 \\ -0.2 & r < -0.1 \\ 0 & \text{otherwise} \end{cases}, \quad \Delta_{\text{drives}} = \begin{cases} -0.05 & \bar{d} > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Clamped to $[-1, 1]$.

### Arousal Update

$$a_t = 0.7 \cdot a_{t-1} + 0.3 \cdot \max(\text{drives})$$

### Emotional Intensity

$$I = \sqrt{v^2 + a^2}$$

---

## 9. Episodic Memory

*Source: `memory/episodic_memory.py`*

### LSH Indexing

4 hash tables, each using 10-bit signatures:

$$h_k(\mathbf{v}) = \text{lsh\_hash}(\mathbf{v}, \text{seed}=k, \text{bits}=10) \quad \text{for } k \in \{0, 1, 2, 3\}$$

### Retrieval Pipeline

1. **LSH bucket lookup**: candidates where $h_k(\mathbf{v}_{\text{episode}}) = h_k(\mathbf{q})$ for any table $k$
2. **1-bit neighbor probing**: for each bit $b \in \{0..7\}$, check $h \oplus 2^b$
3. **Exact ranking**: compute $\text{sim}(\mathbf{q}, \mathbf{v}_{\text{episode}})$ on all candidates
4. **Return top-$k$** by similarity

### Consolidation

Triggered when `len(recent[task]) >= 500`:
1. Older half moved to SQLite via BrainStore
2. LSH index rebuilt for remaining episodes
3. Maximum 10,000 episodes per task

### Situation HV Construction

$$\mathbf{s} = \bigoplus_{p \in \text{active\_predicates}} \mathbf{HV}(p)$$

Active predicates from `GroundingVerifier` are bundled into a single situation vector.

---

## 10. Self-Model

*Source: `cognitive/self_model.py`*

### Calibration Error

$$\text{CE} = \frac{1}{N} \sum_{i=1}^{N} |p_i - o_i|$$

where $p_i$ is the predicted confidence and $o_i \in \{0, 1\}$ is the actual outcome. A perfectly calibrated model has $\text{CE} = 0$.

### Confidence Update

$$c_t = (1 - \alpha) \cdot c_{t-1} + \alpha \cdot o_t$$

where $o_t \in \{0, 1\}$ is the outcome and $\alpha = 0.2$ (smoothing rate).

---

## 11. Curiosity Module

*Source: `learning/curiosity.py`*

### Novelty Score

$$\text{novelty}(\mathbf{q}) = 1 - \max_{\mathbf{v} \in \text{visited}} \text{sim}(\mathbf{q}, \mathbf{v})$$

Novel inputs ($\text{novelty} > 0.5$) trigger exploration.

### Learning Progress

$$\Delta_{\text{learning}} = \frac{1}{W} \sum_{i=t-W}^{t} r_i - \frac{1}{W} \sum_{i=t-2W}^{t-W} r_i$$

Compares recent reward window to previous window. Positive $\Delta$ indicates productive exploration.

### Exploration Decision

Explore if ANY of:
1. $\text{novelty}(\mathbf{q}) > \tau_{\text{novelty}}$ (default 0.5)
2. $\text{confidence} < \tau_{\text{confidence}}$ (default 0.3)
3. $\Delta_{\text{learning}} > 0$ AND $\text{novelty} > 0.3$

---

## 12. Rule Learner

*Source: `reasoning/rule_learner.py`*

### Approximate Predicate Matching

$$J(\text{preds}_1, \text{preds}_2) = \frac{|\text{preds}_1 \cap \text{preds}_2|}{|\text{preds}_1 \cup \text{preds}_2|} \geq 0.6$$

This Jaccard threshold enables rule generalization from fewer examples than exact matching.

### Rule Promotion

A candidate becomes a rule when:
- $\text{frequency} \geq 3$ (minimum observations)
- $\text{success\_rate} > 0$ (net positive outcomes)

### Rule Pruning

Rules are removed when:
- $\text{tenure} > 100$ steps AND $\text{success\_rate} < 0.1$

---

## 13. Analogy Engine

*Source: `reasoning/analogy.py`*

### Auto-Abstraction

Concepts are candidates for abstraction when:

$$\text{structural\_similarity} > 0.52$$

Name-similarity bonus is added for concepts with lexically similar names, facilitating discovery of category-level abstractions.

### Transfer Rule

Given source rule `(condition_src, action_src)` in `domain_src`, transfer to `domain_tgt`:

$$\text{condition}_{\text{tgt}} = \text{map}(\text{condition}_{\text{src}}, \text{domain\_alignment})$$
$$\text{action}_{\text{tgt}} = \text{map}(\text{action}_{\text{src}}, \text{domain\_alignment})$$

Domain alignment is computed via VSA binding: $\text{align} = \text{src\_concept} \otimes \text{tgt\_concept}$.
