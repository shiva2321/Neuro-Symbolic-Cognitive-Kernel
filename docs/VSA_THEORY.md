# VSA Theory & Mathematical Foundations

> Complete mathematical treatment of the Vector Symbolic Architecture underlying NSCK.  
> Includes all formulas, proofs, capacity bounds, and encoding schemes.

---

## Table of Contents

1. [What Is a Vector Symbolic Architecture?](#1-what-is-a-vector-symbolic-architecture)
2. [NSCK's Binary VSA Specification](#2-nscks-binary-vsa-specification)
3. [Core Operations & Algebraic Properties](#3-core-operations--algebraic-properties)
4. [Similarity Measure & Concentration Bounds](#4-similarity-measure--concentration-bounds)
5. [Capacity Bounds](#5-capacity-bounds)
6. [Encoding Schemes](#6-encoding-schemes)
7. [Deterministic Bundling](#7-deterministic-bundling)
8. [Segment-Based Concatenation](#8-segment-based-concatenation)
9. [Delta-P Causal Discovery](#9-delta-p-causal-discovery)
10. [Thermometer Encoding](#10-thermometer-encoding)
11. [Role-Filler Binding](#11-role-filler-binding)
12. [Phrase-Structure Encoding](#12-phrase-structure-encoding)
13. [LSH Indexing for Episodic Memory](#13-lsh-indexing-for-episodic-memory)
14. [VSA Transition Memory](#14-vsa-transition-memory)
15. [Proofs & Derivations](#15-proofs--derivations)
16. [Numerical Examples](#16-numerical-examples)
17. [Comparison to Other VSA Families](#17-comparison-to-other-vsa-families)

---

## 1. What Is a Vector Symbolic Architecture?

A **Vector Symbolic Architecture** (VSA), also called **Hyperdimensional Computing** (HDC), is a computational framework that represents information as high-dimensional vectors and manipulates them using well-defined algebraic operations.

### Historical Context

| Year | Contribution | Reference |
|---|---|---|
| 1988 | Pentti Kanerva proposes sparse distributed memory | Kanerva, 1988 |
| 1994 | Tony Plate introduces Holographic Reduced Representations (HRR) | Plate, 1994 |
| 1996 | Ross Gayler introduces MAP (Multiply-Add-Permute) vectors | Gayler, 1996 |
| 2009 | Kanerva formalises binary spatter codes | Kanerva, 2009 |
| 2020s | VSA/HDC sees renewed interest for edge AI, robotics, biosignal processing | Various |

### Why VSA for Cognitive Architecture?

1. **Unified representation**: Every concept — percept, action, memory, rule — lives in the same vector space
2. **Algebraic compositionality**: Complex structures (role-filler bindings, sequences, trees) are built from three simple operations
3. **Noise tolerance**: High dimensionality provides natural error correction
4. **O(D) computation**: All operations are element-wise, enabling fast CPU execution without GPUs

---

## 2. NSCK's Binary VSA Specification

NSCK uses **Binary Spatter Codes** (BSC), the simplest VSA family:

| Property | Value |
|---|---|
| **Dimension (D)** | 10 240 |
| **Alphabet** | {0, 1} |
| **Storage** | `numpy.int8` array of length 10 240 |
| **Bind (⊗)** | Bitwise XOR |
| **Bundle (⊕)** | Majority vote |
| **Permute (ρ)** | Circular bit rotation (`np.roll`) |
| **Similarity** | Normalised Hamming: $\text{sim}(A,B) = 1 - \frac{d_H(A,B)}{D}$ |
| **Expected similarity (random pair)** | 0.500 |
| **Implementation** | `hypervec_py.py` (84 lines) |

### Why D = 10 240?

Three engineering reasons:

1. **Capacity**: Can reliably store $\sim D / \ln(D) \approx 1\,110$ items in a single bundled vector before retrieval degrades below 50% accuracy (see [Section 5](#5-capacity-bounds))

2. **Discrimination**: The standard deviation of similarity between two random binary vectors is:

$$\sigma = \frac{1}{2\sqrt{D}} = \frac{1}{2\sqrt{10240}} \approx 0.00494$$

This means random pairs cluster tightly around 0.5, making even small deviations (e.g., 0.52) statistically meaningful.

3. **Alignment**: $10\,240 = 160 \times 64$, allowing efficient packing into 160 `uint64` words for the optional Rust backend (`rust_vsa/`).

---

## 3. Core Operations & Algebraic Properties

### 3.1 Binding (XOR)

$$A \otimes B = A \oplus_2 B \quad \text{(bitwise XOR)}$$

**Properties:**

| Property | Formula | Proof |
|---|---|---|
| Self-inverse | $A \otimes A = \mathbf{0}$ (zero vector) | XOR of identical bits always yields 0 |
| Involution | $(A \otimes B) \otimes B = A$ | Follows from self-inverse + associativity |
| Commutativity | $A \otimes B = B \otimes A$ | XOR is commutative |
| Associativity | $(A \otimes B) \otimes C = A \otimes (B \otimes C)$ | XOR is associative |
| Dissimilarity | $\mathbb{E}[\text{sim}(A, A \otimes B)] = 0.5$ for random $B$ | Each bit flips with probability 0.5 |
| Identity | $A \otimes \mathbf{0} = A$ | XOR with 0 is identity |

**Intuition:** Binding creates a new vector that is *unrelated* to both inputs. This is how we create structured representations — binding a role to a filler produces a unique token that can later be unbound to retrieve the filler.

### 3.2 Bundling (Majority Vote)

For vectors $h_1, h_2, \ldots, h_n$, the bundle is computed bit-by-bit:

$$\text{bundle}(h_1, \ldots, h_n)[i] = \begin{cases} 1 & \text{if } \sum_{j=1}^{n} h_j[i] > n/2 \\ 0 & \text{if } \sum_{j=1}^{n} h_j[i] < n/2 \\ h_1[i] & \text{if } \sum_{j=1}^{n} h_j[i] = n/2 \quad \text{(deterministic tiebreak)} \end{cases}$$

**Properties:**

| Property | Formula |
|---|---|
| Similarity to inputs | $\mathbb{E}[\text{sim}(\text{bundle}(A,B), A)] > 0.5$ |
| Commutativity | $\text{bundle}(A,B) = \text{bundle}(B,A)$ (up to tiebreak) |
| Idempotency (n=1) | $\text{bundle}(A) = A$ |
| Capacity-limited | Retrieval degrades as $n$ grows (see [Section 5](#5-capacity-bounds)) |

**Intuition:** Bundling creates a "superposition" — a single vector that is somewhat similar to all its constituents. This enables set-like representations (e.g., "all active predicates").

### 3.3 Permutation (Circular Shift)

$$\rho_k(A)[i] = A[(i + k) \bmod D]$$

Implementation: `np.roll(bits, -k)` (note: numpy roll convention is opposite to mathematical left-shift).

**Properties:**

| Property | Formula |
|---|---|
| Orthogonality | $\mathbb{E}[\text{sim}(A, \rho_k(A))] = 0.5$ for $k \neq 0$ |
| Invertibility | $\rho_{-k}(\rho_k(A)) = A$ |
| Composition | $\rho_j(\rho_k(A)) = \rho_{j+k}(A)$ |
| Preserves Hamming weight | $\|A\|_H = \|\rho_k(A)\|_H$ |

**Intuition:** Permutation assigns *position* or *role* to a vector. Used for encoding sequence order (position 0, 1, 2, …) and distinguishing structural roles.

### 3.4 Similarity

$$\text{sim}(A, B) = 1 - \frac{d_H(A, B)}{D} = 1 - \frac{\sum_{i=0}^{D-1} (A[i] \oplus B[i])}{D}$$

| Range | Meaning |
|---|---|
| 1.0 | Identical vectors |
| > 0.55 | Meaningfully similar (above noise floor) |
| 0.50 ± 0.005 | Random / unrelated (expected for D = 10 240) |
| < 0.45 | Anti-correlated (one is close to the complement of the other) |
| 0.0 | Complement ($B = \overline{A}$) |

---

## 4. Similarity Measure & Concentration Bounds

### 4.1 Distribution of Random Similarity

For two independent random binary vectors $A, B \in \{0,1\}^D$, each bit agrees with probability $1/2$. The Hamming distance $d_H(A,B) = \sum_{i=1}^D X_i$ where $X_i \sim \text{Bernoulli}(1/2)$.

By the Central Limit Theorem:

$$d_H(A,B) \sim \mathcal{N}\left(\frac{D}{2}, \frac{D}{4}\right) \quad \text{for large } D$$

Therefore:

$$\text{sim}(A,B) = 1 - \frac{d_H}{D} \sim \mathcal{N}\left(\frac{1}{2}, \frac{1}{4D}\right)$$

$$\sigma_{\text{sim}} = \frac{1}{2\sqrt{D}} = \frac{1}{2\sqrt{10240}} \approx 0.00494$$

### 4.2 Hoeffding Bound

For any $\epsilon > 0$, the probability that two random vectors have similarity deviating from 0.5 by more than $\epsilon$:

$$\Pr\left[|\text{sim}(A,B) - 0.5| \geq \epsilon\right] \leq 2 \exp\left(-2 D \epsilon^2\right)$$

**Example:** For $D = 10\,240$ and $\epsilon = 0.02$:

$$\Pr[|\text{sim} - 0.5| \geq 0.02] \leq 2 \exp(-2 \times 10240 \times 0.0004) = 2e^{-8.192} \approx 0.00056$$

This means that a similarity of 0.52 or higher is statistically significant — it occurs by chance with probability < 0.06%.

### 4.3 Detection Threshold

To detect a stored item in a bundle with false positive rate $\alpha$:

$$\tau = 0.5 + z_\alpha \cdot \sigma_{\text{sim}}$$

For $\alpha = 0.01$ ($z = 2.326$):

$$\tau = 0.5 + 2.326 \times 0.00494 \approx 0.5115$$

NSCK uses a practical threshold of **0.52** for most similarity-based decisions, which corresponds to $\alpha \approx 0.003$ (3-in-1000 false positive rate).

---

## 5. Capacity Bounds

### 5.1 Bundle Capacity (Superposition)

When $n$ random vectors are bundled, each constituent has expected similarity:

$$\mathbb{E}[\text{sim}(\text{bundle}(h_1, \ldots, h_n), h_i)] \approx \frac{1}{2} + \frac{1}{2} \cdot \frac{1}{\sqrt{n}}$$

For this similarity to remain above the detection threshold $\tau$:

$$\frac{1}{2} + \frac{1}{2\sqrt{n}} > \tau \implies n < \frac{1}{4(\tau - 0.5)^2}$$

For $\tau = 0.5115$: $n < \frac{1}{4 \times 0.0115^2} \approx 1\,893$

A tighter information-theoretic bound (Frady et al., 2018) gives:

$$n_{\max} \approx \frac{D}{\ln D} \approx \frac{10240}{\ln(10240)} \approx \frac{10240}{9.234} \approx 1\,109$$

### 5.2 Practical Capacity in NSCK

| Operation | Capacity | Used In |
|---|---|---|
| Predicate bundling (situation HV) | ~15–30 predicates | `episodic_memory.create_situation_hv()` |
| Keyword bundling (text grounding) | ~100 words | `universal_input.ground_text()` |
| Thermometer encoding | 7 active bins | `universal_input.ground_scalar()` |
| Role-filler bundling (dict grounding) | ~20 key-value pairs | `universal_input.ground_dict()` |
| Prototype memory (curiosity) | 100 prototypes compared individually | `curiosity.compute_novelty()` |
| VSA transition memory | 2 000 transitions (ring buffer) | `world_model.VSATransitionMemory` |

### 5.3 Retrieval Accuracy vs. Number of Items

For a bundle of $n$ items, the probability of correctly retrieving the $i$-th item by maximum similarity:

$$P_{\text{correct}}(n) \approx \Phi\left(\frac{1/\sqrt{n}}{\sigma_{\text{noise}}}\right) \quad \text{where } \sigma_{\text{noise}} = \frac{1}{2\sqrt{D}} + O(1/D)$$

| n | Expected sim to target | P(correct retrieval) |
|---|---|---|
| 2 | 0.854 | > 0.999 |
| 5 | 0.724 | > 0.999 |
| 10 | 0.658 | > 0.99 |
| 50 | 0.571 | > 0.95 |
| 100 | 0.550 | ~0.90 |
| 500 | 0.522 | ~0.70 |
| 1 109 | 0.515 | ~0.50 |

---

## 6. Encoding Schemes

### 6.1 Categorical Encoding

Each category label maps to a unique random HV via deterministic seeding:

$$\text{HV}(\text{label}) = \text{HyperVector}(\text{seed}=\text{hash}(\text{label}))$$

where $\text{hash}$ is a stable 64-bit integer hash of the label string.

**Property:** Any two distinct labels produce HVs with $\text{sim} \approx 0.5$ (orthogonal).

### 6.2 Scalar Encoding (Thermometer Code)

A scalar value $v \in [v_{\min}, v_{\max}]$ is encoded using overlapping bins:

1. **Quantise**: $k = \lfloor \frac{v - v_{\min}}{v_{\max} - v_{\min}} \times (B-1) \rfloor$ where $B$ = number of bins (default 100)
2. **Activate window**: bins $\{k-w, \ldots, k, \ldots, k+w\}$ where $w = 3$ (window half-width)
3. **Bundle**: $\text{HV}(v) = \bigoplus_{j=k-w}^{k+w} \text{HV}(\text{bin}_j)$

**Similarity preservation:**

Two values $v_1, v_2$ with activated bin sets $S_1, S_2$:

$$\text{sim}(\text{HV}(v_1), \text{HV}(v_2)) \propto \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$$

| Bin overlap | Example values (range [0,1]) | Approximate similarity |
|---|---|---|
| 7/7 (identical) | 0.50 vs 0.50 | 1.0 |
| 5/9 | 0.50 vs 0.52 | ~0.62 |
| 3/11 | 0.50 vs 0.54 | ~0.55 |
| 1/13 | 0.50 vs 0.56 | ~0.52 |
| 0/14 | 0.50 vs 0.60 | ~0.50 |

### 6.3 Sequence Encoding (Positional Permutation)

An ordered sequence $[e_1, e_2, \ldots, e_n]$ is encoded as:

$$\text{HV}(\text{seq}) = \bigoplus_{i=1}^{n} \rho_i(\text{HV}(e_i))$$

Each element is permuted by its position index, then all are bundled. This preserves both content and order:

- $\text{HV}([a, b, c]) \neq \text{HV}([b, a, c])$ — order matters
- $\text{HV}([a, b, c])$ is similar to $\text{HV}([a, b, d])$ — shared prefix

---

## 7. Deterministic Bundling

### The Problem

The standard 2-argument `bundle()` in `hypervec_py.py` uses random tie-breaking: when two bits disagree, a random coin flip decides. This means:

$$\text{bundle}(A, B) \neq \text{bundle}(A, B) \quad \text{(in general)}$$

This is catastrophic for grounding, where the same text must always produce the same HV.

### The Solution: `_deterministic_bundle()`

```python
def _deterministic_bundle(hvs: list) -> HyperVector:
    """Majority-vote bundle with first-vector tiebreaking."""
    if len(hvs) == 0:
        return HyperVector(seed=0)
    if len(hvs) == 1:
        return hvs[0]
    
    stacked = np.stack([h.bits for h in hvs], axis=0)  # shape (n, D)
    vote = stacked.sum(axis=0)                          # shape (D,)
    n = len(hvs)
    half = n / 2.0
    
    result = np.where(vote > half, 1,
             np.where(vote < half, 0,
                      stacked[0]))   # tie → first vector
    
    return HyperVectorPy.from_bits(result.astype(np.int8))
```

**Guarantee:** For any fixed set of inputs, `_deterministic_bundle()` always returns the same output.

**Location:** `universal_input.py`, lines 25–43

---

## 8. Segment-Based Concatenation

### The Problem

The naive approach to multi-component HVs is to bundle all components together. But with 4 components of unequal importance, bundling gives each equal weight (~25%), diluting the dominant keyword signal.

### The Solution

Instead of bundling, NSCK **concatenates** components into designated bit segments:

$$\text{HV}_{\text{text}}[0:5120] = \text{keyword}_{\text{hv}}[0:5120] \quad \text{(50\%)}$$
$$\text{HV}_{\text{text}}[5120:6656] = \text{ngram}_{\text{hv}}[0:1536] \quad \text{(15\%)}$$
$$\text{HV}_{\text{text}}[6656:8192] = \text{order}_{\text{hv}}[0:1536] \quad \text{(15\%)}$$
$$\text{HV}_{\text{text}}[8192:10240] = \text{phrase}_{\text{hv}}[0:2048] \quad \text{(20\%)}$$

### Why These Weights?

| Component | Weight | Rationale |
|---|---|---|
| Keyword (bag-of-words) | 50% | Dominant signal for topical similarity. "photosynthesis" should match "photosynthesis" regardless of order. |
| Character n-gram | 15% | Captures morphological similarity. "running" vs "runner" share trigrams run, unn, nni, nin. |
| Word order | 15% | Distinguishes "dog bites man" from "man bites dog" via positional permutation. |
| Phrase structure | 20% | Syntactic composition via POS-based chunking and role-filler binding. |

### Similarity Decomposition

When comparing two text HVs, the overall similarity decomposes into a weighted sum:

$$\text{sim}(T_1, T_2) = 0.50 \cdot \text{sim}_{\text{kw}} + 0.15 \cdot \text{sim}_{\text{ngram}} + 0.15 \cdot \text{sim}_{\text{order}} + 0.20 \cdot \text{sim}_{\text{phrase}}$$

This is because each segment contributes independently to the Hamming distance:

$$d_H(T_1, T_2) = d_H^{\text{kw}} + d_H^{\text{ngram}} + d_H^{\text{order}} + d_H^{\text{phrase}}$$

$$\text{sim}(T_1, T_2) = 1 - \frac{d_H^{\text{kw}} + d_H^{\text{ngram}} + d_H^{\text{order}} + d_H^{\text{phrase}}}{D}$$

Since each segment has its own fractional extent $f_i$ of $D$:

$$= 1 - \sum_i f_i \cdot \frac{d_H^{(i)}}{f_i \cdot D} = 1 - \sum_i f_i \cdot (1 - \text{sim}_i) = \sum_i f_i \cdot \text{sim}_i$$

---

## 9. Delta-P Causal Discovery

### 9.1 Classical Delta-P

The Delta-P statistic (Allan, 1980; Jenkins & Ward, 1965) measures the contingency between a cause $C$ and effect $E$:

$$\Delta P(E|C) = P(E|C) - P(E|\neg C)$$

| Value | Meaning |
|---|---|
| +1.0 | $C$ perfectly predicts $E$ (and $E$ never occurs without $C$) |
| 0.0 | $C$ and $E$ are independent |
| −1.0 | $C$ perfectly prevents $E$ |

### 9.2 Bayesian-Smoothed Delta-P (NSCK Implementation)

Raw frequencies can produce extreme values with few observations. NSCK applies Laplace smoothing with prior $\alpha = 1.0$:

$$P_{\text{smooth}}(E|C) = \frac{n_{CE} + \alpha}{n_C + 2\alpha}$$

$$P_{\text{smooth}}(E|\neg C) = \frac{n_{\neg CE} + \alpha}{n_{\neg C} + 2\alpha} = \frac{(n_E - n_{CE}) + \alpha}{(T - n_C) + 2\alpha}$$

$$\Delta P_{\text{smooth}} = P_{\text{smooth}}(E|C) - P_{\text{smooth}}(E|\neg C)$$

where:
- $n_{CE}$ = number of observations where both $C$ and $E$ occurred
- $n_C$ = number of observations where $C$ occurred
- $n_E$ = number of observations where $E$ occurred
- $T$ = total number of observations
- $\alpha = 1.0$ (Laplace prior)

### 9.3 Thresholds

| Parameter | Full Induction | Incremental Update |
|---|---|---|
| min_confidence (ΔP threshold) | 0.5 | 0.3 |
| min_evidence ($n_C$) | 5 | 2 |

### 9.4 Worked Example

Scenario: Agent observes 20 timesteps. "FOOD_NEAR" occurs 12 times. "REWARD_POSITIVE" occurs 10 times. Both co-occur 9 times.

$$n_{CE} = 9, \quad n_C = 12, \quad n_E = 10, \quad T = 20$$

$$P(E|C) = \frac{9 + 1}{12 + 2} = \frac{10}{14} = 0.714$$

$$P(E|\neg C) = \frac{(10-9) + 1}{(20-12) + 2} = \frac{2}{10} = 0.200$$

$$\Delta P = 0.714 - 0.200 = 0.514$$

Since $\Delta P = 0.514 > 0.5$ (threshold) and evidence $n_C = 12 > 5$ (minimum), this link is added:

```
CausalLink(cause="FOOD_NEAR", effect="REWARD_POSITIVE", 
           relation=CAUSES, strength=0.514, evidence_count=12)
```

---

## 10. Thermometer Encoding

### Definition

A thermometer code maps a scalar $v \in [v_{\min}, v_{\max}]$ to a set of activated "degree" bins:

1. **Normalise**: $\hat{v} = \frac{v - v_{\min}}{v_{\max} - v_{\min}} \in [0, 1]$
2. **Quantise**: $k = \lfloor \hat{v} \times (B-1) \rfloor$ where $B = 100$ bins
3. **Window**: Activate bins $\{k-3, k-2, k-1, k, k+1, k+2, k+3\}$ (clipped to $[0, B-1]$)
4. **Encode**: Each active bin $j$ maps to a deterministic HV via seed $= \text{hash}(\text{``therm\_''} + j)$
5. **Bundle**: $\text{HV}(v) = \text{deterministic\_bundle}(\{HV_j : j \in \text{active bins}\})$

### Similarity Preservation Proof

Let $v_1, v_2$ be two values with quantised bins $k_1, k_2$ and active sets $S_1, S_2$ (each of size $2w+1 = 7$).

The overlap is:

$$|S_1 \cap S_2| = \max(0, 2w + 1 - |k_1 - k_2|)$$

Since HVs in $S_1 \setminus S_2$ and $S_2 \setminus S_1$ are random (orthogonal), and HVs in $S_1 \cap S_2$ are identical:

$$\text{sim}(\text{HV}(v_1), \text{HV}(v_2)) \approx 0.5 + 0.5 \cdot \frac{|S_1 \cap S_2|}{2w+1}$$

This creates a **triangular similarity kernel**: values within 6 bins of each other (6% of range) are positively correlated; values farther apart are orthogonal.

---

## 11. Role-Filler Binding

### The Principle

To encode a structured datum like `{color: red, size: 5.0}`:

$$\text{HV}(\text{datum}) = \bigoplus_i [\text{HV}(\text{key}_i) \otimes \text{HV}(\text{value}_i)]$$

Each key-value pair is bound (XOR) to create a unique constituent, then all constituents are bundled.

### Retrieval

To query "what is the color?":

$$\text{HV}(\text{datum}) \otimes \text{HV}(\text{"color"}) \approx \text{HV}(\text{"red"})$$

**Proof sketch:** Since XOR is self-inverse:

$$\bigoplus_i [\text{HV}(k_i) \otimes \text{HV}(v_i)] \otimes \text{HV}(k_j) = \text{HV}(v_j) + \text{noise}$$

The noise term (from the other key-value pairs) has expected similarity 0.5 and is suppressed by the high dimensionality.

### NSCK Implementation

```python
# universal_input.py
def ground_dict(self, data, domain="default"):
    pair_hvs = []
    for key, value in sorted(data.items()):
        key_hv = self.ground_category(str(key), domain)
        val_hv = self.ground(value, domain)   # auto-detect type
        pair_hvs.append(key_hv.xor(val_hv))   # bind
    return _deterministic_bundle(pair_hvs)     # bundle
```

---

## 12. Phrase-Structure Encoding

### Overview

The phrase-structure component (20% of text HV) encodes syntactic composition using a pipeline:

```
"the big cat sat on the mat"
    │
    ▼ POS Tagging (heuristic)
[DET, ADJ, NOUN, VERB, PREP, DET, NOUN]
    │
    ▼ Chunking (bottom-up)
Subject NP: [the, big, cat]
Verb:        sat
Prep Phrase: [on, [the, mat]]
    │
    ▼ VSA Encoding (role-filler binding)
Subject_HV = ROLE_SUBJECT ⊗ encode_np({det: "the", adj: ["big"], head: "cat"})
Predicate_HV = ROLE_PREDICATE ⊗ HV("sat")
PP_HV = ROLE_PP ⊗ (HV("on") ⊗ encode_np({det: "the", head: "mat"}))
    │
    ▼ Bundle
phrase_hv = deterministic_bundle([Subject_HV, Predicate_HV, PP_HV])
```

### POS Tagger

The heuristic POS tagger (`_pos_tag_simple`) uses closed-class word lists:

| Tag | Detection Method |
|---|---|
| DET | `word in {"the", "a", "an", "this", "that", "these", "those", "my", "your", ...}` |
| PREP | `word in {"in", "on", "at", "to", "from", "by", "with", "for", "of", "about", ...}` |
| CONJ | `word in {"and", "or", "but", "nor", "yet", "so"}` |
| PRON | `word in {"i", "he", "she", "it", "we", "they", "me", "him", ...}` |
| AUX | `word in {"is", "are", "was", "were", "be", "been", "being", "have", "has", ...}` |
| ADJ | `word ends in -ful, -ous, -ive, -able, -ible, -al, -ent, -ant, -ic, -ish, -less, -ly` OR in known set |
| ADV | `word ends in -ly` OR in known set {very, quite, rather, ...} |
| VERB | `word ends in -ing, -ed, -ize, -ify, -ate` OR in known set |
| NOUN | Default fallback |

### Noun Phrase Encoding

A noun phrase like `[the, big, red, cat]` is encoded:

$$\text{NP} = \bigoplus[\text{HV}(\text{``det:the''}), \text{HV}(\text{``adj:big''}), \text{HV}(\text{``adj:red''}), \text{HV}(\text{``head:cat''}, \text{weight}=3)]$$

The head noun is given weight 3 (repeated 3 times in the bundle) to dominate.

### Syntactic Role HVs

Four role HVs are pre-computed with fixed seeds:

| Role | Seed | Purpose |
|---|---|---|
| ROLE_SUBJECT | "SYNTACTIC_ROLE_SUBJECT" | Subject NP |
| ROLE_PREDICATE | "SYNTACTIC_ROLE_PREDICATE" | Main verb |
| ROLE_OBJECT | "SYNTACTIC_ROLE_OBJECT" | Object NP |
| ROLE_PP | "SYNTACTIC_ROLE_PP" | Prepositional phrase |

---

## 13. LSH Indexing for Episodic Memory

### The Problem

Episodic memory stores up to 10 000 episodes. Brute-force Hamming similarity search is O(n × D) per query.

### LSH Hashing

NSCK uses **Locality-Sensitive Hashing** (LSH) to partition episodes into buckets:

```python
# episodic_memory.py
def _lsh_key(self, hv):
    return hv.lsh_hash(seed=42, n_bits=8)
```

This produces an 8-bit hash (256 possible buckets). Episodes with the same hash are likely to have similar HVs.

### Query Protocol

1. Compute LSH hash of query HV
2. Retrieve all episodes in that bucket
3. Also check adjacent buckets (Hamming distance 1 from hash)
4. Compute exact Hamming similarity for candidates
5. Return top-k by similarity

### Analysis

For $N = 10\,000$ episodes uniformly distributed across $2^8 = 256$ buckets:

- Expected bucket size: $N / 256 \approx 39$ episodes
- With adjacent bucket checking (8 Hamming-1 neighbours): $9 \times 39 \approx 351$ candidates
- Exact comparison cost: $351 \times D = 351 \times 10\,240 \approx 3.6$M operations

vs. brute force: $10\,000 \times 10\,240 = 102.4$M operations — a **28× speedup**.

---

## 14. VSA Transition Memory

### Design

The `VSATransitionMemory` class stores state-action → (next-state, reward) mappings using VSA binding as the key:

$$\text{key} = \text{state\_bits} \otimes \text{action\_bits}$$

### Storage

Ring buffer of capacity 2 000:
- `keys[i]`: 10 240-bit binary vector ($\text{state} \otimes \text{action}$)
- `next_states[i]`: 10 240-bit binary vector
- `rewards[i]`: scalar float

### Retrieval (k-NN)

Given a query state-action pair:

1. Compute query key: $q = \text{state} \otimes \text{action}$
2. Compute Hamming similarities to all stored keys: $s_i = \text{sim}(q, \text{keys}[i])$
3. Select top-$k$ (default $k = 3$)
4. Weighted vote: $w_i = \max(0.01, s_i - 0.45)$

$$\text{predicted\_state} = \text{threshold}\left(\frac{\sum_i w_i \cdot \text{next\_states}[i]}{\sum_i w_i}\right)$$

$$\text{predicted\_reward} = \frac{\sum_i w_i \cdot \text{rewards}[i]}{\sum_i w_i}$$

where $\text{threshold}(x) = \mathbf{1}[x > 0.5]$ (binarise each bit).

### Trust criterion

If $\max(s_1, s_2, s_3) \geq 0.55$, the VSA prediction is trusted. Otherwise, the numeric ensemble provides the prediction, with blending:

$$\alpha = \min\left(1.0, \frac{s_{\max} - 0.45}{0.30}\right)$$

$$\text{result} = \alpha \cdot \text{VSA\_prediction} + (1-\alpha) \cdot \text{numeric\_prediction}$$

---

## 15. Proofs & Derivations

### Proof 1: XOR Binding Produces Dissimilar Vectors

**Claim:** For random independent $A, B \in \{0,1\}^D$, $\mathbb{E}[\text{sim}(A, A \otimes B)] = 0.5$.

**Proof:**
$$\text{sim}(A, A \otimes B) = 1 - \frac{d_H(A, A \oplus B)}{D} = 1 - \frac{\sum_i A_i \oplus (A_i \oplus B_i)}{D} = 1 - \frac{\sum_i B_i}{D}$$

Since each $B_i \sim \text{Bernoulli}(0.5)$:
$$\mathbb{E}\left[\sum_i B_i\right] = D/2$$
$$\mathbb{E}[\text{sim}(A, A \otimes B)] = 1 - \frac{D/2}{D} = 0.5 \quad \blacksquare$$

### Proof 2: Bundle Similarity to Constituent

**Claim:** For the majority-vote bundle of $n$ random vectors, $\text{sim}(\text{bundle}, h_j) \approx 0.5 + \frac{1}{2\sqrt{n}}$.

**Proof sketch:** Consider a single bit position $i$. The bundle's value at position $i$ is determined by majority vote. The probability that $\text{bundle}[i] = h_j[i]$ is:

$$P = \Pr\left[\sum_{k \neq j} h_k[i] + h_j[i] > n/2\right] = \Pr\left[\text{Binomial}(n-1, 0.5) > n/2 - h_j[i]\right]$$

For large $n$, using CLT: $\sum_{k \neq j} h_k[i] \approx \mathcal{N}(\frac{n-1}{2}, \frac{n-1}{4})$.

The event that $h_j$ is in the majority includes the case where $h_j$ breaks a near-tie. By standard analysis:

$$P \approx \frac{1}{2} + \frac{1}{2\sqrt{\pi(n-1)/2}} \approx \frac{1}{2} + \frac{1}{\sqrt{2\pi n}} \quad \blacksquare$$

### Proof 3: Permutation Produces Orthogonal Vectors

**Claim:** For random $A \in \{0,1\}^D$ and $k \neq 0$, $\mathbb{E}[\text{sim}(A, \rho_k(A))] = 0.5$.

**Proof:** $\rho_k$ is a fixed permutation. $A[i]$ and $A[(i+k) \bmod D]$ are independent bits (since $A$ is random and $i \neq (i+k) \bmod D$ when $k \neq 0$).

$$\Pr[A[i] = A[(i+k) \bmod D]] = \Pr[0,0] + \Pr[1,1] = 0.25 + 0.25 = 0.5$$

$$\mathbb{E}[\text{sim}(A, \rho_k(A))] = 0.5 \quad \blacksquare$$

---

## 16. Numerical Examples

### Example 1: Grounding "the cat sat on the mat"

```python
from universal_input import UniversalInput
ui = UniversalInput()

hv1 = ui.ground_text("the cat sat on the mat")
hv2 = ui.ground_text("a cat sat upon the mat")
hv3 = ui.ground_text("quantum mechanics violates locality")

# Expected: sim(hv1, hv2) > sim(hv1, hv3)
# Because hv1 and hv2 share keywords {cat, sat, mat} 
# while hv3 shares no keywords.
```

### Example 2: Role-Filler Binding and Retrieval

```python
# Encode: {color: red, shape: circle}
data = {"color": "red", "shape": "circle"}
hv_data = ui.ground_dict(data)

# To check if "color" is "red":
color_role = ui.ground_category("color")
red_hv = ui.ground_category("red")
bound = hv_data.xor(color_role)  # unbind role
sim = bound.similarity(red_hv)    # should be > 0.5
```

### Example 3: Scalar Similarity Preservation

```python
hv_03 = ui.ground_scalar(0.3, 0.0, 1.0)
hv_032 = ui.ground_scalar(0.32, 0.0, 1.0)
hv_09 = ui.ground_scalar(0.9, 0.0, 1.0)

# sim(hv_03, hv_032) >> sim(hv_03, hv_09)
# Because 0.3 and 0.32 share most thermometer bins
# while 0.3 and 0.9 share none
```

### Example 4: Delta-P Calculation

```python
# Observations over 50 timesteps:
# cause "WALL_AHEAD" observed 25 times
# effect "COLLISION" observed 15 times
# co-occurrence: 14 times

n_CE, n_C, n_E, T, alpha = 14, 25, 15, 50, 1.0

P_E_given_C = (n_CE + alpha) / (n_C + 2*alpha)       # = 15/27 = 0.556
P_E_given_notC = ((n_E - n_CE) + alpha) / ((T - n_C) + 2*alpha)  # = 2/27 = 0.074

delta_p = P_E_given_C - P_E_given_notC                # = 0.481

# Since 0.481 < 0.5, this just misses full induction threshold
# but exceeds incremental threshold (0.3)
```

---

## 17. Comparison to Other VSA Families

| VSA Family | Dimension | Alphabet | Bind | Bundle | Notes |
|---|---|---|---|---|---|
| **BSC** (NSCK) | 10 240 | {0,1} | XOR | Majority vote | Simplest, fastest. No normalisation needed. |
| **HRR** (Plate) | ~1 000 | ℝ | Circular convolution | Element-wise addition | Rich algebra but needs normalisation. |
| **MAP** (Gayler) | ~10 000 | {−1,+1} | Element-wise multiply | Element-wise addition + clip | Similar to BSC but uses ±1 representation. |
| **FHRR** (Plate) | ~1 000 | ℂ (unit circle) | Element-wise multiply | Element-wise addition | Unitary binding in frequency domain. |
| **VTB** (Gosmann & Eliasmith) | ~512 | ℝ^(d×d) | Tensor product (projected) | Matrix addition | Used in Neural Engineering Framework (Nengo). |

### Why BSC for NSCK?

1. **Speed**: XOR is a single CPU instruction per 64 bits. The Rust backend processes all 10 240 bits in 160 `u64` XOR operations.
2. **Simplicity**: No floating-point normalisation, no complex arithmetic.
3. **Determinism**: With deterministic bundling, all operations are fully reproducible.
4. **Capacity**: D = 10 240 provides sufficient capacity for NSCK's use cases (max ~30 predicates bundled, max 100 keywords).

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) for system design, [MODULE_REFERENCE.md](MODULE_REFERENCE.md) for API reference.*
