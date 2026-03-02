# NSCK V5: Mathematical Proofs and Formal Foundations

## 1. HV Similarity & Bond Strength

### 1.1 Bipolar Cosine Similarity

Let **x** ∈ {0,1}^D be a binary HV.  Define its bipolar form:

**x̃** = 2**x** − **1** ∈ {−1,+1}^D

The cosine similarity between two HVs **a**, **b** is:

```
sim(a, b) = (ã · b̃) / (‖ã‖ · ‖b̃‖)
```

Since ‖ã‖ = √D for binary HVs, this reduces to:

```
sim(a, b) = (1/D) Σᵢ ãᵢ b̃ᵢ
```

which equals `1 − 2·hamming(a,b)/D`, i.e., cosine similarity and Hamming similarity are
linearly related for binary HVs.

### 1.2 Bond Strength Decomposition

The ValenceEngine bond strength is a convex combination:

```
s(a, b) = 0.60·struct(a,b) + 0.20·valence_compat(a,b) + 0.20·domain_overlap(a,b)
```

where:
- `struct(a,b) = (sim(a,b) + 1) / 2 ∈ [0,1]`  (normalised cosine)
- `valence_compat(a,b) = 1 − |vₐ − v_b| / 2 ∈ [0,1]`
- `domain_overlap(a,b) = |Dₐ ∩ D_b| / |Dₐ ∪ D_b| ∈ [0,1]`  (Jaccard of domain sets)

**Lemma:** s(a,b) ∈ [0,1] for all (a,b) because it is a convex combination of values in [0,1].

---

## 2. Spectral Graph Theory

### 2.1 Normalized Laplacian

Given adjacency matrix **A** and degree matrix **D** = diag(Aᵢ·),
the normalized Laplacian is:

```
L = I − D^{-½} A D^{-½}
```

**Properties:**
- L is symmetric positive semi-definite
- Eigenvalues λ ∈ [0, 2]
- λ₀ = 0 always (corresponding to the constant vector)
- Multiplicity of λ₀ = number of connected components

### 2.2 Spectral Gap

```
gap = λ₁ − λ₀ = λ₁
```

A large spectral gap indicates good community structure (well-separated clusters).
For random graphs on n nodes, gap → 0 as n → ∞.

**Theorem (Cheeger inequality):**
```
gap/2 ≤ h(G) ≤ √(2·gap)
```
where h(G) is the Cheeger constant (minimum cut/volume ratio).

### 2.3 Spectral Coarse-Graining

**Algorithm:**
1. Compute eigenvectors V_k corresponding to the k smallest non-zero eigenvalues
2. Each concept i has feature embedding: **fᵢ** = V_k[i, :] ∈ ℝ^k
3. Apply k-means on {**f₁**, ..., **f_n**} to get cluster assignments
4. Each cluster → supernode

**Correctness:** The eigenvectors of L provide the optimal 2-cut partition (Fiedler vector)
and multi-cut approximations. Clustering in spectral space therefore respects graph structure.

---

## 3. Percolation Theory

### 3.1 Bond Percolation Model

Each edge (u,v) in the bond graph is present if `bond_strength(u,v) ≥ θ_bond`.

The *giant component fraction* P∞ is the fraction of nodes in the largest connected component.

**Phase transition:** For Erdős-Rényi graphs with n nodes and edge probability p:
```
P∞ ≈ 0         for p < 1/n  (subcritical)
P∞ ≈ c·n^{-1/3} for p = 1/n  (critical)
P∞ → β         for p > 1/n  (supercritical, β = fixed point of β = 1 - e^{-⟨k⟩β})
```

For societal knowledge graphs, the critical threshold corresponds to domain emergence.

### 3.2 Transition Detection

The PercolationMonitor triggers `transition_type = "emergence"` when:
```
|P∞(t) - P∞(t-1)| ≥ Δ_threshold = 0.08
and P∞(t) > P∞(t-1)
```

This is a discrete derivative criterion for detecting the onset of a percolation phase transition.

---

## 4. Topological Data Analysis

### 4.1 Witness Complex

Let L ⊂ X be the landmark set, W = X \ L the witnesses.

**Definition:** An edge (l₁, l₂) ∈ L² is in the weak witness complex W(L, X, ε) if
there exists a witness w ∈ W such that:
```
d(w, l₁) ≤ ε  and  d(w, l₂) ≤ ε
```

where d is the HV distance `d(a,b) = (1 - sim(a,b)) / 2 ∈ [0,1]`.

### 4.2 Betti Numbers

For a simplicial complex K:
- **β₀** = number of connected components  (rank of H₀)
- **β₁** = number of independent cycles    (rank of H₁)
- **β₂** = number of independent voids     (rank of H₂)

**Euler characteristic:** χ = β₀ − β₁ + β₂ = V − E + F

For the 1-skeleton (no triangles, no tetrahedra):
```
β₁ = E − V + β₀   (from χ = V - E = β₀ - β₁)
```

### 4.3 Persistence Entropy

Given a persistence diagram with pairs (bᵢ, dᵢ):

```
Lᵢ = dᵢ − bᵢ   (lifetime)
H = −Σᵢ pᵢ log pᵢ   where pᵢ = Lᵢ / Σⱼ Lⱼ
```

**Interpretation:** High entropy → heterogeneous topological structure (healthy).
Low entropy → degenerate topology (single dominant feature or no features).

---

## 5. Zipf Power-Law Validation

### 5.1 Zipf's Law

In a healthy knowledge world, concept activation frequency follows:
```
f(r) = C · r^{−α}
```
where r is the rank (sorted descending) and α ≈ 1 for Zipf's law.

**Log-log regression:**
```
log f(r) = log C − α log r
```
Fit via OLS to get α̂ and R².

### 5.2 Health Criterion

```
is_zipf_like = (0.8 ≤ α̂ ≤ 1.5) and (R² ≥ 0.85)
```

**Health score:**
```
score = 0.60 · R² + 0.40 · (1 − |α̂ − 1| / 1)
```

**Justification:** R² measures goodness-of-fit; the second term penalizes α deviating from
the ideal value of 1. The 60/40 weighting favors fit quality over exact exponent matching.

---

## 6. HV Bundle (Context Hybridization)

The hybridize operation `LHV.hybridize(ctx)` computes:
```
hv_hybrid = bundle(hv_original, hv_context)
```
where `bundle` is the VSA majority-vote (threshold summation):

```
bundle(a, b)ᵢ = ⌊ (aᵢ + bᵢ) / 2 ⌋  (majority of 2 → random tiebreak)
```

**Polysemy:** For a concept c with n context vectors {ctx₁, ..., ctx_n}:
```
hv_c_ctx_i = bundle(hv_c, ctx_i)
```
Each hybrid represents a context-dependent sense. Retrieval queries with context automatically
select the most relevant sense via cosine similarity.

---

## 7. Spreading Activation

### 7.1 Activation Propagation

Given seed concept c₀ with activation a₀ = 1.0:

```
a_v(hop h) = a_u(hop h-1) · bond(u,v) · decay^h
```

where `decay` = spreading_decay = 0.30.

**Convergence:** Since decay < 1 and bond ∈ [0,1]:
```
a_v(h) ≤ decay^h → 0 as h → ∞
```
so the spreading activation is guaranteed to converge.

### 7.2 Domain Affinity Boost

When spreading within domain D, each neighbor v receives a multiplicative boost:
```
spread_boosted = spread · (1 + 0.5 · affinity_D(v))
```

Since affinity_D(v) ∈ [0,1], the boost is in [1.0, 1.5], preserving convergence.
