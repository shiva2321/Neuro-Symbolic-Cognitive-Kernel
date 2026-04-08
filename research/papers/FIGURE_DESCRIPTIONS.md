# Figure Descriptions for Paper 5 — Model Transplantation

## Figure 1 — Transplantation Quality Metrics

**Caption:**
Transplantation quality metrics for two vocabulary sizes (n=100, n=500) on synthetic Gaussian embeddings (d=32, D=10,240). Four metrics are compared across both sizes: Spearman rank correlation ρ (dashed line threshold 0.995), Recall@10 (dashed line threshold 0.90), Recall@50, and Adjusted Rand Index (ARI). All metrics exceed their respective baseline thresholds, confirming near-perfect rank preservation and strong nearest-neighbour recovery in binary HV space.

---

## Figure 2 — Cluster Separation in HV Space

**Caption:**
Intra-cluster vs. inter-cluster HV similarity for a 5-cluster, 20-token-per-cluster synthetic test (d=32, D=10,240). The separation gap (Δ = 0.44, shown by the double-headed arrow) confirms that transplanted HVs maintain cluster identity well above the random baseline of 0.500. This validates that semantic structure from dense embeddings is preserved after binarisation.

---

## Figure 3 — Bit-Flip Robustness

**Caption:**
Recall@10 stability under progressive bit-flip corruption (0–30% flip rate) on a small random vocabulary (n=200, d=32, D=10,240). The flat plateau reflects the near-uniform Hamming landscape of random small vocabularies, not a fundamental failure of the projection: all pairwise similarities cluster near the expected 0.500, leaving no clear nearest neighbour to retrieve. Larger, semantic vocabularies show robust recall even under higher flip rates (see Exp. 5.2).

---

## Figure 4 — Projection Timing

**Caption:**
Left: total projection time vs. vocabulary size on synthetic embeddings (linear scale, d=32, D=10,240), with linear least-squares fit showing slope ≈ 1.37 μs/token. Right: per-token projection cost across all tested embedding sources (synthetic, BERT-base d=768, MiniLM d=384, MiniLM-scaled d=384), computed as total projection time divided by vocabulary size. MiKTeX latency differences reflect Python NumPy overhead and embedding dimensionality; a Rust-native backend would further reduce cost.

---

## Figure 5 — Transplantation Quality: Synthetic vs Real Pretrained Models

**Caption:**
Comparison of three key quality metrics across five embedding sources: two synthetic sizes (n=100, n=500, d=32), contextual BERT-base (n=4,096, d=768), MiniLM-curated (n=280, d=384), and MiniLM-scaled (n=5,000, d=384). All achieve Spearman ρ > 0.98, confirming the projection generalises across embedding sources and scales. Recall@10 varies by source richness (0.898 for curated MiniLM to 0.954 for scaled MiniLM); ARI is sensitive to semantic structure, improving 3× from synthetic (0.18–0.21) to MiniLM (0.61), indicating that transplantation fidelity scales with the semantic coherence of the source embeddings.

---

## Figure 6 — VSA Analogy Test: Ranked Dot Plot + Summary Card

**Caption:**
Left: ranked scatter plot of Top-1 HV similarity per analogy (ascending order, Hamming scale), colour-coded by outcome (green = Top-1 correct, blue = Top-5 only, red = missed). Selected analogy text labels provide semantic anchors for interpretation. Right: summary cards showing aggregate accuracy (Top-1: 39/50 = 78%, Top-5: 48/50 = 96%), per-category outcome breakdown, and the two lowest-similarity misses with expected vs. predicted words. This layout exposes both overall robustness and failure modes across the 50-equation balanced benchmark (5 categories × 10 analogies each).

---

