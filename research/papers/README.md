# NSCK Research Papers

This directory contains eight research papers derived from the NSCK (Neuro-Symbolic Cognitive Kernel) project. Each paper is independent and can be submitted separately.

> **V4 Note (February 2026):** The current canonical release is **NSCK V4** — the consolidated version integrating all features from V3 through V18 development iterations. All benchmark numbers in the papers were produced with Rust backends active (`NSCK_USE_RUST=1`). To reproduce: `cd nsck && make bench-papers`.

**Current implementation status (Feb 2026): 1,669 tests passing (Rust backend active — hypervec\_rs + snn\_rs), 5 skipped, 4 xfailed. 94+ core modules, ~36K LOC Python, ~4.3K LOC Rust. All benchmark numbers measured on real hardware with the Rust backend.**

## Papers

### Paper 1 — Foundation Paper
**"NSCK: A Unified Neuro-Symbolic Cognitive Kernel Using Binary Hypervector Representations"**
- **File:** `paper1_foundation.md`
- **Scope:** Full system architecture, design principles, all 7 layers, benchmarks, V9–V13 extensions
- **Target Venues:** arXiv (immediate), NeurIPS Workshop on Neuro-Symbolic AI, AAAI Student Abstract
- **Priority:** Submit first — this is the paper all future work cites

### Paper 2 — Focused Contribution
**"Auditable Causal Reasoning via Δ-P Discovery and Global Workspace Broadcasting"**
- **File:** `paper2_causal_reasoning.md`
- **Scope:** Δ-P causal discovery, counterfactual simulation, GWT integration, auditability
- **Target Venues:** IJCAI Workshop on Explainable AI, AAAI Workshop on Neuro-Symbolic Learning, Frontiers in AI
- **Priority:** Submit second — the most scientifically clean, narrow contribution

### Paper 3 — Novel Mechanism
**"Zero-Shot Cross-Domain Transfer via VSA Structural Alignment"**
- **File:** `paper3_cross_domain_transfer.md`
- **Scope:** Automatic cross-domain concept alignment, rule lifting, zero-shot transfer
- **Target Venues:** CogSci (directly relevant to Gentner's work), ICLR Workshop on Transfer Learning, JAIR
- **Priority:** Submit third — the most novel but needs strongest experimental support

### Paper 4 — SNN–VSA Bridge
**"Bridging Spikes and Symbols: Bidirectional VSA-SNN Perception with STDP Plasticity"**
- **File:** `paper4_snn_bridge.md`
- **Scope:** LIF SNN perception, STDP weight convergence, rate/temporal encoding, Rust-accelerated benchmarks
- **Key Results:** 100% pattern accuracy @ 1.85 ms/call (Rust, 24× speedup), 90% noise robustness, STDP convergence step 50, sub-linear scaling 0.856→1.846 ms for 64→256 neurons
- **Bug Fixed:** `TemporalCoder` `time_bins` kwarg removed (`snn_perception.py` L491)
- **Errata (this PR):** The legacy `SNNPerceptionModule._apply_stdp()` method contained an O(snn_size × input_dim) nested Python loop. This was dead code in practice (Rust and `PythonSnnCore` fast paths are always preferred) but has been replaced with the vectorized outer-product implementation for correctness. Rust-accelerated results are unaffected.
- **Target Venues:** Neural Networks, Neuromorphic Computing and Engineering, CogSci

### Paper 5 — Model Transplantation
**"Model Transplantation: Absorbing Pretrained Neural Knowledge into Binary Hypervectors"**
- **File:** `paper5_transplantation.md`
- **Scope:** Random Gaussian projection, JL Lemma analysis, SimHash theory, transplant pipeline benchmarks
- **Key Results:** Spearman ρ > 0.995, Recall@10 = 0.910–0.929, cluster separation = 0.440, round-trip fidelity = 1.000, determinism verified
- **Target Venues:** ICLR Workshop on Efficient ML, NeurIPS, AAAI

### Paper 6 — Active Inference + GWT
**"Active Inference Meets Global Workspace Theory: Free-Energy-Guided Coalition Competition in a Cognitive Architecture"**
- **File:** `paper6_active_inference.md`
- **Scope:** FEP action selection, count-based exploration, LIDA competition, mental rehearsal veto
- **Key Results:** World model PE → 0 in 10 updates, 30% exploration rate (60/200 steps), GWT winner at activation 1.350, 10/10 dangerous proposals vetoed
- **Errata (this PR):** The `free_energy()` method previously returned near-constant ≈0.4 in the default no-CuriosityModule configuration. This was caused by `epistemic_value()` hardcoding `return 0.1` regardless of world model state. The fix (using world-model familiarity as an epistemic proxy) improves action differentiation from ~±0.02 salience bias to ~±0.04 for unseen vs. seen transitions. The paper's measured results (mean F = −0.510 with a configured CuriosityModule) remain valid.
- **Target Venues:** Cognitive Science, Journal of Artificial Intelligence Research, Active Inference workshop

### Paper 7 — Memory Architecture
**"Homeostatic Memory Architecture: Four-Store VSA Memory with Self-Regulation and Drift Detection"**
- **File:** `paper7_memory.md`
- **Scope:** Semantic/episodic/procedural/cross-modal memory, drift detection, homeostasis, spreading activation
- **Key Results:** Semantic query 1.41–2.60 ms (sub-linear HNSW), episodic recall 0.072 ms, 100% procedural hit rate, drift alarm at ≥30% flip rate, 100% cross-modal recall, 30/100 spreading activation reach
- **Target Venues:** Cognitive Architecture Workshop, Memory & Cognition, Frontiers in Psychology

### Paper 8 — Cognitive Safety
**"Transparent Cognitive Safety: Declarative Verification, Emotion Modulation, and Theory of Mind in a Neuro-Symbolic Architecture"**
- **File:** `paper8_safety.md`
- **Scope:** Safety rule verification, confidence gating, Plutchik emotion HVs, Theory of Mind, end-to-end audit trail
- **Key Results:** TPR = 1.00 / FPR = 0.00, emotion HVs quasi-orthogonal (mean sim = 0.499), text emotion accuracy = 100% (16 sentences), Sally-Anne test passed, multi-agent ToM correct
- **Target Venues:** AIES (AAAI/ACM AI Ethics and Society), SafeAI Workshop, Frontiers in Artificial Intelligence

---

## Experiment & Results Files

Benchmark experiments are in `research/experiments/`. Run from the `nsck/` directory:

```bash
cd nsck
# Run individual experiment:
python ../research/experiments/paper4_snn_benchmarks.py
# Run all experiments:
python ../research/experiments/run_all_experiments.py
```

Results JSON files are written to `research/results/`:
- `paper4_results.json` — SNN benchmarks (Rust backend)
- `paper5_results.json` — Transplant benchmarks (Rust backend)
- `paper6_results.json` — Active Inference + GWT benchmarks
- `paper7_results.json` — Memory architecture benchmarks
- `paper8_results.json` — Safety layer benchmarks

---

## Submission Guide

### For arXiv
1. Convert Markdown to LaTeX: `pandoc paperX.md -o paperX.tex --standalone`
2. Apply NeurIPS 2025 or similar template
3. Submit at https://arxiv.org — primary category: cs.AI
4. arXiv has no review — establishes priority timestamp immediately

### For Workshops / Conferences
1. Check venue formatting requirements (page limits, template)
2. Most workshops: 4–8 pages; shorten accordingly
3. Focus on core contribution and experiments; move details to appendix

### For Journals (Frontiers in AI, JAIR, Neural Networks)
1. These accept 15–25 page papers
2. Include full experimental details
3. Open access journals maximise visibility

## Formatting Notes

- All papers use Markdown with LaTeX math (`$$...$$` display, `$...$` inline)
- Convert to LaTeX: `pandoc paperX.md -o paperX.tex --standalone`
- To PDF: `pandoc paperX.md -o paperX.pdf --pdf-engine=xelatex`
- Tables: standard Markdown table syntax
- Architecture diagrams: ASCII art (convert to proper figures for final submission)

## What Each Paper Claims (and Does Not Claim)

All papers are grounded in the actual NSCK implementation:
- Every benchmark number comes from measured runs with the Rust backend active
- Every formula corresponds to actual code in the repository
- Limitations are documented honestly in dedicated "Honest Limitations" sections
- We do not claim NSCK is an AGI system or replaces LLMs
- We do not claim biological plausibility beyond the cited theoretical inspirations

## Author Information

**Shivam Prajapati**  
Bachelor of Computer Science  
University of Prince Edward Island  
Charlottetown, PE, Canada

GitHub: https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel

