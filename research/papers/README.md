# NSCK Research Papers

This directory contains three research papers derived from the NSCK (Neuro-Symbolic Cognitive Kernel) project. Each paper is independent and can be submitted separately.

## Papers

### Paper 1 — Foundation Paper
**"NSCK: A Unified Neuro-Symbolic Cognitive Kernel Using Binary Hypervector Representations"**
- **File:** `paper1_foundation.md`
- **Scope:** Full system architecture, design principles, all 8 layers, benchmarks
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

## Submission Guide

### For arXiv
1. Convert Markdown to LaTeX (use pandoc: `pandoc paper1_foundation.md -o paper1.tex`)
2. Apply a standard template (e.g., NeurIPS 2025 style)
3. Submit at https://arxiv.org — choose cs.AI as primary category
4. arXiv has no review — it establishes a priority timestamp

### For Workshops / Conferences
1. Check the specific venue's formatting requirements (page limits, template)
2. Most workshops accept 4–8 page papers; shorten the paper accordingly
3. Focus on the core contribution and experiments; move details to appendix

### For Journals (Frontiers in AI, JAIR)
1. These accept longer papers (15–25 pages)
2. No page limit pressure — include full experimental details
3. Open access journals are good for visibility

## Formatting Notes

- All papers use Markdown with LaTeX math notation (`$$...$$`)
- Convert to LaTeX with: `pandoc paperX.md -o paperX.tex --standalone`
- For PDF: `pandoc paperX.md -o paperX.pdf --pdf-engine=xelatex`
- Tables use standard Markdown table syntax
- Architecture diagrams are in ASCII art (convert to proper figures for final submission)

## What Each Paper Claims (and Does Not Claim)

All three papers are grounded in the actual NSCK implementation:
- Every benchmark number comes from measured runs on real hardware
- Every formula corresponds to actual code in the repository
- Limitations are documented honestly in dedicated sections
- We do not claim NSCK is an AGI system or replaces LLMs

## Author Information

**Shivam Prajapati**
Bachelor of Computer Science
University of Prince Edward Island
Charlottetown, PE, Canada

GitHub: https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel
