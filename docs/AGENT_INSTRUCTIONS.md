# NSCK: Development Guidelines

This document defines development conventions for the NSCK (Neuro-Symbolic Cognitive Kernel) project.

## 1. Primary Directives

*   **Stick to the Roadmap:** All work must align with `ROADMAP_TO_AGI.md` and `docs/IMPLEMENTATION_ROADMAP.md`. Do not invent new features or diverge from the plan without explicit user approval.
*   **Accuracy:** Documentation must reflect the actual state of the codebase. Do not claim capabilities that are not implemented and tested.

## 2. Architectural Principles

*   **Neuro-Symbolic Hybrid:** The system combines spiking neural networks (SNNs/PyTorch) with symbolic reasoning (VSA). Do not replace symbolic components with pure deep learning.
*   **Efficiency First:** The system targets consumer hardware (CPU-only, 4GB+ RAM). Use O(n) VSA operations instead of O(n²) matrix multiplications. Avoid large dense layers (keep dims ≤ 128 for shared layers).
*   **LLM Isolation:** Any LLM integration is a peripheral for communication, NOT the brain. Core logic (decisions, planning) must remain in the Neuro-Symbolic kernel.

## 3. Coding Standards

*   **Language:** Python 3.11+ (type hints required), Rust (for VSA performance-critical components).
*   **Style:** Follow PEP 8. Use meaningful variable names.
*   **Imports:** Always use `import hypervec_shim as hypervec_rs` — never import `hypervec_rs` directly.
*   **Documentation:** All classes and major functions must have docstrings.
*   **Modularity:** Keep components loosely coupled.

## 4. Operational Workflow

1.  **Read Context:** Check `ROADMAP_TO_AGI.md` and `docs/IMPLEMENTATION_ROADMAP.md`.
2.  **Plan:** Outline steps before implementing.
3.  **Implement:** Write code, keeping changes atomic.
4.  **Verify:** Run tests (`python -m pytest nsck-demo/tests/ -v`). Never assume code works without running it.
5.  **Document:** Update code comments and documentation if architecture changes.

## 5. "Do Not" List

*   **DO NOT** introduce heavy matrix multiplications in core VSA paths.
*   **DO NOT** introduce "magic" logic hidden from the cognitive pipeline.
*   **DO NOT** claim capabilities that are not backed by passing tests.
*   **DO NOT** commit binary artifacts (*.db, *.pkl, *.pth, *.csv) — these are in .gitignore.
