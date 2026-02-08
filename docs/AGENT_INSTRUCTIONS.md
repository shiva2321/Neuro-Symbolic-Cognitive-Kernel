# NSCK: Development Guidelines

This document defines development conventions for the NSCK (Neuro-Symbolic Cognitive Kernel) project.

> **For multi-agent coordination** (parallel/sequential coding agents), see [`docs/AGENT_COORDINATION_PLAN.md`](AGENT_COORDINATION_PLAN.md).

## 1. Primary Directives

*   **Stick to the Roadmap:** All work must align with `ROADMAP_TO_AGI.md` and `docs/IMPLEMENTATION_ROADMAP.md`. Do not invent new features or diverge from the plan without explicit user approval.
*   **Accuracy:** Documentation must reflect the actual state of the codebase. Do not claim capabilities that are not implemented and tested.
*   **Role Boundaries:** If you are operating as a coding agent with an assigned role (see `docs/AGENT_COORDINATION_PLAN.md`), stay within your role's file scope. Do not modify files assigned to other roles without documenting the cross-role dependency.

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
*   **Tests:** All new functionality must have corresponding tests in `nsck-demo/tests/`.

## 4. Operational Workflow

1.  **Read Context:** Check `ROADMAP_TO_AGI.md`, `docs/IMPLEMENTATION_ROADMAP.md`, and `docs/AGENT_COORDINATION_PLAN.md`.
2.  **Record Baseline:** Run `python -m pytest nsck-demo/tests/ -v` and note the number of passing tests.
3.  **Plan:** Outline steps before implementing. Verify your plan aligns with your assigned Work Unit.
4.  **Implement:** Write code, keeping changes atomic. Stay within your role's file scope.
5.  **Verify:** Run tests (`python -m pytest nsck-demo/tests/ -v`). Confirm test count has not dropped.
6.  **Document:** Update code comments and documentation if architecture changes.
7.  **Handoff:** Write a session summary (see `docs/AGENT_COORDINATION_PLAN.md` Section 10) in your PR description or commit message.

## 5. "Do Not" List

*   **DO NOT** introduce heavy matrix multiplications in core VSA paths.
*   **DO NOT** introduce "magic" logic hidden from the cognitive pipeline.
*   **DO NOT** claim capabilities that are not backed by passing tests.
*   **DO NOT** commit binary artifacts (*.db, *.pkl, *.pth, *.csv) — these are in .gitignore. Runtime-generated *.png files are also ignored; only commit PNGs explicitly needed for documentation.
*   **DO NOT** modify files outside your assigned role's scope without documenting the cross-role dependency.
*   **DO NOT** add new dependencies without checking for security vulnerabilities and documenting the justification.
*   **DO NOT** change public API signatures of critical modules unless your Work Unit explicitly requires it.

## 6. Agent Coordination Quick Reference

If you are a coding agent working on this project:

1.  **Find your Work Unit** in `docs/AGENT_COORDINATION_PLAN.md` Section 4.
2.  **Check parallelism** — verify no other agent is modifying your target files (Section 5).
3.  **Follow guardrails** — pre-execution checklist, during-execution rules, post-execution verification (Section 6).
4.  **Share context** — read required docs, check recent git history (Section 7).
5.  **Write handoff** — summarize what you did, what's left, and any decisions made (Section 10).

## 7. Related Documents

| Document | Purpose |
|---|---|
| [`AGENT_COORDINATION_PLAN.md`](AGENT_COORDINATION_PLAN.md) | Multi-agent coordination: roles, work units, guardrails, parallelism |
| [`../ROADMAP_TO_AGI.md`](../ROADMAP_TO_AGI.md) | Long-term development roadmap (7 phases) |
| [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) | Technical implementation details |
| [`COMPLETE_MODULE_ANALYSIS.md`](COMPLETE_MODULE_ANALYSIS.md) | Module-by-module status and analysis |
| [`RUST_VSA_ANALYSIS.md`](RUST_VSA_ANALYSIS.md) | Rust VSA performance analysis |
