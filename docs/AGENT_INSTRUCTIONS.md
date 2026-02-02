# NSCK AGI: Agent Instructions & Development Guidelines

This document serves as the "constitution" for all AI agents and human developers working on the NSCK (Neuro-Symbolic Cognitive Kernel) project. Its purpose is to ensure consistency, safety, and adherence to the architectural vision across multiple sessions.

## 1. Primary Directives

*   **Stick to the Roadmap:** All work must align with `docs/IMPLEMENTATION_ROADMAP.md`. Do not invent new features or diverge from the plan without explicit user approval.
*   **Consult Session Handoff:** Always start by reading `docs/SESSION_HANDOFF.md` to understand the immediate context, active tasks, and recent changes.
*   **Update Session Handoff:** Before finishing your session, you MUST update `docs/SESSION_HANDOFF.md` with your progress, next steps, and any open issues.

## 2. Architectural Principles

*   **Neuro-Symbolic Hybrid:** The system MUST combine neural networks (SNNs/PyTorch) with symbolic reasoning (VSA/Rust). Do not replace symbolic components with pure deep learning unless specified.
*   **Biological Inspiration:** Maintain the homeostatic/biological drive model. Agents are not just state machines; they have "needs" (hunger, energy, etc.).
*   **Local & Efficient:** The system targets consumer hardware ("Tiny 11" specs). Optimize for CPU/Memory efficiency. Avoid massive dependencies where lightweight alternatives exist.
*   **LLM Isolation:** The LLM is a *peripheral* for communication, NOT the brain. Core logic (decisions, planning) must remain in the Neuro-Symbolic kernel. The LLM should never directly drive motor actions or bypass the Global Workspace.

## 3. Coding Standards

*   **Language:** Python 3.9+ (Type Hints required), Rust (for VSA/Performance critical components).
*   **Style:** Follow PEP 8. Use meaningful variable names.
*   **Documentation:** All classes and major functions must have docstrings. Update `README.md` and `docs/` if architecture changes.
*   **Modularity:** Keep components loosely coupled (e.g., `CognitiveEngine`, `VisualCortex`, `MotorCortex`).

## 4. Operational Workflow

1.  **Read Context:** Check `SESSION_HANDOFF.md` and `IMPLEMENTATION_ROADMAP.md`.
2.  **Plan:** Use `task_boundary` (if available) or outline steps in chat.
3.  **Implement:** Write code, keeping changes atomic.
4.  **Verify:** Run tests or simulations. *Never* assume code works without running it.
5.  **Document:** Update code comments and documentation files.
6.  **Handoff:** Update `SESSION_HANDOFF.md` for the next agent.

## 5. "Do Not" List

*   **DO NOT** delete existing documentation or history without backup.
*   **DO NOT** introduce "magic" logic hidden from the Global Workspace.
*   **DO NOT** change the directory structure arbitrarily.
*   **DO NOT** ignore existing TODOs or FIXMEs; address them if relevant to your task.
