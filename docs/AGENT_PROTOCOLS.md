# NSCK Agent Protocols: DO NOT IGNORE

**To Future Agents:**
You are contributing to a long-term, complex AGI project. "Wing it" engineering will destroy the architecture. Follow these protocols strictly.

## 1. Context First
*   **ALWAYS** read `docs/NSCK_SPECIFICATION.md` before writing code. This is the source of truth.
*   **ALWAYS** read `docs/ROADMAP.md` to know where you are.
*   **NEVER** deviate from the defined architecture (e.g., do not install a Transformer/LLM when the spec calls for VSA/SDR).

## 2. No "Placeholder" Intelligence
*   Do not fake intelligence with `if-else` statements.
*   If a module (e.g., "Consciousness") is too hard to implement fully in your session, implement the **Data Structure** and the **Interface** correctly, even if the logic is simple.
*   *Bad*: `def get_consciousness(): return "high"`
*   *Good*: Create the `PhiMetric` class, define the `inputs`, and return a standard placeholder value with a `TODO` linking to the spec.

## 3. Preservation of Efficiency
*   The User's Goal is **Efficiency**.
*   **FORBIDDEN**: Importing heavy libraries like `transformers`, `tensorflow` (unless for specific compat), or large pre-trained models > 500MB.
*   We use **PyTorch** for SNNs and **Custom NumPy/Rust** for VSA.

## 4. Handoff Protocol
*   At the end of your session, you **MUST** update `docs/ROADMAP.md` marking what you completed.
*   Leave a clear "Next Steps" note in `task.md`.
*   If you leave the codebase in a broken state (mid-refactor), you MUST explicitly document it in `docs/BROKEN_STATE.md`.

## 5. File Structure Sanctity
*   `python/` is for the Brain (`python_server.py`, `cognitive_engine.py`).
*   `docs/` is for permanent documentation.
*   `rust_vsa/` is for performance-critical hypervector code.
*   Do not create random files in the root.

## 6. Verification & Testing (MANDATORY)
*   **Trust, but Verify**: Every major module change must be accompanied by a proof-of-work.
*   **Unit Tests**: Use `pytest` for logic modules (e.g., "Does VSA binding actually reverse?").
*   **Integration Tests**: Run the `dashboard.py` and confirm the UI updates correctly.
*   **Logs as Proof**: When notifying the user of completion, you MUST include logs or screenshot proof that the feature works. "I have implemented X" is not enough; "Here is the log showing X firing" is required.
