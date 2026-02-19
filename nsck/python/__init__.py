"""
NSCK Python package.

This package holds the symbolic/VSA cognitive components.

Package structure::

    python/
        core/
            vsa/          # HyperVector algebra (Rust + Python fallback)
            memory/       # Episodic + Semantic memory
            reasoning/    # GWT, causal, rules, planner, analogy, engine
            language/     # NLU, NLG, dialogue, text learning
            perception/   # Symbol grounding
            cognitive/    # Metacognition, self-model, theory of mind
            learning/     # Curiosity-driven exploration
            integration/  # Config, persistence, explanation, brain fusion
"""

# Install hypervec_rs compat (adds missing methods to the compiled class)
try:
    from python.core.vsa import hypervec_shim as _hypervec_shim_compat  # noqa: F401
except Exception:
    # If hypervec_rs isn't importable, modules may fall back to hypervec_py.
    pass
