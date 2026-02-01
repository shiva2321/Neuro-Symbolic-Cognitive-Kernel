"""
NSCK Python package.

This package holds the symbolic/VSA cognitive components used by the demo.

IMPORTANT COMPAT NOTE
---------------------
Some environments ship an older `hypervec_rs` binary that lacks helper methods
like `weighted_bundle` and `lsh_hash`.

We install a small compatibility patch on import so the rest of the codebase
(and tests) can rely on these methods.
"""

# Install hypervec_rs compat (adds missing methods to the compiled class)
try:
    from . import hypervec_shim as _hypervec_shim_compat  # noqa: F401
except Exception:
    # If hypervec_rs isn't importable, modules may fall back to hypervec_py.
    pass
