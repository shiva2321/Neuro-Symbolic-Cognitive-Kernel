"""
Shim for Rust-accelerated spreading activation.
Auto-selects Rust or Python backend (same pattern as hypervec_shim.py).

V18: The shim now delegates to ``SemanticMemoryConcurrent.parallel_spread_activation()``
when the Rust backend is available.  Since ``SemanticMemory.add_concept()`` and
``add_relation()`` already mirror writes to the Rust DashMap at write time (Change 1),
the DashMap is always up-to-date and no incremental sync is needed here.

The legacy ``_synced_concepts`` set is retained as a no-op guard for backward
compatibility but is no longer needed for correctness.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

_USE_RUST = False
_rust_instance: Optional[Any] = None
# V4: cache the Rust spreading_activation_step free function if available
_rust_step_fn: Optional[Any] = None
# Retained for backward compatibility — no longer drives the sync logic in V18.
_synced_concepts: Set[str] = set()

try:
    import hypervec_rs as _ext
    if hasattr(_ext, "SemanticMemoryConcurrent") and _ext.SemanticMemoryConcurrent is not None:
        _USE_RUST = True
    # V4: capture the spreading_activation_step free function at import time
    if hasattr(_ext, "spreading_activation_step"):
        _rust_step_fn = _ext.spreading_activation_step
except ImportError:
    pass


def _get_rust_instance():
    """Lazily create and cache the Rust SemanticMemoryConcurrent instance."""
    global _rust_instance
    if not _USE_RUST:
        return None
    if _rust_instance is None:
        try:
            import hypervec_rs as _ext
            _rust_instance = _ext.SemanticMemoryConcurrent()
        except Exception:
            return None
    return _rust_instance


def spread_activation_fast(
    concept_graph,
    start_concepts: List[str],
    relation_weights: Dict[str, float],
    stigmergy: Dict,
    steps: int = 3,
    decay: float = 0.7,
    rust_backend: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """
    Try Rust-accelerated spreading activation.

    V18 fast path: delegates directly to ``SemanticMemoryConcurrent.parallel_spread_activation()``
    which runs all steps inside Rust with Rayon parallelism.  The Rust DashMap graph
    is kept in sync by ``SemanticMemory.add_concept()`` / ``add_relation()`` at write
    time, so no incremental sync is required here.

    ``rust_backend`` should be the **caller's own** ``SemanticMemoryConcurrent`` instance
    (i.e. ``SemanticMemory._rust_backend``).  Passing it explicitly avoids the
    previous bug where the shim used its own module-level instance (always empty)
    instead of the instance that already has all concepts/edges mirrored at write
    time.  When ``rust_backend`` is None the shim falls back to its module-level
    instance (legacy behaviour, kept for callers that don't pass the argument).

    Falls back to the edge-list Python path when Rust is unavailable or when
    stigmergy boosts are active (stigmergy modifies per-edge weights dynamically
    and is not yet modelled in the Rust parallel path).

    Returns None on failure; the caller falls through to the pure-Python implementation.
    """
    # Prefer the caller-supplied backend (already populated); fall back to the
    # module-level singleton only when no backend is provided.
    rust = rust_backend if rust_backend is not None else _get_rust_instance()
    if rust is None:
        return None

    # Stigmergy dynamically modifies per-edge weights — not modelled in Rust path yet.
    # Fall through to Python when stigmergy is active so weights stay correct.
    _stigmergy_boost = bool(stigmergy)

    try:
        # V18 primary path: parallel_spread_activation (all steps inside Rust, no edge list).
        # ONLY use this path when a populated rust_backend was explicitly provided by the
        # caller (i.e. SemanticMemory._rust_backend which was kept in sync at write time).
        # The module-level _rust_instance singleton is always empty — never use it for
        # parallel_spread_activation because it will return only start concepts.
        if (
            not _stigmergy_boost
            and rust_backend is not None          # caller-provided, already populated
            and hasattr(rust, 'parallel_spread_activation')
        ):
            valid_starts = [c for c in start_concepts if c in concept_graph]
            if valid_starts:
                try:
                    result = rust.parallel_spread_activation(
                        start_concepts=valid_starts,
                        steps=steps,
                        decay=decay,
                        min_activation=0.01,
                        bidirectional=False,
                    )
                    return result
                except Exception as _e:
                    logger.warning(
                        "[shim] Rust parallel_spread_activation failed (%s); "
                        "falling back to edge-list path.",
                        _e,
                    )

        # V4 secondary path: Rust spreading_activation_step free function with edge list.
        # Used when parallel_spread_activation is unavailable or when stigmergy is active.
        import python.core.vsa.hypervec_shim as hv_mod
        sync_errors = 0
        new_nodes = [n for n in concept_graph.nodes() if str(n) not in _synced_concepts]
        for node in new_nodes:
            try:
                hv = hv_mod.HyperVector(hash(node) % (2**32))
                rust.add_concept(str(node), hv)
                _synced_concepts.add(str(node))
            except Exception:
                sync_errors += 1
        # If more than half of new nodes failed to sync, bail out to Python path
        n_new = len(new_nodes)
        if n_new > 0 and sync_errors > n_new // 2:
            return None

        # Build initial activations dict
        activation: Dict[str, float] = {c: 1.0 for c in start_concepts if c in concept_graph}
        _MAX_FRONTIER = 200

        # V4: Try Rust spreading_activation_step for each step.
        _active_step_fn = _rust_step_fn  # local alias; set to None on failure

        if _active_step_fn is not None:
            # Build edges list: (from, to, weight) — include stigmergy boost in weight
            edges = []
            for u, v, data in concept_graph.edges(data=True):
                rel = data.get("relation", "similar_to")
                w = relation_weights.get(rel, 0.3)
                if _stigmergy_boost:
                    w = w * (1.0 + stigmergy.get((u, v), 0.0))
                edges.append((str(u), str(v), float(w)))
            # Run each step through Rust
            for _ in range(steps):
                try:
                    activation = _active_step_fn(activation, edges, decay, _MAX_FRONTIER)
                except Exception as _exc:
                    logger.warning(
                        "[shim] Rust spreading_activation_step failed (%s); "
                        "falling back to Python. Ensure snn_rs/hypervec_rs are built correctly.",
                        _exc,
                    )
                    _active_step_fn = None
                    break
            if _active_step_fn is not None:
                return activation

        # Python fallback spreading activation loop
        for _ in range(steps):
            new_activation = activation.copy()
            frontier = sorted(
                ((c, a) for c, a in activation.items() if a >= 0.01),
                key=lambda x: x[1], reverse=True)[:_MAX_FRONTIER]

            for concept, act in frontier:
                for _, neighbor, data in concept_graph.out_edges(concept, data=True):
                    rel = data.get("relation", "similar_to")
                    edge_weight = relation_weights.get(rel, 0.3)
                    if _stigmergy_boost:
                        stig = stigmergy.get((concept, neighbor), 0.0)
                        edge_weight = edge_weight * (1.0 + stig)
                    spread_val = act * decay * edge_weight
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread_val
            activation = new_activation

        return activation
    except Exception:
        return None


__backend__ = "Rust" if _USE_RUST else "Python"

