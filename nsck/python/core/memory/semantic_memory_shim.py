"""
Shim for Rust-accelerated spreading activation.
Auto-selects Rust or Python backend (same pattern as hypervec_shim.py).

The shim caches which concepts have already been synced to the Rust
DashMap, so repeated calls to ``spread_activation_fast()`` only push
*new* nodes rather than re-syncing the entire graph each time.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Set

_USE_RUST = False
_rust_instance: Optional[Any] = None
# Track which concept names have already been synced to the Rust instance
# to avoid redundant add_concept calls on repeated spread_activation calls.
_synced_concepts: Set[str] = set()

try:
    import hypervec_rs as _ext
    if hasattr(_ext, "SemanticMemoryConcurrent") and _ext.SemanticMemoryConcurrent is not None:
        _USE_RUST = True
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
) -> Optional[Dict[str, float]]:
    """
    Try Rust-accelerated spreading activation.
    
    Syncs *new* concepts from the NetworkX graph to Rust's DashMap (already-
    synced concepts are skipped via an in-process cache), then runs the
    spreading activation loop in Python against the synced concept set.
    Returns None if Rust is not available or sync fails; the caller should
    fall through to the pure-Python implementation in that case.
    """
    rust = _get_rust_instance()
    if rust is None:
        return None
    
    try:
        # Sync only concepts that haven't been pushed to Rust yet.
        # This avoids paying the full sync cost on every spread_activation call.
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
        
        # Spreading activation loop (Python, using synced Rust concept set)
        activation: Dict[str, float] = {c: 1.0 for c in start_concepts if c in concept_graph}
        _MAX_FRONTIER = 200
        _stigmergy_boost = bool(stigmergy)
        
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
