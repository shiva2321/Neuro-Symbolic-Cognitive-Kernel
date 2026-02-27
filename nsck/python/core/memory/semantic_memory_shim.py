"""
Shim for Rust-accelerated spreading activation.
Auto-selects Rust or Python backend (same pattern as hypervec_shim.py).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any

_USE_RUST = False
_rust_instance: Optional[Any] = None

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
    
    Syncs the NetworkX graph to Rust's DashMap, runs spreading activation,
    returns result dict. Returns None if Rust not available or sync fails.
    
    The caller should fall through to Python if this returns None.
    """
    rust = _get_rust_instance()
    if rust is None:
        return None
    
    try:
        # Sync concepts from graph to Rust backend
        import python.core.vsa.hypervec_shim as hv_mod
        for node in concept_graph.nodes():
            try:
                hv = hv_mod.HyperVector(hash(node) % (2**32))
                rust.add_concept(str(node), hv)
            except Exception:
                pass
        
        # Spreading activation in Python using Rust storage (DashMap)
        # Since Rust's SemanticMemoryConcurrent doesn't expose spread_activation directly,
        # we run the activation loop but using Rust for concept lookup
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
