"""
Rust Concurrent Memory Shim
============================
Exposes Rust concurrent memory classes with Python fallbacks.
"""
from __future__ import annotations

import sys
import os
from typing import List, Optional, Any, Tuple

USE_RUST_CONCURRENT = False

try:
    _nsck_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    if _nsck_root not in sys.path:
        sys.path.insert(0, _nsck_root)
    import hypervec_rs as _hvrs
    USE_RUST_CONCURRENT = hasattr(_hvrs, 'SemanticMemoryConcurrent')
except ImportError:
    _hvrs = None  # type: ignore


if USE_RUST_CONCURRENT and _hvrs is not None:
    SemanticMemoryConcurrent = _hvrs.SemanticMemoryConcurrent
    EpisodicMemoryConcurrent = _hvrs.EpisodicMemoryConcurrent
    Episode = _hvrs.Episode
    CognitiveWorkerPool = getattr(_hvrs, 'CognitiveWorkerPool', None)
    HyperVectorRegistry = getattr(_hvrs, 'HyperVectorRegistry', None)
    PersistentStorage = getattr(_hvrs, 'PersistentStorage', None)
    parallel_bundle = getattr(_hvrs, 'parallel_bundle', None)
    batch_parallel_similarity_search = getattr(_hvrs, 'batch_parallel_similarity_search', None)
    batch_similarity_matrix = getattr(_hvrs, 'batch_similarity_matrix', None)
else:
    # ── Python fallbacks ──────────────────────────────────────────────────────

    from dataclasses import dataclass, field as dc_field

    @dataclass
    class Episode:
        """Simple episode dataclass (Python fallback)."""
        timestamp: float = 0.0
        task_tag: str = "global"
        situation_hv: Any = None
        action: str = "none"
        outcome: str = "unknown"
        reward: float = 0.0
        impact_score: float = 0.0

    class SemanticMemoryConcurrent:
        """Python fallback for Rust SemanticMemoryConcurrent."""

        def __init__(self):
            self._concepts: dict = {}
            try:
                import networkx as _nx
                self._graph = _nx.DiGraph()
            except ImportError:
                self._graph = None

        def add_concept(self, name: str, hv):
            self._concepts[name] = hv
            if self._graph is not None:
                self._graph.add_node(name)

        # Alias for convenience
        def store_concept(self, name: str, hv):
            self.add_concept(name, hv)

        def get_concept(self, name: str):
            return self._concepts.get(name)

        def add_relation(self, src: str, rel: str, dst: str, weight: float = 1.0):
            if self._graph is not None:
                self._graph.add_edge(src, dst, relation=rel, weight=weight)

        def parallel_semantic_search(self, query_hv, k: int = 5) -> list:
            results = []
            for name, hv in self._concepts.items():
                try:
                    sim = hv.similarity(query_hv)
                    results.append((name, sim))
                except Exception:
                    results.append((name, 0.0))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        # Alias
        def similarity_search(self, query_hv, k: int = 5) -> list:
            return self.parallel_semantic_search(query_hv, k)

        def concept_count(self) -> int:
            return len(self._concepts)

    class EpisodicMemoryConcurrent:
        """Python fallback for Rust EpisodicMemoryConcurrent."""

        def __init__(self, max_hot_size: int = 1000):
            self._episodes: List[Episode] = []
            self._max_size = max_hot_size

        def add_episode(self, episode):
            self._episodes.append(episode)
            if len(self._episodes) > self._max_size:
                self._episodes = self._episodes[-self._max_size:]

        def get_recent_episodes(self, n: int = 10) -> List:
            return self._episodes[-n:]

        # Alias
        def query_recent(self, n: int = 10) -> List:
            return self.get_recent_episodes(n)

        def search_by_task(self, task_tag: str) -> List:
            return [e for e in self._episodes if e.task_tag == task_tag]

        # Alias
        def query_by_task(self, task_tag: str) -> List:
            return self.search_by_task(task_tag)

        def size(self) -> int:
            return len(self._episodes)

        # Alias
        def episode_count(self) -> int:
            return self.size()

    class CognitiveWorkerPool:
        """Python fallback — raises RuntimeError to indicate Rust is required."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("Rust required for CognitiveWorkerPool")

    class HyperVectorRegistry:
        """Python fallback registry."""

        def __init__(self):
            self._registry: dict = {}

        def register(self, name: str, hv):
            self._registry[name] = hv

        def get(self, name: str):
            return self._registry.get(name)

    class PersistentStorage:
        """Python fallback persistent storage (in-memory)."""

        def __init__(self, db_path: str = ":memory:", batch_size: int = 100):
            self._store: dict = {}
            self.db_path = db_path

        def store_hypervector(self, key: str, hv):
            self._store[key] = hv

        def load_hypervector(self, key: str):
            return self._store.get(key)

    def parallel_bundle(hvs):
        """Sequential bundle fallback."""
        if not hvs:
            return None
        result = hvs[0]
        for hv in hvs[1:]:
            result = result.bundle(hv)
        return result

    def batch_parallel_similarity_search(query, hvs: list, k: int = 5) -> list:
        """Sequential k-NN search fallback."""
        results = []
        for i, hv in enumerate(hvs):
            try:
                sim = hv.similarity(query)
            except Exception:
                sim = 0.0
            results.append((i, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def batch_similarity_matrix(hvs: list) -> list:
        """Compute pairwise similarity matrix (fallback)."""
        n = len(hvs)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                try:
                    sim = hvs[i].similarity(hvs[j])
                except Exception:
                    sim = 1.0 if i == j else 0.0
                row.append(sim)
            matrix.append(row)
        return matrix


def get_status() -> dict:
    """Return status of the concurrent memory shim."""
    available = []
    for name in [
        "SemanticMemoryConcurrent", "EpisodicMemoryConcurrent", "Episode",
        "CognitiveWorkerPool", "HyperVectorRegistry", "PersistentStorage",
        "parallel_bundle", "batch_parallel_similarity_search", "batch_similarity_matrix",
    ]:
        available.append(name)
    return {
        "use_rust": USE_RUST_CONCURRENT,
        "available_classes": available,
    }
