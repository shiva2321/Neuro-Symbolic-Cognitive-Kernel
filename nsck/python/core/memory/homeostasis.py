"""Memory Homeostasis module for NSCK V3 - keeps memory healthy."""
from __future__ import annotations
import time
from typing import List, TYPE_CHECKING
import python.core.vsa.hypervec_shim as hypervec_rs

if TYPE_CHECKING:
    from python.core.memory.semantic_memory import SemanticMemory


class MemoryHomeostasis:
    def __init__(self):
        self.target_edge_density = 0.1
        self.max_contradiction_rate = 0.2
        self.max_staleness_days = 30.0
        self.target_activation_entropy = 2.0

    def regulate(self, memory) -> List[str]:
        """Measure metrics, take corrective actions, return list of actions taken."""
        actions: List[str] = []
        density = self._measure_edge_density(memory)
        if density > self.target_edge_density:
            pruned = self._prune_low_degree_edges(memory)
            if pruned:
                actions.append(f"pruned_edges:{pruned}")
        stale = self._forget_stale_concepts(memory)
        if stale:
            actions.append(f"forgot_stale:{stale}")
        new_categories = self._auto_categorize(memory)
        for cat in new_categories:
            actions.append(f"created_category:{cat}")
        return actions

    def _measure_edge_density(self, memory) -> float:
        g = memory.concept_graph
        n = g.number_of_nodes()
        e = g.number_of_edges()
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)
        return e / max_edges if max_edges > 0 else 0.0

    def _measure_contradiction_rate(self, memory) -> float:
        g = memory.concept_graph
        contradiction_edges = sum(
            1 for _, _, data in g.edges(data=True)
            if data.get("relation") == "contradicts"
        )
        total = g.number_of_edges()
        return contradiction_edges / total if total > 0 else 0.0

    def _prune_low_degree_edges(self, memory, min_degree: int = 1) -> int:
        g = memory.concept_graph
        to_remove = []
        for u, v, data in list(g.edges(data=True)):
            if g.degree(u) <= min_degree and g.degree(v) <= min_degree:
                to_remove.append((u, v))
        for u, v in to_remove:
            g.remove_edge(u, v)
        return len(to_remove)

    def _forget_stale_concepts(self, memory) -> int:
        g = memory.concept_graph
        now = time.time()
        cutoff = now - self.max_staleness_days * 86400
        to_remove = []
        for node, data in list(g.nodes(data=True)):
            last_access = data.get("last_access", data.get("timestamp", now))
            if last_access < cutoff and g.degree(node) == 0:
                to_remove.append(node)
        for node in to_remove:
            g.remove_node(node)
            memory.concept_hvs.pop(node, None)
        return len(to_remove)

    def _auto_categorize(self, memory, threshold: float = 0.7) -> List[str]:
        """Find clusters of similar concepts without shared is_a parent, create prototype."""
        new_categories: List[str] = []
        concepts = list(memory.concept_hvs.keys())
        if len(concepts) < 3:
            return new_categories
        uncategorized = []
        for concept in concepts:
            has_parent = any(
                data.get("relation") == "is_a"
                for _, _, data in memory.concept_graph.out_edges(concept, data=True)
            )
            if not has_parent:
                uncategorized.append(concept)
        if len(uncategorized) < 3:
            return new_categories
        clusters = self._simple_cluster(memory, uncategorized, threshold)
        for cluster in clusters:
            if len(cluster) >= 3:
                cat_name = f"auto_category_{hash(frozenset(cluster)) % 100000}"
                if cat_name not in memory.concept_hvs:
                    proto_hv = memory.concept_hvs[cluster[0]]
                    for c in cluster[1:]:
                        proto_hv = proto_hv.bundle(memory.concept_hvs[c])
                    memory.add_concept(cat_name, {"auto_generated": True}, hv_override=proto_hv)
                    for c in cluster:
                        memory.add_relation(c, "is_a", cat_name)
                    new_categories.append(cat_name)
        return new_categories

    def _simple_cluster(self, memory, concepts: list, threshold: float) -> List[List[str]]:
        """Greedy clustering by HV similarity."""
        used = set()
        clusters = []
        for i, c1 in enumerate(concepts):
            if c1 in used:
                continue
            cluster = [c1]
            used.add(c1)
            hv1 = memory.concept_hvs.get(c1)
            if hv1 is None:
                continue
            for c2 in concepts[i+1:]:
                if c2 in used:
                    continue
                hv2 = memory.concept_hvs.get(c2)
                if hv2 is None:
                    continue
                try:
                    sim = float(hv1.similarity(hv2))
                except Exception:
                    sim = 0.0
                if sim >= threshold:
                    cluster.append(c2)
                    used.add(c2)
            clusters.append(cluster)
        return clusters
