"""
PercolationMonitor — phase-transition detection for NSCK V5.

Monitors the bond network for percolation events (giant component
emergence/merge/split) that signal domain formation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nsck.percolation")

_TRANSITION_DELTA_THRESHOLD = 0.08   # min Δ(giant_fraction) per step to flag


class PercolationMonitor:
    """Detects percolation / phase-transition events in the concept bond graph.

    Parameters
    ----------
    bond_threshold:
        Minimum bond weight to include as an edge (default 0.30).
    min_cluster_fraction:
        Minimum fraction for a component to be considered "large" (default 0.10).
    phase_history_len:
        How many steps to keep in the giant-fraction history (default 50).
    """

    def __init__(
        self,
        bond_threshold: float = 0.30,
        min_cluster_fraction: float = 0.10,
        phase_history_len: int = 50,
    ) -> None:
        self.bond_threshold = float(bond_threshold)
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.phase_history_len = int(phase_history_len)
        self._history: List[float] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def compute_bond_graph(self, concepts: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build adjacency list from concept bonds.

        Returns
        -------
        Dict mapping concept_id → list of bonded concept_ids
        (only edges with weight > bond_threshold).
        """
        adj: Dict[str, List[str]] = {cid: [] for cid in concepts}
        for cid, lhv in concepts.items():
            for bonded_id, weight in lhv.bonds.items():
                if bonded_id in concepts and weight >= self.bond_threshold:
                    adj[cid].append(bonded_id)
        return adj

    # ------------------------------------------------------------------
    # Giant component
    # ------------------------------------------------------------------

    def detect_giant_component(
        self, adj: Dict[str, List[str]]
    ) -> Tuple[float, List[str]]:
        """Find the largest connected component.

        Returns
        -------
        (giant_fraction, list_of_node_ids_in_giant)
        """
        n = len(adj)
        if n == 0:
            return 0.0, []

        visited: set = set()
        components: List[List[str]] = []

        for start in adj:
            if start in visited:
                continue
            # BFS
            comp: List[str] = []
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in adj.get(node, []):
                    if nb not in visited:
                        queue.append(nb)
            components.append(comp)

        if not components:
            return 0.0, []

        giant = max(components, key=len)
        return float(len(giant) / n), giant

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def detect_phase_transition(
        self,
        adj: Dict[str, List[str]],
        history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Classify the current phase-transition state.

        Returns
        -------
        Dict with: is_transitioning, giant_fraction, order_parameter,
        transition_type (emergence | merge | split | stable), description.
        """
        gf, giant_nodes = self.detect_giant_component(adj)
        n = len(adj)

        # Count components
        n_components = self._count_components(adj)
        order_param = gf  # commonly used order parameter

        hist = history if history is not None else self._history
        transition_type = "stable"
        description = "Stable network — no significant phase change."

        if hist:
            prev_gf = hist[-1]
            delta = gf - prev_gf
            if abs(delta) >= _TRANSITION_DELTA_THRESHOLD:
                if delta > 0:
                    transition_type = "emergence"
                    description = (
                        f"Giant component growing rapidly "
                        f"(Δ={delta:+.3f}): domain emergence detected."
                    )
                else:
                    # Check if we had a large component that split
                    if prev_gf > 0.4 and gf < prev_gf - 0.1:
                        transition_type = "split"
                        description = (
                            f"Giant component shrinking (Δ={delta:+.3f}): "
                            "domain split detected."
                        )
                    else:
                        transition_type = "merge"
                        description = (
                            f"Component structure shifting (Δ={delta:+.3f})."
                        )
            elif gf > 0.6 and n_components <= 2:
                transition_type = "stable"
                description = (
                    f"Large stable giant component ({gf:.1%} of nodes)."
                )
        elif gf > 0.5:
            transition_type = "emergence"
            description = f"Large initial giant component ({gf:.1%})."

        return {
            "is_transitioning": transition_type != "stable",
            "giant_fraction": float(gf),
            "order_parameter": float(order_param),
            "n_components": int(n_components),
            "n_giant": len(giant_nodes),
            "transition_type": transition_type,
            "description": description,
        }

    def _count_components(self, adj: Dict[str, List[str]]) -> int:
        """Count connected components via BFS."""
        visited: set = set()
        count = 0
        for start in adj:
            if start in visited:
                continue
            count += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nb in adj.get(node, []):
                    if nb not in visited:
                        queue.append(nb)
        return count

    # ------------------------------------------------------------------
    # Threshold scan
    # ------------------------------------------------------------------

    def scan_threshold(
        self,
        concepts: Dict[str, Any],
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan bond thresholds and return giant-fraction at each.

        Parameters
        ----------
        concepts:
            Dict of concept_id → LivingHyperVector.
        thresholds:
            List of bond-weight thresholds to evaluate.

        Returns
        -------
        List of dicts: {threshold, giant_fraction, n_components}.
        """
        if thresholds is None:
            thresholds = [i / 10.0 for i in range(0, 11)]

        results = []
        orig = self.bond_threshold
        for thr in thresholds:
            self.bond_threshold = thr
            adj = self.compute_bond_graph(concepts)
            gf, _ = self.detect_giant_component(adj)
            n_comps = self._count_components(adj)
            results.append({
                "threshold": float(thr),
                "giant_fraction": float(gf),
                "n_components": int(n_comps),
            })
        self.bond_threshold = orig
        return results

    # ------------------------------------------------------------------
    # Monitoring tick
    # ------------------------------------------------------------------

    def monitor_tick(self, concepts: Dict[str, Any]) -> Dict[str, Any]:
        """One-step monitoring: build graph, detect transition, update history."""
        adj = self.compute_bond_graph(concepts)
        result = self.detect_phase_transition(adj, self._history)

        # Update history
        self._history.append(result["giant_fraction"])
        if len(self._history) > self.phase_history_len:
            self._history.pop(0)

        return result
