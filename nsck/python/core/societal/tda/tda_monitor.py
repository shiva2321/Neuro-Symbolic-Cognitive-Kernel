"""
TDAHealthMonitor — approximate persistent homology via witness complexes for NSCK V5.

Computes topological Betti numbers (β₀, β₁, β₂) and persistence diagnostics
to monitor the topological health of the concept space.

Background analysis can be scheduled via a daemon thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from python.core.societal.living_hypervector import _hv_cosine_sim

logger = logging.getLogger("nsck.tda_monitor")


# ---------------------------------------------------------------------------
# Union-Find for Betti-0 / persistence
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ---------------------------------------------------------------------------
# TDAHealthMonitor
# ---------------------------------------------------------------------------

class TDAHealthMonitor:
    """Approximate topological health monitor using witness complexes.

    Parameters
    ----------
    n_landmarks:
        Number of landmark concepts to sample (default 50).
    max_edge_length:
        Maximum cosine-distance for edges in the complex (default 0.7).
        Note: distance = (1 - cosine_sim) / 2.
    alert_threshold:
        Health score below which alerts are triggered (default 0.3).
    background_interval:
        Seconds between background analysis runs (default 60.0).
    """

    def __init__(
        self,
        n_landmarks: int = 50,
        max_edge_length: float = 0.70,
        alert_threshold: float = 0.30,
        background_interval: float = 60.0,
    ) -> None:
        self.n_landmarks = int(n_landmarks)
        self.max_edge_length = float(max_edge_length)
        self.alert_threshold = float(alert_threshold)
        self.background_interval = float(background_interval)
        self.last_analysis: Optional[Dict[str, Any]] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Landmark selection
    # ------------------------------------------------------------------

    def select_landmarks(
        self, concepts: Dict[str, Any], n: int
    ) -> List[str]:
        """Maxmin landmark sampling.

        Start from a random concept; each subsequent landmark is the concept
        farthest (lowest similarity) from the current landmark set.

        Parameters
        ----------
        concepts:
            Dict concept_id → LivingHyperVector.
        n:
            Desired number of landmarks (capped at len(concepts)).

        Returns
        -------
        List of landmark concept_ids.
        """
        ids = list(concepts.keys())
        k = min(n, len(ids))
        if k == 0:
            return []
        if k == 1:
            return [ids[0]]

        # Precompute bit arrays for fast similarity
        try:
            bits = np.stack(
                [np.asarray(concepts[cid].hv.bits, dtype=np.float32) for cid in ids]
            )
            bipolar = bits * 2.0 - 1.0
            norms = np.linalg.norm(bipolar, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized = bipolar / norms  # (N, D)
            have_matrix = True
        except Exception:
            have_matrix = False
            normalized = None

        rng = np.random.default_rng(42)
        first_idx = int(rng.integers(0, len(ids)))
        landmarks = [first_idx]
        # min distance to landmark set for each point
        if have_matrix:
            min_dists = np.ones(len(ids), dtype=np.float32)
            for _ in range(k - 1):
                last = landmarks[-1]
                sims = (normalized @ normalized[last]).astype(np.float32)
                # dist = (1 - sim) / 2
                dists = (1.0 - sims) / 2.0
                min_dists = np.minimum(min_dists, dists)
                next_lm = int(np.argmax(min_dists))
                landmarks.append(next_lm)
        else:
            # Fallback: random selection
            landmarks = list(rng.choice(len(ids), size=k, replace=False))

        return [ids[i] for i in landmarks]

    # ------------------------------------------------------------------
    # Witness complex
    # ------------------------------------------------------------------

    def build_witness_complex(
        self, concepts: Dict[str, Any], landmarks: List[str]
    ) -> Dict[str, Any]:
        """Build a witness complex from landmarks.

        For each non-landmark concept (witness), find its two nearest landmarks.
        Add an edge (l₁, l₂) if they share a witness with max_edge_length tolerance.

        Returns
        -------
        Dict with keys: vertices (landmark_ids), edges (list of 2-tuples).
        """
        lm_set = set(landmarks)
        witnesses = [cid for cid in concepts if cid not in lm_set]

        # Precompute landmark vectors
        lm_vecs: List[np.ndarray] = []
        for lm in landmarks:
            try:
                bits = np.asarray(concepts[lm].hv.bits, dtype=np.float32)
                bip = bits * 2.0 - 1.0
                norm = float(np.linalg.norm(bip))
                lm_vecs.append(bip / norm if norm > 0 else bip)
            except Exception:
                lm_vecs.append(np.zeros(1, dtype=np.float32))

        edges: Set[Tuple[int, int]] = set()

        for w_id in witnesses:
            try:
                w_bits = np.asarray(concepts[w_id].hv.bits, dtype=np.float32)
                w_bip = w_bits * 2.0 - 1.0
                w_norm = float(np.linalg.norm(w_bip))
                w_vec = w_bip / w_norm if w_norm > 0 else w_bip
            except Exception:
                continue

            # Similarities to all landmarks
            sims = [float(np.dot(w_vec, lv)) if len(lv) > 1 else 0.0
                    for lv in lm_vecs]
            dists = [(1.0 - s) / 2.0 for s in sims]

            # Nearest two landmarks
            sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i])
            if len(sorted_idx) < 2:
                continue
            i1, i2 = sorted_idx[0], sorted_idx[1]
            # Add edge if witness is "close enough" to both
            if dists[i1] <= self.max_edge_length and dists[i2] <= self.max_edge_length:
                e = (min(i1, i2), max(i1, i2))
                edges.add(e)

        return {
            "vertices": list(landmarks),
            "edges": [(landmarks[i], landmarks[j]) for i, j in edges],
            "n_witnesses": len(witnesses),
        }

    # ------------------------------------------------------------------
    # Betti numbers
    # ------------------------------------------------------------------

    def compute_betti_numbers(
        self, complex_dict: Dict[str, Any]
    ) -> Tuple[int, int, int]:
        """Approximate Betti numbers β₀, β₁, β₂.

        β₀ = number of connected components (union-find on 1-skeleton).
        β₁ ≈ number of independent cycles  (V - E + β₀ via Euler characteristic).
        β₂ = 0 (no 3-simplices in witness complex).
        """
        vertices = complex_dict.get("vertices", [])
        edges = complex_dict.get("edges", [])
        n_v = len(vertices)
        n_e = len(edges)

        if n_v == 0:
            return 0, 0, 0

        # β₀: connected components
        v_idx = {v: i for i, v in enumerate(vertices)}
        uf = _UnionFind(n_v)
        for va, vb in edges:
            ia = v_idx.get(va, -1)
            ib = v_idx.get(vb, -1)
            if ia >= 0 and ib >= 0:
                uf.union(ia, ib)

        b0 = len({uf.find(i) for i in range(n_v)})
        # β₁: from Euler: χ = V - E = β₀ - β₁  → β₁ = β₀ - (V - E)
        b1 = max(0, b0 - (n_v - n_e))
        b2 = 0

        return b0, b1, b2

    # ------------------------------------------------------------------
    # Persistence diagram
    # ------------------------------------------------------------------

    def compute_persistence_diagram(
        self, concepts: Dict[str, Any]
    ) -> List[Tuple[float, float, int]]:
        """Approximate persistence by adding edges in order of weight.

        Returns list of (birth, death, dimension=0) tuples.
        """
        ids = list(concepts.keys())
        n = len(ids)
        if n < 2:
            return []

        # Collect all bond pairs with weights
        pairs: List[Tuple[float, int, int]] = []
        id_idx = {cid: i for i, cid in enumerate(ids)}
        for i, cid in enumerate(ids):
            for bonded_id, w in concepts[cid].bonds.items():
                j = id_idx.get(bonded_id, -1)
                if j > i:
                    # Use distance = 1 - w
                    pairs.append((1.0 - w, i, j))

        pairs.sort()

        uf = _UnionFind(n)
        diagram = []
        # Each component starts at birth=0
        for dist, i, j in pairs:
            if uf.union(i, j):
                # A component merges: record (birth=0, death=dist)
                diagram.append((0.0, float(dist), 0))

        return diagram

    # ------------------------------------------------------------------
    # Persistence entropy
    # ------------------------------------------------------------------

    def compute_persistence_entropy(
        self, persistence_diagram: List[Tuple[float, float, int]]
    ) -> float:
        """Entropy of lifetimes in the persistence diagram."""
        lifetimes = np.array([d - b for b, d, _ in persistence_diagram], dtype=np.float64)
        lifetimes = lifetimes[lifetimes > 0]
        if len(lifetimes) == 0:
            return 0.0
        total = lifetimes.sum()
        p = lifetimes / total
        return float(-np.sum(p * np.log(p + 1e-12)))

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_analysis(self, concepts: Dict[str, Any]) -> Dict[str, Any]:
        """Full topological health analysis.

        Returns
        -------
        Dict with beta0, beta1, beta2, persistence_entropy,
        n_landmarks, n_witnesses, n_edges, alerts, health_score.
        """
        n = len(concepts)
        n_lm = min(self.n_landmarks, n)
        landmarks = self.select_landmarks(concepts, n_lm)

        complex_dict = self.build_witness_complex(concepts, landmarks)
        b0, b1, b2 = self.compute_betti_numbers(complex_dict)
        pd = self.compute_persistence_diagram(concepts)
        pe = self.compute_persistence_entropy(pd)

        # Health score
        # Ideal: connected (b0=1) + has structure (b1 > 0)
        connectivity_score = 1.0 / b0 if b0 > 0 else 0.0
        structure_score = min(1.0, b1 / max(1, n_lm // 5))
        entropy_score = min(1.0, pe / 3.0)  # normalise roughly
        health_score = 0.50 * connectivity_score + 0.30 * structure_score + 0.20 * entropy_score

        result: Dict[str, Any] = {
            "beta0": int(b0),
            "beta1": int(b1),
            "beta2": int(b2),
            "persistence_entropy": float(pe),
            "n_landmarks": len(landmarks),
            "n_witnesses": int(complex_dict.get("n_witnesses", 0)),
            "n_edges": len(complex_dict.get("edges", [])),
            "n_persistence_pairs": len(pd),
            "health_score": float(health_score),
            "alerts": [],
        }

        result["alerts"] = self.check_alerts(result)
        self.last_analysis = result
        return result

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def check_alerts(self, analysis: Dict[str, Any]) -> List[str]:
        """Return alert strings for unhealthy topology."""
        alerts = []
        b0 = analysis.get("beta0", 0)
        b1 = analysis.get("beta1", 0)
        pe = analysis.get("persistence_entropy", 0.0)
        hs = analysis.get("health_score", 1.0)

        if b0 > 5:
            alerts.append(f"HIGH_FRAGMENTATION: β₀={b0} (concept space is fragmented into {b0} components)")
        if b1 == 0 and analysis.get("n_landmarks", 0) > 5:
            alerts.append("NO_CYCLES: β₁=0 (no cyclic structure — world may be too tree-like)")
        if pe < 0.1 and analysis.get("n_persistence_pairs", 0) > 0:
            alerts.append(f"LOW_PERSISTENCE_ENTROPY: {pe:.3f} (degenerate topology)")
        if hs < self.alert_threshold:
            alerts.append(f"LOW_HEALTH_SCORE: {hs:.3f} < threshold {self.alert_threshold}")

        return alerts

    # ------------------------------------------------------------------
    # Background scheduling
    # ------------------------------------------------------------------

    def schedule_background(
        self, world: Any, interval: Optional[float] = None
    ) -> None:
        """Start a daemon thread running run_analysis periodically."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return

        self._stop_event.clear()
        interval_ = interval if interval is not None else self.background_interval

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.run_analysis(world.concepts)
                except Exception as exc:
                    logger.warning("TDA background analysis failed: %s", exc)
                self._stop_event.wait(interval_)

        self._bg_thread = threading.Thread(target=_loop, daemon=True, name="tda-monitor")
        self._bg_thread.start()
        logger.info("TDAHealthMonitor background thread started (interval=%.1fs)", interval_)

    def stop_background(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5.0)
            self._bg_thread = None
