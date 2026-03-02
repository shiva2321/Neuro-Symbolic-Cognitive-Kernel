"""
Unit tests for NSCK V5 Emergence: SpectralLaplacianRG, PercolationMonitor, ZipfValidator.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.emergence.spectral_rg import SpectralLaplacianRG
from python.core.societal.emergence.percolation import PercolationMonitor
from python.core.societal.emergence.zipf_validator import ZipfValidator


def _make_lhv(cid: str, seed: int, stability: float = 0.5) -> LivingHyperVector:
    hv = hypervec_rs.HyperVector(seed=seed)
    return LivingHyperVector(cid, hv, {"stability": stability})


def _make_small_world(n: int = 10) -> SocietalKnowledgeWorld:
    world = SocietalKnowledgeWorld()
    for i in range(n):
        world.register_concept(f"c{i}", hypervec_rs.HyperVector(seed=i))
    return world


# ============================================================
# SpectralLaplacianRG
# ============================================================

class TestSpectralLaplacianRG:

    def setup_method(self):
        self.rg = SpectralLaplacianRG(n_eigenvectors=8, similarity_threshold=0.1)

    def test_similarity_graph_shape(self):
        world = _make_small_world(6)
        ids, adj = self.rg.build_similarity_graph(world.concepts)
        assert len(ids) == 6
        assert adj.shape == (6, 6)
        np.testing.assert_array_equal(np.diag(adj), np.zeros(6))

    def test_similarity_graph_single(self):
        world = _make_small_world(1)
        ids, adj = self.rg.build_similarity_graph(world.concepts)
        assert len(ids) == 1
        assert adj.shape == (1, 1)

    def test_laplacian_shape(self):
        adj = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]], dtype=np.float32)
        L = self.rg.compute_laplacian(adj)
        assert L.shape == (3, 3)

    def test_laplacian_diagonal_one(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
        L = self.rg.compute_laplacian(adj)
        np.testing.assert_allclose(np.diag(L), np.ones(3), atol=1e-6)

    def test_eigendecompose(self):
        L = np.eye(4, dtype=np.float32)
        evals, evecs = self.rg.eigendecompose(L)
        assert len(evals) == 4
        assert evecs.shape == (4, 4)
        # Should be sorted ascending
        assert (np.diff(evals) >= -1e-6).all()

    def test_spectral_gap(self):
        evals = np.array([0.0, 0.1, 0.5, 0.9], dtype=np.float32)
        gap = self.rg.spectral_gap(evals)
        assert abs(gap - 0.1) < 1e-6

    def test_spectral_gap_single(self):
        gap = self.rg.spectral_gap(np.array([0.5]))
        assert gap == 0.0

    def test_coarse_grain_returns_groups(self):
        world = _make_small_world(8)
        ids, adj = self.rg.build_similarity_graph(world.concepts)
        L = self.rg.compute_laplacian(adj)
        evals, evecs = self.rg.eigendecompose(L)
        supernodes = self.rg.coarse_grain(ids, evals, evecs)
        assert isinstance(supernodes, dict)
        all_cids = [cid for cids in supernodes.values() for cid in cids]
        assert set(all_cids) == set(ids)

    def test_build_supernode_graph(self):
        world = _make_small_world(6)
        supernodes = {"sn_0": ["c0", "c1"], "sn_1": ["c2", "c3"], "sn_2": ["c4", "c5"]}
        sn_graph = self.rg.build_supernode_graph(world.concepts, supernodes)
        assert len(sn_graph) == 3
        for v in sn_graph.values():
            assert "centroid_concept" in v
            assert "n_concepts" in v

    def test_run_full_pipeline(self):
        world = _make_small_world(12)
        result = self.rg.run(world.concepts)
        assert "supernodes" in result
        assert "spectral_gap" in result
        assert "n_concepts" in result
        assert result["n_concepts"] == 12

    def test_run_empty(self):
        result = self.rg.run({})
        assert result["n_concepts"] == 0
        assert result["n_supernodes"] == 0


# ============================================================
# PercolationMonitor
# ============================================================

class TestPercolationMonitor:

    def setup_method(self):
        self.monitor = PercolationMonitor(bond_threshold=0.2)

    def _make_world_with_bonds(self, n: int, bond_all: bool = False) -> SocietalKnowledgeWorld:
        world = _make_small_world(n)
        if bond_all:
            ids = list(world.concepts.keys())
            for i in range(n - 1):
                world.concepts[ids[i]].add_bond(ids[i + 1], 0.7)
                world.concepts[ids[i + 1]].add_bond(ids[i], 0.7)
        return world

    def test_compute_bond_graph_empty(self):
        world = _make_small_world(5)
        adj = self.monitor.compute_bond_graph(world.concepts)
        assert isinstance(adj, dict)
        assert len(adj) == 5

    def test_compute_bond_graph_with_bonds(self):
        world = self._make_world_with_bonds(4, bond_all=True)
        adj = self.monitor.compute_bond_graph(world.concepts)
        # All concepts linked in a chain → c0→c1, c1→c2, etc.
        n_edges = sum(len(v) for v in adj.values())
        assert n_edges > 0

    def test_giant_component_connected(self):
        world = self._make_world_with_bonds(5, bond_all=True)
        adj = self.monitor.compute_bond_graph(world.concepts)
        gf, giant = self.monitor.detect_giant_component(adj)
        assert abs(gf - 1.0) < 1e-9
        assert len(giant) == 5

    def test_giant_component_isolated(self):
        world = _make_small_world(5)  # no bonds
        adj = self.monitor.compute_bond_graph(world.concepts)
        gf, giant = self.monitor.detect_giant_component(adj)
        assert abs(gf - 0.2) < 1e-9  # each node is its own component

    def test_phase_transition_emergence(self):
        world = self._make_world_with_bonds(10, bond_all=True)
        adj = self.monitor.compute_bond_graph(world.concepts)
        self.monitor._history = [0.1]  # was fragmented
        result = self.monitor.detect_phase_transition(adj, [0.1])
        assert result["is_transitioning"] is True
        assert result["transition_type"] == "emergence"

    def test_phase_transition_stable(self):
        world = _make_small_world(5)
        adj = self.monitor.compute_bond_graph(world.concepts)
        result = self.monitor.detect_phase_transition(adj, [0.2])
        assert isinstance(result["is_transitioning"], bool)

    def test_scan_threshold(self):
        world = self._make_world_with_bonds(6, bond_all=True)
        results = self.monitor.scan_threshold(world.concepts, [0.0, 0.5, 1.0])
        assert len(results) == 3
        assert all("threshold" in r and "giant_fraction" in r for r in results)

    def test_monitor_tick(self):
        world = self._make_world_with_bonds(6, bond_all=True)
        result = self.monitor.monitor_tick(world.concepts)
        assert "giant_fraction" in result
        assert "transition_type" in result
        assert len(self.monitor._history) == 1


# ============================================================
# ZipfValidator
# ============================================================

class TestZipfValidator:

    def setup_method(self):
        self.validator = ZipfValidator(min_concepts=5)

    def _make_world_with_activations(self, n: int, zipf: bool = True) -> SocietalKnowledgeWorld:
        world = _make_small_world(n)
        ids = list(world.concepts.keys())
        for i, cid in enumerate(ids):
            if zipf:
                # Zipf distribution: f(r) = 1/r
                act = 1.0 / (i + 1)
            else:
                act = 0.5  # uniform (not Zipf)
            for _ in range(5):
                world.concepts[cid].update_activation(act)
        return world

    def test_get_activation_counts_sorted(self):
        world = self._make_world_with_activations(10)
        freqs = self.validator.get_activation_counts(world.concepts)
        assert len(freqs) == 10
        # Should be sorted descending
        assert (np.diff(freqs) <= 1e-9).all()

    def test_fit_power_law_zipf(self):
        # Ideal Zipf: f(r) = 1/r → alpha=1
        n = 50
        ranks = np.arange(1, n + 1, dtype=np.float64)
        freqs = (1.0 / ranks).astype(np.float32)
        alpha, _, r2 = self.validator.fit_power_law(freqs)
        assert abs(alpha - 1.0) < 0.1
        assert r2 > 0.99

    def test_fit_power_law_short(self):
        alpha, _, r2 = self.validator.fit_power_law(np.array([1.0]))
        assert alpha == 0.0

    def test_validate_zipf_world(self):
        world = self._make_world_with_activations(20, zipf=True)
        result = self.validator.validate(world.concepts)
        assert "is_zipf_like" in result
        assert "alpha" in result
        assert "health_score" in result
        assert 0.0 <= result["health_score"] <= 1.0

    def test_validate_too_few(self):
        world = _make_small_world(3)
        result = self.validator.validate(world.concepts)
        assert result["is_zipf_like"] is False
        assert result["n_concepts"] == 3

    def test_entropy_uniform(self):
        freqs = np.ones(10, dtype=np.float32)
        entropy = self.validator.compute_entropy(freqs)
        assert abs(entropy - 1.0) < 1e-6  # maximum entropy for uniform

    def test_entropy_concentrated(self):
        freqs = np.zeros(10, dtype=np.float32)
        freqs[0] = 1.0
        entropy = self.validator.compute_entropy(freqs)
        assert entropy < 0.01  # near zero

    def test_entropy_empty(self):
        assert self.validator.compute_entropy(np.array([])) == 0.0
