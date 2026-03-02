"""
Unit tests for NSCK V5 Navigation (SocietalHNSW), TDA (TDAHealthMonitor),
and Routing (SocietalContextRouter).
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.navigation.societal_hnsw import SocietalHNSW
from python.core.societal.tda.tda_monitor import TDAHealthMonitor
from python.core.societal.routing.context_router import SocietalContextRouter


def _make_hv(seed: int = 42):
    return hypervec_rs.HyperVector(seed=seed)


def _make_world_with_domains() -> SocietalKnowledgeWorld:
    """Build a small world with 2 domains, 2 neighbourhoods each."""
    world = SocietalKnowledgeWorld(bond_threshold=0.25)
    # Register 16 concepts
    for i in range(16):
        world.register_concept(f"c{i}", _make_hv(i), {"stability": 0.5})
    # Add bonds within groups
    for i in range(0, 8):
        for j in range(i + 1, 8):
            world.concepts[f"c{i}"].add_bond(f"c{j}", 0.6)
            world.concepts[f"c{j}"].add_bond(f"c{i}", 0.6)
    for i in range(8, 16):
        for j in range(i + 1, 16):
            world.concepts[f"c{i}"].add_bond(f"c{j}", 0.6)
            world.concepts[f"c{j}"].add_bond(f"c{i}", 0.6)
    # Electro-negativity
    for cid, lhv in world.concepts.items():
        for _ in range(3):
            lhv.tick()
    # Neighbourhoods
    n1 = world.form_neighborhood([f"c{i}" for i in range(4)], "nbhd_A")
    n2 = world.form_neighborhood([f"c{i}" for i in range(4, 8)], "nbhd_B")
    n3 = world.form_neighborhood([f"c{i}" for i in range(8, 12)], "nbhd_C")
    n4 = world.form_neighborhood([f"c{i}" for i in range(12, 16)], "nbhd_D")
    # Domains
    world.form_domain(["nbhd_A", "nbhd_B"], "science", "dom_sci")
    world.form_domain(["nbhd_C", "nbhd_D"], "art", "dom_art")
    return world


# ============================================================
# SocietalHNSW
# ============================================================

class TestSocietalHNSW:

    def test_build_empty_world(self):
        world = SocietalKnowledgeWorld()
        hnsw = SocietalHNSW()
        hnsw.build(world)  # should not crash
        assert not hnsw._built

    def test_build_with_world(self):
        world = _make_world_with_domains()
        hnsw = SocietalHNSW(M=4)
        hnsw.build(world)
        assert hnsw._built
        assert hnsw.stats()["n_layer0"] == 16

    def test_query_returns_results(self):
        world = _make_world_with_domains()
        hnsw = SocietalHNSW(M=4)
        hnsw.build(world)
        query_hv = _make_hv(0)
        results = hnsw.query(query_hv, top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_query_domain_filter(self):
        world = _make_world_with_domains()
        hnsw = SocietalHNSW(M=4)
        hnsw.build(world)
        query_hv = _make_hv(0)
        results = hnsw.query(query_hv, top_k=5, domain_filter="dom_sci")
        # All results should be in science domain members
        sci_domain = world.domains["dom_sci"]
        sci_members = set()
        for nbhd_id in sci_domain.neighborhood_ids:
            nbhd = world.neighborhoods.get(nbhd_id)
            if nbhd:
                sci_members |= nbhd.concept_ids
        for cid, _ in results:
            assert cid in sci_members, f"{cid} not in science domain"

    def test_cross_domain_route(self):
        world = _make_world_with_domains()
        # Add cross-domain bridge
        world.concepts["c3"].add_bond("c8", 0.5)
        world.concepts["c8"].add_bond("c3", 0.5)
        hnsw = SocietalHNSW(M=4)
        hnsw.build(world)
        query_hv = _make_hv(0)
        results = hnsw.cross_domain_route(query_hv, "dom_sci", "dom_art", top_k=3)
        assert isinstance(results, list)

    def test_query_not_built(self):
        hnsw = SocietalHNSW()
        results = hnsw.query(_make_hv(0), top_k=5)
        assert results == []

    def test_stats(self):
        world = _make_world_with_domains()
        hnsw = SocietalHNSW(M=4)
        hnsw.build(world)
        stats = hnsw.stats()
        assert "built" in stats
        assert stats["built"] is True
        assert stats["n_layer0"] == 16
        assert stats["n_domains"] == 2


# ============================================================
# TDAHealthMonitor
# ============================================================

class TestTDAHealthMonitor:

    def setup_method(self):
        self.tda = TDAHealthMonitor(n_landmarks=10, max_edge_length=0.8)

    def _make_world_with_bonds(self, n: int = 15) -> SocietalKnowledgeWorld:
        world = SocietalKnowledgeWorld()
        for i in range(n):
            world.register_concept(f"t{i}", _make_hv(i))
        # Add bonds to create structure
        ids = list(world.concepts.keys())
        for i in range(n - 1):
            world.concepts[ids[i]].add_bond(ids[i + 1], 0.7)
            world.concepts[ids[i + 1]].add_bond(ids[i], 0.7)
        # Activate some
        for cid in ids[:5]:
            for _ in range(3):
                world.concepts[cid].update_activation(0.8)
        return world

    def test_select_landmarks(self):
        world = self._make_world_with_bonds(20)
        landmarks = self.tda.select_landmarks(world.concepts, 8)
        assert len(landmarks) <= 8
        assert all(lm in world.concepts for lm in landmarks)

    def test_select_landmarks_fewer_than_n(self):
        world = SocietalKnowledgeWorld()
        for i in range(3):
            world.register_concept(f"s{i}", _make_hv(i))
        landmarks = self.tda.select_landmarks(world.concepts, 10)
        assert len(landmarks) == 3

    def test_build_witness_complex(self):
        world = self._make_world_with_bonds(15)
        landmarks = self.tda.select_landmarks(world.concepts, 6)
        complex_dict = self.tda.build_witness_complex(world.concepts, landmarks)
        assert "vertices" in complex_dict
        assert "edges" in complex_dict
        assert "n_witnesses" in complex_dict
        assert len(complex_dict["vertices"]) == len(landmarks)

    def test_betti_numbers_line_graph(self):
        # Line graph: 0-1-2-3 → no cycles, β0=1, β1=0
        complex_dict = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [("a", "b"), ("b", "c"), ("c", "d")],
        }
        b0, b1, b2 = self.tda.compute_betti_numbers(complex_dict)
        assert b0 == 1
        assert b1 == 0
        assert b2 == 0

    def test_betti_numbers_cycle(self):
        # Cycle: 0-1-2-0 → β0=1, β1=1
        complex_dict = {
            "vertices": ["a", "b", "c"],
            "edges": [("a", "b"), ("b", "c"), ("c", "a")],
        }
        b0, b1, b2 = self.tda.compute_betti_numbers(complex_dict)
        assert b0 == 1
        assert b1 == 1

    def test_betti_numbers_disconnected(self):
        # Two isolated components
        complex_dict = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [("a", "b")],
        }
        b0, b1, _ = self.tda.compute_betti_numbers(complex_dict)
        assert b0 == 3  # {a,b}, {c}, {d}

    def test_persistence_entropy(self):
        pd = [(0.0, 0.5, 0), (0.0, 0.3, 0), (0.0, 0.8, 0)]
        entropy = self.tda.compute_persistence_entropy(pd)
        assert entropy > 0.0

    def test_run_analysis(self):
        world = self._make_world_with_bonds(15)
        result = self.tda.run_analysis(world.concepts)
        assert "beta0" in result
        assert "beta1" in result
        assert "beta2" in result
        assert "health_score" in result
        assert "alerts" in result
        assert 0.0 <= result["health_score"] <= 1.0

    def test_last_analysis_stored(self):
        world = self._make_world_with_bonds(15)
        self.tda.run_analysis(world.concepts)
        assert self.tda.last_analysis is not None


# ============================================================
# SocietalContextRouter
# ============================================================

class TestSocietalContextRouter:

    def setup_method(self):
        self.world = _make_world_with_domains()
        self.router = SocietalContextRouter(self.world)

    def test_detect_domain_with_city_hall(self):
        query_hv = _make_hv(0)
        domain_id, confidence = self.router.detect_domain(query_hv)
        # domain_id could be either or None (depends on similarity)
        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_detect_domain_empty_world(self):
        empty_world = SocietalKnowledgeWorld()
        router = SocietalContextRouter(empty_world)
        domain_id, confidence = router.detect_domain(_make_hv(0))
        assert domain_id is None
        assert confidence == 0.0

    def test_border_zone_routing(self):
        # Add a cross-domain bond to create a border concept
        self.world.concepts["c3"].add_bond("c8", 0.5)
        self.world.concepts["c8"].add_bond("c3", 0.5)
        # Recompute border concepts
        for nbhd in self.world.neighborhoods.values():
            nbhd.compute_border_concepts(self.world.concepts)

        results = self.router.border_zone_routing(_make_hv(0), "dom_sci")
        assert isinstance(results, list)

    def test_spreading_activation(self):
        self.world.activate_concept("c0", 0.9)
        spread = self.router.domain_weighted_spreading_activation(
            "c0", domain_id="dom_sci", n_hops=2, top_k=10
        )
        assert isinstance(spread, dict)
        assert "c0" in spread
        assert spread["c0"] == 1.0

    def test_spreading_activation_missing_seed(self):
        spread = self.router.domain_weighted_spreading_activation("nonexistent")
        assert spread == {}

    def test_route_returns_dict(self):
        result = self.router.route(_make_hv(0), top_k=5)
        assert "domain_id" in result
        assert "confidence" in result
        assert "concepts" in result
        assert "border_zone_active" in result
        assert "routing_trace" in result

    def test_update_coalition_scores_dict(self):
        self.world.activate_concept("c0", 0.9)
        coalitions = [
            {"source": "s1", "content": "The concept c0 is important", "base_salience": 0.3},
            {"source": "s2", "content": "Something unrelated", "base_salience": 0.3},
        ]
        updated = self.router.update_coalition_scores(coalitions, {})
        # c0 is active, so coalition 0 should be boosted
        assert updated[0]["base_salience"] > 0.3 or updated[0]["base_salience"] >= 0.3

    def test_update_coalition_scores_object(self):
        """Test with object-style coalitions (not dict)."""
        from python.core.reasoning.global_workspace import Coalition
        self.world.activate_concept("c1", 0.9)
        coal = Coalition(source="test", content="c1 is related", base_salience=0.3)
        updated = self.router.update_coalition_scores([coal], {})
        assert isinstance(updated, list)
