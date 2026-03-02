"""
Unit tests for NSCK V5 Societal Core.
Tests: LivingHyperVector, ValenceEngine, KnowledgeNeighborhood,
       KnowledgeDomain, SocietalKnowledgeWorld
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.valence_engine import ValenceEngine
from python.core.societal.knowledge_neighborhood import (
    KnowledgeNeighborhood,
    KnowledgeDomain,
)
from python.core.societal.societal_world import SocietalKnowledgeWorld


def _make_hv(seed: int = 42):
    return hypervec_rs.HyperVector(seed=seed)


# ============================================================
# LivingHyperVector
# ============================================================

class TestLivingHyperVector:

    def test_creation_defaults(self):
        hv = _make_hv(42)
        lhv = LivingHyperVector("cat", hv)
        assert lhv.concept_id == "cat"
        assert lhv.age == 0
        assert 0.0 <= lhv.stability <= 1.0
        assert lhv.valence == 0.0
        assert lhv.activation == 0.0
        assert lhv.hybridization_state == "free"
        assert lhv.stability_class in ("volatile", "active", "stable", "crystallized")

    def test_creation_with_metadata(self):
        hv = _make_hv(1)
        lhv = LivingHyperVector("dog", hv, {"stability": 0.9, "valence": 0.5})
        assert abs(lhv.stability - 0.9) < 1e-9
        assert abs(lhv.valence - 0.5) < 1e-9
        assert lhv.stability_class == "crystallized"

    def test_tick_increments_age(self):
        hv = _make_hv(2)
        lhv = LivingHyperVector("bird", hv)
        lhv.update_activation(0.8)
        act_before = lhv.activation
        lhv.tick()
        assert lhv.age == 1
        assert lhv.activation < act_before

    def test_activation_history(self):
        hv = _make_hv(3)
        lhv = LivingHyperVector("fish", hv)
        for v in [0.1, 0.5, 0.9]:
            lhv.update_activation(v)
        assert len(lhv.activation_history) == 3
        assert abs(lhv.activation - 0.9) < 1e-9

    def test_stability_class_volatile(self):
        hv = _make_hv(4)
        lhv = LivingHyperVector("x", hv, {"stability": 0.1})
        assert lhv.stability_class == "volatile"

    def test_stability_class_active(self):
        hv = _make_hv(5)
        lhv = LivingHyperVector("x", hv, {"stability": 0.45})
        assert lhv.stability_class == "active"

    def test_stability_class_stable(self):
        hv = _make_hv(6)
        lhv = LivingHyperVector("x", hv, {"stability": 0.72})
        assert lhv.stability_class == "stable"

    def test_stability_class_crystallized(self):
        hv = _make_hv(7)
        lhv = LivingHyperVector("x", hv, {"stability": 0.95})
        assert lhv.stability_class == "crystallized"

    def test_add_remove_bond(self):
        hv = _make_hv(8)
        lhv = LivingHyperVector("a", hv)
        lhv.add_bond("b", 0.7)
        assert "b" in lhv.bonds
        assert abs(lhv.bonds["b"] - 0.7) < 1e-9
        lhv.remove_bond("b")
        assert "b" not in lhv.bonds

    def test_domain_affinity(self):
        hv = _make_hv(9)
        lhv = LivingHyperVector("a", hv)
        assert lhv.get_affinity("science") == 0.0
        lhv.set_affinity("science", 0.8)
        assert abs(lhv.get_affinity("science") - 0.8) < 1e-9
        assert lhv.primary_domain == "science"

    def test_hybridize_returns_new_instance(self):
        hv_a = _make_hv(10)
        hv_ctx = _make_hv(11)
        lhv = LivingHyperVector("a", hv_a)
        hybrid = lhv.hybridize(hv_ctx)
        assert hybrid is not lhv
        assert hybrid.hybridization_state == "hybridized"
        assert hybrid.concept_id == "a"

    def test_to_from_dict(self):
        hv = _make_hv(12)
        lhv = LivingHyperVector("z", hv, {"stability": 0.7, "valence": -0.3})
        lhv.add_bond("y", 0.5)
        d = lhv.to_dict()
        assert d["concept_id"] == "z"
        assert "bonds" in d
        reconstructed = LivingHyperVector.from_dict(d, hv)
        assert reconstructed.concept_id == "z"
        assert abs(reconstructed.stability - 0.7) < 1e-9

    def test_cosine_similarity_to_self(self):
        hv = _make_hv(13)
        lhv = LivingHyperVector("self", hv)
        sim = lhv.cosine_similarity_to(lhv)
        assert sim > 0.99


# ============================================================
# ValenceEngine
# ============================================================

class TestValenceEngine:

    def setup_method(self):
        self.engine = ValenceEngine(bond_threshold=0.3, break_threshold=0.1)

    def _make_lhv(self, cid, seed):
        hv = _make_hv(seed)
        return LivingHyperVector(cid, hv)

    def test_compute_bond_strength_range(self):
        a = self._make_lhv("a", 42)
        b = self._make_lhv("b", 43)
        s = self.engine.compute_bond_strength(a, b)
        assert 0.0 <= s <= 1.0

    def test_compute_bond_strength_identical_hvs(self):
        hv = _make_hv(42)
        a = LivingHyperVector("a", hv)
        b = LivingHyperVector("b", hv)
        s = self.engine.compute_bond_strength(a, b)
        assert s > 0.5  # should be high for identical HVs

    def test_try_form_bond_same_hv(self):
        hv = _make_hv(42)
        a = LivingHyperVector("a", hv)
        b = LivingHyperVector("b", hv)
        strength = self.engine.try_form_bond(a, b)
        assert strength is not None
        assert "b" in a.bonds
        assert "a" in b.bonds

    def test_try_form_bond_orthogonal_hv(self):
        # Two random seeds produce dissimilar vectors
        a = self._make_lhv("a", 0)
        b = self._make_lhv("b", 12345)
        # May or may not form depending on similarity; just check no crash
        _ = self.engine.try_form_bond(a, b)

    def test_try_break_bond(self):
        hv = _make_hv(42)
        a = LivingHyperVector("a", hv)
        b = LivingHyperVector("b", hv)
        a.add_bond("b", 0.05)  # below break threshold
        b.add_bond("a", 0.05)
        result = self.engine.try_break_bond(a, b)
        assert result is True
        assert "b" not in a.bonds

    def test_bulk_initialize(self):
        concepts = [self._make_lhv(f"c{i}", i) for i in range(5)]
        sim_mat = np.full((5, 5), 0.9, dtype=np.float32)
        np.fill_diagonal(sim_mat, 0.0)
        bonds_formed = self.engine.bulk_initialize(concepts, sim_mat)
        assert bonds_formed > 0

    def test_context_conditioned_retrieval(self):
        concepts = [self._make_lhv(f"c{i}", i) for i in range(10)]
        query_hv = _make_hv(0)
        results = self.engine.context_conditioned_retrieval(query_hv, concepts, top_k=5)
        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


# ============================================================
# KnowledgeNeighborhood
# ============================================================

class TestKnowledgeNeighborhood:

    def test_creation(self):
        nbhd = KnowledgeNeighborhood("nbhd_test")
        assert nbhd.neighborhood_id == "nbhd_test"
        assert len(nbhd.concept_ids) == 0

    def test_add_concept(self):
        nbhd = KnowledgeNeighborhood()
        hv = _make_hv(1)
        lhv = LivingHyperVector("cat", hv)
        nbhd.add_concept("cat", lhv)
        assert "cat" in nbhd.concept_ids
        assert lhv.neighborhood_id == nbhd.neighborhood_id

    def test_remove_concept(self):
        nbhd = KnowledgeNeighborhood()
        nbhd.add_concept("cat")
        nbhd.remove_concept("cat")
        assert "cat" not in nbhd.concept_ids

    def test_elect_anchor(self):
        nbhd = KnowledgeNeighborhood()
        for i, cid in enumerate(["a", "b", "c"]):
            hv = _make_hv(i)
            lhv = LivingHyperVector(cid, hv, {"stability": float(i) / 3.0})
            lhv.tick(); lhv.tick()  # build up electronegativity
            nbhd.add_concept(cid, lhv)

        concepts = {
            "a": LivingHyperVector("a", _make_hv(0), {"stability": 0.1}),
            "b": LivingHyperVector("b", _make_hv(1), {"stability": 0.5}),
            "c": LivingHyperVector("c", _make_hv(2), {"stability": 0.9}),
        }
        for cid in concepts:
            concepts[cid].tick(); concepts[cid].tick()
        anchor = nbhd.elect_anchor(concepts)
        assert anchor in ("a", "b", "c")

    def test_merge(self):
        n1 = KnowledgeNeighborhood()
        n1.add_concept("a"); n1.add_concept("b")
        n2 = KnowledgeNeighborhood()
        n2.add_concept("c"); n2.add_concept("d")
        merged = n1.merge_with(n2)
        assert len(merged.concept_ids) == 4

    def test_stats(self):
        nbhd = KnowledgeNeighborhood("nbhd_stats")
        nbhd.add_concept("a")
        stats = nbhd.stats()
        assert stats["n_concepts"] == 1
        assert stats["neighborhood_id"] == "nbhd_stats"


# ============================================================
# KnowledgeDomain
# ============================================================

class TestKnowledgeDomain:

    def test_creation(self):
        domain = KnowledgeDomain("science", "dom_sci")
        assert domain.domain_id == "dom_sci"
        assert domain.name == "science"

    def test_add_neighborhood(self):
        domain = KnowledgeDomain("science")
        nbhd = KnowledgeNeighborhood("nbhd_1")
        domain.add_neighborhood(nbhd)
        assert "nbhd_1" in domain.neighborhood_ids
        assert nbhd.domain_id == domain.domain_id

    def test_update_city_hall(self):
        domain = KnowledgeDomain("science")
        nbhd = KnowledgeNeighborhood("nbhd_1")
        hv = _make_hv(42)
        lhv = LivingHyperVector("atom", hv)
        lhv.electronegativity = 0.8
        nbhd.add_concept("atom", lhv)
        nbhd.anchor_id = "atom"
        domain.add_neighborhood(nbhd)
        concepts = {"atom": lhv}
        nbhds = {"nbhd_1": nbhd}
        domain.update_city_hall(nbhds, concepts)
        assert domain.city_hall_hv is not None

    def test_stats(self):
        domain = KnowledgeDomain("technology")
        stats = domain.stats()
        assert "domain_id" in stats
        assert stats["n_neighborhoods"] == 0


# ============================================================
# SocietalKnowledgeWorld
# ============================================================

class TestSocietalKnowledgeWorld:

    def setup_method(self):
        self.world = SocietalKnowledgeWorld(bond_threshold=0.3)

    def test_register_concept(self):
        hv = _make_hv(1)
        lhv = self.world.register_concept("cat", hv)
        assert "cat" in self.world.concepts
        assert lhv.concept_id == "cat"

    def test_register_duplicate(self):
        hv1 = _make_hv(1)
        hv2 = _make_hv(2)
        self.world.register_concept("cat", hv1)
        lhv = self.world.register_concept("cat", hv2)
        assert lhv.hv is hv2

    def test_activate_concept(self):
        hv = _make_hv(1)
        self.world.register_concept("cat", hv)
        lhv = self.world.activate_concept("cat", strength=0.9)
        assert lhv is not None
        assert abs(lhv.activation - 0.9) < 1e-6

    def test_activate_missing(self):
        result = self.world.activate_concept("nonexistent")
        assert result is None

    def test_query_returns_results(self):
        for i in range(5):
            self.world.register_concept(f"c{i}", _make_hv(i))
        results = self.world.query(_make_hv(0), top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)

    def test_tick_increments(self):
        for i in range(3):
            self.world.register_concept(f"c{i}", _make_hv(i))
        assert self.world.tick_count == 0
        self.world.run_societal_tick()
        assert self.world.tick_count == 1
        for lhv in self.world.concepts.values():
            assert lhv.age == 1

    def test_form_neighborhood(self):
        for i in range(4):
            self.world.register_concept(f"n{i}", _make_hv(i))
        nbhd = self.world.form_neighborhood(["n0", "n1", "n2"], "my_nbhd")
        assert "my_nbhd" in self.world.neighborhoods
        assert len(nbhd.concept_ids) == 3

    def test_form_domain(self):
        for i in range(6):
            self.world.register_concept(f"d{i}", _make_hv(i))
        n1 = self.world.form_neighborhood(["d0", "d1", "d2"])
        n2 = self.world.form_neighborhood(["d3", "d4", "d5"])
        domain = self.world.form_domain([n1.neighborhood_id, n2.neighborhood_id], "physics")
        assert domain.domain_id in self.world.domains
        assert len(domain.neighborhood_ids) == 2

    def test_get_concept_domain(self):
        hv = _make_hv(1)
        lhv = self.world.register_concept("cat", hv, {"primary_domain": "biology"})
        result = self.world.get_concept_domain("cat")
        assert result == "biology"

    def test_stats_report_structure(self):
        for i in range(3):
            self.world.register_concept(f"x{i}", _make_hv(i))
        stats = self.world.stats_report()
        assert "n_concepts" in stats
        assert "n_neighborhoods" in stats
        assert "n_domains" in stats
        assert "tick_count" in stats
        assert stats["n_concepts"] == 3

    def test_multiple_ticks(self):
        for i in range(5):
            self.world.register_concept(f"t{i}", _make_hv(i))
        for _ in range(20):
            self.world.run_societal_tick()
        assert self.world.tick_count == 20
        for lhv in self.world.concepts.values():
            assert lhv.age == 20
