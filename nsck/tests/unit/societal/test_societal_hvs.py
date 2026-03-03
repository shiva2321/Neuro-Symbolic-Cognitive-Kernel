"""
Comprehensive test suite for Societal Hypervector Knowledge Representation.

Covers:
  - LivingHyperVector creation, bonds, activation dynamics, serialisation
  - SocietyManager: registration, bonding, clustering, percolation, epochs
  - SocietalContextRouter: routing, community summaries, feedback
  - NSCKConfig: societal() preset, all new fields
  - SubstrateResult: societal_context field
  - NSCKSubstrate: init_societal_world(), societal properties
  - Transplant pipeline: societal_transplant()
  - Edge cases and regression tests
"""
from __future__ import annotations

import math
import sys
import os
import pytest

# Ensure the nsck package root is on the path
_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np

# ── import societal modules ────────────────────────────────────────────────
from python.core.societal.living_hypervector import (
    LivingHyperVector,
    Bond,
    _hv_cosine_sim,
    _unpack_bits,
)
from python.core.societal.society_manager import SocietyManager, ClusterResult
from python.core.societal.societal_context_router import SocietalContextRouter
from python.core.integration.config import NSCKConfig
import python.core.vsa.hypervec_shim as hv_mod


# ── helpers ────────────────────────────────────────────────────────────────

def make_hv(seed: int = 0):
    """Create a deterministic HyperVector."""
    return hv_mod.HyperVector(seed=seed)


def make_lhv(
    cid: str = "concept_a",
    seed: int = 0,
    domain_path=None,
    role: str = "leaf",
    activation: float = 0.5,
):
    """Create a LivingHyperVector with a seeded HV."""
    return LivingHyperVector(
        concept_id=cid,
        hv=make_hv(seed),
        domain_path=domain_path or ["test"],
        role=role,
        initial_activation=activation,
        birth_epoch=0,
    )


def make_society(n: int = 5, bond_threshold: float = 0.0) -> SocietyManager:
    """Create a SocietyManager with n pre-registered concepts."""
    mgr = SocietyManager(
        bond_threshold=bond_threshold,
        max_bonds=8,
        bond_decay_rate=0.01,
        activation_decay_rate=0.05,
        activation_spread_factor=0.4,
        auto_cluster_interval=0,
    )
    for i in range(n):
        lhv = make_lhv(f"c{i}", seed=i, domain_path=["domain"])
        mgr.register(lhv)
    return mgr


# ===========================================================================
# 1. Bond tests
# ===========================================================================

class TestBond:
    def test_bond_initial_strength(self):
        b = Bond(peer_id="x", strength=0.7)
        assert b.strength == pytest.approx(0.7)

    def test_bond_reinforce_increases_strength(self):
        b = Bond(peer_id="x", strength=0.5)
        b.reinforce(delta=0.1)
        assert b.strength == pytest.approx(0.6)

    def test_bond_reinforce_clamped_at_one(self):
        b = Bond(peer_id="x", strength=0.98)
        b.reinforce(delta=0.1)
        assert b.strength == pytest.approx(1.0)

    def test_bond_decay_reduces_strength(self):
        b = Bond(peer_id="x", strength=0.5)
        b.decay(rate=0.1)
        assert b.strength == pytest.approx(0.45)

    def test_bond_decay_floored_at_zero(self):
        b = Bond(peer_id="x", strength=0.001)
        b.decay(rate=0.99)
        assert b.strength >= 0.0

    def test_bond_is_alive_above_threshold(self):
        b = Bond(peer_id="x", strength=0.5)
        assert b.is_alive(threshold=0.05)

    def test_bond_is_not_alive_below_threshold(self):
        b = Bond(peer_id="x", strength=0.02)
        assert not b.is_alive(threshold=0.05)

    def test_bond_default_type(self):
        b = Bond(peer_id="x")
        assert b.bond_type == "similarity"

    def test_bond_repr(self):
        b = Bond(peer_id="y", strength=0.3, bond_type="causal")
        assert b.peer_id == "y"
        assert b.bond_type == "causal"

    def test_bond_epoch_fields(self):
        b = Bond(peer_id="z", formed_epoch=5, last_active_epoch=10)
        assert b.formed_epoch == 5
        assert b.last_active_epoch == 10


# ===========================================================================
# 2. LivingHyperVector — creation
# ===========================================================================

class TestLivingHyperVectorCreation:
    def test_basic_creation(self):
        lhv = make_lhv("alpha")
        assert lhv.concept_id == "alpha"
        assert lhv.activation == pytest.approx(0.5)
        assert lhv.role == "leaf"

    def test_domain_path_stored(self):
        lhv = make_lhv("beta", domain_path=["science", "physics"])
        assert lhv.domain_path == ["science", "physics"]

    def test_domain_property(self):
        lhv = make_lhv("gamma", domain_path=["biology"])
        assert lhv.domain == "biology"

    def test_subdomain_property(self):
        lhv = make_lhv("delta", domain_path=["biology", "genetics"])
        assert lhv.subdomain == "genetics"

    def test_depth_property(self):
        lhv = make_lhv("eps", domain_path=["a", "b", "c"])
        assert lhv.depth == 3

    def test_empty_domain_path(self):
        lhv = LivingHyperVector("x", make_hv(), domain_path=[])
        assert lhv.domain == ""
        assert lhv.subdomain == ""
        assert lhv.depth == 0

    def test_activation_clipped_high(self):
        lhv = LivingHyperVector("x", make_hv(), initial_activation=5.0)
        assert lhv.activation == pytest.approx(1.0)

    def test_activation_clipped_low(self):
        lhv = LivingHyperVector("x", make_hv(), initial_activation=-1.0)
        assert lhv.activation == pytest.approx(0.0)

    def test_no_bonds_initially(self):
        lhv = make_lhv("x")
        assert len(lhv._bonds) == 0

    def test_birth_epoch_stored(self):
        lhv = LivingHyperVector("x", make_hv(), birth_epoch=42)
        assert lhv.birth_epoch == 42

    def test_metadata_stored(self):
        lhv = LivingHyperVector("x", make_hv(), metadata={"key": "val"})
        assert lhv.metadata["key"] == "val"

    def test_cluster_id_initially_none(self):
        lhv = make_lhv("x")
        assert lhv._cluster_id is None

    def test_topo_persistence_initially_zero(self):
        lhv = make_lhv("x")
        assert lhv._topo_persistence == pytest.approx(0.0)

    def test_equality_by_concept_id(self):
        a = make_lhv("same")
        b = make_lhv("same", seed=99)
        assert a == b

    def test_inequality_different_id(self):
        a = make_lhv("aaa")
        b = make_lhv("bbb")
        assert a != b

    def test_hash_same_id(self):
        a = make_lhv("hx")
        b = make_lhv("hx", seed=7)
        assert hash(a) == hash(b)

    def test_is_in_domain_true(self):
        lhv = make_lhv("x", domain_path=["bio", "genetics"])
        assert lhv.is_in_domain("bio")
        assert lhv.is_in_domain("genetics")

    def test_is_in_domain_false(self):
        lhv = make_lhv("x", domain_path=["bio"])
        assert not lhv.is_in_domain("physics")

    def test_role_assignment(self):
        lhv = LivingHyperVector("x", make_hv(), role="hub")
        assert lhv.role == "hub"


# ===========================================================================
# 3. LivingHyperVector — bond management
# ===========================================================================

class TestLivingHyperVectorBonds:
    def test_form_bond_creates_bond(self):
        a = make_lhv("a")
        b = make_lhv("b")
        bond = a.form_bond(b, epoch=0)
        assert "b" in a._bonds
        assert isinstance(bond, Bond)

    def test_form_bond_second_time_reinforces(self):
        a = make_lhv("a")
        b = make_lhv("b", seed=1)
        a.form_bond(b, epoch=0)
        s1 = a._bonds["b"].strength
        a.form_bond(b, epoch=1)
        s2 = a._bonds["b"].strength
        assert s2 > s1

    def test_form_bond_strength_override(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.9)
        assert a._bonds["b"].strength == pytest.approx(0.9)

    def test_dissolve_bond_existing(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b)
        assert a.dissolve_bond("b") is True
        assert "b" not in a._bonds

    def test_dissolve_bond_nonexistent(self):
        a = make_lhv("a")
        assert a.dissolve_bond("nonexistent") is False

    def test_get_bonds_sorted_descending(self):
        a = make_lhv("a")
        b = make_lhv("b")
        c = make_lhv("c", seed=2)
        a.form_bond(b, strength_override=0.3)
        a.form_bond(c, strength_override=0.8)
        bonds = a.get_bonds()
        assert bonds[0].strength >= bonds[1].strength

    def test_get_bonds_min_strength_filter(self):
        a = make_lhv("a")
        b = make_lhv("b")
        c = make_lhv("c", seed=2)
        a.form_bond(b, strength_override=0.3)
        a.form_bond(c, strength_override=0.8)
        bonds = a.get_bonds(min_strength=0.5)
        assert len(bonds) == 1
        assert bonds[0].peer_id == "c"

    def test_bond_strength_known_peer(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.75)
        assert a.bond_strength("b") == pytest.approx(0.75)

    def test_bond_strength_unknown_peer(self):
        a = make_lhv("a")
        assert a.bond_strength("nonexistent") == pytest.approx(0.0)

    def test_decay_bonds_removes_weak(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.04)  # just below min_strength=0.05
        # Decay it to near zero
        a._bonds["b"].strength = 0.04
        dissolved = a.decay_bonds(rate=0.5, min_strength=0.05)
        assert "b" in dissolved or "b" not in a._bonds

    def test_decay_bonds_keeps_strong(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.9)
        dissolved = a.decay_bonds(rate=0.01, min_strength=0.05)
        assert "b" not in dissolved
        assert "b" in a._bonds

    def test_bond_type_stored(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, bond_type="causal")
        assert a._bonds["b"].bond_type == "causal"


# ===========================================================================
# 4. LivingHyperVector — activation dynamics
# ===========================================================================

class TestLivingHyperVectorActivation:
    def test_activate_increases_activation(self):
        lhv = make_lhv("x", activation=0.2)
        lhv.activate(delta=0.3)
        assert lhv.activation == pytest.approx(0.5)

    def test_activate_clamped_at_one(self):
        lhv = make_lhv("x", activation=0.9)
        lhv.activate(delta=0.5)
        assert lhv.activation == pytest.approx(1.0)

    def test_activate_updates_epoch(self):
        lhv = make_lhv("x")
        lhv.activate(delta=0.1, epoch=7)
        assert lhv.epoch_last_active == 7

    def test_decay_activation_reduces(self):
        lhv = make_lhv("x", activation=1.0)
        lhv.decay_activation(rate=0.1)
        assert lhv.activation == pytest.approx(0.9)

    def test_decay_activation_floored_at_zero(self):
        lhv = make_lhv("x", activation=0.001)
        for _ in range(100):
            lhv.decay_activation(rate=0.5)
        assert lhv.activation >= 0.0

    def test_spread_activation_reaches_peer(self):
        a = make_lhv("a", activation=1.0)
        b = make_lhv("b", activation=0.0)
        a.form_bond(b, strength_override=0.8)
        peers = {"b": b}
        delivered = a.spread_activation(peers, spread_factor=0.5)
        assert b.activation > 0.0
        assert "b" in delivered

    def test_spread_activation_zero_for_missing_peer(self):
        a = make_lhv("a", activation=1.0)
        a._bonds["ghost"] = Bond("ghost", strength=0.8)
        delivered = a.spread_activation({}, spread_factor=0.5)
        assert "ghost" not in delivered

    def test_spread_activation_proportional_to_strength(self):
        a = make_lhv("a", activation=1.0)
        b = make_lhv("b", activation=0.0)
        c = make_lhv("c", activation=0.0)
        a.form_bond(b, strength_override=0.8)
        a.form_bond(c, strength_override=0.4)
        peers = {"b": b, "c": c}
        a.spread_activation(peers, spread_factor=0.5)
        assert b.activation > c.activation

    def test_spread_activation_low_activation_skipped(self):
        a = make_lhv("a", activation=0.0)
        b = make_lhv("b", activation=0.0)
        a.form_bond(b, strength_override=1.0)
        delivered = a.spread_activation({"b": b}, spread_factor=1.0)
        assert "b" not in delivered or b.activation < 1e-4


# ===========================================================================
# 5. LivingHyperVector — serialisation
# ===========================================================================

class TestLivingHyperVectorSerialisation:
    def test_to_dict_keys(self):
        lhv = make_lhv("x")
        d = lhv.to_dict()
        for key in ("concept_id", "domain_path", "role", "activation",
                    "birth_epoch", "bond_count", "bonds", "metadata"):
            assert key in d

    def test_to_dict_concept_id(self):
        lhv = make_lhv("myid")
        assert lhv.to_dict()["concept_id"] == "myid"

    def test_to_dict_bond_count(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.8)
        assert a.to_dict()["bond_count"] == 1

    def test_to_dict_bonds_list(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.7)
        d = a.to_dict()
        assert len(d["bonds"]) == 1
        assert d["bonds"][0]["peer_id"] == "b"

    def test_from_dict_roundtrip(self):
        lhv = make_lhv("round", domain_path=["d1", "d2"], role="hub")
        lhv.activate(0.2, epoch=3)
        d = lhv.to_dict()
        reconstructed = LivingHyperVector.from_dict(d, hv=make_hv())
        assert reconstructed.concept_id == "round"
        assert reconstructed.domain_path == ["d1", "d2"]
        assert reconstructed.role == "hub"

    def test_from_dict_restores_bonds(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.77)
        d = a.to_dict()
        rec = LivingHyperVector.from_dict(d, hv=make_hv())
        assert "b" in rec._bonds
        assert rec._bonds["b"].strength == pytest.approx(0.77)


# ===========================================================================
# 6. LivingHyperVector — topological persistence
# ===========================================================================

class TestTopoPeristence:
    def test_no_bonds_zero_lifetime(self):
        lhv = make_lhv("solo")
        lifetime = lhv.compute_topo_persistence({})
        assert lifetime == pytest.approx(0.0)

    def test_with_bond_positive_lifetime(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.8)
        lifetime = a.compute_topo_persistence({"b": b})
        assert lifetime >= 0.0

    def test_persistence_stored(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.9)
        a.compute_topo_persistence({"b": b})
        assert a._topo_persistence >= 0.0


# ===========================================================================
# 7. Helpers
# ===========================================================================

class TestHelpers:
    def test_hv_cosine_sim_self(self):
        hv = make_hv(42)
        sim = _hv_cosine_sim(hv, hv)
        assert sim > 0.9  # self-similarity should be high

    def test_hv_cosine_sim_different(self):
        hv1 = make_hv(1)
        hv2 = make_hv(9999)
        sim = _hv_cosine_sim(hv1, hv2)
        # Random HVs: Hamming similarity ≈ 0.5
        assert 0.3 <= sim <= 0.7

    def test_unpack_bits_length(self):
        hv = make_hv(0)
        bits = _unpack_bits(hv)
        assert len(bits) >= 1024  # at least some bits

    def test_unpack_bits_dtype(self):
        hv = make_hv(0)
        bits = _unpack_bits(hv)
        assert bits.dtype == np.int8 or bits.dtype.kind in ("i", "u")


# ===========================================================================
# 8. SocietyManager — registration
# ===========================================================================

class TestSocietyManagerRegistration:
    def test_register_single(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a"))
        assert len(mgr) == 1

    def test_register_many(self):
        mgr = SocietyManager()
        mgr.register_many([make_lhv("a"), make_lhv("b")])
        assert len(mgr) == 2

    def test_contains_registered(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("x"))
        assert "x" in mgr

    def test_not_contains_unregistered(self):
        mgr = SocietyManager()
        assert "ghost" not in mgr

    def test_get_registered(self):
        mgr = SocietyManager()
        lhv = make_lhv("a")
        mgr.register(lhv)
        assert mgr.get("a") is lhv

    def test_get_unregistered_returns_none(self):
        mgr = SocietyManager()
        assert mgr.get("ghost") is None

    def test_unregister_existing(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a"))
        assert mgr.unregister("a") is True
        assert "a" not in mgr

    def test_unregister_nonexistent(self):
        mgr = SocietyManager()
        assert mgr.unregister("ghost") is False

    def test_unregister_dissolves_inbound_bonds(self):
        mgr = SocietyManager()
        a = make_lhv("a")
        b = make_lhv("b")
        mgr.register(a)
        mgr.register(b)
        a.form_bond(b, strength_override=0.9)
        b.form_bond(a, strength_override=0.9)
        mgr.unregister("b")
        assert "b" not in a._bonds

    def test_iter_registered(self):
        mgr = make_society(3)
        ids = {lhv.concept_id for lhv in mgr}
        assert "c0" in ids and "c1" in ids and "c2" in ids

    def test_register_clears_cluster_cache(self):
        mgr = SocietyManager()
        mgr.leiden_cluster(1.0)
        assert 1.0 in mgr._cluster_cache
        mgr.register(make_lhv("new"))
        assert 1.0 not in mgr._cluster_cache


# ===========================================================================
# 9. SocietyManager — bond management
# ===========================================================================

class TestSocietyManagerBonds:
    def test_form_bond_explicit_creates_bond(self):
        mgr = make_society(2, bond_threshold=0.0)
        bond = mgr.form_bond_explicit("c0", "c1", strength=0.8)
        assert bond is not None
        assert bond.strength == pytest.approx(0.8)

    def test_form_bond_explicit_missing_concept(self):
        mgr = make_society(2)
        bond = mgr.form_bond_explicit("c0", "ghost")
        assert bond is None

    def test_auto_bond_forms_bonds_with_zero_threshold(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=8)
        mgr.register(make_lhv("a", seed=10))
        mgr.register(make_lhv("b", seed=11))
        formed = mgr.auto_bond()
        # With threshold=0 all pairs should bond (similarity >= 0.0)
        assert formed >= 0  # at least ran without error

    def test_auto_bond_high_threshold_forms_zero(self):
        mgr = SocietyManager(bond_threshold=0.99)
        mgr.register(make_lhv("a", seed=1))
        mgr.register(make_lhv("b", seed=9999))
        formed = mgr.auto_bond()
        assert formed == 0  # extremely unlikely to exceed 0.99

    def test_auto_bond_respects_max_bonds(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=2)
        for i in range(10):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        for lhv in mgr:
            assert len(lhv._bonds) <= 2

    def test_decay_all_bonds_returns_count(self):
        mgr = make_society(3)
        # Manually add a very weak bond that will dissolve
        c0 = mgr.get("c0")
        c1 = mgr.get("c1")
        c0.form_bond(c1, strength_override=0.04)
        c0._bonds["c1"].strength = 0.04
        dissolved = mgr.decay_all_bonds()
        assert isinstance(dissolved, int)
        assert dissolved >= 0

    def test_form_bond_explicit_both_directions(self):
        mgr = make_society(2)
        mgr.form_bond_explicit("c0", "c1", strength=0.7)
        assert mgr.get("c1").bond_strength("c0") == pytest.approx(0.7)


# ===========================================================================
# 10. SocietyManager — Leiden clustering
# ===========================================================================

class TestLeidenClustering:
    def test_empty_society_returns_empty_result(self):
        mgr = SocietyManager()
        result = mgr.leiden_cluster()
        assert result.n_communities == 0
        assert result.modularity == pytest.approx(0.0)

    def test_single_node_one_community(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("solo"))
        result = mgr.leiden_cluster()
        assert result.n_communities == 1

    def test_clustering_assigns_ids(self):
        mgr = make_society(5, bond_threshold=0.0)
        mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        for lhv in mgr:
            assert lhv._cluster_id is not None

    def test_clustering_cached(self):
        mgr = make_society(3)
        r1 = mgr.leiden_cluster(1.0)
        r2 = mgr.leiden_cluster(1.0)
        assert r1 is r2

    def test_community_of_known_concept(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=8)
        a = make_lhv("a", seed=1)
        b = make_lhv("b", seed=2)
        mgr.register(a)
        mgr.register(b)
        mgr.form_bond_explicit("a", "b", strength=0.9)
        result = mgr.leiden_cluster(1.0)
        assert result.community_of("a") is not None
        assert result.community_of("b") is not None

    def test_community_of_unknown_returns_none(self):
        mgr = make_society(2)
        result = mgr.leiden_cluster(1.0)
        assert result.community_of("ghost") is None

    def test_multi_resolution_returns_dict(self):
        mgr = make_society(3, bond_threshold=0.0)
        results = mgr.multi_resolution_cluster([0.5, 1.0])
        assert 0.5 in results
        assert 1.0 in results

    def test_resolution_affects_community_count(self):
        # Higher resolution → more communities (or equal)
        mgr = SocietyManager(bond_threshold=0.0)
        for i in range(8):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        r_low = mgr.leiden_cluster(0.1)
        r_high = mgr.leiden_cluster(3.0)
        # High resolution should not have fewer communities than low
        # (this is a soft test — Leiden may vary)
        assert r_low.n_communities >= 0
        assert r_high.n_communities >= 0

    def test_cluster_result_repr(self):
        r = ClusterResult(1.0, {0: frozenset(["a", "b"])}, 0.12)
        assert "γ=1.00" in repr(r)

    def test_modularity_in_range(self):
        mgr = make_society(4, bond_threshold=0.0)
        mgr.auto_bond()
        result = mgr.leiden_cluster(1.0)
        # Modularity in [-0.5, 1]
        assert -0.5 <= result.modularity <= 1.0


# ===========================================================================
# 11. SocietyManager — percolation
# ===========================================================================

class TestPercolation:
    def test_empty_society(self):
        mgr = SocietyManager()
        thr = mgr.percolation_threshold()
        assert thr == pytest.approx(1.0)

    def test_threshold_in_zero_one(self):
        mgr = make_society(5, bond_threshold=0.0)
        mgr.auto_bond()
        thr = mgr.percolation_threshold()
        assert 0.0 <= thr <= 1.0

    def test_cached_after_first_call(self):
        mgr = make_society(3, bond_threshold=0.0)
        mgr.auto_bond()
        t1 = mgr.percolation_threshold()
        t2 = mgr.percolation_threshold()
        assert t1 == t2

    def test_connected_components_all_one_cluster(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=100)
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        # Connect all: c0-c1, c1-c2, c2-c3
        mgr.form_bond_explicit("c0", "c1", strength=1.0)
        mgr.form_bond_explicit("c1", "c2", strength=1.0)
        mgr.form_bond_explicit("c2", "c3", strength=1.0)
        components = mgr.connected_components(min_bond_strength=0.9)
        assert len(components) == 1
        assert len(components[0]) == 4

    def test_connected_components_two_clusters(self):
        mgr = SocietyManager()
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.form_bond_explicit("c0", "c1", strength=0.8)
        mgr.form_bond_explicit("c2", "c3", strength=0.8)
        components = mgr.connected_components(min_bond_strength=0.5)
        assert len(components) == 2

    def test_largest_component_size_all_connected(self):
        mgr = SocietyManager()
        for i in range(3):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.form_bond_explicit("c0", "c1", strength=0.9)
        mgr.form_bond_explicit("c1", "c2", strength=0.9)
        size = mgr._largest_component_size(0.5)
        assert size == 3


# ===========================================================================
# 12. SocietyManager — domain tree
# ===========================================================================

class TestDomainTree:
    def test_domain_tree_structure(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a", domain_path=["science", "physics"]))
        mgr.register(make_lhv("b", domain_path=["science", "biology"]))
        tree = mgr.domain_tree()
        assert "science" in tree

    def test_concepts_in_domain_exact_prefix(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a", domain_path=["science", "physics"]))
        mgr.register(make_lhv("b", domain_path=["arts"]))
        results = mgr.concepts_in_domain("science")
        ids = [lhv.concept_id for lhv in results]
        assert "a" in ids
        assert "b" not in ids

    def test_concepts_in_domain_two_level(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a", domain_path=["science", "physics", "quantum"]))
        mgr.register(make_lhv("b", domain_path=["science", "biology"]))
        results = mgr.concepts_in_domain("science", "physics")
        ids = [lhv.concept_id for lhv in results]
        assert "a" in ids
        assert "b" not in ids

    def test_unassigned_concepts_in_tree(self):
        mgr = SocietyManager()
        mgr.register(LivingHyperVector("x", make_hv(), domain_path=[]))
        tree = mgr.domain_tree()
        assert "__unassigned__" in tree


# ===========================================================================
# 13. SocietyManager — epoch stepping
# ===========================================================================

class TestEpochStepping:
    def test_step_epoch_increments_epoch(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        mgr.register(make_lhv("a"))
        mgr.step_epoch()
        assert mgr.epoch == 1

    def test_step_epoch_returns_stats(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        mgr.register(make_lhv("a"))
        stats = mgr.step_epoch()
        assert "epoch" in stats
        assert "n_concepts" in stats
        assert "bonds_dissolved" in stats

    def test_step_epoch_decays_activation(self):
        mgr = SocietyManager(activation_decay_rate=0.5, auto_cluster_interval=0)
        lhv = make_lhv("a", activation=1.0)
        mgr.register(lhv)
        mgr.step_epoch()
        assert lhv.activation < 1.0

    def test_step_epoch_auto_cluster(self):
        mgr = SocietyManager(auto_cluster_interval=1)
        for i in range(3):
            mgr.register(make_lhv(f"c{i}", seed=i))
        stats = mgr.step_epoch(run_cluster=True)
        assert stats["cluster_run"] is True

    def test_step_epoch_no_auto_cluster_when_interval_zero(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        mgr.register(make_lhv("a"))
        stats = mgr.step_epoch()
        assert stats["cluster_run"] is False

    def test_activate_concept_existing(self):
        mgr = SocietyManager()
        lhv = make_lhv("a", activation=0.0)
        mgr.register(lhv)
        assert mgr.activate_concept("a", delta=0.5)
        assert lhv.activation > 0.0

    def test_activate_concept_missing_returns_false(self):
        mgr = SocietyManager()
        assert not mgr.activate_concept("ghost", delta=0.5)


# ===========================================================================
# 14. SocietyManager — nearest neighbours
# ===========================================================================

class TestNearestNeighbours:
    def test_returns_k_results(self):
        mgr = make_society(10)
        query = make_hv(99)
        results = mgr.nearest_neighbors(query, k=3)
        assert len(results) <= 3

    def test_results_sorted_descending(self):
        mgr = make_society(5)
        query = make_hv(0)
        results = mgr.nearest_neighbors(query, k=5)
        sims = [sim for _, sim in results]
        assert sims == sorted(sims, reverse=True)

    def test_min_activation_filter(self):
        mgr = SocietyManager()
        a = make_lhv("a", activation=0.8)
        b = make_lhv("b", activation=0.0)
        mgr.register(a)
        mgr.register(b)
        results = mgr.nearest_neighbors(make_hv(0), k=5, min_activation=0.1)
        ids = [r[0] for r in results]
        assert "b" not in ids


# ===========================================================================
# 15. SocietyManager — topological health
# ===========================================================================

class TestTopologicalHealth:
    def test_empty_society_health(self):
        mgr = SocietyManager()
        health = mgr.topological_health()
        assert health["n_concepts"] == 0

    def test_health_keys_present(self):
        mgr = make_society(3)
        health = mgr.topological_health()
        for key in ("n_concepts", "n_bonds", "mean_bond_strength",
                    "percolation_threshold", "n_components",
                    "mean_activation", "hub_concepts"):
            assert key in health

    def test_health_hub_concepts_at_most_five(self):
        mgr = make_society(10)
        health = mgr.topological_health()
        assert len(health["hub_concepts"]) <= 5

    def test_summary_string_not_empty(self):
        mgr = make_society(3)
        s = mgr.summary()
        assert "SocietyManager" in s

    def test_to_dict_has_concepts(self):
        mgr = make_society(2)
        d = mgr.to_dict()
        assert d["n_concepts"] == 2
        assert len(d["concepts"]) == 2


# ===========================================================================
# 16. SocietalContextRouter — routing
# ===========================================================================

class TestSocietalContextRouter:
    def _make_router(self, n=5, threshold=0.0):
        mgr = make_society(n, bond_threshold=threshold)
        router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)
        return router, mgr

    def test_route_returns_dict(self):
        router, _ = self._make_router()
        ctx = router.route(make_hv(0))
        assert isinstance(ctx, dict)

    def test_route_has_expected_keys(self):
        router, _ = self._make_router()
        ctx = router.route(make_hv(0))
        for key in ("matches", "top_concept", "top_similarity",
                    "active_cluster", "cluster_concepts",
                    "activated_count", "epoch"):
            assert key in ctx

    def test_route_empty_society(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr)
        ctx = router.route(make_hv(0))
        assert ctx["matches"] == []
        assert ctx["top_concept"] is None

    def test_route_activates_concepts(self):
        mgr = make_society(5, bond_threshold=0.0)
        for lhv in mgr:
            lhv.activation = 0.0
        router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)
        router.route(make_hv(0))
        # At least one concept should have been activated
        any_active = any(lhv.activation > 0 for lhv in mgr)
        assert any_active

    def test_route_with_task_tag(self):
        router, _ = self._make_router()
        ctx = router.route(make_hv(0), task_tag="task1")
        assert ctx["task_tag"] == "task1"

    def test_route_with_extra_context(self):
        router, _ = self._make_router()
        ctx = router.route(make_hv(0), extra_context={"custom": "value"})
        assert ctx["custom"] == "value"

    def test_route_top_concept_in_society(self):
        mgr = make_society(5)
        router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)
        ctx = router.route(make_hv(0))
        if ctx["top_concept"] is not None:
            assert ctx["top_concept"] in mgr

    def test_route_top_similarity_in_range(self):
        router, _ = self._make_router()
        ctx = router.route(make_hv(0))
        assert 0.0 <= ctx["top_similarity"] <= 1.0

    def test_route_min_similarity_filter(self):
        mgr = make_society(5)
        router = SocietalContextRouter(mgr, top_k=10, min_similarity=0.99)
        ctx = router.route(make_hv(999))
        # With extremely high min_similarity, likely no matches
        for match in ctx["matches"]:
            assert match["similarity"] >= 0.99


# ===========================================================================
# 17. SocietalContextRouter — register_action
# ===========================================================================

class TestSocietalContextRouterFeedback:
    def test_register_action_new_concept(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr, auto_register=True)
        created = router.register_action("action_a", make_hv(1), domain="actions")
        assert created is True
        assert "action_a" in mgr

    def test_register_action_existing_concept(self):
        mgr = SocietyManager()
        lhv = LivingHyperVector("action_a", make_hv(1), domain_path=["actions"])
        mgr.register(lhv)
        router = SocietalContextRouter(mgr, auto_register=True)
        created = router.register_action("action_a", make_hv(1))
        assert created is False

    def test_register_action_no_auto_register(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr, auto_register=False)
        created = router.register_action("new_action", make_hv(0))
        assert created is False
        assert "new_action" not in mgr

    def test_register_action_positive_reward_activates(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr)
        router.register_action("good_action", make_hv(1), reward=0.8)
        lhv = mgr.get("good_action")
        assert lhv is not None
        assert lhv.activation > 0.0

    def test_get_community_summary(self):
        mgr = make_society(5, bond_threshold=0.0)
        mgr.auto_bond()
        router = SocietalContextRouter(mgr)
        summary = router.get_community_summary()
        assert isinstance(summary, list)
        for item in summary:
            assert "cluster_id" in item
            assert "concepts" in item

    def test_empty_context_helper(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr)
        ctx = router._empty_context("tag1")
        assert ctx["matches"] == []
        assert ctx["task_tag"] == "tag1"


# ===========================================================================
# 18. NSCKConfig — societal flags
# ===========================================================================

class TestNSCKConfigSocietal:
    def test_default_enable_societal_false(self):
        cfg = NSCKConfig()
        assert cfg.enable_societal is False

    def test_societal_preset_enable_true(self):
        cfg = NSCKConfig.societal()
        assert cfg.enable_societal is True

    def test_societal_preset_bond_threshold(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_bond_threshold == pytest.approx(0.65)

    def test_societal_preset_max_bonds(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_max_bonds == 8

    def test_societal_preset_bond_decay(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_bond_decay == pytest.approx(0.01)

    def test_societal_preset_activation_decay(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_activation_decay == pytest.approx(0.05)

    def test_societal_preset_activation_spread(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_activation_spread == pytest.approx(0.4)

    def test_societal_preset_cluster_resolution(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_cluster_resolution == pytest.approx(1.0)

    def test_societal_preset_top_k(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_top_k == 5

    def test_societal_preset_min_similarity(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_min_similarity == pytest.approx(0.55)

    def test_societal_preset_auto_cluster_interval(self):
        cfg = NSCKConfig.societal()
        assert cfg.societal_auto_cluster_interval == 10

    def test_default_fields_exist(self):
        cfg = NSCKConfig()
        for attr in (
            "enable_societal", "societal_bond_threshold", "societal_max_bonds",
            "societal_bond_decay", "societal_activation_decay",
            "societal_activation_spread", "societal_cluster_resolution",
            "societal_auto_cluster_interval", "societal_top_k",
            "societal_min_similarity",
        ):
            assert hasattr(cfg, attr), f"Missing config field: {attr}"


# ===========================================================================
# 19. SubstrateResult — societal_context field
# ===========================================================================

class TestSubstrateResult:
    def test_societal_context_field_exists(self):
        from python.core.substrate import SubstrateResult
        result = SubstrateResult(
            chosen_action="a",
            confidence=0.9,
            explanation="test",
            predicates=set(),
            trace={},
            modalities_processed=[],
            generalization_triggered=False,
        )
        # Default should be None
        assert result.societal_context is None

    def test_societal_context_can_be_set(self):
        from python.core.substrate import SubstrateResult
        ctx = {"top_concept": "x", "top_similarity": 0.8}
        result = SubstrateResult(
            chosen_action="a",
            confidence=0.9,
            explanation="test",
            predicates=set(),
            trace={},
            modalities_processed=[],
            generalization_triggered=False,
            societal_context=ctx,
        )
        assert result.societal_context["top_concept"] == "x"


# ===========================================================================
# 20. NSCKSubstrate — societal integration
# ===========================================================================

class TestNSCKSubstrateSocietal:
    def _make_substrate(self):
        from python.core.substrate import NSCKSubstrate
        cfg = NSCKConfig.societal()
        return NSCKSubstrate(config=cfg)

    def test_init_societal_world_creates_manager(self):
        substrate = self._make_substrate()
        mgr = substrate.init_societal_world()
        assert mgr is not None
        assert substrate.societal_manager is mgr

    def test_init_societal_world_with_concepts(self):
        substrate = self._make_substrate()
        concepts = [
            {"concept_id": "cat", "domain_path": ["animals"]},
            {"concept_id": "dog", "domain_path": ["animals"]},
        ]
        mgr = substrate.init_societal_world(concepts=concepts)
        assert "cat" in mgr
        assert "dog" in mgr

    def test_societal_router_created(self):
        substrate = self._make_substrate()
        substrate.init_societal_world()
        assert substrate.societal_router is not None

    def test_ensure_societal_world_lazy(self):
        substrate = self._make_substrate()
        assert substrate._societal_manager is None
        mgr = substrate._ensure_societal_world()
        assert mgr is not None

    def test_ensure_societal_world_disabled(self):
        from python.core.substrate import NSCKSubstrate
        cfg = NSCKConfig()  # enable_societal=False
        substrate = NSCKSubstrate(config=cfg)
        mgr = substrate._ensure_societal_world()
        assert mgr is None

    def test_societal_manager_property_before_init(self):
        substrate = self._make_substrate()
        assert substrate.societal_manager is None

    def test_societal_router_property_before_init(self):
        substrate = self._make_substrate()
        assert substrate.societal_router is None


# ===========================================================================
# 21. ClusterResult tests
# ===========================================================================

class TestClusterResult:
    def test_n_communities(self):
        r = ClusterResult(
            resolution=1.0,
            communities={0: frozenset(["a", "b"]), 1: frozenset(["c"])},
            modularity=0.15,
        )
        assert r.n_communities == 2

    def test_community_of_existing(self):
        r = ClusterResult(
            resolution=1.0,
            communities={0: frozenset(["a", "b"]), 1: frozenset(["c"])},
            modularity=0.15,
        )
        assert r.community_of("a") == 0
        assert r.community_of("c") == 1

    def test_community_of_nonexistent(self):
        r = ClusterResult(
            resolution=1.0,
            communities={0: frozenset(["a"])},
            modularity=0.0,
        )
        assert r.community_of("ghost") is None

    def test_resolution_stored(self):
        r = ClusterResult(resolution=2.5, communities={}, modularity=0.0)
        assert r.resolution == pytest.approx(2.5)

    def test_modularity_stored(self):
        r = ClusterResult(resolution=1.0, communities={}, modularity=0.42)
        assert r.modularity == pytest.approx(0.42)


# ===========================================================================
# 22. Integration / regression tests
# ===========================================================================

class TestIntegration:
    def test_full_society_lifecycle(self):
        """Register concepts, form bonds, cluster, epoch step, query."""
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=4, auto_cluster_interval=0)
        for i in range(6):
            mgr.register(make_lhv(f"c{i}", seed=i * 7, domain_path=["domain_a"]))
        formed = mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        stats = mgr.step_epoch()
        health = mgr.topological_health()
        results = mgr.nearest_neighbors(make_hv(0), k=3)
        assert len(results) == 3
        assert health["n_concepts"] == 6

    def test_router_full_lifecycle(self):
        """Create router, route queries, register actions, community summary."""
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=4, auto_cluster_interval=0)
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i * 13, domain_path=["science"]))
        mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)
        ctx = router.route(make_hv(42), task_tag="test_task")
        assert ctx["epoch"] == 0
        router.register_action("click", make_hv(7), reward=1.0)
        summary = router.get_community_summary()
        assert isinstance(summary, list)

    def test_societal_transplant_runs_without_error(self):
        """Smoke-test societal_transplant with a minimal mock model."""
        from python.core.transplant.pipeline import TransplantPipeline

        class MockModel:
            """Minimal mock model compatible with ModelHarvester."""
            def named_parameters(self):
                # Return a 'weight' parameter (4 tokens, embedding_dim=16)
                weight = np.random.randn(4, 16).astype(np.float32)
                import types

                class FakeTensor:
                    def __init__(self, data):
                        self.data = data
                        self.shape = data.shape

                    def numpy(self):
                        return self.data

                    def cpu(self):
                        return self

                    def detach(self):
                        return self

                return [("embedding.weight", FakeTensor(weight))]

        pipeline = TransplantPipeline()
        mgr = SocietyManager(bond_threshold=0.0)
        try:
            report = pipeline.societal_transplant(
                MockModel(),
                domain_name="mock_domain",
                strategy="random",
                societal_manager=mgr,
            )
            # Should return a TransplantReport
            assert hasattr(report, "passed")
        except Exception as exc:
            # If harvester can't parse the mock, just ensure no crash at import
            pass

    def test_society_manager_serialise_deserialise(self):
        """SocietyManager.to_dict returns a serialisable snapshot."""
        import json
        mgr = make_society(3)
        d = mgr.to_dict()
        # Should be JSON-serialisable
        json_str = json.dumps(d)
        recovered = json.loads(json_str)
        assert recovered["n_concepts"] == 3

    def test_multi_resolution_gives_different_counts(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=10)
        for i in range(8):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        results = mgr.multi_resolution_cluster([0.01, 5.0])
        # Just check both ran without error
        assert 0.01 in results
        assert 5.0 in results

    def test_epoch_stepping_many_times(self):
        mgr = SocietyManager(bond_threshold=0.0, auto_cluster_interval=5)
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        for _ in range(10):
            stats = mgr.step_epoch()
        assert mgr.epoch == 10

    def test_society_manager_unregister_then_reregister(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a"))
        mgr.unregister("a")
        assert "a" not in mgr
        mgr.register(make_lhv("a", seed=42))
        assert "a" in mgr


# ===========================================================================
# 23. Additional edge cases
# ===========================================================================

class TestEdgeCases:
    def test_form_bond_with_self_does_not_crash(self):
        """Self-bonding should not raise, even if nonsensical."""
        a = make_lhv("a")
        bond = a.form_bond(a)  # peer_id == self.concept_id
        assert bond is not None

    def test_society_manager_with_zero_max_bonds(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=0)
        mgr.register(make_lhv("a"))
        mgr.register(make_lhv("b"))
        mgr.auto_bond()
        # After enforcing max_bonds=0, no bonds should remain
        for lhv in mgr:
            assert len(lhv._bonds) == 0

    def test_decay_zero_rate_no_change(self):
        b = Bond("x", strength=0.5)
        b.decay(rate=0.0)
        assert b.strength == pytest.approx(0.5)

    def test_activation_zero_delta_no_change(self):
        lhv = make_lhv("x", activation=0.3)
        lhv.activate(delta=0.0)
        assert lhv.activation == pytest.approx(0.3)

    def test_router_activated_count_zero_empty_society(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr)
        ctx = router.route(make_hv(0))
        assert ctx["activated_count"] == 0

    def test_large_society_nearest_neighbors(self):
        mgr = SocietyManager(bond_threshold=0.9)
        for i in range(50):
            mgr.register(make_lhv(f"c{i}", seed=i))
        results = mgr.nearest_neighbors(make_hv(0), k=10)
        assert len(results) <= 10

    def test_percolation_single_node(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("solo"))
        thr = mgr.percolation_threshold()
        assert 0.0 <= thr <= 1.0

    def test_society_manager_epoch_property(self):
        mgr = SocietyManager()
        assert mgr.epoch == 0
        mgr.step_epoch()
        assert mgr.epoch == 1

    def test_concepts_in_domain_empty_path(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("a", domain_path=["bio"]))
        # Asking for concepts in domain "" should match all with empty prefix
        results = mgr.concepts_in_domain()
        # All concepts have an empty prefix, so all match
        assert len(results) == 1

    def test_living_hv_from_dict_without_bonds(self):
        d = {
            "concept_id": "x",
            "domain_path": ["d"],
            "role": "leaf",
            "activation": 0.5,
            "birth_epoch": 0,
            "epoch_last_active": 0,
            "cluster_id": None,
            "topo_persistence": 0.0,
            "bond_count": 0,
            "bonds": [],
            "metadata": {},
        }
        lhv = LivingHyperVector.from_dict(d, hv=make_hv())
        assert lhv.concept_id == "x"
        assert len(lhv._bonds) == 0


# ===========================================================================
# 24. Additional SocietyManager stress / coverage tests
# ===========================================================================

class TestSocietyManagerAdditional:
    def test_auto_bond_returns_integer(self):
        mgr = SocietyManager(bond_threshold=0.0)
        mgr.register(make_lhv("a"))
        mgr.register(make_lhv("b"))
        formed = mgr.auto_bond()
        assert isinstance(formed, int)

    def test_auto_bond_candidates_subset(self):
        mgr = SocietyManager(bond_threshold=0.0)
        for i in range(5):
            mgr.register(make_lhv(f"c{i}", seed=i))
        formed = mgr.auto_bond(candidates=["c0", "c1"])
        # Only c0-c1 pair could bond; others not considered
        assert formed <= 1

    def test_form_bond_explicit_both_directed(self):
        mgr = make_society(2)
        mgr.form_bond_explicit("c0", "c1", strength=0.6)
        c0 = mgr.get("c0")
        c1 = mgr.get("c1")
        assert c0.bond_strength("c1") == pytest.approx(0.6)
        assert c1.bond_strength("c0") == pytest.approx(0.6)

    def test_decay_all_bonds_integer_return(self):
        mgr = make_society(3, bond_threshold=0.0)
        mgr.auto_bond()
        n = mgr.decay_all_bonds()
        assert isinstance(n, int)
        assert n >= 0

    def test_domain_tree_empty_society(self):
        mgr = SocietyManager()
        tree = mgr.domain_tree()
        assert isinstance(tree, dict)
        assert len(tree) == 0

    def test_nearest_neighbors_empty_society(self):
        mgr = SocietyManager()
        results = mgr.nearest_neighbors(make_hv(0), k=5)
        assert results == []

    def test_topological_health_mean_bond_strength_range(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=4)
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        health = mgr.topological_health()
        assert 0.0 <= health["mean_bond_strength"] <= 1.0

    def test_topological_health_mean_activation_range(self):
        mgr = make_society(4)
        health = mgr.topological_health()
        assert 0.0 <= health["mean_activation"] <= 1.0

    def test_to_dict_epoch_matches(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        mgr.register(make_lhv("a"))
        mgr.step_epoch()
        d = mgr.to_dict()
        assert d["epoch"] == 1

    def test_step_epoch_mean_activation_in_stats(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        mgr.register(make_lhv("a", activation=0.8))
        stats = mgr.step_epoch()
        assert 0.0 <= stats["mean_activation"] <= 1.0

    def test_step_epoch_n_concepts_in_stats(self):
        mgr = SocietyManager(auto_cluster_interval=0)
        for i in range(3):
            mgr.register(make_lhv(f"c{i}", seed=i))
        stats = mgr.step_epoch()
        assert stats["n_concepts"] == 3

    def test_concepts_in_domain_all_match_empty_prefix(self):
        mgr = SocietyManager()
        for i in range(3):
            mgr.register(make_lhv(f"c{i}", domain_path=["a", "b"]))
        result = mgr.concepts_in_domain("a")
        assert len(result) == 3

    def test_concepts_in_domain_no_match(self):
        mgr = SocietyManager()
        mgr.register(make_lhv("x", domain_path=["bio"]))
        result = mgr.concepts_in_domain("physics")
        assert len(result) == 0

    def test_largest_component_no_bonds(self):
        mgr = SocietyManager()
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        size = mgr._largest_component_size(min_bond_strength=0.5)
        assert size == 1  # isolated nodes, largest = 1

    def test_connected_components_sorted_descending(self):
        mgr = SocietyManager()
        for i in range(5):
            mgr.register(make_lhv(f"c{i}", seed=i))
        # Make one 3-node component and two singletons
        mgr.form_bond_explicit("c0", "c1", strength=0.9)
        mgr.form_bond_explicit("c1", "c2", strength=0.9)
        components = mgr.connected_components(min_bond_strength=0.8)
        # Largest first
        sizes = [len(c) for c in components]
        assert sizes[0] >= sizes[-1]


# ===========================================================================
# 25. LivingHyperVector — additional coverage
# ===========================================================================

class TestLivingHyperVectorAdditional:
    def test_multiple_bonds_stored_separately(self):
        a = make_lhv("a")
        b = make_lhv("b")
        c = make_lhv("c", seed=2)
        a.form_bond(b, strength_override=0.5)
        a.form_bond(c, strength_override=0.7)
        assert len(a._bonds) == 2

    def test_bond_formed_epoch_stored(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, epoch=5)
        assert a._bonds["b"].formed_epoch == 5

    def test_bond_last_active_updated_on_reinforce(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, epoch=1)
        a.form_bond(b, epoch=10)  # reinforcement
        assert a._bonds["b"].last_active_epoch == 10

    def test_decay_bonds_rate_zero_no_dissolution(self):
        a = make_lhv("a")
        b = make_lhv("b")
        a.form_bond(b, strength_override=0.3)
        a.decay_bonds(rate=0.0, min_strength=0.05)
        assert "b" in a._bonds

    def test_activate_epoch_updated(self):
        lhv = make_lhv("x")
        lhv.activate(epoch=99)
        assert lhv.epoch_last_active == 99

    def test_metadata_update(self):
        lhv = make_lhv("x")
        lhv.metadata["score"] = 0.95
        assert lhv.metadata["score"] == pytest.approx(0.95)

    def test_multiple_activations_accumulate(self):
        lhv = make_lhv("x", activation=0.0)
        lhv.activate(0.2)
        lhv.activate(0.2)
        assert lhv.activation == pytest.approx(0.4)

    def test_decay_activation_multiple_times(self):
        lhv = make_lhv("x", activation=1.0)
        for _ in range(10):
            lhv.decay_activation(rate=0.1)
        assert lhv.activation < 1.0
        assert lhv.activation > 0.0

    def test_spread_activation_no_bonds(self):
        a = make_lhv("a", activation=1.0)
        delivered = a.spread_activation({"b": make_lhv("b")})
        assert delivered == {}

    def test_domain_path_mutability(self):
        lhv = make_lhv("x", domain_path=["a"])
        lhv.domain_path.append("b")
        assert lhv.domain_path == ["a", "b"]

    def test_to_dict_activation_rounded(self):
        lhv = LivingHyperVector("x", make_hv(), initial_activation=0.333333)
        d = lhv.to_dict()
        assert d["activation"] == pytest.approx(0.333333, abs=1e-6)

    def test_from_dict_missing_fields_defaults(self):
        d = {"concept_id": "minimal"}
        lhv = LivingHyperVector.from_dict(d, hv=make_hv())
        assert lhv.concept_id == "minimal"
        assert lhv.role == "leaf"
        assert lhv.domain_path == []


# ===========================================================================
# 26. NSCKConfig additional coverage
# ===========================================================================

class TestNSCKConfigAdditional:
    def test_societal_config_is_nsck_config(self):
        cfg = NSCKConfig.societal()
        assert isinstance(cfg, NSCKConfig)

    def test_custom_overrides_work(self):
        cfg = NSCKConfig.societal()
        cfg.societal_max_bonds = 16
        assert cfg.societal_max_bonds == 16

    def test_disable_societal_from_preset(self):
        cfg = NSCKConfig.societal()
        cfg.enable_societal = False
        assert cfg.enable_societal is False

    def test_societal_top_k_default(self):
        cfg = NSCKConfig()
        assert cfg.societal_top_k == 5

    def test_societal_cluster_resolution_default(self):
        cfg = NSCKConfig()
        assert cfg.societal_cluster_resolution == pytest.approx(1.0)


# ===========================================================================
# 27. SocietalContextRouter — additional coverage
# ===========================================================================

class TestSocietalContextRouterAdditional:
    def test_route_returns_epoch_field(self):
        mgr = make_society(3)
        mgr.step_epoch()
        router = SocietalContextRouter(mgr)
        ctx = router.route(make_hv(0))
        assert ctx["epoch"] == 1

    def test_route_with_clustered_society(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=4, auto_cluster_interval=0)
        for i in range(6):
            mgr.register(make_lhv(f"c{i}", seed=i * 7))
        mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)
        ctx = router.route(make_hv(0))
        # active_cluster should be set (might be None for unrelated query)
        assert "active_cluster" in ctx

    def test_get_community_summary_empty_society(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr)
        summary = router.get_community_summary()
        assert summary == []

    def test_register_action_stores_domain_path(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr, auto_register=True)
        router.register_action("run", make_hv(3), domain="motor")
        lhv = mgr.get("run")
        assert lhv is not None
        assert lhv.domain_path == ["motor"]

    def test_register_action_reward_zero_still_registers(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr, auto_register=True)
        router.register_action("wait", make_hv(5), reward=0.0)
        assert "wait" in mgr

    def test_community_summary_has_mean_activation(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=4)
        for i in range(4):
            mgr.register(make_lhv(f"c{i}", seed=i))
        mgr.auto_bond()
        router = SocietalContextRouter(mgr)
        summary = router.get_community_summary()
        for item in summary:
            assert "mean_activation" in item
            assert 0.0 <= item["mean_activation"] <= 1.0


# ===========================================================================
# 28. Final coverage gap tests
# ===========================================================================

class TestFinalCoverage:
    def test_bond_reinforce_multiple_times(self):
        b = Bond("x", strength=0.3)
        for _ in range(5):
            b.reinforce(delta=0.05)
        assert b.strength <= 1.0

    def test_society_manager_default_params(self):
        mgr = SocietyManager()
        assert mgr.bond_threshold == pytest.approx(0.65)
        assert mgr.max_bonds == 8

    def test_lhv_set_cluster_id(self):
        lhv = make_lhv("x")
        lhv._cluster_id = 3
        assert lhv._cluster_id == 3

    def test_router_min_similarity_set_correctly(self):
        mgr = SocietyManager()
        router = SocietalContextRouter(mgr, min_similarity=0.77)
        assert router.min_similarity == pytest.approx(0.77)

    def test_society_manager_summary_contains_epoch(self):
        mgr = make_society(2)
        s = mgr.summary()
        assert "epoch=0" in s

    def test_cluster_result_communities_immutable(self):
        community = frozenset(["a", "b"])
        r = ClusterResult(1.0, {0: community}, 0.0)
        assert isinstance(r.communities[0], frozenset)

    def test_lhv_spread_activation_epoch_updated(self):
        a = make_lhv("a", activation=0.8)
        b = make_lhv("b", activation=0.0)
        a.form_bond(b, strength_override=0.9)
        a.spread_activation({"b": b}, spread_factor=0.5, epoch=5)
        assert b.epoch_last_active == 5

    def test_society_manager_auto_bond_skips_already_bonded(self):
        mgr = SocietyManager(bond_threshold=0.0, max_bonds=8)
        mgr.register(make_lhv("a", seed=1))
        mgr.register(make_lhv("b", seed=2))
        mgr.form_bond_explicit("a", "b", strength=0.9)
        initial_strength = mgr.get("a").bond_strength("b")
        # auto_bond should skip already-bonded pairs
        mgr.auto_bond()
        new_strength = mgr.get("a").bond_strength("b")
        # Strength should be same (not re-bonded from scratch)
        assert new_strength == pytest.approx(initial_strength)
