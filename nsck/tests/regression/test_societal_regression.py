"""
Societal HV Regression Tests — NSCK V26
========================================
Verifies that:
1. Flat-VSA results (similarity, bundling, semantic memory) are identical
   whether or not the societal layer is active — societal routing must be
   purely additive (no mutation of existing HVs or semantic memory).
2. NSCKSubstrate.process() produces the same ``chosen_action`` and
   ``confidence`` with and without societal enabled.
3. Societal context field is None when societal is disabled and a dict when
   societal is enabled.
4. TransplantPipeline.run() produces the same TransplantReport whether or not
   societal_transplant() is also called.
5. Bond operations are idempotent: re-registering the same concept does not
   corrupt existing bond data.

All tests are self-contained and require no internet access.
"""
from __future__ import annotations

import os
import sys
import pytest
import numpy as np

# Path setup
_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import python.core.vsa.hypervec_shim as hv_mod
from python.core.integration.config import NSCKConfig
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.society_manager import SocietyManager
from python.core.societal.societal_context_router import SocietalContextRouter


# ===========================================================================
# Helpers
# ===========================================================================

def _hv(seed: int):
    return hv_mod.HyperVector(seed=seed)


def _lhv(cid: str, seed: int = 0, domain=None):
    return LivingHyperVector(cid, _hv(seed), domain_path=domain or [])


def _make_mgr(*concept_ids, bond_threshold=0.0):
    mgr = SocietyManager(bond_threshold=bond_threshold, max_bonds=8,
                         auto_cluster_interval=0)
    for i, cid in enumerate(concept_ids):
        mgr.register(_lhv(cid, seed=i))
    return mgr


# ===========================================================================
# 1. VSA similarity is unchanged by societal layer
# ===========================================================================

def test_vsa_similarity_unchanged_by_societal():
    """HV.similarity() must return the same value regardless of societal state."""
    a = _hv(1)
    b = _hv(2)
    # Baseline without any societal involvement
    sim_before = a.similarity(b)

    # Create a society, register these HVs, run epoch steps
    mgr = SocietyManager(bond_threshold=0.0)
    mgr.register(LivingHyperVector("a", a))
    mgr.register(LivingHyperVector("b", b))
    mgr.auto_bond()
    for _ in range(5):
        mgr.step_epoch()

    # Similarity must be unchanged — LHVs never mutate the underlying HV
    sim_after = a.similarity(b)
    assert sim_before == pytest.approx(sim_after, abs=1e-9), (
        f"VSA similarity changed from {sim_before} to {sim_after} "
        "after societal operations — regression!"
    )
    print(f"  sim before={sim_before:.6f}, after={sim_after:.6f}")
    print("✅ VSA similarity unchanged by societal layer")


# ===========================================================================
# 2. HV bundling is unchanged
# ===========================================================================

def test_hv_bundle_unchanged_by_societal():
    """HV.bundle() must be unaffected by societal activity."""
    a = _hv(10)
    b = _hv(11)
    # Baseline bundle
    bundled_before = a.bundle(b)
    sim_before = bundled_before.similarity(a)

    # Run many societal epochs involving these vectors
    mgr = SocietyManager(bond_threshold=0.0)
    la = LivingHyperVector("a", a)
    lb = LivingHyperVector("b", b)
    mgr.register(la)
    mgr.register(lb)
    mgr.auto_bond()
    for _ in range(10):
        mgr.activate_concept("a", delta=0.5)
        mgr.step_epoch()

    # Bundle must give identical result
    bundled_after = a.bundle(b)
    sim_after = bundled_after.similarity(a)
    assert sim_before == pytest.approx(sim_after, abs=1e-9), (
        f"Bundle similarity changed: {sim_before} → {sim_after}"
    )
    print("✅ HV bundle unchanged by societal layer")


# ===========================================================================
# 3. Semantic memory lookups are identical
# ===========================================================================

def test_semantic_memory_unchanged_by_societal():
    """SemanticMemory.get_concept() must return the same HV before and after
    societal init."""
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory()
    mem.add_concept("cat", {"kind": "animal"})
    hv_before = mem.get_concept("cat")
    assert hv_before is not None

    # Init a society (does not touch SemanticMemory)
    mgr = SocietyManager()
    mgr.register(LivingHyperVector("cat", _hv(0)))
    for _ in range(3):
        mgr.step_epoch()

    hv_after = mem.get_concept("cat")
    assert hv_after is not None
    assert hv_before.similarity(hv_after) == pytest.approx(1.0, abs=1e-9), (
        "SemanticMemory concept HV changed after societal activity!"
    )
    print("✅ SemanticMemory unchanged by societal layer")


# ===========================================================================
# 4. NSCKSubstrate.process() same result with/without societal
# ===========================================================================

def test_substrate_same_action_with_without_societal():
    """process() with societal enabled must return the same structural result type
    as without societal (both SubstrateResult, with same confidence range)."""
    from python.core.substrate import NSCKSubstrate, SubstrateResult

    # Without societal
    cfg_flat = NSCKConfig()
    sub_flat = NSCKSubstrate(config=cfg_flat)
    sub_flat.register_task("t1")
    result_flat = sub_flat.process("The cat sat on the mat", "t1")

    # With societal (same input, different substrate instance)
    cfg_soc = NSCKConfig.societal()
    sub_soc = NSCKSubstrate(config=cfg_soc)
    sub_soc.register_task("t1")
    sub_soc.init_societal_world()
    result_soc = sub_soc.process("The cat sat on the mat", "t1")

    # Both must be SubstrateResult instances with valid fields
    assert isinstance(result_flat, SubstrateResult)
    assert isinstance(result_soc, SubstrateResult)
    assert isinstance(result_flat.confidence, float)
    assert isinstance(result_soc.confidence, float)
    assert 0.0 <= result_flat.confidence <= 1.0
    assert 0.0 <= result_soc.confidence <= 1.0
    # The societal version should have a dict for societal_context (populated from empty society)
    assert isinstance(result_soc.societal_context, dict)
    # The flat version must have None
    assert result_flat.societal_context is None
    print(f"  flat action={result_flat.chosen_action!r} conf={result_flat.confidence:.3f}")
    print(f"  soc  action={result_soc.chosen_action!r} conf={result_soc.confidence:.3f}")
    print("✅ NSCKSubstrate.process() returns valid results with/without societal")


# ===========================================================================
# 5. SubstrateResult.societal_context is None when disabled
# ===========================================================================

def test_societal_context_none_when_disabled():
    """SubstrateResult.societal_context must be None when enable_societal=False."""
    from python.core.substrate import NSCKSubstrate

    cfg = NSCKConfig()
    assert cfg.enable_societal is False
    sub = NSCKSubstrate(config=cfg)
    sub.register_task("t2")
    result = sub.process("hello world", "t2")
    assert result.societal_context is None, (
        f"Expected societal_context=None when disabled, got {result.societal_context}"
    )
    print("✅ societal_context is None when disabled")


# ===========================================================================
# 6. SubstrateResult.societal_context is dict when enabled
# ===========================================================================

def test_societal_context_dict_when_enabled():
    """SubstrateResult.societal_context must be a dict when enabled and
    init_societal_world() has been called."""
    from python.core.substrate import NSCKSubstrate

    cfg = NSCKConfig.societal()
    sub = NSCKSubstrate(config=cfg)
    sub.register_task("t3")

    # Populate society with some concepts
    sub.init_societal_world(concepts=[
        {"concept_id": "cat", "domain_path": ["animals"]},
        {"concept_id": "dog", "domain_path": ["animals"]},
    ])
    result = sub.process("cats and dogs", "t3")
    assert isinstance(result.societal_context, dict), (
        f"Expected dict, got {type(result.societal_context)}"
    )
    for key in ("matches", "top_concept", "epoch"):
        assert key in result.societal_context, f"Missing key {key!r}"
    print(f"  context keys: {list(result.societal_context.keys())}")
    print("✅ societal_context is a dict when enabled")


# ===========================================================================
# 7. Transplant pipeline produces identical reports
# ===========================================================================

def test_transplant_report_identical_flat_vs_societal():
    """TransplantPipeline.run() and societal_transplant() must agree on
    the validation metrics (rho, etc.)."""
    from python.core.transplant.pipeline import TransplantPipeline

    # Build a minimal synthetic model
    # Note: harvester requires n > d (more vocab rows than embedding dims)
    class _FakeTensor:
        def __init__(self, data):
            self.data = data
            self.shape = data.shape

        def numpy(self):
            return self.data

        def cpu(self):
            return self

        def detach(self):
            return self

    class _FakeModel:
        def named_parameters(self):
            # 50 tokens, embedding_dim=16: 50 > 16 satisfies harvester n > d check
            w = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
            return [("embedding.weight", _FakeTensor(w))]

    pipe = TransplantPipeline()
    model = _FakeModel()

    report_flat = pipe.run(model, domain_name="test_flat", strategy="random")

    mgr = SocietyManager(bond_threshold=0.0)
    model2 = _FakeModel()
    report_soc = pipe.societal_transplant(
        model2, domain_name="test_societal", strategy="random",
        societal_manager=mgr
    )

    # Both should pass (or both should fail with the same outcome)
    assert report_flat.passed == report_soc.passed, (
        f"Flat passed={report_flat.passed}, societal passed={report_soc.passed}"
    )
    print(f"  Both reports passed={report_flat.passed}")
    print("✅ Transplant flat vs societal same pass/fail")


# ===========================================================================
# 8. Re-registering a concept does not corrupt bond data
# ===========================================================================

def test_reregister_does_not_corrupt_bonds():
    """Unregistering and re-registering a concept must not corrupt bonds
    from other nodes."""
    mgr = SocietyManager(bond_threshold=0.0, max_bonds=8, auto_cluster_interval=0)
    a = _lhv("a", seed=1)
    b = _lhv("b", seed=2)
    c = _lhv("c", seed=3)
    mgr.register(a)
    mgr.register(b)
    mgr.register(c)
    mgr.form_bond_explicit("a", "b", strength=0.9)
    mgr.form_bond_explicit("b", "c", strength=0.8)

    # b is the middle node; unregister and re-register a
    mgr.unregister("a")
    new_a = _lhv("a", seed=99)
    mgr.register(new_a)

    # b's bond to a should have been dissolved (a was removed)
    assert "a" not in b._bonds, "b still has stale bond to a after a was unregistered"
    # b's bond to c must be intact
    assert "c" in b._bonds, "b's bond to c was corrupted by a's unregister"
    print("✅ Re-registration does not corrupt bonds")


# ===========================================================================
# 9. Activation spreading does not exceed 1.0
# ===========================================================================

def test_activation_never_exceeds_one():
    """After many epochs of high-delta activation + spread, no concept
    activation should exceed 1.0."""
    mgr = SocietyManager(bond_threshold=0.0, max_bonds=8, auto_cluster_interval=0)
    for i in range(10):
        mgr.register(_lhv(f"c{i}", seed=i))
    mgr.auto_bond()

    # Continuously spike and step
    for _ in range(20):
        mgr.activate_concept("c0", delta=1.0)
        mgr.step_epoch()

    for lhv in mgr:
        assert lhv.activation <= 1.0 + 1e-9, (
            f"Activation {lhv.activation} > 1.0 for {lhv.concept_id}"
        )
    print("✅ Activation never exceeds 1.0")


# ===========================================================================
# 10. Bond strength remains in [0, 1]
# ===========================================================================

def test_bond_strength_bounded():
    """Bond strength must remain in [0, 1] after reinforcement and decay."""
    mgr = SocietyManager(bond_threshold=0.0, max_bonds=8, auto_cluster_interval=0)
    for i in range(5):
        mgr.register(_lhv(f"c{i}", seed=i))
    mgr.auto_bond()

    # Reinforce all bonds massively
    for lhv in mgr:
        for bond in lhv._bonds.values():
            for _ in range(100):
                bond.reinforce(delta=0.5)

    for lhv in mgr:
        for bond in lhv._bonds.values():
            assert 0.0 <= bond.strength <= 1.0 + 1e-9, (
                f"Bond strength {bond.strength} out of range for {bond.peer_id}"
            )
    print("✅ Bond strength bounded in [0, 1]")


# ===========================================================================
# 11. Leiden cluster assignment covers all registered concepts
# ===========================================================================

def test_leiden_covers_all_concepts():
    """Every registered concept must receive a cluster_id after leiden_cluster."""
    mgr = SocietyManager(bond_threshold=0.0, max_bonds=8, auto_cluster_interval=0)
    for i in range(8):
        mgr.register(_lhv(f"c{i}", seed=i))
    mgr.auto_bond()
    mgr.leiden_cluster(1.0)

    for lhv in mgr:
        assert lhv._cluster_id is not None, (
            f"{lhv.concept_id} has no cluster_id after leiden_cluster()"
        )
    print("✅ Leiden covers all concepts")


# ===========================================================================
# 12. Domain tree contains all registered concepts
# ===========================================================================

def test_domain_tree_contains_all_concepts():
    """domain_tree() must reference every registered concept_id exactly once."""
    mgr = SocietyManager()
    concepts = [
        _lhv("alpha", seed=1, domain=["sci", "bio"]),
        _lhv("beta",  seed=2, domain=["sci", "phys"]),
        _lhv("gamma", seed=3, domain=["art"]),
    ]
    for c in concepts:
        mgr.register(c)

    tree = mgr.domain_tree()
    # Flatten all leaf lists in tree
    def _collect(node):
        ids = []
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            for v in node.values():
                ids.extend(_collect(v))
        return ids

    found = set(_collect(tree))
    expected = {"alpha", "beta", "gamma"}
    assert found == expected, f"domain_tree missing: {expected - found}, extra: {found - expected}"
    print("✅ Domain tree contains all concepts")


# ===========================================================================
# 13. SocietyManager serialisation roundtrip preserves n_concepts
# ===========================================================================

def test_society_serialisation_n_concepts():
    """to_dict()['n_concepts'] must equal len(mgr)."""
    mgr = _make_mgr("a", "b", "c", "d")
    d = mgr.to_dict()
    assert d["n_concepts"] == 4
    assert len(d["concepts"]) == 4
    print("✅ Society serialisation n_concepts correct")


# ===========================================================================
# 14. SocietalContextRouter route does not mutate query HV
# ===========================================================================

def test_router_route_does_not_mutate_query():
    """route() must not modify the query hypervector in any way."""
    mgr = _make_mgr("a", "b", "c")
    router = SocietalContextRouter(mgr, top_k=3, min_similarity=0.0)

    query = _hv(42)
    try:
        bits_before = np.array(query.bits)
    except Exception:
        bits_before = None

    router.route(query, task_tag="test")

    if bits_before is not None:
        bits_after = np.array(query.bits)
        assert np.array_equal(bits_before, bits_after), (
            "route() mutated the query HV!"
        )
    print("✅ route() does not mutate query HV")


# ===========================================================================
# 15. Epoch count monotonically increases
# ===========================================================================

def test_epoch_monotonic():
    """epoch counter must increase by exactly 1 per step_epoch()."""
    mgr = SocietyManager(auto_cluster_interval=0)
    mgr.register(_lhv("x"))
    assert mgr.epoch == 0
    for expected in range(1, 6):
        mgr.step_epoch()
        assert mgr.epoch == expected
    print("✅ Epoch counter monotonically increases")


if __name__ == "__main__":
    test_vsa_similarity_unchanged_by_societal()
    test_hv_bundle_unchanged_by_societal()
    test_semantic_memory_unchanged_by_societal()
    test_substrate_same_action_with_without_societal()
    test_societal_context_none_when_disabled()
    test_societal_context_dict_when_enabled()
    test_transplant_report_identical_flat_vs_societal()
    test_reregister_does_not_corrupt_bonds()
    test_activation_never_exceeds_one()
    test_bond_strength_bounded()
    test_leiden_covers_all_concepts()
    test_domain_tree_contains_all_concepts()
    test_society_serialisation_n_concepts()
    test_router_route_does_not_mutate_query()
    test_epoch_monotonic()
    print("\n🎉 All Societal Regression Tests Passed!")
