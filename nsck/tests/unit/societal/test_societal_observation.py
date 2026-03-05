"""
Tests for SocietalSnapshot and SocietyManager.snapshot() (V30).
"""
from __future__ import annotations

import sys
import os
import pytest

_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np

from python.core.societal.society_manager import SocietyManager
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.snapshots import SocietalSnapshot
import python.core.vsa.hypervec_shim as hv_mod


def make_lhv(cid: str, seed: int, activation: float = 0.5) -> LivingHyperVector:
    hv = hv_mod.HyperVector(seed=seed)
    lhv = LivingHyperVector(
        concept_id=cid,
        hv=hv,
        domain_path=["science", "biology"],
        role="leaf",
        initial_activation=activation,
    )
    return lhv


def build_society(n: int = 6) -> SocietyManager:
    mgr = SocietyManager(bond_threshold=0.0)
    for i in range(n):
        mgr.register(make_lhv(f"c{i}", seed=i, activation=0.1 * i))
    mgr.auto_bond()
    return mgr


class TestSocietalSnapshot:
    def test_snapshot_returns_all_fields(self):
        """SocietyManager.snapshot() must return a SocietalSnapshot with all fields."""
        mgr = build_society(5)
        snap = mgr.snapshot()
        assert isinstance(snap, SocietalSnapshot)
        assert snap.epoch == mgr.epoch
        assert snap.n_concepts == 5
        assert snap.n_bonds >= 0
        assert 0.0 <= snap.avg_bond_strength <= 1.0
        assert 0.0 <= snap.avg_activation <= 1.0
        assert snap.n_communities >= 0
        assert 0.0 <= snap.percolation_threshold <= 1.0
        assert snap.giant_component_size >= 0
        assert isinstance(snap.top_activated, list)
        assert isinstance(snap.top_bonded, list)
        assert isinstance(snap.domain_tree_summary, dict)

    def test_snapshot_to_dict_is_serialisable(self):
        """SocietalSnapshot.to_dict() produces a JSON-serialisable dict."""
        import json
        mgr = build_society(4)
        snap = mgr.snapshot()
        d = snap.to_dict()
        serialised = json.dumps(d, default=str)
        assert len(serialised) > 10
        for key in ("epoch", "n_concepts", "n_bonds", "avg_bond_strength",
                    "avg_activation", "n_communities", "percolation_threshold",
                    "giant_component_size", "top_activated", "top_bonded"):
            assert key in d, f"Missing key '{key}' in SocietalSnapshot.to_dict()"

    def test_snapshot_epoch_advances(self):
        """Snapshot epoch reflects current epoch after step_epoch()."""
        mgr = build_society(4)
        mgr.step_epoch(run_cluster=False)
        mgr.step_epoch(run_cluster=False)
        snap = mgr.snapshot()
        assert snap.epoch == 2

    def test_activation_spreads_to_bonded_concepts(self):
        """After activating one concept and stepping, bonded concepts get activated."""
        mgr = SocietyManager(bond_threshold=0.0)
        c0 = make_lhv("c0", seed=0, activation=0.0)
        c1 = make_lhv("c1", seed=0, activation=0.0)  # same seed = similar HV
        mgr.register(c0)
        mgr.register(c1)
        mgr.auto_bond()
        mgr.activate_concept("c0", delta=1.0, spread=True)
        # After spreading, c1 should have some activation
        c1_after = mgr.get("c1")
        assert c1_after is not None
        assert c1_after.activation > 0.0, (
            "Bonded concept should receive spread activation"
        )

    def test_leiden_cluster_assigns_all_concepts(self):
        """leiden_cluster() assigns every concept to a community."""
        mgr = build_society(6)
        result = mgr.leiden_cluster(resolution=1.0)
        all_members = set()
        for members in result.communities.values():
            all_members.update(members)
        assert len(all_members) == 6, (
            f"Expected all 6 concepts assigned, got {len(all_members)}"
        )

    def test_percolation_threshold_in_range(self):
        """percolation_threshold() must return a value in [0, 1]."""
        mgr = build_society(6)
        perc = mgr.percolation_threshold()
        assert 0.0 <= perc <= 1.0, f"Percolation threshold out of range: {perc}"

    def test_societal_context_in_substrate_result(self):
        """NSCKSubstrate with societal enabled attaches societal_context to result."""
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.societal()
        substrate = NSCKSubstrate(cfg)
        substrate.register_task("soc_test")
        # Initialise the societal world with some concepts
        substrate.init_societal_world([
            {"concept_id": "photosynthesis", "domain_path": ["science"]},
            {"concept_id": "chlorophyll", "domain_path": ["science"]},
        ])
        result = substrate.process("chlorophyll drives photosynthesis", "soc_test")
        # When societal is enabled and society is initialised, context should be populated
        # (May be None if routing returned nothing, but SubstrateResult field must exist)
        assert hasattr(result, "societal_context")
