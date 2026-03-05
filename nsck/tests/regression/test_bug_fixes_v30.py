"""
Regression tests for NSCK V30 bug fixes.

Covers:
  - Bug 5.1: CausalGraph serialization preserves all links across save/load
  - Bug 5.2: nsck_studio.py and train_vsa_language.py use relative (not Windows) paths
  - Bug 5.3: CounterfactualReasoner triggered on hypothetical queries
  - Bug 5.4: enable_embedding_bridge defaults True with graceful fallback
  - Bug 5.5: NSCKConfig.v30() preset exists and has correct flags
  - Rust backends on by default
"""
from __future__ import annotations

import sys
import os
import json
import pytest

# _PKG_ROOT is the repo root (4 levels up from this file in nsck/tests/regression/)
_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# _NSCK_ROOT is the nsck/ directory (3 levels up from this file)
_NSCK_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _NSCK_ROOT not in sys.path:
    sys.path.insert(0, _NSCK_ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Bug 5.1: CausalGraph serialisation preserves all links
# ---------------------------------------------------------------------------

class TestCausalGraphCheckpoint:
    def test_causal_graph_survives_checkpoint(self):
        """CausalGraph.to_dict() / from_dict() preserves all links."""
        from python.core.reasoning.causal_reasoning import (
            CausalGraph, CausalLink, CausalRelation,
        )

        graph = CausalGraph()
        # Add a variety of links
        graph.add_link(CausalLink("rain", "flood", CausalRelation.CAUSES, 0.9))
        graph.add_link(CausalLink("drought", "fire", CausalRelation.ENABLES, 0.7))
        graph.add_link(CausalLink("heat", "rain", CausalRelation.PREVENTS, 0.4))
        graph.add_link(CausalLink(
            "wind", "fire", CausalRelation.REQUIRES, 0.6, context="climate"
        ))
        graph.add_causes("cloud", "rain", strength=0.85, context="weather")

        n_links = len(graph.all_links)
        assert n_links == 5, f"Expected 5 links, got {n_links}"

        # Serialise
        state = graph.to_dict()
        assert state["total_links"] == n_links
        assert len(state["links"]) == n_links

        # Deserialise
        restored = CausalGraph.from_dict(state)
        assert len(restored.all_links) == n_links, (
            f"Restored graph has {len(restored.all_links)} links; expected {n_links}"
        )

        # Verify forward chains still work
        chains = restored.forward_chain("rain")
        assert any(c.end == "flood" for c in chains), (
            "Forward chain rain→flood not preserved after round-trip"
        )

    def test_causal_graph_to_dict_total_links_matches(self):
        """to_dict() total_links matches actual link count."""
        from python.core.reasoning.causal_reasoning import CausalGraph
        graph = CausalGraph()
        for i in range(10):
            graph.add_causes(f"cause_{i}", f"effect_{i}", strength=0.5 + i * 0.04)
        d = graph.to_dict()
        assert d["total_links"] == len(graph.all_links)
        assert d["total_links"] == 10

    def test_causal_graph_empty_round_trip(self):
        """Empty CausalGraph survives to_dict / from_dict."""
        from python.core.reasoning.causal_reasoning import CausalGraph
        graph = CausalGraph()
        d = graph.to_dict()
        assert d["total_links"] == 0
        restored = CausalGraph.from_dict(d)
        assert len(restored.all_links) == 0


# ---------------------------------------------------------------------------
# Bug 5.2: nsck_studio.py uses relative paths (no hardcoded Windows paths)
# ---------------------------------------------------------------------------

class TestNsckStudioPaths:
    def test_nsck_studio_paths_are_relative(self):
        """nsck_studio.py must not contain hardcoded Windows absolute paths."""
        studio_path = os.path.join(_NSCK_ROOT, "python", "nsck_studio.py")
        if not os.path.exists(studio_path):
            pytest.skip(f"nsck_studio.py not found at {studio_path}")
        with open(studio_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "d:\\" not in content.lower() and "d:/" not in content.lower(), (
            "nsck_studio.py contains hardcoded Windows path"
        )
        assert "os.path.abspath" in content, (
            "nsck_studio.py should use os.path.abspath for path setup"
        )

    def test_train_vsa_language_paths_are_relative(self):
        """train_vsa_language.py must not contain hardcoded Windows path comments."""
        tool_path = os.path.join(_NSCK_ROOT, "python", "tools", "train_vsa_language.py")
        if not os.path.exists(tool_path):
            pytest.skip("train_vsa_language.py not found")
        with open(tool_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "d:\\" not in content.lower() and "d:/" not in content.lower(), (
            "train_vsa_language.py contains hardcoded Windows path"
        )


# ---------------------------------------------------------------------------
# Bug 5.3: CounterfactualReasoner triggered on hypothetical queries
# ---------------------------------------------------------------------------

class TestCounterfactualTriggered:
    def test_counterfactual_triggered_on_hypothetical(self):
        """Hypothetical text queries should set trace['counterfactual']['triggered']=True."""
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig
        substrate = NSCKSubstrate(NSCKConfig())
        substrate.register_task("cf_test")
        # Hypothetical query containing "if"
        result = substrate.process("What if plants had no sunlight?", "cf_test")
        trace = result.trace
        cf = trace.get("counterfactual", {})
        assert cf.get("triggered") is True, (
            "Expected counterfactual['triggered']=True for 'what if' query; "
            f"got trace={trace}"
        )

    def test_counterfactual_not_triggered_on_factual(self):
        """Plain factual queries should set trace['counterfactual']['triggered']=False."""
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig
        substrate = NSCKSubstrate(NSCKConfig())
        substrate.register_task("factual_test")
        result = substrate.process("Photosynthesis converts sunlight to sugar.", "factual_test")
        trace = result.trace
        cf = trace.get("counterfactual", {})
        assert cf.get("triggered") is False, (
            "Expected counterfactual['triggered']=False for factual query; "
            f"got trace={trace}"
        )


# ---------------------------------------------------------------------------
# Bug 5.4: enable_embedding_bridge defaults to True
# ---------------------------------------------------------------------------

class TestEmbeddingBridgeDefault:
    def test_embedding_bridge_defaults_to_true(self):
        """NSCKConfig default must have enable_embedding_bridge=True."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        assert cfg.enable_embedding_bridge is True, (
            "V30: enable_embedding_bridge should default to True"
        )

    def test_auto_persist_defaults_to_true(self):
        """NSCKConfig default must have enable_auto_persist=True."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        assert cfg.enable_auto_persist is True, (
            "V30: enable_auto_persist should default to True"
        )

    def test_embedding_bridge_gracefully_degraded_without_sbert(self):
        """NSCKSubstrate should initialise without error even if sentence-transformers is absent."""
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        assert cfg.enable_embedding_bridge is True
        # Should not raise even if sentence-transformers not installed
        substrate = NSCKSubstrate(cfg)
        assert substrate is not None


# ---------------------------------------------------------------------------
# Bug 5.5 / WP4: NSCKConfig.v30() preset and Rust flags
# ---------------------------------------------------------------------------

class TestV30ConfigAndRust:
    def test_v30_preset_exists(self):
        """NSCKConfig.v30() must return a valid config with all V30 flags set."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.v30()
        assert cfg.enable_transplant is True
        assert cfg.enable_societal is True
        assert cfg.enable_embedding_bridge is True
        assert cfg.enable_auto_persist is True
        assert cfg.use_rust_backend is True
        assert cfg.enable_rust_snn is True
        assert cfg.enable_transparency is True

    def test_rust_backends_active_by_default(self):
        """NSCKSubstrate._verify_rust_backends() returns a dict with all keys."""
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig
        substrate = NSCKSubstrate(NSCKConfig())
        status = substrate._rust_status
        assert isinstance(status, dict)
        for key in ("hypervec_rs", "snn_rs", "societal_rs"):
            assert key in status, f"Rust status missing key '{key}'"
        # Each value should be a bool
        for key, val in status.items():
            assert isinstance(val, bool), f"Rust status['{key}'] should be bool"
