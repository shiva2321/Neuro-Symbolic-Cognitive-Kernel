"""
Unit tests for NSCK V5 Transplant Extension.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.transplant_extension import SocietalTransplantStage


def _make_embeddings(tokens: list, dim: int = 32, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {t: rng.standard_normal(dim).astype(np.float32) for t in tokens}


# ============================================================
# SocietalTransplantStage
# ============================================================

class TestSocietalTransplantStage:

    def setup_method(self):
        self.world = SocietalKnowledgeWorld()
        self.stage = SocietalTransplantStage(
            self.world, n_domain_clusters=3, seed=42
        )

    def test_init(self):
        assert self.stage.n_domain_clusters == 3
        assert self.stage.seed == 42

    def test_run_stage7_basic(self):
        tokens = [f"tok_{i}" for i in range(15)]
        embs = _make_embeddings(tokens)
        result = self.stage.run_stage7(
            transplant_report=None,
            embeddings=embs,
            domain_name="test_domain",
        )
        assert result["n_concepts_registered"] == 15
        assert result["n_neighborhoods"] == 3
        assert result["domain_id"] is not None
        assert result["domain_id"] in self.world.domains

    def test_run_stage7_registers_concepts(self):
        tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
        embs = _make_embeddings(tokens)
        self.stage.run_stage7(None, embs, "vocabulary")
        for tok in tokens:
            assert tok in self.world.concepts

    def test_run_stage7_creates_domain(self):
        tokens = [f"w{i}" for i in range(12)]
        embs = _make_embeddings(tokens)
        result = self.stage.run_stage7(None, embs, "linguistics")
        domain_id = result["domain_id"]
        assert domain_id in self.world.domains
        domain = self.world.domains[domain_id]
        assert domain.name == "linguistics"

    def test_run_stage7_empty(self):
        result = self.stage.run_stage7(None, {}, "empty_domain")
        assert result["n_concepts_registered"] == 0
        assert result["domain_id"] is None

    def test_run_stage7_with_codebook(self):
        tokens = [f"v{i}" for i in range(10)]
        embs = _make_embeddings(tokens)
        codebook = {t: hypervec_rs.HyperVector(seed=i) for i, t in enumerate(tokens)}
        result = self.stage.run_stage7(None, embs, "coded_domain", codebook=codebook)
        assert result["n_concepts_registered"] == 10

    def test_stats_delta(self):
        before = {
            "n_concepts": 5, "n_neighborhoods": 1,
            "n_domains": 0, "total_bonds": 3,
            "avg_bonds_per_concept": 0.6,
        }
        after = {
            "n_concepts": 15, "n_neighborhoods": 4,
            "n_domains": 1, "total_bonds": 12,
            "avg_bonds_per_concept": 0.8,
        }
        delta = SocietalTransplantStage.compute_societal_stats_delta(before, after)
        assert delta["d_n_concepts"] == 10
        assert delta["d_n_neighborhoods"] == 3
        assert delta["d_n_domains"] == 1

    def test_societal_stats_in_result(self):
        tokens = [f"s{i}" for i in range(9)]
        embs = _make_embeddings(tokens)
        result = self.stage.run_stage7(None, embs, "stats_test")
        assert "societal_stats" in result
        assert "delta" in result
        assert result["societal_stats"]["n_concepts"] >= 9
