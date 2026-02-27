"""Unit tests for TransplantValidator."""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.validator import TransplantValidator, TransplantReport
from python.core.transplant.projector import SVDFactoredProjector, RandomProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clustered_embeddings(n_clusters=5, per_cluster=20, d=32, seed=0):
    """Create embeddings with well-separated cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 5.0
    emb = np.vstack([
        centers[k] + rng.standard_normal((per_cluster, d)) * 0.2
        for k in range(n_clusters)
    ]).astype(np.float32)
    return emb


def _random_embeddings(n=100, d=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _vocab_mapping(n):
    return {f"token_{i}": i for i in range(n)}


def _random_codebook(n: int):
    """Return a codebook of fully random (unrelated) HVs."""
    from python.core.vsa.hypervec_shim import HyperVector
    return {f"token_{i}": HyperVector(i * 1234567 + 999) for i in range(n)}


def _good_codebook(emb, vm, n_components=8, n_bins=32, seed=0):
    """SVD-factored codebook on well-structured data."""
    proj = SVDFactoredProjector(emb.shape[1], n_components=n_components, n_bins=n_bins, seed=seed)
    return proj.project(emb, vm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransplantReport:
    def test_all_fields_present(self):
        r = TransplantReport(
            spearman_rho=0.85,
            recall_at_10=0.75,
            recall_at_50=0.65,
            ari=0.70,
            passed=True,
            phase_timings={},
            n_concepts=100,
            worst_concepts=[],
            best_concepts=[],
            svd_variance_explained=None,
            calibration_quality_curve=None,
        )
        assert r.passed is True
        assert r.spearman_rho == 0.85


class TestTransplantValidator:

    def test_metrics_in_valid_range_good(self):
        emb = _clustered_embeddings(n_clusters=5, per_cluster=20, d=32)
        vm = _vocab_mapping(len(emb))
        cb = _good_codebook(emb, vm, n_components=8, n_bins=32)
        v = TransplantValidator(
            rho_threshold=0.0,
            recall10_threshold=0.0,
            recall50_threshold=0.0,
            ari_threshold=0.0,
        )
        report = v.validate(emb, cb, vm, n_sample_pairs=500)
        assert -1.0 <= report.spearman_rho <= 1.0
        assert 0.0 <= report.recall_at_10 <= 1.0
        assert 0.0 <= report.recall_at_50 <= 1.0
        assert -1.0 <= report.ari <= 1.0

    def test_passed_true_when_all_thresholds_met(self):
        emb = _clustered_embeddings(n_clusters=8, per_cluster=25, d=32)
        vm = _vocab_mapping(len(emb))
        cb = _good_codebook(emb, vm, n_components=16, n_bins=64)
        # Use all-negative thresholds: any finite metric value should pass
        v = TransplantValidator(
            rho_threshold=-1.0,
            recall10_threshold=-1.0,
            recall50_threshold=-1.0,
            ari_threshold=-1.0,
        )
        report = v.validate(emb, cb, vm, n_sample_pairs=500)
        assert report.passed is True

    def test_passed_false_random_hvs(self):
        """Completely random HVs should fail strict thresholds."""
        emb = _random_embeddings(100, 32)
        vm = _vocab_mapping(100)
        cb = _random_codebook(100)
        v = TransplantValidator(
            rho_threshold=0.80,
            recall10_threshold=0.70,
            recall50_threshold=0.60,
            ari_threshold=0.65,
        )
        report = v.validate(emb, cb, vm, n_sample_pairs=300)
        assert report.passed is False

    def test_report_has_best_worst_concepts(self):
        emb = _clustered_embeddings(n_clusters=3, per_cluster=10, d=16)
        vm = _vocab_mapping(len(emb))
        cb = _good_codebook(emb, vm, n_components=4, n_bins=16)
        v = TransplantValidator(rho_threshold=0.0, recall10_threshold=0.0,
                                recall50_threshold=0.0, ari_threshold=0.0)
        report = v.validate(emb, cb, vm)
        assert isinstance(report.worst_concepts, list)
        assert isinstance(report.best_concepts, list)

    def test_insufficient_tokens_returns_empty_report(self):
        emb = np.zeros((1, 4), dtype=np.float32)
        vm = {"token_0": 0}
        cb = _random_codebook(1)
        v = TransplantValidator()
        report = v.validate(emb, cb, vm)
        assert report.n_concepts <= 1
        assert report.passed is False

    def test_deterministic_same_seed(self):
        emb = _clustered_embeddings(n_clusters=3, per_cluster=10, d=16)
        vm = _vocab_mapping(len(emb))
        cb = _good_codebook(emb, vm, n_components=4, n_bins=16)
        v1 = TransplantValidator(seed=0)
        v2 = TransplantValidator(seed=0)
        r1 = v1.validate(emb, cb, vm, n_sample_pairs=100)
        r2 = v2.validate(emb, cb, vm, n_sample_pairs=100)
        assert abs(r1.spearman_rho - r2.spearman_rho) < 1e-9
        assert abs(r1.recall_at_10 - r2.recall_at_10) < 1e-9

    def test_phase_timings_keys_present(self):
        emb = _clustered_embeddings(n_clusters=3, per_cluster=5, d=8)
        vm = _vocab_mapping(len(emb))
        cb = _good_codebook(emb, vm, n_components=4, n_bins=8)
        v = TransplantValidator()
        report = v.validate(emb, cb, vm, n_sample_pairs=50)
        assert "spearman" in report.phase_timings
        assert "recall" in report.phase_timings
        assert "ari" in report.phase_timings

    def test_n_concepts_correct(self):
        n = 30
        emb = _random_embeddings(n, 16)
        vm = _vocab_mapping(n)
        cb = _good_codebook(emb, vm, n_components=4, n_bins=8)
        v = TransplantValidator()
        report = v.validate(emb, cb, vm)
        assert report.n_concepts == n
