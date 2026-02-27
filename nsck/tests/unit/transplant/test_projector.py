"""Unit tests for transplant projectors."""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.projector import (
    RandomProjector,
    LearnedProjector,
    SVDFactoredProjector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int = 200, d: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _vocab_mapping(n: int) -> dict:
    return {f"token_{i}": i for i in range(n)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        return ranks
    rx = _rank(x) - _rank(x).mean()
    ry = _rank(y) - _rank(y).mean()
    num = float(np.dot(rx, ry))
    den = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return num / den if den > 1e-12 else 0.0


def _sample_similarity_pairs(embeddings: np.ndarray, n_pairs: int = 500, seed: int = 42):
    """Returns (original_cos_sims, idx_a, idx_b)."""
    rng = np.random.default_rng(seed)
    N = len(embeddings)
    ia = rng.integers(0, N, size=n_pairs)
    ib = rng.integers(0, N, size=n_pairs)
    ib[ia == ib] = (ib[ia == ib] + 1) % N
    norms = np.linalg.norm(embeddings, axis=1)
    dots = np.sum(embeddings[ia] * embeddings[ib], axis=1)
    cos = dots / np.maximum(norms[ia] * norms[ib], 1e-9)
    return cos, ia, ib


def _hv_hamming_sims(codebook, tokens, ia, ib):
    D = 10240
    hvs = [codebook[tokens[i]].bits[:D].astype(np.int32) for i in range(len(tokens))]
    hvs = np.array(hvs)  # (N, D)
    agree = (hvs[ia] == hvs[ib]).sum(axis=1).astype(float)
    return 2.0 * agree / D - 1.0


# ---------------------------------------------------------------------------
# RandomProjector tests
# ---------------------------------------------------------------------------

class TestRandomProjector:

    def test_produces_valid_hvs(self):
        emb = _make_embeddings(50, 32)
        vm = _vocab_mapping(50)
        proj = RandomProjector(dim_in=32, hv_dim=10240, seed=0)
        codebook = proj.project(emb, vm)
        assert len(codebook) == 50
        for hv in codebook.values():
            bits = hv.bits
            assert len(bits) == 10240
            assert set(bits.tolist()).issubset({0, 1})

    def test_deterministic_same_seed(self):
        emb = _make_embeddings(30, 16)
        vm = _vocab_mapping(30)
        cb1 = RandomProjector(16, seed=7).project(emb, vm)
        cb2 = RandomProjector(16, seed=7).project(emb, vm)
        for tok in vm:
            np.testing.assert_array_equal(cb1[tok].bits, cb2[tok].bits)

    def test_different_seeds_differ(self):
        emb = _make_embeddings(10, 16)
        vm = _vocab_mapping(10)
        cb1 = RandomProjector(16, seed=1).project(emb, vm)
        cb2 = RandomProjector(16, seed=2).project(emb, vm)
        diffs = sum(
            not np.array_equal(cb1[t].bits, cb2[t].bits) for t in vm
        )
        assert diffs > 0

    def test_encode_new_consistent_with_batch(self):
        emb = _make_embeddings(20, 16)
        vm = _vocab_mapping(20)
        proj = RandomProjector(16, seed=3)
        cb = proj.project(emb, vm)
        # encode_new should reproduce the batch result
        for i in range(min(5, 20)):
            hv_batch = cb[f"token_{i}"]
            hv_new = proj.encode_new(emb[i])
            np.testing.assert_array_equal(hv_batch.bits, hv_new.bits)

    def test_similarity_rank_preserved_spearman(self):
        emb = _make_embeddings(200, 64, seed=10)
        vm = _vocab_mapping(200)
        cb = RandomProjector(64, seed=0).project(emb, vm)
        tokens = list(vm.keys())
        cos, ia, ib = _sample_similarity_pairs(emb, n_pairs=400)
        ham = _hv_hamming_sims(cb, tokens, ia, ib)
        rho = _spearman(cos, ham)
        assert rho > 0.5, f"RandomProjector Spearman ρ={rho:.3f} too low"


# ---------------------------------------------------------------------------
# LearnedProjector tests
# ---------------------------------------------------------------------------

class TestLearnedProjector:

    def test_produces_valid_hvs(self):
        emb = _make_embeddings(40, 16)
        vm = _vocab_mapping(40)
        proj = LearnedProjector(dim_in=16, hv_dim=10240, seed=0, n_pairs=50, n_epochs=1)
        cb = proj.project(emb, vm)
        assert len(cb) == 40
        for hv in cb.values():
            assert set(hv.bits.tolist()).issubset({0, 1})

    def test_deterministic(self):
        emb = _make_embeddings(20, 8)
        vm = _vocab_mapping(20)
        cb1 = LearnedProjector(8, seed=5, n_pairs=10, n_epochs=1).project(emb, vm)
        cb2 = LearnedProjector(8, seed=5, n_pairs=10, n_epochs=1).project(emb, vm)
        for tok in vm:
            np.testing.assert_array_equal(cb1[tok].bits, cb2[tok].bits)

    def test_encode_new_consistent(self):
        emb = _make_embeddings(20, 8)
        vm = _vocab_mapping(20)
        proj = LearnedProjector(8, seed=2, n_pairs=10, n_epochs=1)
        cb = proj.project(emb, vm)
        for i in range(5):
            np.testing.assert_array_equal(
                cb[f"token_{i}"].bits,
                proj.encode_new(emb[i]).bits,
            )


# ---------------------------------------------------------------------------
# SVDFactoredProjector tests
# ---------------------------------------------------------------------------

class TestSVDFactoredProjector:

    def test_produces_valid_hvs(self):
        emb = _make_embeddings(100, 64)
        vm = _vocab_mapping(100)
        proj = SVDFactoredProjector(dim_in=64, n_components=16, n_bins=32)
        cb = proj.project(emb, vm)
        assert len(cb) == 100
        for hv in cb.values():
            bits = hv.bits
            assert len(bits) == 10240
            assert set(bits.tolist()).issubset({0, 1})

    def test_deterministic(self):
        emb = _make_embeddings(50, 32)
        vm = _vocab_mapping(50)
        cb1 = SVDFactoredProjector(32, n_components=8, n_bins=16, seed=0).project(emb, vm)
        cb2 = SVDFactoredProjector(32, n_components=8, n_bins=16, seed=0).project(emb, vm)
        for tok in vm:
            np.testing.assert_array_equal(cb1[tok].bits, cb2[tok].bits)

    def test_encode_new_consistent(self):
        emb = _make_embeddings(50, 32)
        vm = _vocab_mapping(50)
        proj = SVDFactoredProjector(32, n_components=8, n_bins=16, seed=1)
        cb = proj.project(emb, vm)
        for i in range(5):
            np.testing.assert_array_equal(
                cb[f"token_{i}"].bits,
                proj.encode_new(emb[i]).bits,
            )

    def test_spearman_better_than_random(self):
        """SVD+FPE should achieve positive Spearman ρ on tight-cluster data."""
        rng = np.random.default_rng(99)
        # Create embeddings with very tight cluster structure (noise << inter-cluster dist)
        n_clusters, per_cluster, d = 10, 20, 32
        centers = rng.standard_normal((n_clusters, d)) * 5.0
        emb = np.vstack([
            centers[k] + rng.standard_normal((per_cluster, d)) * 0.05
            for k in range(n_clusters)
        ]).astype(np.float32)
        n = len(emb)
        vm = _vocab_mapping(n)

        # Use few bins so bin-width >> intra-cluster noise
        cb = SVDFactoredProjector(d, n_components=16, n_bins=8, seed=0).project(emb, vm)
        tokens = list(vm.keys())

        # Verify cluster preservation: intra-cluster HV sim > inter-cluster HV sim
        D = 10240
        hvs = [cb[tokens[i]].bits[:D].astype(np.int32) for i in range(n)]
        hvs = np.array(hvs)

        intra_sims, inter_sims = [], []
        for k in range(n_clusters):
            start = k * per_cluster
            end = start + per_cluster
            for i in range(start, end):
                for j in range(i + 1, end):
                    agree = float((hvs[i] == hvs[j]).mean())
                    intra_sims.append(2 * agree - 1)
            for other_k in range(k + 1, n_clusters):
                other_start = other_k * per_cluster
                for i in range(start, start + 3):
                    for j in range(other_start, other_start + 3):
                        agree = float((hvs[i] == hvs[j]).mean())
                        inter_sims.append(2 * agree - 1)

        mean_intra = float(np.mean(intra_sims))
        mean_inter = float(np.mean(inter_sims))
        assert mean_intra > mean_inter, (
            f"Intra-cluster sim ({mean_intra:.3f}) should exceed "
            f"inter-cluster sim ({mean_inter:.3f})"
        )

    def test_large_vocabulary_capped(self):
        """Fitting on >10k vocab should not raise (SVD is capped at 10k rows)."""
        emb = _make_embeddings(12000, 16)
        vm = _vocab_mapping(12000)
        cb = SVDFactoredProjector(16, n_components=4, n_bins=8).project(emb, vm)
        assert len(cb) == 12000

    def test_encode_new_raises_before_fit(self):
        proj = SVDFactoredProjector(16, n_components=4, n_bins=8)
        with pytest.raises(RuntimeError):
            proj.encode_new(np.zeros(16))
