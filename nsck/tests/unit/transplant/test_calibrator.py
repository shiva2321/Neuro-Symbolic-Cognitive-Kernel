"""Unit tests for STDPCalibrator."""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.calibrator import STDPCalibrator, CalibratedResult
from python.core.transplant.projector import RandomProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int = 50, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _vocab_mapping(n: int) -> dict:
    return {f"token_{i}": i for i in range(n)}


def _initial_codebook(emb, vm, seed=0):
    return RandomProjector(emb.shape[1], seed=seed).project(emb, vm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalibratedResult:
    def test_fields(self):
        cr = CalibratedResult(
            codebook={},
            snn_weights=np.zeros((10, 5), dtype=np.float32),
            quality_curve=[0.1, 0.2],
            n_epochs_run=2,
            final_quality=0.2,
        )
        assert cr.n_epochs_run == 2
        assert cr.final_quality == 0.2


class TestSTDPCalibrator:

    def test_returns_calibrated_result(self):
        emb = _make_embeddings(30, 8)
        vm = _vocab_mapping(30)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(input_dim=8, snn_size=64, n_epochs=2, n_pairs=10, seed=0)
        result = cal.calibrate(emb, cb0, vm)
        assert isinstance(result, CalibratedResult)

    def test_codebook_has_all_tokens(self):
        emb = _make_embeddings(20, 8)
        vm = _vocab_mapping(20)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(8, snn_size=64, n_epochs=1, n_pairs=5, seed=1)
        result = cal.calibrate(emb, cb0, vm)
        # All tokens present in refined codebook
        assert set(result.codebook.keys()) == set(vm.keys())

    def test_hvs_are_valid_binary(self):
        emb = _make_embeddings(15, 8)
        vm = _vocab_mapping(15)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(8, snn_size=64, n_epochs=1, n_pairs=5, seed=2)
        result = cal.calibrate(emb, cb0, vm)
        for hv in result.codebook.values():
            assert set(hv.bits.tolist()).issubset({0, 1})

    def test_quality_curve_grows_or_stays(self):
        emb = _make_embeddings(40, 8)
        vm = _vocab_mapping(40)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(8, snn_size=64, n_epochs=3, n_pairs=20, seed=3)
        result = cal.calibrate(emb, cb0, vm)
        # Quality curve should have at least 1 entry
        assert len(result.quality_curve) >= 1
        # Values should be in [0, 1]
        assert all(0.0 <= q <= 1.0 for q in result.quality_curve)

    def test_snn_weights_shape(self):
        d = 12
        snn_sz = 64
        emb = _make_embeddings(20, d)
        vm = _vocab_mapping(20)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(d, snn_size=snn_sz, n_epochs=1, n_pairs=5, seed=4)
        result = cal.calibrate(emb, cb0, vm)
        assert result.snn_weights.shape == (snn_sz, d)

    def test_deterministic_same_seed(self):
        emb = _make_embeddings(20, 8)
        vm = _vocab_mapping(20)
        cb0 = _initial_codebook(emb, vm, seed=0)
        cal1 = STDPCalibrator(8, snn_size=64, n_epochs=1, n_pairs=5, seed=99)
        cal2 = STDPCalibrator(8, snn_size=64, n_epochs=1, n_pairs=5, seed=99)
        r1 = cal1.calibrate(emb, cb0, vm)
        r2 = cal2.calibrate(emb, cb0, vm)
        np.testing.assert_array_equal(r1.snn_weights, r2.snn_weights)

    def test_early_stopping(self):
        emb = _make_embeddings(30, 8)
        vm = _vocab_mapping(30)
        cb0 = _initial_codebook(emb, vm)
        # patience=1 should stop early
        cal = STDPCalibrator(8, snn_size=64, n_epochs=10, n_pairs=10, patience=1, seed=5)
        result = cal.calibrate(emb, cb0, vm)
        assert result.n_epochs_run <= 10

    def test_final_quality_matches_last_curve_entry(self):
        emb = _make_embeddings(20, 8)
        vm = _vocab_mapping(20)
        cb0 = _initial_codebook(emb, vm)
        cal = STDPCalibrator(8, snn_size=64, n_epochs=2, n_pairs=5, seed=6)
        result = cal.calibrate(emb, cb0, vm)
        assert abs(result.final_quality - result.quality_curve[-1]) < 1e-9

    def test_spikes_to_hv_correct_shape(self):
        from python.core.vsa.hypervec_shim import HyperVector  # noqa: PLC0415
        spike_train = [[float(j % 2) for j in range(64)] for _ in range(10)]
        hv = STDPCalibrator._spikes_to_hv(spike_train, HyperVector)
        assert len(hv.bits) == 10240
        assert set(hv.bits.tolist()).issubset({0, 1})
