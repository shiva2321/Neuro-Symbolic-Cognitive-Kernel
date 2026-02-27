"""STDP Calibrator — fine-tune HV codebook using spiking neural network dynamics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class CalibratedResult:
    """Output from :class:`STDPCalibrator`."""
    codebook: Dict[str, "HyperVector"]  # type: ignore[name-defined]
    snn_weights: np.ndarray
    quality_curve: List[float]
    n_epochs_run: int
    final_quality: float


class STDPCalibrator:
    """Refine an initial HV codebook via STDP-driven SNN simulation.

    For each epoch, random pairs of embeddings are fed through a
    ``PythonSnnCore``, spike trains are converted to HyperVectors, and
    neighbourhood-preservation quality is measured.  Training stops early when
    quality plateaus.

    Parameters
    ----------
    input_dim:        Embedding dimensionality.
    snn_size:         Number of SNN neurons (default 10 240).
    n_epochs:         Maximum training epochs (default 10).
    n_pairs:          Pairs sampled per epoch (default 500).
    seed:             RNG seed.
    patience:         Epochs without ``min_improvement`` before early stop.
    min_improvement:  Minimum quality delta to reset patience counter.
    """

    def __init__(
        self,
        input_dim: int,
        snn_size: int = 10240,
        n_epochs: int = 10,
        n_pairs: int = 500,
        seed: int = 42,
        patience: int = 3,
        min_improvement: float = 0.005,
    ) -> None:
        self._input_dim = input_dim
        self._snn_size = snn_size
        self._n_epochs = n_epochs
        self._n_pairs = n_pairs
        self._seed = seed
        self._patience = patience
        self._min_improvement = min_improvement

    # ------------------------------------------------------------------

    def calibrate(
        self,
        embeddings: np.ndarray,
        initial_codebook: Dict[str, "HyperVector"],  # type: ignore[name-defined]
        vocab_mapping: Dict[str, int],
    ) -> CalibratedResult:
        """Run STDP calibration and return a refined codebook.

        Parameters
        ----------
        embeddings:       Float embedding matrix, shape (N, dim).
        initial_codebook: Initial token → HV mapping from a projector.
        vocab_mapping:    Token → embedding-row index.
        """
        from python.core.perception.snn_perception import PythonSnnCore  # noqa: PLC0415
        from python.core.vsa.hypervec_shim import HyperVector             # noqa: PLC0415

        rng = np.random.default_rng(self._seed)
        snn = PythonSnnCore(
            input_dim=self._input_dim,
            snn_size=self._snn_size,
            seed=self._seed,
        )

        N = len(embeddings)
        tokens = list(vocab_mapping.keys())
        indices = np.array([vocab_mapping[t] for t in tokens], dtype=np.int64)
        valid_mask = indices < N
        tokens = [t for t, v in zip(tokens, valid_mask) if v]
        indices = indices[valid_mask]

        quality_curve: List[float] = []
        best_quality = -1.0
        no_improve = 0
        codebook = dict(initial_codebook)  # will be updated each epoch

        for epoch in range(self._n_epochs):
            # Sample random pairs
            pair_a = rng.integers(0, len(indices), size=self._n_pairs)
            pair_b = rng.integers(0, len(indices), size=self._n_pairs)
            same = pair_a == pair_b
            pair_b[same] = (pair_b[same] + 1) % len(indices)

            for a_pos, b_pos in zip(pair_a, pair_b):
                e_a = embeddings[indices[a_pos]].tolist()
                e_b = embeddings[indices[b_pos]].tolist()
                snn.simulate(e_a, n_steps=10, learn=True)
                snn.simulate(e_b, n_steps=10, learn=True)

            # Re-encode all tokens through calibrated SNN
            new_codebook: Dict[str, "HyperVector"] = {}
            for token, idx in zip(tokens, indices):
                spike_train = snn.simulate(
                    embeddings[idx].tolist(), n_steps=10, learn=False
                )
                hv = self._spikes_to_hv(spike_train, HyperVector)
                new_codebook[token] = hv

            quality = self._measure_quality(embeddings, indices, new_codebook, tokens)
            quality_curve.append(quality)

            if quality - best_quality > self._min_improvement:
                best_quality = quality
                codebook = new_codebook
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self._patience:
                    break

        final_weights = np.asarray(snn.input_weights, dtype=np.float32)

        return CalibratedResult(
            codebook=codebook,
            snn_weights=final_weights,
            quality_curve=quality_curve,
            n_epochs_run=len(quality_curve),
            final_quality=quality_curve[-1] if quality_curve else 0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spikes_to_hv(
        spike_train: list, HyperVector: type
    ) -> "HyperVector":  # type: ignore[name-defined]
        """Convert an (n_steps × snn_size) spike train to a HyperVector."""
        arr = np.asarray(spike_train, dtype=np.float32)
        if arr.ndim == 1:
            rates = arr
        elif arr.ndim == 2:
            rates = arr.mean(axis=0)
        else:
            rates = np.zeros(10240, dtype=np.float32)

        # Pad or trim to 10 240 bits
        D = 10240
        if len(rates) < D:
            rates = np.concatenate([rates, np.zeros(D - len(rates), dtype=np.float32)])
        else:
            rates = rates[:D]

        threshold = float(rates.mean())
        bits = (rates >= threshold).astype(np.int8)
        return HyperVector.from_bits(bits)

    @staticmethod
    def _measure_quality(
        embeddings: np.ndarray,
        indices: np.ndarray,
        codebook: Dict[str, "HyperVector"],  # type: ignore[name-defined]
        tokens: List[str],
        n_queries: int = 50,
        k: int = 10,
        seed: int = 7,
    ) -> float:
        """Recall@k: fraction of true top-k neighbours found in HV top-k."""
        N = len(tokens)
        if N <= k:
            return 1.0

        rng = np.random.default_rng(seed)
        query_pos = rng.choice(N, size=min(n_queries, N), replace=False)

        E = embeddings[indices].astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.maximum(norms, 1e-9)

        # Build HV bits matrix
        hv_bits = np.zeros((N, 10240), dtype=np.int8)
        for i, tok in enumerate(tokens):
            hv = codebook.get(tok)
            if hv is not None:
                hv_bits[i] = hv.bits[:10240]

        total_recall = 0.0
        for qp in query_pos:
            # True top-k by cosine similarity
            cos_sims = E_norm @ E_norm[qp]
            true_top = set(np.argsort(-cos_sims)[1 : k + 1].tolist())

            # HV top-k by Hamming similarity (popcount of agreement)
            q_bits = hv_bits[qp].astype(np.int32)
            ham_sims = (hv_bits.astype(np.int32) == q_bits).sum(axis=1)
            hv_top = set(np.argsort(-ham_sims)[1 : k + 1].tolist())

            total_recall += len(true_top & hv_top) / k

        return total_recall / len(query_pos)
