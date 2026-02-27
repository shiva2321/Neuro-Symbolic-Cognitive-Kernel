"""Projectors — map dense embedding matrices to HyperVector codebooks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List

import numpy as np


class BaseProjector(ABC):
    """Abstract base for all embedding → HyperVector projectors."""

    @abstractmethod
    def project(
        self, embeddings: np.ndarray, vocab_mapping: Dict[str, int]
    ) -> Dict[str, "HyperVector"]:  # type: ignore[name-defined]
        """Project every token embedding into a HyperVector codebook."""

    @abstractmethod
    def encode_new(self, embedding: np.ndarray) -> "HyperVector":  # type: ignore[name-defined]
        """Encode a single new embedding that was not seen during fitting."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_hv():
    from python.core.vsa.hypervec_shim import HyperVector  # noqa: PLC0415
    return HyperVector


def _binarize(proj: np.ndarray) -> np.ndarray:
    """Map sign(proj) to {0, 1} int8 array."""
    return (proj >= 0).astype(np.int8)


def _cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batch cosine similarity between paired rows."""
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    denom = np.maximum(na * nb, 1e-9)
    return np.sum(A * B, axis=1) / denom.ravel()


# ---------------------------------------------------------------------------
# RandomProjector
# ---------------------------------------------------------------------------

class RandomProjector(BaseProjector):
    """Johnson–Lindenstrauss random projection into binary HV space.

    Parameters
    ----------
    dim_in:   Input embedding dimensionality.
    hv_dim:   Target HV dimensionality (default 10 240).
    seed:     Random seed for reproducibility.
    """

    def __init__(self, dim_in: int, hv_dim: int = 10240, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._P: np.ndarray = rng.standard_normal((dim_in, hv_dim)).astype(np.float32)
        self._hv_dim = hv_dim

    def project(
        self, embeddings: np.ndarray, vocab_mapping: Dict[str, int]
    ) -> Dict[str, "HyperVector"]:
        HyperVector = _import_hv()
        codebook: Dict[str, "HyperVector"] = {}
        proj = embeddings @ self._P  # (N, hv_dim)
        for token, idx in vocab_mapping.items():
            if idx >= len(embeddings):
                continue
            bits = _binarize(proj[idx])
            codebook[token] = HyperVector.from_bits(bits)
        return codebook

    def encode_new(self, embedding: np.ndarray) -> "HyperVector":
        HyperVector = _import_hv()
        e = np.asarray(embedding, dtype=np.float32).ravel()
        bits = _binarize(e @ self._P)
        return HyperVector.from_bits(bits)


# ---------------------------------------------------------------------------
# LearnedProjector
# ---------------------------------------------------------------------------

class LearnedProjector(BaseProjector):
    """Gradient-descent projection that preserves cosine neighbourhood structure.

    Training uses a straight-through-inspired update: pairs of embeddings are
    sampled, target cosine similarities computed, and the projection matrix P
    is updated to minimise (sim_pred − sim_target)² using floating-point
    pre-binarization activations.

    Parameters
    ----------
    dim_in, hv_dim, seed:
        Same as :class:`RandomProjector`.
    n_pairs:
        Number of random pairs sampled per epoch.
    n_epochs:
        Training epochs.
    lr:
        Learning rate.
    """

    def __init__(
        self,
        dim_in: int,
        hv_dim: int = 10240,
        seed: int = 42,
        n_pairs: int = 1000,
        n_epochs: int = 5,
        lr: float = 0.01,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim_in)
        self._P: np.ndarray = (
            rng.standard_normal((dim_in, hv_dim)).astype(np.float32) * scale
        )
        self._hv_dim = hv_dim
        self._n_pairs = n_pairs
        self._n_epochs = n_epochs
        self._lr = lr
        self._seed = seed

    def fit(self, embeddings: np.ndarray) -> None:
        """Train projection matrix on *embeddings*."""
        rng = np.random.default_rng(self._seed + 1)
        N = len(embeddings)
        if N < 2:
            return

        for _ in range(self._n_epochs):
            # Sample n_pairs random pairs
            idx_a = rng.integers(0, N, size=self._n_pairs)
            idx_b = rng.integers(0, N, size=self._n_pairs)
            same = idx_a == idx_b
            idx_b[same] = (idx_b[same] + 1) % N

            E_a = embeddings[idx_a].astype(np.float32)  # (n_pairs, dim_in)
            E_b = embeddings[idx_b].astype(np.float32)

            # Target cosine similarities in embedding space
            sim_target = _cosine_rows(E_a, E_b)  # (n_pairs,)

            # Float projections (pre-sign) for gradient computation
            Z_a = E_a @ self._P  # (n_pairs, hv_dim)
            Z_b = E_b @ self._P

            # Predicted similarity using float projections (differentiable proxy)
            nz_a = np.linalg.norm(Z_a, axis=1, keepdims=True)
            nz_b = np.linalg.norm(Z_b, axis=1, keepdims=True)
            denom = np.maximum(nz_a * nz_b, 1e-9)
            sim_pred = np.sum(Z_a * Z_b, axis=1) / denom.ravel()  # (n_pairs,)

            # Loss gradient: dL/d(sim_pred) = 2*(sim_pred - sim_target)
            err = (sim_pred - sim_target).astype(np.float32)  # (n_pairs,)

            # Gradient of cosine similarity w.r.t. Z_a
            # d(z_a·z_b / |z_a||z_b|) / dZ_a ≈ (Z_b - sim*Z_a) / (|z_a||z_b|)
            denom2 = denom.ravel()[:, np.newaxis]  # (n_pairs, 1)
            dL_dZa = err[:, np.newaxis] * (
                Z_b / denom2 - sim_pred[:, np.newaxis] * Z_a / (nz_a ** 2)
            )  # (n_pairs, hv_dim)

            # Chain to P: dL/dP += E_a^T @ dL_dZa
            grad_P = E_a.T @ dL_dZa / self._n_pairs  # (dim_in, hv_dim)
            self._P -= self._lr * grad_P

    def project(
        self, embeddings: np.ndarray, vocab_mapping: Dict[str, int]
    ) -> Dict[str, "HyperVector"]:
        self.fit(embeddings)
        HyperVector = _import_hv()
        codebook: Dict[str, "HyperVector"] = {}
        proj = embeddings.astype(np.float32) @ self._P
        for token, idx in vocab_mapping.items():
            if idx >= len(embeddings):
                continue
            bits = _binarize(proj[idx])
            codebook[token] = HyperVector.from_bits(bits)
        return codebook

    def encode_new(self, embedding: np.ndarray) -> "HyperVector":
        HyperVector = _import_hv()
        e = np.asarray(embedding, dtype=np.float32).ravel()
        bits = _binarize(e @ self._P)
        return HyperVector.from_bits(bits)


# ---------------------------------------------------------------------------
# SVDFactoredProjector  (recommended default)
# ---------------------------------------------------------------------------

class SVDFactoredProjector(BaseProjector):
    """SVD-based dimensionality reduction followed by FPE codebook encoding.

    Matches the FPE pattern used in ``image_adapter.py`` and
    ``audio_adapter.py``:

    * Reduce embedding to ``n_components`` principal components via SVD.
    * For each component, quantise the scalar value to one of ``n_bins`` bins.
    * Bind each bin HV with a per-component role HV (XOR).
    * Bundle all bound HVs into a single HyperVector.

    Parameters
    ----------
    dim_in:       Input embedding dimensionality.
    hv_dim:       HV space dimensionality (must match HyperVector, default 10 240).
    n_components: Number of SVD components to keep (default 128).
    n_bins:       FPE quantisation bins per component (default 256).
    seed:         Codebook seed base (default 42, not directly used — seeds
                  are derived from component indices to match adapter pattern).
    """

    def __init__(
        self,
        dim_in: int = 768,
        hv_dim: int = 10240,
        n_components: int = 128,
        n_bins: int = 256,
        seed: int = 42,
    ) -> None:
        self._dim_in = dim_in
        self._hv_dim = hv_dim
        self._n_components = n_components
        self._n_bins = n_bins
        self._seed = seed

        # Populated by fit()
        self._V_proj: np.ndarray | None = None        # (n_components, dim_in)
        self._mean: np.ndarray | None = None           # (dim_in,)
        self._comp_min: np.ndarray | None = None       # (n_components,)
        self._comp_max: np.ndarray | None = None       # (n_components,)
        self._codebooks: List[List] | None = None      # list[n_components][n_bins]
        self._role_hvs: List | None = None             # list[n_components]

        # Fallback projector used when SVD fails
        self._fallback: RandomProjector | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit SVD and build per-component FPE codebooks."""
        HyperVector = _import_hv()

        E = embeddings.astype(np.float32)
        # For large vocabularies, fit SVD on a capped subset
        fit_E = E[: min(10000, len(E))]

        # 1. Centre
        self._mean = fit_E.mean(axis=0)
        E_c = fit_E - self._mean

        # 2. SVD — catch degenerate matrices
        try:
            _U, _S, Vt = np.linalg.svd(E_c, full_matrices=False)
            k = min(self._n_components, Vt.shape[0])
            self._V_proj = Vt[:k].copy()  # (k, dim_in)
        except np.linalg.LinAlgError:
            import warnings  # noqa: PLC0415
            warnings.warn(
                "SVDFactoredProjector: SVD failed (degenerate matrix). "
                "Falling back to RandomProjector — similarity preservation may be degraded.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fallback = RandomProjector(self._dim_in, self._hv_dim, self._seed)
            return

        # Project all data to get component ranges for normalisation
        Z_all = (E - self._mean) @ self._V_proj.T  # (N, k)
        self._comp_min = Z_all.min(axis=0)
        self._comp_max = Z_all.max(axis=0)

        k = self._V_proj.shape[0]

        # 3. Build per-component codebooks (FPE pattern from image_adapter.py)
        self._codebooks = []
        for j in range(k):
            cb_seed_base = j * 31 + 7777  # mirrors image_adapter seed formula
            cb = [HyperVector(i * 31 + cb_seed_base) for i in range(self._n_bins)]
            self._codebooks.append(cb)

        # 4. Build role HVs (mirrors audio_adapter role formula)
        self._role_hvs = [
            HyperVector((j * 1013 + 5003) % (2 ** 32)) for j in range(k)
        ]

    def _encode_projected(self, z_vec: np.ndarray) -> "HyperVector":
        """Encode a projected (reduced-dim) vector using FPE + XOR + bundle."""
        HyperVector = _import_hv()

        assert self._codebooks is not None and self._role_hvs is not None
        assert self._comp_min is not None and self._comp_max is not None

        k = len(self._codebooks)
        acc = None

        for j in range(k):
            rng_j = self._comp_max[j] - self._comp_min[j]
            if rng_j < 1e-9:
                norm_val = 0.0
            else:
                norm_val = float(np.clip((z_vec[j] - self._comp_min[j]) / rng_j, 0.0, 1.0))

            bin_idx = int(norm_val * (self._n_bins - 1) + 0.5)
            bin_idx = max(0, min(self._n_bins - 1, bin_idx))

            val_hv = self._codebooks[j][bin_idx]
            bound = val_hv.xor(self._role_hvs[j])
            acc = bound if acc is None else acc.bundle(bound)

        return acc if acc is not None else HyperVector(self._seed)

    def project(
        self, embeddings: np.ndarray, vocab_mapping: Dict[str, int]
    ) -> Dict[str, "HyperVector"]:
        self.fit(embeddings)

        if self._fallback is not None:
            return self._fallback.project(embeddings, vocab_mapping)

        codebook: Dict[str, "HyperVector"] = {}
        E = embeddings.astype(np.float32)
        Z = (E - self._mean) @ self._V_proj.T  # (N, k)

        for token, idx in vocab_mapping.items():
            if idx >= len(embeddings):
                continue
            codebook[token] = self._encode_projected(Z[idx])
        return codebook

    def encode_new(self, embedding: np.ndarray) -> "HyperVector":
        if self._fallback is not None:
            return self._fallback.encode_new(embedding)
        if self._V_proj is None or self._mean is None:
            raise RuntimeError("SVDFactoredProjector.fit() must be called before encode_new()")
        e = np.asarray(embedding, dtype=np.float32).ravel()
        z = (e - self._mean) @ self._V_proj.T
        return self._encode_projected(z)
