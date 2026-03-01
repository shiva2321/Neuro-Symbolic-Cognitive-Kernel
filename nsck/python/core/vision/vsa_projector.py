"""VSAProjector — deterministic random projection into binary HyperVector space.

Uses the Johnson-Lindenstrauss lemma to project dense vision embeddings into a
10,240-bit binary HyperVector space while approximately preserving pairwise
cosine similarities.  Each model_id gets a deterministic seed matrix derived
from the model_id hash, guaranteeing reproducibility across runs.

JL guarantee: for ε ∈ (0, 0.5) and any two unit vectors u, v,
    P[|sim(Pu, Pv) − sim(u, v)| > ε] ≤ 2 exp(−(ε² − ε³) · hv_dim / 4)
With hv_dim=10240 this gives tight preservation for typical ε=0.1 pairs.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import hashlib

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.transplant.projector import SVDFactoredProjector, RandomProjector


def _derive_seed(model_id: str) -> int:
    """Deterministically derive an integer seed from a model identifier."""
    digest = hashlib.sha256(model_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _binarize(proj: np.ndarray) -> np.ndarray:
    """Map sign(proj) to {0,1} int8 array."""
    return (proj >= 0).astype(np.int8)


class VSAProjector:
    """Deterministic projection from float feature space to binary HV space.

    Mirrors the SVDFactoredProjector pattern but is specialised for vision
    models and labelled feature batches.

    Parameters
    ----------
    dim_in:      Input embedding dimensionality.
    hv_dim:      Target HV dimensionality (default 10 240).
    model_id:    Identifier used to derive the deterministic seed matrix.
    seed:        Override seed; derived from model_id hash when None.
    n_components: SVD components to keep (passed to SVDFactoredProjector).
    n_bins:       FPE quantisation bins (passed to SVDFactoredProjector).
    """

    def __init__(
        self,
        dim_in: int,
        hv_dim: int = 10240,
        model_id: str = "default",
        seed: Optional[int] = None,
        n_components: int = 128,
        n_bins: int = 256,
    ) -> None:
        self._dim_in = dim_in
        self._hv_dim = hv_dim
        self._model_id = model_id
        self._seed: int = seed if seed is not None else _derive_seed(model_id)
        self._n_components = n_components
        self._n_bins = n_bins

        # Primary projector: SVD-factored (better similarity preservation)
        self._svd_proj = SVDFactoredProjector(
            dim_in=dim_in,
            hv_dim=hv_dim,
            n_components=n_components,
            n_bins=n_bins,
            seed=self._seed,
        )

        # Fallback: plain random projection via JL
        self._rng_proj = RandomProjector(dim_in=dim_in, hv_dim=hv_dim, seed=self._seed)

        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_seed(self) -> int:
        """Return the deterministic seed used for this model_id."""
        return self._seed

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit SVD-based projection on a corpus of embeddings.

        Parameters
        ----------
        embeddings: (N, dim_in) float32 array of feature vectors.
        """
        E = np.asarray(embeddings, dtype=np.float32)
        # Normalise to unit sphere before fitting (cosine-invariant)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        E_norm = E / norms
        self._svd_proj.fit(E_norm)
        self._fitted = True

    def project(
        self, embeddings: np.ndarray, labels: List[str]
    ) -> Dict[str, "hypervec_rs.HyperVector"]:
        """Project a labelled batch of embeddings into the HV codebook.

        Parameters
        ----------
        embeddings: (N, dim_in) float32 array.
        labels:     length-N list of string labels.

        Returns
        -------
        Dict[label → HyperVector]
        """
        E = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        E_norm = E / norms

        if not self._fitted:
            self.fit(E_norm)

        vocab_mapping: Dict[str, int] = {label: i for i, label in enumerate(labels)}
        return self._svd_proj.project(E_norm, vocab_mapping)

    def encode_new(self, embedding: np.ndarray) -> "hypervec_rs.HyperVector":
        """Encode a single new embedding for live inference.

        The input is normalised to unit sphere then projected using the
        fitted SVD projection (or random fallback if not fitted).

        Parameters
        ----------
        embedding: 1-D float array of length dim_in.

        Returns
        -------
        HyperVector
        """
        e = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(e)
        if norm > 1e-9:
            e = e / norm

        if self._fitted:
            return self._svd_proj.encode_new(e)
        # Fallback: random projection (still JL-compliant)
        return self._rng_proj.encode_new(e)
