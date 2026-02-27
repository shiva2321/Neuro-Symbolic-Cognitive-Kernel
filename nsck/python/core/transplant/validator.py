"""Transplant Validator — measures neighbourhood-preservation quality of HV codebooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TransplantReport:
    """Quality report produced by :class:`TransplantValidator`."""
    spearman_rho: float
    recall_at_10: float
    recall_at_50: float
    ari: float
    passed: bool                                    # all thresholds met
    phase_timings: Dict[str, float]
    n_concepts: int
    worst_concepts: List[str]
    best_concepts: List[str]
    svd_variance_explained: Optional[float]
    calibration_quality_curve: Optional[List[float]]
    metadata: Dict = field(default_factory=dict)


class TransplantValidator:
    """Compute neighbourhood-preservation metrics for a HV codebook.

    Parameters
    ----------
    rho_threshold:      Minimum Spearman ρ (default 0.80).
    recall10_threshold: Minimum Recall@10 (default 0.70).
    recall50_threshold: Minimum Recall@50 (default 0.60).
    ari_threshold:      Minimum ARI (default 0.65).
    """

    def __init__(
        self,
        rho_threshold: float = 0.80,
        recall10_threshold: float = 0.70,
        recall50_threshold: float = 0.60,
        ari_threshold: float = 0.65,
        seed: int = 0,
    ) -> None:
        self._rho_thresh = rho_threshold
        self._rec10_thresh = recall10_threshold
        self._rec50_thresh = recall50_threshold
        self._ari_thresh = ari_threshold
        self._seed = seed

    # ------------------------------------------------------------------

    def validate(
        self,
        embeddings: np.ndarray,
        codebook: Dict[str, "HyperVector"],  # type: ignore[name-defined]
        vocab_mapping: Dict[str, int],
        n_sample_pairs: int = 2000,
    ) -> TransplantReport:
        """Compute and return a :class:`TransplantReport`.

        Parameters
        ----------
        embeddings:     Float embedding matrix (vocab_size, embedding_dim).
        codebook:       Token → HyperVector mapping.
        vocab_mapping:  Token → embedding row index.
        n_sample_pairs: Number of random pairs for Spearman / ARI.
        """
        import time  # noqa: PLC0415

        timings: Dict[str, float] = {}

        # Align tokens with valid indices
        tokens = [t for t in vocab_mapping if vocab_mapping[t] < len(embeddings) and t in codebook]
        N = len(tokens)

        if N < 2:
            return self._empty_report(timings, N)

        indices = np.array([vocab_mapping[t] for t in tokens], dtype=np.int64)
        E = embeddings[indices].astype(np.float32)

        # Build HV bits matrix (N, 10240)
        hv_bits = self._build_bits_matrix(tokens, codebook)

        # ---- Spearman ρ ------------------------------------------------
        t0 = time.perf_counter()
        rho = self._spearman(E, hv_bits, n_sample_pairs, seed=self._seed)
        timings["spearman"] = time.perf_counter() - t0

        # ---- Recall@k --------------------------------------------------
        t0 = time.perf_counter()
        rec10 = self._recall_at_k(E, hv_bits, k=10, seed=self._seed + 1)
        rec50 = self._recall_at_k(E, hv_bits, k=50, seed=self._seed + 1)
        timings["recall"] = time.perf_counter() - t0

        # ---- ARI -------------------------------------------------------
        t0 = time.perf_counter()
        ari = self._compute_ari(E, hv_bits, n_clusters=10, seed=self._seed)
        timings["ari"] = time.perf_counter() - t0

        # ---- Per-concept quality (for best/worst lists) ----------------
        t0 = time.perf_counter()
        per_token_quality = self._per_token_quality(E, hv_bits, tokens, k=10, seed=self._seed + 2)
        timings["per_token"] = time.perf_counter() - t0

        sorted_tokens = sorted(per_token_quality, key=lambda t: per_token_quality[t])
        worst = sorted_tokens[:10]
        best = sorted_tokens[-10:][::-1]

        passed = (
            rho >= self._rho_thresh
            and rec10 >= self._rec10_thresh
            and rec50 >= self._rec50_thresh
            and ari >= self._ari_thresh
        )

        return TransplantReport(
            spearman_rho=rho,
            recall_at_10=rec10,
            recall_at_50=rec50,
            ari=ari,
            passed=passed,
            phase_timings=timings,
            n_concepts=N,
            worst_concepts=worst,
            best_concepts=best,
            svd_variance_explained=None,
            calibration_quality_curve=None,
            metadata={
                "thresholds": {
                    "rho": self._rho_thresh,
                    "recall10": self._rec10_thresh,
                    "recall50": self._rec50_thresh,
                    "ari": self._ari_thresh,
                }
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bits_matrix(
        tokens: List[str],
        codebook: Dict[str, "HyperVector"],  # type: ignore[name-defined]
    ) -> np.ndarray:
        D = 10240
        mat = np.zeros((len(tokens), D), dtype=np.int8)
        for i, tok in enumerate(tokens):
            hv = codebook.get(tok)
            if hv is not None:
                b = hv.bits
                mat[i, : min(D, len(b))] = b[:D]
        return mat

    @staticmethod
    def _cosine_sim_matrix_rows(E: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(E, axis=1)
        na = norms[idx_a]
        nb = norms[idx_b]
        dots = np.sum(E[idx_a] * E[idx_b], axis=1)
        denom = np.maximum(na * nb, 1e-9)
        return dots / denom

    @staticmethod
    def _hamming_sim_rows(
        bits: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
    ) -> np.ndarray:
        D = bits.shape[1]
        agree = (bits[idx_a] == bits[idx_b]).sum(axis=1).astype(np.float32)
        return 2.0 * agree / D - 1.0

    @staticmethod
    def _spearman_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
        """Pure-numpy Spearman rank correlation."""
        def _rank(a: np.ndarray) -> np.ndarray:
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        rx -= rx.mean()
        ry -= ry.mean()
        num = float(np.dot(rx, ry))
        den = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
        if den < 1e-12:
            return 0.0
        return num / den

    def _spearman(
        self, E: np.ndarray, bits: np.ndarray, n_pairs: int, seed: int = 0
    ) -> float:
        N = len(E)
        rng = np.random.default_rng(seed)
        idx_a = rng.integers(0, N, size=n_pairs)
        idx_b = rng.integers(0, N, size=n_pairs)
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % N

        cos_sims = self._cosine_sim_matrix_rows(E, idx_a, idx_b)
        ham_sims = self._hamming_sim_rows(bits, idx_a, idx_b)
        return self._spearman_from_arrays(cos_sims, ham_sims)

    def _recall_at_k(
        self, E: np.ndarray, bits: np.ndarray, k: int, n_queries: int = 200, seed: int = 1
    ) -> float:
        N = len(E)
        if N <= k:
            return 1.0
        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=min(n_queries, N), replace=False)

        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.maximum(norms, 1e-9)

        total = 0.0
        for q in queries:
            cos = E_norm @ E_norm[q]
            true_top = set(np.argsort(-cos)[1 : k + 1].tolist())

            q_bits = bits[q].astype(np.int32)
            ham = (bits.astype(np.int32) == q_bits).sum(axis=1)
            hv_top = set(np.argsort(-ham)[1 : k + 1].tolist())

            total += len(true_top & hv_top) / k

        return total / len(queries)

    def _compute_ari(
        self, E: np.ndarray, bits: np.ndarray, n_clusters: int = 10, seed: int = 0
    ) -> float:
        """Adjusted Rand Index between embedding-space and HV-space clusters."""
        N = len(E)
        n_clusters = min(n_clusters, N)
        if n_clusters < 2:
            return 0.5  # neutral

        labels_e = self._kmeans(E, n_clusters, seed=seed)
        labels_h = self._kmeans(bits.astype(np.float32), n_clusters, seed=seed)
        return self._ari_numpy(labels_e, labels_h)

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 30) -> np.ndarray:
        """Lightweight pure-numpy k-means, returns cluster labels."""
        rng = np.random.default_rng(seed)
        N = len(X)
        # Subsample for speed
        cap = min(5000, N)
        idx = rng.choice(N, size=cap, replace=False)
        Xs = X[idx].astype(np.float64)

        # Initialise centroids via k-means++
        c_idx = [int(rng.integers(0, cap))]
        for _ in range(k - 1):
            dists = np.array(
                [min(float(np.sum((Xs[i] - Xs[ci]) ** 2)) for ci in c_idx) for i in range(cap)]
            )
            total = dists.sum()
            if total < 1e-12:
                dists = np.ones(cap, dtype=np.float64) / cap
            else:
                dists /= total
            c_idx.append(int(rng.choice(cap, p=dists)))
        centroids = Xs[c_idx]

        labels = np.zeros(cap, dtype=np.int64)
        for _ in range(max_iter):
            # Assignment
            diffs = Xs[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dists2 = (diffs ** 2).sum(axis=2)
            new_labels = dists2.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = Xs[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)

        # Assign all N points
        all_labels = np.zeros(N, dtype=np.int64)
        Xf = X.astype(np.float64)
        for i in range(0, N, 256):
            chunk = Xf[i : i + 256]
            diffs = chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dists2 = (diffs ** 2).sum(axis=2)
            all_labels[i : i + 256] = dists2.argmin(axis=1)
        return all_labels

    @staticmethod
    def _ari_numpy(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
        """Pure-numpy Adjusted Rand Index."""
        N = len(labels_a)
        ka = int(labels_a.max()) + 1
        kb = int(labels_b.max()) + 1

        # Contingency table
        contingency = np.zeros((ka, kb), dtype=np.float64)
        for i in range(N):
            contingency[labels_a[i], labels_b[i]] += 1

        sum_comb_c = float(np.sum(contingency * (contingency - 1) / 2))
        a = contingency.sum(axis=1)
        b = contingency.sum(axis=0)
        sum_comb_a = float(np.sum(a * (a - 1) / 2))
        sum_comb_b = float(np.sum(b * (b - 1) / 2))
        n_comb = float(N * (N - 1) / 2)

        expected = sum_comb_a * sum_comb_b / max(n_comb, 1e-12)
        max_val = (sum_comb_a + sum_comb_b) / 2.0
        denom = max_val - expected
        if abs(denom) < 1e-12:
            return 1.0 if abs(sum_comb_c - expected) < 1e-9 else 0.0
        return (sum_comb_c - expected) / denom

    def _per_token_quality(
        self,
        E: np.ndarray,
        bits: np.ndarray,
        tokens: List[str],
        k: int = 10,
        n_queries: int = 500,
        seed: int = 2,
    ) -> Dict[str, float]:
        """Per-token Recall@k quality score."""
        N = len(tokens)
        if N <= k:
            return {t: 1.0 for t in tokens}

        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=min(n_queries, N), replace=False)

        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.maximum(norms, 1e-9)

        scores: Dict[str, float] = {}
        for q in queries:
            cos = E_norm @ E_norm[q]
            true_top = set(np.argsort(-cos)[1 : k + 1].tolist())

            q_bits = bits[q].astype(np.int32)
            ham = (bits.astype(np.int32) == q_bits).sum(axis=1)
            hv_top = set(np.argsort(-ham)[1 : k + 1].tolist())

            scores[tokens[q]] = len(true_top & hv_top) / k

        return scores

    def _empty_report(self, timings: Dict[str, float], n: int) -> TransplantReport:
        return TransplantReport(
            spearman_rho=0.0,
            recall_at_10=0.0,
            recall_at_50=0.0,
            ari=0.0,
            passed=False,
            phase_timings=timings,
            n_concepts=n,
            worst_concepts=[],
            best_concepts=[],
            svd_variance_explained=None,
            calibration_quality_curve=None,
            metadata={"error": "insufficient tokens"},
        )
