"""Benchmark suite for the V15 Model Transplantation Pipeline.

Usage (from repository root)::

    python nsck/eval/bench_transplant.py

Compares the three projection strategies (random, learned, svd_factored) on
synthetic vocabularies of varying scale and reports time, memory, and quality
metrics.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.projector import (
    RandomProjector,
    LearnedProjector,
    SVDFactoredProjector,
)
from python.core.transplant.validator import TransplantValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_embeddings(vocab: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = max(2, vocab // 20)
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    per_cluster = vocab // n_clusters
    remainder = vocab - per_cluster * n_clusters
    rows = []
    for k in range(n_clusters):
        count = per_cluster + (1 if k < remainder else 0)
        rows.append(centers[k] + rng.standard_normal((count, dim)) * 0.3)
    emb = np.vstack(rows).astype(np.float32)
    return emb


def _vocab_mapping(n: int) -> Dict[str, int]:
    return {f"token_{i}": i for i in range(n)}


def _measure(strategy_name: str, emb: np.ndarray, vm: Dict[str, int]) -> Dict:
    dim = emb.shape[1]
    vocab = len(emb)

    if strategy_name == "random":
        proj = RandomProjector(dim, seed=0)
    elif strategy_name == "learned":
        proj = LearnedProjector(dim, seed=0, n_pairs=min(500, vocab), n_epochs=2)
    else:
        n_comp = min(64, dim)
        proj = SVDFactoredProjector(dim, n_components=n_comp, n_bins=128, seed=0)

    t0 = time.perf_counter()
    codebook = proj.project(emb, vm)
    t_project = time.perf_counter() - t0

    v = TransplantValidator(
        rho_threshold=0.0, recall10_threshold=0.0,
        recall50_threshold=0.0, ari_threshold=0.0,
        seed=0,
    )
    t0 = time.perf_counter()
    report = v.validate(emb, codebook, vm, n_sample_pairs=min(1000, vocab * (vocab - 1) // 2))
    t_validate = time.perf_counter() - t0

    return {
        "strategy": strategy_name,
        "vocab": vocab,
        "dim": dim,
        "t_project_s": round(t_project, 3),
        "t_validate_s": round(t_validate, 3),
        "spearman_rho": round(report.spearman_rho, 4),
        "recall@10": round(report.recall_at_10, 4),
        "recall@50": round(report.recall_at_50, 4),
        "ari": round(report.ari, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks() -> List[Dict]:
    scales = [
        (100,  64),
        (1000, 64),
        (10000, 64),
    ]
    strategies = ["random", "learned", "svd_factored"]
    results = []

    for vocab, dim in scales:
        emb = _make_synthetic_embeddings(vocab, dim)
        vm = _vocab_mapping(vocab)
        print(f"\n── vocab={vocab}, dim={dim} ──")
        for strat in strategies:
            row = _measure(strat, emb, vm)
            results.append(row)
            print(
                f"  {strat:14s}  project={row['t_project_s']:.3f}s  "
                f"validate={row['t_validate_s']:.3f}s  "
                f"ρ={row['spearman_rho']:.3f}  "
                f"R@10={row['recall@10']:.3f}  "
                f"R@50={row['recall@50']:.3f}  "
                f"ARI={row['ari']:.3f}"
            )

    print("\n── Final Quality Report ──")
    print(f"{'Strategy':14s} {'Vocab':>6s} {'ρ':>6s} {'R@10':>6s} {'R@50':>6s} {'ARI':>6s}")
    for row in results:
        print(
            f"{row['strategy']:14s} {row['vocab']:>6d} "
            f"{row['spearman_rho']:>6.3f} {row['recall@10']:>6.3f} "
            f"{row['recall@50']:>6.3f} {row['ari']:>6.3f}"
        )
    return results


if __name__ == "__main__":
    run_benchmarks()
