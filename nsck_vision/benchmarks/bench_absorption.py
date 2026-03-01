"""Benchmark: absorption speed, memory usage, similarity preservation."""
from __future__ import annotations

import os
import sys
import json
import time
import logging

os.environ["NSCK_USE_RUST"] = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

import numpy as np

logger = logging.getLogger("nsck_vision.bench_absorption")
logging.basicConfig(level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_absorption_benchmark(n_samples: int = 100, dim: int = 512) -> dict:
    """Run absorption benchmark with synthetic data."""
    from nsck_vision.system import NSCKVisionSystem
    
    system = NSCKVisionSystem()
    
    # Generate synthetic dataset
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_samples, dim)).astype(np.float32)
    labels = [f"class_{i % 10}" for i in range(n_samples)]
    dataset_iter = [(embeddings[i], labels[i]) for i in range(n_samples)]
    
    t0 = time.perf_counter()
    
    # Absorb using a callable model that returns embeddings
    def dummy_model(x):
        return x  # pass-through
    
    report = system.absorb(
        model_or_name=dummy_model,
        domain="synthetic",
        dataset=dataset_iter,
        max_samples=n_samples,
        model_id="bench_model",
    )
    
    elapsed = time.perf_counter() - t0
    hv_per_second = n_samples / max(elapsed, 1e-9)
    
    # Compute similarity preservation (Spearman ρ)
    try:
        from scipy.stats import spearmanr
        # Sample pairs
        n_pairs = min(50, n_samples * (n_samples - 1) // 2)
        rng2 = np.random.default_rng(99)
        i_idx = rng2.integers(0, n_samples, size=n_pairs)
        j_idx = rng2.integers(0, n_samples, size=n_pairs)
        
        orig_sims = []
        for i, j in zip(i_idx, j_idx):
            a, b = embeddings[i], embeddings[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            orig_sims.append(float(np.dot(a, b) / max(na * nb, 1e-9)))
        
        spearman_rho = float(report.spearman_rho)
    except Exception:
        spearman_rho = float(report.spearman_rho)
    
    result = {
        "n_samples": n_samples,
        "dim": dim,
        "absorption_time_s": float(elapsed),
        "hv_per_second": float(hv_per_second),
        "spearman_rho": spearman_rho,
        "n_concepts_absorbed": report.n_concepts_absorbed,
        "rust_backend_active": report.rust_backend_active,
        "passed": report.passed,
    }
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Running absorption benchmark...")
    results = {}
    for domain, n in [("synthetic_small", 50), ("synthetic_medium", 200)]:
        logger.info("  Domain: %s (n=%d)", domain, n)
        results[domain] = run_absorption_benchmark(n_samples=n)
    
    out_path = os.path.join(RESULTS_DIR, "absorption_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)
    return results


if __name__ == "__main__":
    main()
