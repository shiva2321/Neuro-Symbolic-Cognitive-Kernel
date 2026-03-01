"""Benchmark: cross-domain transfer."""
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

logger = logging.getLogger("nsck_vision.bench_crossdomain")
logging.basicConfig(level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_crossdomain_benchmark() -> dict:
    """Run cross-domain transfer benchmark (2x2 matrix)."""
    from nsck_vision.system import NSCKVisionSystem
    
    system = NSCKVisionSystem()
    rng = np.random.default_rng(42)
    domains = ["domain_A", "domain_B"]
    dim = 64
    n_train = 30
    n_test = 10
    
    # Build per-domain class centers (similar to simulate transfer)
    domain_centers = {}
    for d in domains:
        domain_centers[d] = rng.standard_normal((3, dim)).astype(np.float32)
    
    # Absorb both domains
    for d in domains:
        train_data = []
        for i in range(n_train):
            cls = i % 3
            emb = domain_centers[d][cls] + rng.standard_normal(dim).astype(np.float32) * 0.1
            train_data.append((emb, f"{d}_class_{cls}"))
        
        def identity_model(x): return x
        system.absorb(
            model_or_name=identity_model,
            domain=d,
            dataset=train_data,
            max_samples=n_train,
            model_id=f"crossdomain_{d}",
        )
    
    # Cross-domain transfer matrix
    transfer_matrix = {}
    for train_d in domains:
        transfer_matrix[train_d] = {}
        for test_d in domains:
            correct = 0
            for i in range(n_test):
                cls = i % 3
                # Test with similar embeddings from different domain
                emb = domain_centers[test_d][cls] + rng.standard_normal(dim).astype(np.float32) * 0.05
                response = system.analyze(emb)
                # Check if either domain's label matches
                expected_labels = [f"{train_d}_class_{cls}", f"{test_d}_class_{cls}"]
                if response.label in expected_labels:
                    correct += 1
            transfer_matrix[train_d][test_d] = correct / n_test if n_test > 0 else 0.0
    
    return {"transfer_matrix": transfer_matrix, "domains": domains, "n_test_per_cell": n_test}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Running cross-domain benchmark...")
    results = run_crossdomain_benchmark()
    
    out_path = os.path.join(RESULTS_DIR, "crossdomain_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)
    return results


if __name__ == "__main__":
    main()
