"""
Research benchmark: NSCK-UPMA Vision Absorber (Paper 6).

Absorbs synthetic ResNet-18-like features, measures Top-1 Accuracy, Spearman ρ,
Recall@10, and latency. Produces publication-ready results saved to
research/results/paper6_results.json.
"""
import os
import sys
import json
import time
import logging

os.environ["NSCK_USE_RUST"] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

logger = logging.getLogger("paper6")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_recall_at_k(absorbed_hvs, query_hvs, true_labels, query_labels, k=10):
    """Compute Recall@K: fraction of queries for which the true label appears in top-K."""
    hits = 0
    total = len(query_hvs)
    if total == 0:
        return 0.0

    for q_hv, q_label in zip(query_hvs, query_labels):
        sims = [(absorbed_hvs[j].similarity_robust(q_hv), true_labels[j])
                for j in range(len(absorbed_hvs))]
        sims.sort(key=lambda x: -x[0])
        top_k_labels = [lbl for _, lbl in sims[:k]]
        if q_label in top_k_labels:
            hits += 1
    return hits / total


def run_paper6_benchmark():
    from nsck_vision.system import NSCKVisionSystem

    logger.info("=== Paper 6 Benchmark: NSCK-UPMA Vision Absorber ===")

    system = NSCKVisionSystem()
    rng = np.random.default_rng(42)

    # --- Experiment 1: ResNet-like absorption (synthetic) ---
    n_classes = 10
    n_train = 100
    n_test = 50
    dim = 512

    logger.info("Generating synthetic ResNet-like features (dim=%d, n_train=%d)...", dim, n_train)
    class_centers = rng.standard_normal((n_classes, dim)).astype(np.float32)
    for i in range(n_classes):
        class_centers[i] /= np.linalg.norm(class_centers[i])

    train_data = []
    for i in range(n_train):
        cls = i % n_classes
        noise = rng.standard_normal(dim).astype(np.float32) * 0.1
        emb = class_centers[cls] + noise
        emb /= np.linalg.norm(emb)
        train_data.append((emb, f"imagenet_class_{cls:03d}"))

    def identity_model(x): return x

    t0 = time.perf_counter()
    report = system.absorb(
        model_or_name=identity_model,
        domain="imagenet_synthetic",
        dataset=train_data,
        max_samples=n_train,
        model_id="resnet18_synthetic",
    )
    absorption_time = time.perf_counter() - t0

    logger.info(
        "Absorption: %d concepts in %.2fs (%.0f HV/s)",
        report.n_concepts_absorbed, absorption_time, report.hv_per_second,
    )
    logger.info("Spearman ρ: %.4f, Rust active: %s", report.spearman_rho, report.rust_backend_active)

    # --- Experiment 2: Classification accuracy + Recall@10 ---
    logger.info("Running classification evaluation (n_test=%d)...", n_test)
    correct = 0
    latencies = []
    test_embs = []
    test_labels = []

    for i in range(n_test):
        cls = i % n_classes
        noise = rng.standard_normal(dim).astype(np.float32) * 0.05
        test_emb = class_centers[cls] + noise
        test_emb /= np.linalg.norm(test_emb)
        test_embs.append(test_emb)
        test_labels.append(f"imagenet_class_{cls:03d}")

        t0 = time.perf_counter()
        response = system.analyze(test_emb)
        latencies.append((time.perf_counter() - t0) * 1000)

        expected = f"imagenet_class_{cls:03d}"
        if response.label == expected:
            correct += 1

    top1_acc = correct / n_test
    mean_latency = float(np.mean(latencies))

    # Compute Recall@10 using absorbed HVs from memory
    try:
        from python.core.vision.vsa_projector import VSAProjector
        import python.core.vsa.hypervec_shim as hvs
        absorbed_mem = system._substrate._absorption_memory
        if absorbed_mem and len(absorbed_mem._records) > 0:
            absorbed_hvs = [r.hv for r in absorbed_mem._records]
            absorbed_labels = [r.label for r in absorbed_mem._records]
            proj_key = absorbed_mem._records[0].model_id
            projector = system._substrate._vision_absorber._projectors.get(proj_key)
            if projector is not None:
                query_hvs = [projector.encode_new(e) for e in test_embs]
                recall_at_10 = compute_recall_at_k(
                    absorbed_hvs, query_hvs, absorbed_labels, test_labels, k=10
                )
            else:
                recall_at_10 = report.recall_at_10
        else:
            recall_at_10 = report.recall_at_10
    except Exception:
        recall_at_10 = report.recall_at_10

    logger.info("Top-1 Accuracy: %.4f (%d/%d)", top1_acc, correct, n_test)
    logger.info("Recall@10: %.4f", recall_at_10)
    logger.info("Mean latency: %.1fms", mean_latency)

    # --- Compile results ---
    results = {
        "experiment": "paper6_nsck_upma_vision",
        "timestamp": time.time(),
        "config": {
            "n_classes": n_classes,
            "n_train": n_train,
            "n_test": n_test,
            "feature_dim": dim,
        },
        "absorption": {
            "n_concepts_absorbed": report.n_concepts_absorbed,
            "absorption_time_s": absorption_time,
            "hv_per_second": report.hv_per_second,
            "spearman_rho": report.spearman_rho,
            "rust_backend_active": report.rust_backend_active,
            "strategy": report.strategy,
        },
        "classification": {
            "top1_accuracy": top1_acc,
            "correct": correct,
            "total": n_test,
            "recall_at_10": recall_at_10,
        },
        "latency": {
            "mean_ms": mean_latency,
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "target_ms": 500.0,
            "target_met": mean_latency < 500.0,
        },
    }

    # Save results
    results_dir = os.path.join(_REPO_ROOT, "research", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "paper6_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", out_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("NSCK-UPMA Paper 6 Benchmark Results")
    print("=" * 60)
    print(f"  Top-1 Accuracy:     {top1_acc:.4f} ({correct}/{n_test})")
    print(f"  Recall@10:          {recall_at_10:.4f}")
    print(f"  Spearman ρ:         {report.spearman_rho:.4f}")
    print(f"  HV/s:               {report.hv_per_second:.0f}")
    print(f"  Mean Latency:       {mean_latency:.1f}ms")
    print(f"  Target (<500ms):    {'✓ MET' if mean_latency < 500 else '✗ NOT MET'}")
    print(f"  Rust Backend:       {'Active' if report.rust_backend_active else 'Python fallback'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_paper6_benchmark()
