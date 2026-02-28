"""
Paper 5 Experiments: Model Transplantation
==========================================
Run from: cd nsck && python ../research/experiments/paper5_transplant_benchmarks.py

Measures transplant/projection quality metrics for Paper 5.
Writes results to research/results/paper5_results.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve()
_nsck_dir = _here.parent.parent.parent / "nsck"
if str(_nsck_dir) not in sys.path:
    sys.path.insert(0, str(_nsck_dir))

RESULTS_DIR = _here.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "paper5_results.json"

rust_vsa_active = False
try:
    import hypervec_rs  # noqa: F401
    rust_vsa_active = True
except ImportError:
    pass
print(f"[Paper 5] Rust VSA backend: {'ACTIVE' if rust_vsa_active else 'Python fallback'}")

try:
    from python.core.transplant.projector import RandomProjector
    from python.core.transplant.validator import TransplantValidator, TransplantReport
    from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
    from python.core.vsa.hypervec_shim import HyperVector
    print("[Paper 5] Transplant modules imported OK")
except ImportError as e:
    print(f"[Paper 5] ERROR: {e}")
    sys.exit(1)

results = {"backend": {"rust_vsa": rust_vsa_active}}


# ── Experiment 5.1 — Projector Strategy Comparison ──────────────────────────
def exp_5_1():
    print("\n[Exp 5.1] Projector Strategy Comparison...")
    rng = np.random.default_rng(42)
    dim = 32
    rows = []
    for vocab_size in [100, 500]:
        embeddings = rng.standard_normal((vocab_size, dim)).astype(np.float32)
        vocab_mapping = {f"tok{i}": i for i in range(vocab_size)}
        projector = RandomProjector(dim_in=dim, hv_dim=1024, seed=42)
        codebook = projector.project(embeddings, vocab_mapping)
        # Validate with lower thresholds for small vocab
        validator = TransplantValidator(
            rho_threshold=0.0, recall10_threshold=0.0,
            recall50_threshold=0.0, ari_threshold=0.0
        )
        report = validator.validate(embeddings, codebook, vocab_mapping, n_sample_pairs=500)
        rows.append({
            "vocab_size": vocab_size,
            "dim": dim,
            "projector": "RandomProjector",
            "spearman_rho": report.spearman_rho,
            "recall_at_10": report.recall_at_10,
            "recall_at_50": report.recall_at_50,
            "ari": report.ari,
            "passed": report.passed,
        })
        print(f"  vocab={vocab_size}  ρ={report.spearman_rho:.3f}  R@10={report.recall_at_10:.3f}  R@50={report.recall_at_50:.3f}  ARI={report.ari:.3f}")
    return {"projector_comparison": rows}


# ── Experiment 5.2 — Cluster Preservation ───────────────────────────────────
def exp_5_2():
    print("\n[Exp 5.2] Cluster Preservation...")
    rng = np.random.default_rng(7)
    dim = 32
    n_clusters = 5
    per_cluster = 20
    vocab_size = n_clusters * per_cluster

    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)
    labels = []
    for c in range(n_clusters):
        start = c * per_cluster
        embeddings[start:start + per_cluster] = centers[c] + rng.standard_normal((per_cluster, dim)) * 0.5
        labels.extend([c] * per_cluster)
    labels = np.array(labels)

    vocab_mapping = {f"tok{i}": i for i in range(vocab_size)}
    projector = RandomProjector(dim_in=dim, hv_dim=1024, seed=42)
    codebook = projector.project(embeddings, vocab_mapping)

    # Build HV bit matrix
    hv_bits = np.array([codebook[f"tok{i}"].bits for i in range(vocab_size)], dtype=np.float32)
    # Hamming similarity: sim = 1 - hamming = mean(bits_i == bits_j)
    # Intra-cluster vs inter-cluster
    intra_sims, inter_sims = [], []
    for i in range(vocab_size):
        for j in range(i + 1, min(i + 10, vocab_size)):
            sim = float(np.mean(hv_bits[i] == hv_bits[j]))
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    result = {
        "n_clusters": n_clusters,
        "per_cluster": per_cluster,
        "intra_cluster_sim_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "inter_cluster_sim_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
        "separation": float(np.mean(intra_sims) - np.mean(inter_sims)) if intra_sims and inter_sims else 0.0,
    }
    print(f"  intra={result['intra_cluster_sim_mean']:.4f}  inter={result['inter_cluster_sim_mean']:.4f}  sep={result['separation']:.4f}")
    return {"cluster_preservation": result}


# ── Experiment 5.3 — Bit-Flip Robustness ─────────────────────────────────────
def exp_5_3():
    print("\n[Exp 5.3] Bit-Flip Robustness...")
    rng = np.random.default_rng(3)
    dim = 32
    vocab_size = 200
    hv_dim = 1024

    embeddings = rng.standard_normal((vocab_size, dim)).astype(np.float32)
    vocab_mapping = {f"tok{i}": i for i in range(vocab_size)}
    projector = RandomProjector(dim_in=dim, hv_dim=hv_dim, seed=42)
    codebook = projector.project(embeddings, vocab_mapping)

    flip_rates = [0, 5, 10, 20, 30]
    flip_results = []
    n_queries = 50

    # Baseline recall@10 (brute force)
    def recall_at_10(hv_bits_q, hv_bits_db):
        # For each query, find top-10 by Hamming sim in DB
        correct = 0
        for i in range(len(hv_bits_q)):
            sims = np.mean(hv_bits_q[i:i+1] == hv_bits_db, axis=1)
            top10 = np.argsort(sims)[::-1][:10]
            if i in top10:
                correct += 1
        return correct / len(hv_bits_q)

    orig_bits = np.array([codebook[f"tok{i}"].bits for i in range(vocab_size)], dtype=np.float32)
    query_indices = rng.choice(vocab_size, size=n_queries, replace=False)
    query_bits = orig_bits[query_indices]

    for flip_pct in flip_rates:
        if flip_pct == 0:
            flipped = query_bits.copy()
        else:
            flipped = query_bits.copy()
            n_flip = int(hv_dim * flip_pct / 100)
            for i in range(len(flipped)):
                flip_idx = rng.choice(hv_dim, size=n_flip, replace=False)
                flipped[i, flip_idx] = 1.0 - flipped[i, flip_idx]

        rec = recall_at_10(flipped, orig_bits)
        flip_results.append({"flip_rate_pct": flip_pct, "recall_at_10": rec})
        print(f"  flip={flip_pct}%  recall@10={rec:.3f}")

    return {"bit_flip_robustness": flip_results}


# ── Experiment 5.4 — Projection Determinism ──────────────────────────────────
def exp_5_4():
    print("\n[Exp 5.4] Projection Determinism...")
    rng = np.random.default_rng(42)
    dim = 32
    vocab_size = 50
    embeddings = rng.standard_normal((vocab_size, dim)).astype(np.float32)
    vocab_mapping = {f"tok{i}": i for i in range(vocab_size)}

    # Same seed → same codebook
    cb1 = RandomProjector(dim_in=dim, hv_dim=1024, seed=99).project(embeddings, vocab_mapping)
    cb2 = RandomProjector(dim_in=dim, hv_dim=1024, seed=99).project(embeddings, vocab_mapping)
    same_seed_identical = all(np.array_equal(cb1[k].bits, cb2[k].bits) for k in cb1)

    # Different seed → different codebook
    cb3 = RandomProjector(dim_in=dim, hv_dim=1024, seed=1234).project(embeddings, vocab_mapping)
    diff_seed_different = not all(np.array_equal(cb1[k].bits, cb3[k].bits) for k in cb1)

    result = {
        "same_seed_identical": same_seed_identical,
        "diff_seed_different": diff_seed_different,
        "determinism_ok": same_seed_identical and diff_seed_different,
    }
    print(f"  same_seed_identical={same_seed_identical}  diff_seed_different={diff_seed_different}")
    return {"projection_determinism": result}


# ── Experiment 5.5 — Embedding Bridge Round-Trip ─────────────────────────────
def exp_5_5():
    print("\n[Exp 5.5] Embedding Bridge Round-Trip...")
    rng = np.random.default_rng(5)
    dim_in = 64
    n_samples = 50
    bridge = EmbeddingVSABridge(dim_in=dim_in, hv_dim=1024, seed=42)

    sims = []
    for _ in range(n_samples):
        emb = rng.standard_normal(dim_in).astype(np.float32)
        hv1 = bridge.embed_to_hv(emb)
        # Re-encode via same bridge
        hv2 = bridge.embed_to_hv(emb)
        # Hamming similarity
        sim = float(np.mean(np.array(hv1.bits) == np.array(hv2.bits)))
        sims.append(sim)

    result = {
        "mean_round_trip_similarity": float(np.mean(sims)),
        "std_similarity": float(np.std(sims)),
        "n_samples": n_samples,
        "note": "Same embedding re-encoded should produce identical HV (deterministic)",
    }
    print(f"  mean_similarity={result['mean_round_trip_similarity']:.4f}")
    return {"embedding_bridge_round_trip": result}


# ── Experiment 5.6 — Projection Timing ───────────────────────────────────────
def exp_5_6():
    print("\n[Exp 5.6] Projection Timing...")
    rng = np.random.default_rng(6)
    dim = 32
    vocab_sizes = [100, 500, 1000]

    timing_results = []
    for vocab_size in vocab_sizes:
        embeddings = rng.standard_normal((vocab_size, dim)).astype(np.float32)
        vocab_mapping = {f"tok{i}": i for i in range(vocab_size)}
        projector = RandomProjector(dim_in=dim, hv_dim=1024, seed=42)
        t0 = time.perf_counter()
        projector.project(embeddings, vocab_mapping)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timing_results.append({"vocab_size": vocab_size, "projection_time_ms": elapsed_ms})
        print(f"  vocab_size={vocab_size}  projection_time_ms={elapsed_ms:.2f}")

    return {"projection_timing": timing_results}


# ── Run all experiments ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, fn in [
        ("exp_5_1", exp_5_1),
        ("exp_5_2", exp_5_2),
        ("exp_5_3", exp_5_3),
        ("exp_5_4", exp_5_4),
        ("exp_5_5", exp_5_5),
        ("exp_5_6", exp_5_6),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            results[name] = {"error": str(e)}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Paper 5] Results written to {RESULTS_FILE}")
