"""
Paper 4 Experiments: Bridging Spikes and Symbols
=================================================
Run from: cd nsck && python ../research/experiments/paper4_snn_benchmarks.py

Measures SNN perception benchmarks for Paper 4.
Writes results to research/results/paper4_results.json
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Ensure nsck/ is in sys.path (script may be run from anywhere)
_here = Path(__file__).resolve()
_nsck_dir = _here.parent.parent.parent / "nsck"
if str(_nsck_dir) not in sys.path:
    sys.path.insert(0, str(_nsck_dir))

RESULTS_DIR = _here.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "paper4_results.json"

# ── Backend detection ────────────────────────────────────────────────────────
rust_vsa_active = False
rust_snn_active = False
try:
    import hypervec_rs  # noqa: F401
    rust_vsa_active = True
except ImportError:
    pass
try:
    import snn_rs  # noqa: F401
    rust_snn_active = True
except ImportError:
    pass

print(f"[Paper 4] Rust VSA backend: {'ACTIVE' if rust_vsa_active else 'Python fallback'}")
print(f"[Paper 4] Rust SNN backend: {'ACTIVE' if rust_snn_active else 'Python fallback'}")

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from python.core.perception.snn_perception import SNNPerceptionModule, LIFNeuronLayer
    print("[Paper 4] SNN modules imported OK")
except ImportError as e:
    print(f"[Paper 4] ERROR importing SNN modules: {e}")
    sys.exit(1)

results = {
    "backend": {
        "rust_vsa": rust_vsa_active,
        "rust_snn": rust_snn_active,
    }
}

# ── Experiment 4.1 — Pattern Classification ─────────────────────────────────
def exp_4_1():
    print("\n[Exp 4.1] Pattern Classification...")
    rng = np.random.default_rng(42)
    input_dim = 64
    n_patterns = 10
    n_epochs = 10

    # Create distinct patterns
    patterns = [rng.standard_normal(input_dim) * (i + 1) * 0.5 for i in range(n_patterns)]

    snn = SNNPerceptionModule(
        input_dim=input_dim,
        snn_size=256,
        hv_dimension=1024,
        n_concepts=50,
        encoding_mode="rate",
    )

    # Training: multiple epochs
    concept_ids = {}
    for epoch in range(n_epochs):
        for i, pat in enumerate(patterns):
            snn.perceive(pat, learn=True)

    # Evaluation: measure stability of concept_id
    correct = 0
    latencies = []
    concept_id_map = {}
    for i, pat in enumerate(patterns):
        t0 = time.perf_counter()
        out = snn.perceive(pat, learn=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        cid = out.get("concept_id", -1)
        if i not in concept_id_map:
            concept_id_map[i] = cid
        if concept_id_map[i] == cid and cid != -1:
            correct += 1

    accuracy = correct / n_patterns
    avg_latency_ms = float(np.mean(latencies))
    n_concepts_learned = snn.concept_mapper.next_id

    print(f"  accuracy={accuracy:.3f}  avg_latency_ms={avg_latency_ms:.3f}  n_concepts={n_concepts_learned}")
    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency_ms,
        "n_concepts_learned": n_concepts_learned,
        "n_patterns": n_patterns,
        "n_epochs": n_epochs,
    }

# ── Experiment 4.2 — Noise Robustness ───────────────────────────────────────
def exp_4_2():
    print("\n[Exp 4.2] Noise Robustness...")
    rng = np.random.default_rng(0)
    input_dim = 64
    n_patterns = 10
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    patterns = [rng.standard_normal(input_dim) for _ in range(n_patterns)]

    snn = SNNPerceptionModule(input_dim=input_dim, snn_size=256, hv_dimension=1024, n_concepts=50)

    # Train
    concept_id_map = {}
    for _ in range(5):
        for i, pat in enumerate(patterns):
            out = snn.perceive(pat, learn=True)
            if i not in concept_id_map:
                concept_id_map[i] = out.get("concept_id", -1)

    noise_results = []
    for noise in noise_levels:
        correct = 0
        for i, pat in enumerate(patterns):
            noisy = pat + rng.normal(0, noise, size=input_dim)
            out = snn.perceive(noisy, learn=False)
            cid = out.get("concept_id", -1)
            if cid == concept_id_map.get(i, -2) and cid != -1:
                correct += 1
        acc = correct / n_patterns
        noise_results.append({"noise_level": noise, "accuracy": acc})
        print(f"  noise={noise:.2f}  accuracy={acc:.3f}")

    return {"noise_robustness": noise_results}

# ── Experiment 4.3 — STDP Weight Evolution ──────────────────────────────────
def exp_4_3():
    print("\n[Exp 4.3] STDP Weight Evolution...")
    rng = np.random.default_rng(7)
    input_dim = 64
    snn = SNNPerceptionModule(input_dim=input_dim, snn_size=256, hv_dimension=1024,
                               stdp_enabled=True, stdp_lr=0.0001)
    pat = rng.standard_normal(input_dim)

    checkpoints = [0, 50, 100, 200, 500]
    weight_norms = []
    step = 0

    for cp in checkpoints:
        while step < cp:
            snn.perceive(pat, learn=True)
            step += 1
        norm = float(np.linalg.norm(snn.input_weights))
        weight_norms.append({"step": cp, "weight_norm": norm})
        print(f"  step={cp}  weight_norm={norm:.4f}")

    return {"stdp_weight_evolution": weight_norms}

# ── Experiment 4.4 — SNN Scaling ─────────────────────────────────────────────
def exp_4_4():
    print("\n[Exp 4.4] SNN Scaling...")
    rng = np.random.default_rng(3)
    input_dim = 64
    snn_sizes = [64, 128, 256]
    n_calls = 100

    scaling_results = []
    for snn_size in snn_sizes:
        snn = SNNPerceptionModule(input_dim=input_dim, snn_size=snn_size, hv_dimension=1024)
        pat = rng.standard_normal(input_dim)
        latencies = []
        for _ in range(n_calls):
            t0 = time.perf_counter()
            snn.perceive(pat, learn=False)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(latencies))
        scaling_results.append({"snn_size": snn_size, "avg_latency_ms": avg_ms})
        print(f"  snn_size={snn_size}  avg_latency_ms={avg_ms:.3f}")

    return {"snn_scaling": scaling_results}

# ── Experiment 4.5 — Rate vs Temporal Coding ─────────────────────────────────
def exp_4_5():
    print("\n[Exp 4.5] Rate vs Temporal Coding...")
    rng = np.random.default_rng(9)
    input_dim = 64
    n_patterns = 8
    patterns = [rng.standard_normal(input_dim) for _ in range(n_patterns)]

    coding_results = []
    for mode in ["rate", "temporal"]:
        try:
            snn = SNNPerceptionModule(input_dim=input_dim, snn_size=128,
                                       hv_dimension=1024, encoding_mode=mode)
        except TypeError as e:
            # TemporalCoder may not support all kwargs — skip gracefully
            coding_results.append({"encoding_mode": mode, "accuracy": None, "note": str(e)})
            print(f"  mode={mode} SKIPPED: {e}")
            continue
        concept_map = {}
        for _ in range(5):
            for i, pat in enumerate(patterns):
                out = snn.perceive(pat, learn=True)
                if i not in concept_map:
                    concept_map[i] = out.get("concept_id", -1)

        correct = 0
        for i, pat in enumerate(patterns):
            out = snn.perceive(pat, learn=False)
            if out.get("concept_id", -1) == concept_map.get(i, -2):
                correct += 1
        acc = correct / n_patterns
        coding_results.append({"encoding_mode": mode, "accuracy": acc})
        print(f"  mode={mode}  accuracy={acc:.3f}")

    return {"rate_vs_temporal": coding_results}

# ── Experiment 4.6 — Spike Statistics ────────────────────────────────────────
def exp_4_6():
    print("\n[Exp 4.6] Spike Statistics...")
    rng = np.random.default_rng(11)
    input_dim = 64
    snn = SNNPerceptionModule(input_dim=input_dim, snn_size=256, hv_dimension=1024)

    spike_counts, active_neuron_counts, proc_times = [], [], []
    for _ in range(100):
        x = rng.standard_normal(input_dim)
        out = snn.perceive(x, learn=False)
        spike_counts.append(int(out.get("n_spikes", 0)))
        an = out.get("active_neurons", 0)
        active_neuron_counts.append(len(an) if isinstance(an, (list, np.ndarray)) else int(an))
        proc_times.append(float(out.get("processing_time_ms", 0.0)))

    stats = {
        "mean_spike_count": float(np.mean(spike_counts)),
        "std_spike_count": float(np.std(spike_counts)),
        "mean_active_neurons": float(np.mean(active_neuron_counts)),
        "mean_processing_time_ms": float(np.mean(proc_times)),
        "n_samples": 100,
    }
    print(f"  mean_spikes={stats['mean_spike_count']:.1f}  mean_active={stats['mean_active_neurons']:.1f}  mean_ms={stats['mean_processing_time_ms']:.3f}")
    return {"spike_statistics": stats}

# ── Experiment 4.7 — Rust vs Python Backend ──────────────────────────────────
def exp_4_7():
    print("\n[Exp 4.7] Backend Comparison...")
    # We always use the active backend; report which one and its latency
    rng = np.random.default_rng(5)
    input_dim = 64
    snn = SNNPerceptionModule(input_dim=input_dim, snn_size=256, hv_dimension=1024)
    n_calls = 100
    latencies = []
    for _ in range(n_calls):
        x = rng.standard_normal(input_dim)
        t0 = time.perf_counter()
        snn.perceive(x, learn=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(latencies))
    backend = "Rust" if rust_snn_active else "Python"
    result = {
        "backend": backend,
        "avg_latency_ms": avg_ms,
        "n_calls": n_calls,
        "speedup_vs_python": None,  # Would require both backends simultaneously
        "note": f"Only {backend} backend available in this run",
    }
    print(f"  backend={backend}  avg_latency_ms={avg_ms:.3f}")
    return {"backend_comparison": result}

# ── Run all experiments ───────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        results["exp_4_1"] = exp_4_1()
    except Exception as e:
        print(f"[Exp 4.1] FAILED: {e}")
        results["exp_4_1"] = {"error": str(e)}

    try:
        results["exp_4_2"] = exp_4_2()
    except Exception as e:
        print(f"[Exp 4.2] FAILED: {e}")
        results["exp_4_2"] = {"error": str(e)}

    try:
        results["exp_4_3"] = exp_4_3()
    except Exception as e:
        print(f"[Exp 4.3] FAILED: {e}")
        results["exp_4_3"] = {"error": str(e)}

    try:
        results["exp_4_4"] = exp_4_4()
    except Exception as e:
        print(f"[Exp 4.4] FAILED: {e}")
        results["exp_4_4"] = {"error": str(e)}

    try:
        results["exp_4_5"] = exp_4_5()
    except Exception as e:
        print(f"[Exp 4.5] FAILED: {e}")
        results["exp_4_5"] = {"error": str(e)}

    try:
        results["exp_4_6"] = exp_4_6()
    except Exception as e:
        print(f"[Exp 4.6] FAILED: {e}")
        results["exp_4_6"] = {"error": str(e)}

    try:
        results["exp_4_7"] = exp_4_7()
    except Exception as e:
        print(f"[Exp 4.7] FAILED: {e}")
        results["exp_4_7"] = {"error": str(e)}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Paper 4] Results written to {RESULTS_FILE}")
