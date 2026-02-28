"""
Paper 7 Experiments: Homeostatic Memory Architecture
=====================================================
Run from: cd nsck && python ../research/experiments/paper7_memory_benchmarks.py

Writes results to research/results/paper7_results.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

_here = Path(__file__).resolve()
_nsck_dir = _here.parent.parent.parent / "nsck"
if str(_nsck_dir) not in sys.path:
    sys.path.insert(0, str(_nsck_dir))

RESULTS_DIR = _here.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "paper7_results.json"

rust_vsa_active = False
try:
    import hypervec_rs  # noqa: F401
    rust_vsa_active = True
except ImportError:
    pass
print(f"[Paper 7] Rust VSA backend: {'ACTIVE' if rust_vsa_active else 'Python fallback'}")

try:
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
    from python.core.memory.procedural_memory import ProceduralMemory
    from python.core.memory.concept_drift_detector import ConceptDriftDetector
    from python.core.memory.homeostasis import MemoryHomeostasis
    from python.core.memory.cross_modal_associative_memory import CrossModalAssociativeMemory
    from python.core.vsa.hypervec_shim import HyperVector
    print("[Paper 7] Memory modules imported OK")
except ImportError as e:
    print(f"[Paper 7] ERROR: {e}")
    sys.exit(1)

results = {"backend": {"rust_vsa": rust_vsa_active}}


# ── Experiment 7.1 — Semantic Memory Scaling ─────────────────────────────────
def exp_7_1():
    print("\n[Exp 7.1] Semantic Memory Scaling...")
    rng = np.random.default_rng(42)
    n_concept_levels = [100, 500, 1000, 2000]
    n_queries = 100
    scaling_results = []

    for n_concepts in n_concept_levels:
        mem = SemanticMemory()
        for i in range(n_concepts):
            mem.add_concept(f"concept_{i}", {"type": "test", "index": i})

        query_seeds = rng.integers(0, n_concepts, size=n_queries)
        latencies = []
        for seed in query_seeds:
            qhv = HyperVector(seed=int(seed))
            t0 = time.perf_counter()
            mem.query(qhv, k=5)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = float(np.mean(latencies))
        scaling_results.append({"n_concepts": n_concepts, "avg_query_ms": avg_ms})
        print(f"  n_concepts={n_concepts}  avg_query_ms={avg_ms:.4f}")

    return {"semantic_memory_scaling": scaling_results}


# ── Experiment 7.2 — Episodic Memory Recall ──────────────────────────────────
def exp_7_2():
    print("\n[Exp 7.2] Episodic Memory Recall...")
    rng = np.random.default_rng(7)
    mem = EpisodicMemory(use_rust=False)
    n_episodes = 500
    tag = "test_task"

    hvs = [HyperVector(seed=int(i)) for i in range(n_episodes)]
    for i in range(n_episodes):
        ep = LiveEpisode(
            timestamp=float(i),
            task_tag=tag,
            situation_hv=hvs[i],
            state={"idx": i},
            action=f"act_{i % 10}",
            outcome="ok",
            reward=float(rng.uniform(-1, 1)),
        )
        mem.record(ep)

    n_queries = 50
    query_indices = rng.choice(n_episodes, size=n_queries, replace=False)
    latencies = []
    top5_sims = []

    for qi in query_indices:
        qhv = hvs[qi]
        t0 = time.perf_counter()
        results_ep = mem.recall_similar(qhv, tag, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        if results_ep:
            sims = [float(ep.situation_hv.similarity(qhv)) for ep in results_ep]
            top5_sims.append(float(np.mean(sims)))

    avg_recall_ms = float(np.mean(latencies))
    mean_sim = float(np.mean(top5_sims)) if top5_sims else 0.0
    print(f"  avg_recall_ms={avg_recall_ms:.3f}  mean_similarity_top5={mean_sim:.4f}")
    return {
        "episodic_recall": {
            "n_episodes": n_episodes,
            "n_queries": n_queries,
            "avg_recall_ms": avg_recall_ms,
            "mean_similarity_of_top5": mean_sim,
        }
    }


# ── Experiment 7.3 — Procedural Memory Fast-Path ─────────────────────────────
def exp_7_3():
    print("\n[Exp 7.3] Procedural Memory Fast-Path...")
    rng = np.random.default_rng(3)
    mem = ProceduralMemory(max_skills=200, familiarity_threshold=0.7)
    n_skills = 50

    skill_hvs = [HyperVector(seed=int(i + 100)) for i in range(n_skills)]
    for i, hv in enumerate(skill_hvs):
        mem.cache_skill(hv, action=f"action_{i}", reward=float(rng.uniform(0, 1)))

    # 50 matching queries (known contexts), 50 novel queries
    n_hit_queries = 50
    n_miss_queries = 50
    latencies = []
    hits = 0

    for i in range(n_hit_queries):
        hv = skill_hvs[i % n_skills]
        t0 = time.perf_counter()
        res = mem.recall_action(hv)
        latencies.append((time.perf_counter() - t0) * 1000)
        if res is not None:
            hits += 1

    for i in range(n_miss_queries):
        hv = HyperVector(seed=int(i + 10000))  # Novel context
        t0 = time.perf_counter()
        res = mem.recall_action(hv)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = mem.get_statistics()
    hit_rate = hits / n_hit_queries
    avg_ms = float(np.mean(latencies))
    print(f"  hit_rate={hit_rate:.3f}  avg_lookup_ms={avg_ms:.4f}  stats_hit_rate={stats['hit_rate']:.3f}")
    return {
        "procedural_fast_path": {
            "n_skills_cached": n_skills,
            "hit_rate": hit_rate,
            "avg_lookup_ms": avg_ms,
            "library_hit_rate": stats["hit_rate"],
        }
    }


# ── Experiment 7.4 — Concept Drift Detection ─────────────────────────────────
def exp_7_4():
    print("\n[Exp 7.4] Concept Drift Detection...")
    rng = np.random.default_rng(5)
    n_concepts = 20
    n_drifting = 10
    detector = ConceptDriftDetector(drift_threshold=0.2)

    # Create and snapshot 20 concepts
    original_hvs = [HyperVector(seed=int(i + 300)) for i in range(n_concepts)]
    concept_names = [f"concept_{i}" for i in range(n_concepts)]
    for name, hv in zip(concept_names, original_hvs):
        detector.snapshot(name, hv)

    # Drift detection at various flip rates
    flip_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    drift_results = []

    for flip_rate in flip_rates:
        # Reset and re-snapshot
        detector2 = ConceptDriftDetector(drift_threshold=0.2)
        for name, hv in zip(concept_names, original_hvs):
            detector2.snapshot(name, hv)

        alarms = 0
        for i in range(n_concepts):
            bits = np.array(original_hvs[i].bits, dtype=np.int8).copy()
            if i < n_drifting and flip_rate > 0:
                # Flip a fraction of bits
                n_flip = int(len(bits) * flip_rate)
                flip_idx = rng.choice(len(bits), size=n_flip, replace=False)
                bits[flip_idx] = 1 - bits[flip_idx]
                drifted_hv = HyperVector.from_bits(bits)
            else:
                drifted_hv = original_hvs[i]
            event = detector2.check(concept_names[i], drifted_hv)
            if event.alarm:
                alarms += 1

        alarm_rate = alarms / n_concepts
        drift_results.append({"flip_rate": flip_rate, "alarm_rate": alarm_rate, "n_alarms": alarms})
        print(f"  flip_rate={flip_rate:.2f}  alarm_rate={alarm_rate:.3f}  alarms={alarms}/{n_concepts}")

    return {"concept_drift_detection": drift_results}


# ── Experiment 7.5 — Homeostatic Regulation ──────────────────────────────────
def exp_7_5():
    print("\n[Exp 7.5] Homeostatic Regulation...")
    rng = np.random.default_rng(11)
    n_concepts_init = 200
    mem = SemanticMemory()
    homeostasis = MemoryHomeostasis()

    # Add concepts and random edges
    concepts = [f"node_{i}" for i in range(n_concepts_init)]
    for name in concepts:
        mem.add_concept(name, {"val": rng.integers(0, 100).item()})

    # Add random edges
    for _ in range(600):
        a, b = rng.choice(n_concepts_init, size=2, replace=False)
        mem.add_relation(concepts[a], "related_to", concepts[b])

    n_cycles = 10
    regulation_trace = []
    for cycle in range(n_cycles):
        actions = homeostasis.regulate(mem)
        density = homeostasis._measure_edge_density(mem)
        n_curr = len(mem.concept_graph.nodes())
        regulation_trace.append({
            "cycle": cycle,
            "edge_density": density,
            "n_concepts": n_curr,
            "actions_taken": len(actions),
        })
        print(f"  cycle={cycle}  density={density:.4f}  n_concepts={n_curr}  actions={len(actions)}")

    return {"homeostatic_regulation": regulation_trace}


# ── Experiment 7.6 — Cross-Modal Binding ─────────────────────────────────────
def exp_7_6():
    print("\n[Exp 7.6] Cross-Modal Binding...")
    rng = np.random.default_rng(13)
    mem = CrossModalAssociativeMemory()
    n_pairs = 20

    visual_hvs = [HyperVector(seed=int(i + 400)) for i in range(n_pairs)]
    audio_hvs = [HyperVector(seed=int(i + 500)) for i in range(n_pairs)]

    for i in range(n_pairs):
        mem.bind(
            modality_a="visual",
            hv_a=visual_hvs[i],
            modality_b="audio",
            hv_b=audio_hvs[i],
            label=f"pair_{i}",
        )

    correct = 0
    sims = []
    for i in range(n_pairs):
        results_recall = mem.recall(
            query_hv=visual_hvs[i],
            from_modality="visual",
            to_modality="audio",
            top_k=1,
        )
        if results_recall:
            recalled_hv, strength, label = results_recall[0]
            sim = float(recalled_hv.similarity(audio_hvs[i]))
            sims.append(sim)
            if sim > 0.9:
                correct += 1

    recall_accuracy = correct / n_pairs
    avg_sim = float(np.mean(sims)) if sims else 0.0
    print(f"  recall_accuracy={recall_accuracy:.3f}  avg_similarity={avg_sim:.4f}")
    return {
        "cross_modal_binding": {
            "n_pairs": n_pairs,
            "recall_accuracy": recall_accuracy,
            "avg_similarity": avg_sim,
        }
    }


# ── Experiment 7.7 — Spreading Activation ────────────────────────────────────
def exp_7_7():
    print("\n[Exp 7.7] Spreading Activation...")
    rng = np.random.default_rng(17)
    n_concepts = 100
    mem = SemanticMemory()
    concept_names = [f"node_{i}" for i in range(n_concepts)]
    for name in concept_names:
        mem.add_concept(name, {"val": 0})

    # Add edges
    for _ in range(300):
        a, b = rng.choice(n_concepts, size=2, replace=False)
        mem.add_relation(concept_names[a], "related_to", concept_names[b])

    start_nodes = [concept_names[0], concept_names[25], concept_names[50]]
    activation_map = mem.spread_activation(start_concepts=start_nodes, steps=3, decay=0.7)

    activations = list(activation_map.values())
    n_activated = sum(1 for v in activations if v > 0.01)
    result = {
        "n_start_nodes": len(start_nodes),
        "n_activated_nodes": n_activated,
        "mean_activation": float(np.mean(activations)) if activations else 0.0,
        "max_activation": float(np.max(activations)) if activations else 0.0,
        "activation_distribution": {
            "p25": float(np.percentile(activations, 25)) if activations else 0.0,
            "p50": float(np.percentile(activations, 50)) if activations else 0.0,
            "p75": float(np.percentile(activations, 75)) if activations else 0.0,
        },
    }
    print(f"  n_activated={n_activated}/{n_concepts}  mean={result['mean_activation']:.4f}")
    return {"spreading_activation": result}


# ── Run all experiments ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, fn in [
        ("exp_7_1", exp_7_1),
        ("exp_7_2", exp_7_2),
        ("exp_7_3", exp_7_3),
        ("exp_7_4", exp_7_4),
        ("exp_7_5", exp_7_5),
        ("exp_7_6", exp_7_6),
        ("exp_7_7", exp_7_7),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"[{name}] FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Paper 7] Results written to {RESULTS_FILE}")
