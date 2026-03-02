"""
NSCK V5 Societal Benchmarks.

Measures:
  - Concept registration speed
  - Query speed
  - Tick speed
  - Spectral RG time
  - TDA analysis time

Outputs JSON to nsck/benchmarks/societal_bench_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "nsck"))

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.emergence.spectral_rg import SpectralLaplacianRG
from python.core.societal.tda.tda_monitor import TDAHealthMonitor
from python.core.societal.navigation.societal_hnsw import SocietalHNSW


def _make_hv(seed: int):
    return hypervec_rs.HyperVector(seed=seed)


# Performance targets (ms)
_TARGETS = {
    "register_per_concept_ms": 10.0,
    "query_ms": 50.0,
    "tick_100_ms": 500.0,
    "spectral_rg_50_ms": 5000.0,
    "tda_analysis_ms": 10000.0,
}


def bench_registration(n: int = 100) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    t0 = time.perf_counter()
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_concept = elapsed_ms / n
    return {
        "name": "registration",
        "n_concepts": n,
        "total_ms": round(elapsed_ms, 3),
        "per_concept_ms": round(per_concept, 4),
        "target_per_concept_ms": _TARGETS["register_per_concept_ms"],
        "passed": per_concept < _TARGETS["register_per_concept_ms"],
    }


def bench_query(n: int = 50, n_queries: int = 100) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))
    query_hvs = [_make_hv(n + i) for i in range(n_queries)]

    t0 = time.perf_counter()
    for qhv in query_hvs:
        world.query(qhv, top_k=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_query = elapsed_ms / n_queries
    return {
        "name": "query",
        "n_concepts": n,
        "n_queries": n_queries,
        "total_ms": round(elapsed_ms, 3),
        "per_query_ms": round(per_query, 4),
        "target_per_query_ms": _TARGETS["query_ms"],
        "passed": per_query < _TARGETS["query_ms"],
    }


def bench_tick(n: int = 30, n_ticks: int = 100) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))
    for i in range(n - 1):
        world.concepts[f"c{i}"].add_bond(f"c{i+1}", float(rng.uniform(0.3, 0.9)))
    world.form_neighborhood([f"c{i}" for i in range(n // 2)], "nbhd_1")
    world.form_neighborhood([f"c{i}" for i in range(n // 2, n)], "nbhd_2")

    t0 = time.perf_counter()
    for _ in range(n_ticks):
        world.run_societal_tick()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "name": "tick",
        "n_concepts": n,
        "n_ticks": n_ticks,
        "total_ms": round(elapsed_ms, 3),
        "per_tick_ms": round(elapsed_ms / n_ticks, 4),
        "target_100_ticks_ms": _TARGETS["tick_100_ms"],
        "passed": elapsed_ms < _TARGETS["tick_100_ms"],
    }


def bench_spectral_rg(n: int = 50) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))

    rg = SpectralLaplacianRG(similarity_threshold=0.1)
    t0 = time.perf_counter()
    result = rg.run(world.concepts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "name": "spectral_rg",
        "n_concepts": n,
        "n_supernodes": result["n_supernodes"],
        "spectral_gap": result["spectral_gap"],
        "elapsed_ms": round(elapsed_ms, 3),
        "target_ms": _TARGETS["spectral_rg_50_ms"],
        "passed": elapsed_ms < _TARGETS["spectral_rg_50_ms"],
    }


def bench_tda(n: int = 30) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))
    for i in range(n - 1):
        world.concepts[f"c{i}"].add_bond(f"c{i+1}", float(rng.uniform(0.3, 0.9)))
        for _ in range(5):
            world.concepts[f"c{i}"].update_activation(float(rng.uniform(0.1, 1.0)))

    tda = TDAHealthMonitor(n_landmarks=15)
    t0 = time.perf_counter()
    analysis = tda.run_analysis(world.concepts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "name": "tda_analysis",
        "n_concepts": n,
        "n_landmarks": analysis["n_landmarks"],
        "health_score": analysis["health_score"],
        "elapsed_ms": round(elapsed_ms, 3),
        "target_ms": _TARGETS["tda_analysis_ms"],
        "passed": elapsed_ms < _TARGETS["tda_analysis_ms"],
    }


def bench_hnsw_build_query(n: int = 50) -> Dict[str, Any]:
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i))
    for i in range(0, n // 2 - 1):
        world.concepts[f"c{i}"].add_bond(f"c{i+1}", 0.6)
    n1 = world.form_neighborhood([f"c{i}" for i in range(n // 2)])
    n2 = world.form_neighborhood([f"c{i}" for i in range(n // 2, n)])
    world.form_domain([n1.neighborhood_id, n2.neighborhood_id], "test")
    for lhv in world.concepts.values():
        for _ in range(3):
            lhv.tick()

    hnsw = SocietalHNSW(M=8)
    t0 = time.perf_counter()
    hnsw.build(world)
    build_ms = (time.perf_counter() - t0) * 1000.0

    queries = [_make_hv(n + i) for i in range(20)]
    t0 = time.perf_counter()
    for qhv in queries:
        hnsw.query(qhv, top_k=5)
    query_ms = (time.perf_counter() - t0) * 1000.0 / 20

    return {
        "name": "hnsw",
        "n_concepts": n,
        "build_ms": round(build_ms, 3),
        "per_query_ms": round(query_ms, 4),
        "stats": hnsw.stats(),
    }


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 55)
    print("NSCK V5 Societal Benchmarks")
    print("=" * 55)

    t_total = time.perf_counter()
    results = []
    for bench_fn, label in [
        (bench_registration, "Registration  (100 concepts)"),
        (bench_query,        "Query         (50 concepts, 100 queries)"),
        (bench_tick,         "Tick          (30 concepts, 100 ticks)"),
        (bench_spectral_rg,  "Spectral RG   (50 concepts)"),
        (bench_tda,          "TDA Analysis  (30 concepts)"),
        (bench_hnsw_build_query, "HNSW Build+Query (50 concepts)"),
    ]:
        r = bench_fn()
        results.append(r)
        status = "✓" if r.get("passed", True) else "✗"
        key = "per_concept_ms" if "per_concept_ms" in r else (
              "per_query_ms" if "per_query_ms" in r else (
              "per_tick_ms" if "per_tick_ms" in r else (
              "elapsed_ms" if "elapsed_ms" in r else "total_ms")))
        print(f"  {status} {label}: {r.get(key, r.get('total_ms', 0)):.2f} ms")

    total_ms = (time.perf_counter() - t_total) * 1000.0
    n_pass = sum(1 for r in results if r.get("passed", True))
    print(f"\nPassed: {n_pass}/{len(results)} benchmarks")
    print(f"Total time: {total_ms:.1f} ms")

    report: Dict[str, Any] = {
        "suite": "NSCK V5 Societal Benchmarks",
        "version": "5.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_ms": round(total_ms, 2),
        "n_pass": n_pass,
        "n_total": len(results),
        "performance_targets": _TARGETS,
        "results": results,
    }
    return report


if __name__ == "__main__":
    report = run_benchmarks()
    out_dir = _REPO / "nsck" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "societal_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")
