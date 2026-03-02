"""
Paper 8 Societal Benchmarks — reproduces Table 4.1 from paper8_societal_hvs.md.

Usage:
    cd /path/to/repo && PYTHONPATH=nsck python research/experiments/paper8_societal_benchmarks.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "nsck"))

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.emergence.spectral_rg import SpectralLaplacianRG
from python.core.societal.tda.tda_monitor import TDAHealthMonitor
from python.core.societal.navigation.societal_hnsw import SocietalHNSW


def _make_hv(seed):
    return hypervec_rs.HyperVector(seed=seed)


def run_benchmarks():
    print("\nTable 4.1: NSCK V5 Performance Benchmarks")
    print("-" * 55)
    print(f"{'Operation':<40} {'World':>8} {'Time':>10}")
    print("-" * 55)

    rows = []

    # Registration
    world = SocietalKnowledgeWorld()
    t0 = time.perf_counter()
    for i in range(100):
        world.register_concept(f"c{i}", _make_hv(i))
    ms_per = (time.perf_counter() - t0) * 1000 / 100
    print(f"{'Register 100 concepts':<40} {'—':>8} {ms_per:.2f} ms/c")
    rows.append({"op": "register", "world": 100, "time_ms": round(ms_per, 3), "unit": "ms/concept"})

    # Query
    world = SocietalKnowledgeWorld()
    for i in range(50):
        world.register_concept(f"c{i}", _make_hv(i))
    t0 = time.perf_counter()
    for _ in range(100):
        world.query(_make_hv(99), top_k=10)
    ms_per = (time.perf_counter() - t0) * 1000 / 100
    print(f"{'Query (top-10)':<40} {'50':>8} {ms_per:.2f} ms")
    rows.append({"op": "query", "world": 50, "time_ms": round(ms_per, 3), "unit": "ms/query"})

    # Tick
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(30):
        world.register_concept(f"c{i}", _make_hv(i))
    for i in range(29):
        world.concepts[f"c{i}"].add_bond(f"c{i+1}", float(rng.uniform(0.3, 0.8)))
    world.form_neighborhood([f"c{i}" for i in range(15)])
    world.form_neighborhood([f"c{i}" for i in range(15, 30)])
    t0 = time.perf_counter()
    for _ in range(100):
        world.run_societal_tick()
    ms_per = (time.perf_counter() - t0) * 1000 / 100
    print(f"{'Societal tick':<40} {'30':>8} {ms_per:.2f} ms")
    rows.append({"op": "tick", "world": 30, "time_ms": round(ms_per, 3), "unit": "ms/tick"})

    # Spectral RG
    world = SocietalKnowledgeWorld()
    for i in range(50):
        world.register_concept(f"c{i}", _make_hv(i))
    rg = SpectralLaplacianRG(similarity_threshold=0.1)
    t0 = time.perf_counter()
    rg.run(world.concepts)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{'Spectral RG':<40} {'50':>8} {ms:.2f} ms")
    rows.append({"op": "spectral_rg", "world": 50, "time_ms": round(ms, 3), "unit": "ms"})

    # TDA
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(30):
        world.register_concept(f"c{i}", _make_hv(i))
        for _ in range(5):
            world.concepts[f"c{i}"].update_activation(float(rng.uniform(0.1, 1.0)))
    tda = TDAHealthMonitor(n_landmarks=15)
    t0 = time.perf_counter()
    tda.run_analysis(world.concepts)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{'TDA Analysis':<40} {'30':>8} {ms:.2f} ms")
    rows.append({"op": "tda_analysis", "world": 30, "time_ms": round(ms, 3), "unit": "ms"})

    # HNSW
    world = SocietalKnowledgeWorld()
    rng = np.random.default_rng(42)
    for i in range(50):
        world.register_concept(f"c{i}", _make_hv(i))
    for i in range(49):
        world.concepts[f"c{i}"].add_bond(f"c{i+1}", 0.6)
    world.form_neighborhood([f"c{i}" for i in range(25)])
    world.form_neighborhood([f"c{i}" for i in range(25, 50)])
    world.form_domain(
        [list(world.neighborhoods.keys())[0]], "dom_a"
    )
    for lhv in world.concepts.values():
        for _ in range(3):
            lhv.tick()
    hnsw = SocietalHNSW(M=8)
    hnsw.build(world)
    t0 = time.perf_counter()
    for i in range(20):
        hnsw.query(_make_hv(100 + i), top_k=5)
    ms = (time.perf_counter() - t0) * 1000 / 20
    print(f"{'HNSW Query':<40} {'50':>8} {ms:.2f} ms")
    rows.append({"op": "hnsw_query", "world": 50, "time_ms": round(ms, 3), "unit": "ms/query"})

    print("-" * 55)

    # Save
    out_dir = _REPO / "research" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper8_bench_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "paper": "paper8_societal_hvs",
            "table": "4.1",
            "rows": rows,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmarks()
