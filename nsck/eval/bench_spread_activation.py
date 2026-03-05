"""Benchmark spreading activation at different graph scales.

V18 update: compares Rust parallel_spread_activation vs Python fallback.
"""
from __future__ import annotations
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def build_graph(n_nodes: int, n_edges_per_node: int = 3, use_rust: bool = False):
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory(use_rust=use_rust)
    for i in range(n_nodes):
        mem.add_concept(f"concept_{i}", {"index": i})
    import random
    random.seed(42)
    for i in range(n_nodes):
        for _ in range(n_edges_per_node):
            j = random.randint(0, n_nodes - 1)
            if i != j:
                mem.add_relation(f"concept_{i}", "similar_to", f"concept_{j}")
    return mem


def benchmark_scale(n_nodes: int, use_rust: bool = False) -> dict:
    mem = build_graph(n_nodes, use_rust=use_rust)

    # Warm-up
    mem.spread_activation(["concept_0"], steps=3, decay=0.7)

    # Measure
    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        result = mem.spread_activation(["concept_0", "concept_1"], steps=3, decay=0.7)
    elapsed = (time.perf_counter() - start) / runs

    backend = "rust" if use_rust and mem._rust_backend is not None else "python"
    return {
        "n_nodes": n_nodes,
        "backend": backend,
        "latency_ms": elapsed * 1000,
        "activated_count": len(result),
    }


def main():
    import python.core.vsa.hypervec_shim as hv_mod
    rust_available = hv_mod.__backend__ == "Rust"

    scales = [100, 1_000, 10_000]

    # Python benchmarks
    py_results = {}
    print("\nPython backend:")
    print(f"{'Nodes':>8}  {'Latency (ms)':>14}  {'Activated':>10}")
    print("-" * 40)
    for n in scales:
        r = benchmark_scale(n, use_rust=False)
        py_results[n] = r["latency_ms"]
        print(f"{r['n_nodes']:>8}  {r['latency_ms']:>14.2f}  {r['activated_count']:>10}")

    # Rust benchmarks (if available)
    if rust_available:
        print("\nRust backend (parallel_spread_activation):")
        print(f"{'Nodes':>8}  {'Latency (ms)':>14}  {'Activated':>10}  {'Speedup':>9}")
        print("-" * 50)
        for n in scales:
            r = benchmark_scale(n, use_rust=True)
            speedup = py_results[n] / r["latency_ms"] if r["latency_ms"] > 0 else float("inf")
            print(
                f"{r['n_nodes']:>8}  {r['latency_ms']:>14.2f}  "
                f"{r['activated_count']:>10}  {speedup:>8.1f}x"
            )
    else:
        print("\nRust backend not available (compile with: cd nsck/rust_vsa && maturin develop --release)")

    print()


if __name__ == "__main__":
    main()

