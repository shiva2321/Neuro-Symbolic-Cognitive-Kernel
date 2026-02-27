"""Benchmark spreading activation at different graph scales."""
from __future__ import annotations
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def build_graph(n_nodes: int, n_edges_per_node: int = 3):
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory(use_rust=False)
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


def benchmark_scale(n_nodes: int) -> dict:
    mem = build_graph(n_nodes)
    
    # Warm-up
    mem.spread_activation(["concept_0"], steps=3, decay=0.7)
    
    # Measure
    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        result = mem.spread_activation(["concept_0", "concept_1"], steps=3, decay=0.7)
    elapsed = (time.perf_counter() - start) / runs
    
    return {
        "n_nodes": n_nodes,
        "latency_ms": elapsed * 1000,
        "activated_count": len(result),
    }


def main():
    scales = [100, 1_000, 10_000]
    print(f"{'Nodes':>8}  {'Latency (ms)':>14}  {'Activated':>10}")
    print("-" * 40)
    for n in scales:
        r = benchmark_scale(n)
        print(f"{r['n_nodes']:>8}  {r['latency_ms']:>14.2f}  {r['activated_count']:>10}")


if __name__ == "__main__":
    main()
