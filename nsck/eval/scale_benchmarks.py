"""Scale validation benchmarks for NSCK."""
from __future__ import annotations
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def build_mem(n: int):
    from python.core.memory.semantic_memory import SemanticMemory
    import random
    random.seed(0)
    mem = SemanticMemory(use_rust=False)
    for i in range(n):
        mem.add_concept(f"c{i}", {"idx": i})
    for i in range(min(n, 2000)):
        j = random.randint(0, n - 1)
        if i != j:
            mem.add_relation(f"c{i}", "similar_to", f"c{j}")
    return mem


def bench_spread(n: int) -> float:
    mem = build_mem(n)
    mem.spread_activation([f"c0"], steps=2, decay=0.7)  # warm-up
    t = time.perf_counter()
    mem.spread_activation([f"c0", f"c1"], steps=2, decay=0.7)
    return (time.perf_counter() - t) * 1000


def bench_query(n: int) -> float:
    from python.core.memory.semantic_memory import SemanticMemory
    import python.core.vsa.hypervec_shim as hv_mod
    mem = SemanticMemory(use_rust=False)
    for i in range(n):
        mem.add_concept(f"c{i}", {"idx": i})
    query = hv_mod.HyperVector(0)
    t = time.perf_counter()
    mem.query(query, k=10)
    return (time.perf_counter() - t) * 1000


def main():
    scales = [100, 500, 1_000, 5_000]
    print(f"{'Nodes':>6}  {'Spread (ms)':>12}  {'Query (ms)':>11}")
    print("-" * 35)
    for n in scales:
        spread_ms = bench_spread(n)
        query_ms = bench_query(n)
        print(f"{n:>6}  {spread_ms:>12.2f}  {query_ms:>11.2f}")


if __name__ == "__main__":
    main()
