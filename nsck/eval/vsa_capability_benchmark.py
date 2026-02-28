"""
VSA Capability Benchmark — V17

Measures throughput of V17 enrichment modules (no Rust dependency required).
Run::

    python nsck/eval/vsa_capability_benchmark.py

Results are printed to stdout and optionally saved to nsck/eval/results/v17_capability.json.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to path so we can import from nsck/python/core
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "nsck"))


def _bench(fn, n: int = 1000) -> float:
    """Run fn() n times and return throughput in ops/s."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    return n / elapsed


def bench_causal_enricher() -> dict:
    from python.core.reasoning.causal_enricher import CausalEnricher
    e = CausalEnricher()
    ops = _bench(lambda: e.enrich("fire", "smoke"), n=5000)
    chain_ops = _bench(lambda: e.enrich_chain(["a", "b", "c", "d"]), n=2000)
    return {"causal_enrich_ops_s": int(ops), "causal_chain_ops_s": int(chain_ops)}


def bench_perceptual_enricher() -> dict:
    from unittest.mock import MagicMock
    from python.core.perception.perceptual_enricher import PerceptualEnricher
    e = PerceptualEnricher()
    pkt = MagicMock()
    pkt.modality = "text"
    pkt.situation_hv = None
    ops = _bench(lambda: e.enrich(pkt), n=5000)
    return {"perceptual_enrich_ops_s": int(ops)}


def bench_semantic_enricher() -> dict:
    from python.core.memory.semantic_enricher import SemanticEnricher
    e = SemanticEnricher(semantic_memory=None)
    ops = _bench(lambda: e.enrich_concept("dog", "is_a", "animal"), n=5000)
    bulk_triples = [("a", "is_a", "b")] * 10
    bulk_ops = _bench(lambda: e.enrich_bulk(bulk_triples), n=500)
    return {"semantic_enrich_ops_s": int(ops), "semantic_bulk_10_ops_s": int(bulk_ops)}


def bench_glass_box_tracer() -> dict:
    from python.core.cognitive.glass_box_tracer import GlassBoxTracer
    t = GlassBoxTracer()

    def one_decision():
        t.begin_decision()
        with t.span("perception"):
            t.record("M", "msg", confidence=0.9)
        with t.span("reasoning"):
            t.record("N", "msg2")
        t.end_decision()

    ops = _bench(one_decision, n=2000)
    return {"glass_box_decisions_s": int(ops)}


def bench_crossmodal_enricher() -> dict:
    from python.core.memory.crossmodal_enricher import CrossModalEnricher
    e = CrossModalEnricher(crossmodal_memory=None)
    ops = _bench(
        lambda: e.link_modalities("dog", [("vision", "v"), ("audio", "a")]),
        n=2000,
    )
    return {"crossmodal_link_ops_s": int(ops)}


def run_all() -> dict:
    results: dict = {
        "version": "V17",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("NSCK V17 Capability Benchmark")
    print("=" * 40)
    for fn in [
        bench_causal_enricher,
        bench_perceptual_enricher,
        bench_semantic_enricher,
        bench_glass_box_tracer,
        bench_crossmodal_enricher,
    ]:
        r = fn()
        results.update(r)
        for k, v in r.items():
            print(f"  {k}: {v:,}")
    print("=" * 40)
    return results


if __name__ == "__main__":
    results = run_all()
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "v17_capability.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")
