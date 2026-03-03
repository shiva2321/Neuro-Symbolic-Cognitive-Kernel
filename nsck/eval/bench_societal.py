"""
Societal Hypervector Benchmark Suite — NSCK V26
================================================
Benchmarks all major operations of the Societal HV Knowledge Representation
system.  All measurements are latency (µs or ms) averaged over multiple runs.

Usage (from repository root)::

    python nsck/eval/bench_societal.py

Output: console table + JSON results written to
``nsck/eval/results/bench_societal_results.json``.

No internet access required; no external datasets needed.
All workloads are synthetically generated.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import python.core.vsa.hypervec_shim as hv_mod
from python.core.societal.living_hypervector import LivingHyperVector, Bond
from python.core.societal.society_manager import SocietyManager
from python.core.societal.societal_context_router import SocietalContextRouter

# ── Helpers ──────────────────────────────────────────────────────────────────

def _us(t0: float) -> float:
    """Elapsed µs since t0."""
    return (time.perf_counter() - t0) * 1e6


def _ms(t0: float) -> float:
    """Elapsed ms since t0."""
    return (time.perf_counter() - t0) * 1e3


def _make_hv(seed: int) -> Any:
    return hv_mod.HyperVector(seed=seed)


def _make_society(n: int, bond_threshold: float = 0.0, max_bonds: int = 8) -> SocietyManager:
    mgr = SocietyManager(
        bond_threshold=bond_threshold,
        max_bonds=max_bonds,
        auto_cluster_interval=0,
    )
    for i in range(n):
        lhv = LivingHyperVector(
            concept_id=f"c{i}",
            hv=_make_hv(i),
            domain_path=["bench", f"grp{i % 5}"],
            role="leaf",
            initial_activation=0.5,
        )
        mgr.register(lhv)
    return mgr


def _bench(fn, n_warmup: int = 2, n_runs: int = 10) -> Dict[str, float]:
    """Run fn() n_warmup times to warm up, then n_runs times and return stats."""
    for _ in range(n_warmup):
        fn()
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)  # µs
    arr = np.array(times)
    return {
        "mean_us": float(np.mean(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "std_us": float(np.std(arr)),
        "n_runs": n_runs,
    }


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _print_row(label: str, stats: Dict[str, float], unit: str = "µs") -> None:
    mul = 1.0 if unit == "µs" else 1e-3
    m = stats["mean_us"] * mul
    mn = stats["min_us"] * mul
    mx = stats["max_us"] * mul
    print(f"  {label:<38s} {m:8.2f} {unit}  (min={mn:.2f}, max={mx:.2f})")


# ── Benchmark sections ───────────────────────────────────────────────────────

def bench_lhv_creation() -> Dict[str, Any]:
    """LivingHyperVector construction."""
    _print_section("1. LivingHyperVector creation")
    hv = _make_hv(0)
    results = {}

    # Single LHV creation
    r = _bench(lambda: LivingHyperVector("x", hv, domain_path=["a", "b"]))
    results["lhv_create_single"] = r
    _print_row("LHV create (single)", r)

    # Bulk creation (100 nodes)
    def _bulk():
        for i in range(100):
            LivingHyperVector(f"c{i}", hv, domain_path=["bench"])
    r = _bench(_bulk, n_runs=5)
    results["lhv_create_bulk_100"] = r
    _print_row("LHV create (bulk×100)", r)

    return results


def bench_bond_operations() -> Dict[str, Any]:
    """Bond formation, reinforcement, decay."""
    _print_section("2. Bond operations")
    a = LivingHyperVector("a", _make_hv(1))
    b = LivingHyperVector("b", _make_hv(2))
    results = {}

    # Bond formation
    def _form():
        a._bonds.clear()
        a.form_bond(b, strength_override=0.7)
    r = _bench(_form)
    results["bond_form"] = r
    _print_row("Bond formation", r)

    # Bond reinforcement (existing bond)
    a.form_bond(b, strength_override=0.7)
    r = _bench(lambda: a.form_bond(b))
    results["bond_reinforce"] = r
    _print_row("Bond reinforcement", r)

    # Bond decay
    r = _bench(lambda: a.decay_bonds(rate=0.01, min_strength=0.0))
    results["bond_decay"] = r
    _print_row("Bond decay", r)

    return results


def bench_activation_dynamics() -> Dict[str, Any]:
    """Activation spike and spread."""
    _print_section("3. Activation dynamics")
    results = {}

    # Activation spike
    lhv = LivingHyperVector("x", _make_hv(0))
    r = _bench(lambda: lhv.activate(0.1))
    results["activation_spike"] = r
    _print_row("Activation spike", r)

    # Activation decay
    r = _bench(lambda: lhv.decay_activation(0.05))
    results["activation_decay"] = r
    _print_row("Activation decay", r)

    # Spread activation (10 peers)
    peers_map = {}
    for i in range(10):
        peer = LivingHyperVector(f"p{i}", _make_hv(i + 10))
        lhv.form_bond(peer, strength_override=0.7)
        peers_map[f"p{i}"] = peer
    hub = LivingHyperVector("hub", _make_hv(99), initial_activation=1.0)
    hub._bonds = lhv._bonds.copy()
    r = _bench(lambda: hub.spread_activation(peers_map, spread_factor=0.4))
    results["activation_spread_10_peers"] = r
    _print_row("Activation spread (10 peers)", r)

    return results


def bench_society_manager(sizes: List[int] = None) -> Dict[str, Any]:
    """SocietyManager operations at various scales."""
    _print_section("4. SocietyManager operations")
    if sizes is None:
        sizes = [10, 50, 100]
    results = {}

    for n in sizes:
        mgr = _make_society(n, bond_threshold=0.0)
        # Force-add bonds so clustering has something to work with
        ids = list(mgr._concepts.keys())
        for i in range(min(n, 20)):
            mgr.form_bond_explicit(ids[i], ids[(i + 1) % n], strength=0.8)

        # Nearest neighbors
        query = _make_hv(999)
        r = _bench(lambda: mgr.nearest_neighbors(query, k=5))
        results[f"nearest_neighbors_n{n}"] = r
        unit = "ms" if r["mean_us"] > 1000 else "µs"
        _print_row(f"  Nearest neighbors (N={n})", r, unit=unit)

        # Leiden clustering
        r = _bench(lambda: mgr.leiden_cluster.__wrapped__(mgr, 1.0) if hasattr(mgr.leiden_cluster, '__wrapped__') else _leiden_no_cache(mgr), n_warmup=1, n_runs=5)
        results[f"leiden_cluster_n{n}"] = r
        unit = "ms" if r["mean_us"] > 1000 else "µs"
        _print_row(f"  Leiden cluster (N={n})", r, unit=unit)

        # Epoch step
        r = _bench(lambda: mgr.step_epoch(run_cluster=False), n_runs=5)
        results[f"epoch_step_n{n}"] = r
        unit = "ms" if r["mean_us"] > 1000 else "µs"
        _print_row(f"  Epoch step (N={n})", r, unit=unit)

    return results


def _leiden_no_cache(mgr: SocietyManager) -> None:
    """Run Leiden without hitting the cache."""
    mgr._cluster_cache.clear()
    mgr.leiden_cluster(1.0)


def bench_auto_bond(sizes: List[int] = None) -> Dict[str, Any]:
    """Auto-bond at various scales."""
    _print_section("5. Auto-bond")
    if sizes is None:
        sizes = [10, 30, 50]
    results = {}

    for n in sizes:
        mgr = _make_society(n, bond_threshold=0.0)

        def _run():
            for lhv in mgr:
                lhv._bonds.clear()
            mgr.auto_bond()

        r = _bench(_run, n_warmup=1, n_runs=5)
        results[f"auto_bond_n{n}"] = r
        unit = "ms" if r["mean_us"] > 1000 else "µs"
        _print_row(f"  Auto-bond (N={n})", r, unit=unit)

    return results


def bench_router() -> Dict[str, Any]:
    """SocietalContextRouter routing latency."""
    _print_section("6. SocietalContextRouter routing")
    results = {}

    for n in [10, 50, 100]:
        mgr = _make_society(n, bond_threshold=0.0)
        mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        router = SocietalContextRouter(mgr, top_k=5, min_similarity=0.0)
        query = _make_hv(42)

        r = _bench(lambda: router.route(query, task_tag="bench"))
        results[f"router_route_n{n}"] = r
        unit = "ms" if r["mean_us"] > 500 else "µs"
        _print_row(f"  Route (N={n})", r, unit=unit)

    return results


def bench_serialisation() -> Dict[str, Any]:
    """to_dict / from_dict roundtrip."""
    _print_section("7. Serialisation")
    results = {}

    mgr = _make_society(20, bond_threshold=0.0)
    mgr.auto_bond()

    # to_dict for one LHV
    lhv = list(mgr)[0]
    r = _bench(lambda: lhv.to_dict())
    results["lhv_to_dict"] = r
    _print_row("LHV.to_dict()", r)

    # SocietyManager.to_dict (full)
    r = _bench(lambda: mgr.to_dict(), n_runs=5)
    results["manager_to_dict_n20"] = r
    _print_row("SocietyManager.to_dict() (N=20)", r)

    return results


def bench_percolation() -> Dict[str, Any]:
    """Percolation threshold computation."""
    _print_section("8. Percolation threshold")
    results = {}

    for n in [10, 50]:
        mgr = _make_society(n, bond_threshold=0.0)
        mgr.auto_bond()

        def _run():
            mgr._percolation_threshold = None
            mgr.percolation_threshold()

        r = _bench(_run, n_warmup=1, n_runs=5)
        results[f"percolation_n{n}"] = r
        unit = "ms" if r["mean_us"] > 1000 else "µs"
        _print_row(f"  Percolation threshold (N={n})", r, unit=unit)

    return results


def bench_topo_health() -> Dict[str, Any]:
    """Topological health report."""
    _print_section("9. Topological health")
    results = {}

    mgr = _make_society(50, bond_threshold=0.0)
    mgr.auto_bond()

    r = _bench(lambda: mgr.topological_health(), n_runs=5)
    results["topo_health_n50"] = r
    unit = "ms" if r["mean_us"] > 1000 else "µs"
    _print_row("Topological health (N=50)", r, unit=unit)

    return results


def bench_multi_resolution() -> Dict[str, Any]:
    """Multi-resolution clustering."""
    _print_section("10. Multi-resolution clustering")

    mgr = _make_society(30, bond_threshold=0.0)
    mgr.auto_bond()
    resolutions = [0.5, 1.0, 1.5, 2.0]

    def _run():
        mgr._cluster_cache.clear()
        mgr.multi_resolution_cluster(resolutions)

    r = _bench(_run, n_warmup=1, n_runs=5)
    unit = "ms" if r["mean_us"] > 1000 else "µs"
    _print_row(f"Multi-resolution (γ={resolutions}, N=30)", r, unit=unit)
    return {"multi_resolution_n30": r}


# ── Main entry point ─────────────────────────────────────────────────────────

def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmark sections and return combined results dict."""
    print("\n" + "═" * 60)
    print("  NSCK V26 — Societal HV Benchmark Suite")
    print("═" * 60)

    import python.core.vsa.hypervec_shim as shim
    print(f"  Backend: {shim.__backend__}")

    results: Dict[str, Any] = {
        "backend": shim.__backend__,
    }

    results.update(bench_lhv_creation())
    results.update(bench_bond_operations())
    results.update(bench_activation_dynamics())
    results.update(bench_society_manager())
    results.update(bench_auto_bond())
    results.update(bench_router())
    results.update(bench_serialisation())
    results.update(bench_percolation())
    results.update(bench_topo_health())
    results.update(bench_multi_resolution())

    # Summary table
    print("\n" + "═" * 60)
    print("  SUMMARY (mean latency)")
    print("═" * 60)
    key_ops = [
        ("lhv_create_single",        "LHV create (single)",       "µs"),
        ("bond_form",                 "Bond form",                  "µs"),
        ("bond_reinforce",            "Bond reinforce",             "µs"),
        ("activation_spike",          "Activation spike",           "µs"),
        ("activation_spread_10_peers","Activation spread×10",       "µs"),
        ("nearest_neighbors_n100",    "Nearest neighbors (N=100)",  "µs"),
        ("leiden_cluster_n100",       "Leiden cluster (N=100)",     "µs"),
        ("epoch_step_n100",           "Epoch step (N=100)",         "µs"),
        ("router_route_n100",         "Router route (N=100)",       "µs"),
        ("percolation_n50",           "Percolation threshold (N=50)","µs"),
    ]
    for key, label, unit in key_ops:
        if key in results:
            m = results[key]["mean_us"]
            if unit == "ms":
                print(f"  {label:<40s} {m/1000:8.3f} ms")
            else:
                print(f"  {label:<40s} {m:8.2f} µs")

    return results


def save_results(results: Dict[str, Any]) -> str:
    """Save benchmark results to JSON file."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_societal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    return out_path


if __name__ == "__main__":
    results = run_all_benchmarks()
    save_results(results)
