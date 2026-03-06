"""
NSCK V30 — Societal Living HyperVector Observation & Analysis
==============================================================
7-phase comprehensive observation of what the Societal HV system is
actually doing — bond formation, community structure, percolation,
activation spread, Rust speedup, and whether it helps or hurts reasoning.

Usage::

    PYTHONPATH=nsck python nsck/eval/societal_observation.py

Output:
    - Full console analysis report
    - JSON saved to nsck/eval/results/societal_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("NSCK_USE_RUST", "1")
warnings.filterwarnings("ignore")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import python.core.vsa.hypervec_shim as hv_mod
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.society_manager import SocietyManager, ClusterResult
from python.core.societal.societal_context_router import SocietalContextRouter
from python.core.societal.snapshots import SocietalSnapshot


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3

def _hdr(msg: str) -> None:
    w = 72
    print(f"\n{'═' * w}\n  {msg}\n{'═' * w}")

def _sub(msg: str) -> None:
    print(f"\n  ── {msg} {'─' * max(0, 58 - len(msg))}")

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}")

def _info(msg: str) -> None:
    print(f"        {msg}")

def _make_lhv(cid: str, seed: int, domain: List[str], activation: float = 0.1) -> LivingHyperVector:
    return LivingHyperVector(
        concept_id=cid,
        hv=hv_mod.HyperVector(seed=seed),
        domain_path=domain,
        role="leaf",
        initial_activation=activation,
    )


# ── S1: Bootstrap ─────────────────────────────────────────────────────────────

def phase_s1_bootstrap() -> Tuple[SocietyManager, Dict[str, Any]]:
    _hdr("Phase S1 — Bootstrap: Build Societal World from Realistic Corpus")

    # Build concept set from realistic knowledge domains
    KNOWLEDGE = {
        "science.biology":   ["photosynthesis", "dna", "neuron", "mitochondria",
                               "evolution", "protein", "chromosome", "enzyme",
                               "chlorophyll", "ribosome", "metabolism", "cell",
                               "organism", "ecology", "biodiversity"],
        "science.physics":   ["gravity", "quantum", "spacetime", "electron",
                               "entropy", "radiation", "momentum", "energy",
                               "wave", "particle", "magnetism", "thermodynamics",
                               "relativity", "photon", "nucleus"],
        "technology.ml":     ["neural_network", "backpropagation", "gradient",
                               "transformer", "attention", "embedding", "tensor",
                               "optimiser", "loss", "dataset", "inference",
                               "training", "overfitting", "regularisation", "batch"],
        "technology.cs":     ["algorithm", "database", "api", "encryption",
                               "compiler", "operating_system", "memory",
                               "process", "network", "protocol", "thread",
                               "cache", "binary", "pointer", "stack"],
        "philosophy":        ["epistemology", "ethics", "consciousness",
                               "rationalism", "empiricism", "metaphysics",
                               "free_will", "justice", "knowledge", "truth",
                               "logic", "inference", "reality", "mind", "reason"],
        "history":           ["democracy", "empire", "revolution", "colonisation",
                               "industrialisation", "nationalism", "warfare",
                               "treaty", "monarchy", "republic", "slavery",
                               "trade", "migration", "culture", "civilisation"],
    }

    _sub(f"Registering concepts")
    mgr = SocietyManager(bond_threshold=0.0, max_bonds=12,
                          bond_decay_rate=0.005, activation_spread_factor=0.4,
                          auto_cluster_interval=10)
    concept_list = []
    for domain_path_str, concepts in KNOWLEDGE.items():
        domain_path = domain_path_str.split(".")
        for i, cid in enumerate(concepts):
            seed = abs(hash(cid)) % (2**30)
            activation = 0.1 + 0.05 * (i % 5)
            lhv = _make_lhv(cid, seed, domain_path, activation)
            mgr.register(lhv)
            concept_list.append(cid)

    n_concepts = len(mgr)
    _ok(f"Registered {n_concepts} concepts across {len(KNOWLEDGE)} domain paths")

    # Initial bonding
    _sub("Initial auto_bond (threshold=0.0)")
    t0 = time.perf_counter()
    mgr.auto_bond()
    bond_ms = _ms(t0)

    # Step epochs (100 epochs, snapshots every 10)
    _sub("Running 100 epochs, snapshotting every 10")
    snapshots = []
    snap_data = []
    for epoch in range(1, 101):
        mgr.step_epoch(run_cluster=(epoch % 20 == 0))
        if epoch % 10 == 0:
            snap = mgr.snapshot()
            snapshots.append(snap)
            snap_data.append(snap.to_dict())
            _info(f"  Epoch {epoch:3d}: n_bonds={snap.n_bonds:4d}  "
                  f"avg_activation={snap.avg_activation:.3f}  "
                  f"n_communities={snap.n_communities}")

    final_snap = snapshots[-1] if snapshots else mgr.snapshot()
    _ok(f"Bootstrap complete: {final_snap.n_concepts} concepts, "
        f"{final_snap.n_bonds} bonds, {final_snap.n_communities} communities")
    _ok(f"auto_bond latency: {bond_ms:.1f}ms")

    result = {
        "n_concepts": n_concepts,
        "auto_bond_ms": round(bond_ms, 2),
        "n_bonds_final": final_snap.n_bonds,
        "n_communities_final": final_snap.n_communities,
        "avg_activation_final": final_snap.avg_activation,
        "percolation_threshold": final_snap.percolation_threshold,
        "epoch_snapshots": snap_data,
    }
    return mgr, result


# ── S2: Community Analysis ────────────────────────────────────────────────────

def phase_s2_community_analysis(mgr: SocietyManager) -> Dict[str, Any]:
    _hdr("Phase S2 — Community Analysis (Leiden at 3 Resolutions)")
    result_s2 = {}

    for resolution in [0.5, 1.0, 2.0]:
        _sub(f"Leiden clustering resolution={resolution}")
        t0 = time.perf_counter()
        cr = mgr.leiden_cluster(resolution=resolution)
        cluster_ms = _ms(t0)
        _ok(f"  resolution={resolution}: {cr.n_communities} communities, "
            f"modularity Q={cr.modularity:.4f} ({cluster_ms:.1f}ms)")

        # Top-3 hub concepts per community (highest bond count)
        hub_info = []
        for comm_id, members in cr.communities.items():
            hubs = sorted(
                [(cid, len(mgr.get(cid)._bonds) if mgr.get(cid) else 0)
                 for cid in members],
                key=lambda x: x[1], reverse=True
            )[:3]
            hub_info.append({
                "community": comm_id,
                "size": len(members),
                "hubs": [(cid, bonds) for cid, bonds in hubs],
            })
            if hubs:
                _info(f"    Comm {comm_id} ({len(members)} members): "
                      f"hubs={[h[0] for h in hubs[:3]]}")

        # Cross-community bridge concepts (bonded to concepts in other communities)
        bridges = []
        for cid, lhv in list(mgr._concepts.items())[:80]:  # sample for speed
            comm_self = cr.community_of(cid)
            if comm_self is None:
                continue
            n_cross = sum(
                1 for bonded_id in lhv._bonds
                if cr.community_of(bonded_id) not in (None, comm_self)
            )
            if n_cross >= 2:
                bridges.append((cid, n_cross))
        bridges.sort(key=lambda x: x[1], reverse=True)
        _info(f"    Bridge concepts (top 3): {bridges[:3]}")

        result_s2[f"res_{resolution}"] = {
            "n_communities": cr.n_communities,
            "modularity": round(cr.modularity, 4),
            "cluster_ms": round(cluster_ms, 2),
            "top_communities": hub_info[:5],
            "bridge_concepts": bridges[:5],
        }

    return result_s2


# ── S3: Activation Dynamics ───────────────────────────────────────────────────

def phase_s3_activation_dynamics(mgr: SocietyManager) -> Dict[str, Any]:
    _hdr("Phase S3 — Activation Dynamics (Spreading Activation Trace)")
    result_s3 = {}

    seed_queries = ["photosynthesis", "democracy", "neural_network"]
    THRESHOLD = 0.3

    for seed_concept in seed_queries:
        _sub(f"Seed concept: '{seed_concept}', threshold={THRESHOLD}")
        # Reset activations
        for lhv in mgr._concepts.values():
            lhv.activation = 0.05

        lhv_seed = mgr.get(seed_concept)
        if lhv_seed is None:
            _warn(f"  '{seed_concept}' not in society — skipping")
            continue

        # Inject activation and trace for 5 epochs
        epoch_traces = []
        mgr.activate_concept(seed_concept, delta=0.9, spread=False)
        for epoch in range(1, 6):
            # Spread activation for one step
            lhv_seed.spread_activation(
                mgr._concepts,
                spread_factor=mgr.activation_spread_factor,
                epoch=mgr.epoch,
            )
            mgr.step_epoch(run_cluster=False)
            above_threshold = [
                (cid, round(lhv.activation, 3))
                for cid, lhv in mgr._concepts.items()
                if lhv.activation >= THRESHOLD
            ]
            above_threshold.sort(key=lambda x: x[1], reverse=True)
            epoch_traces.append({
                "epoch": epoch,
                "n_activated": len(above_threshold),
                "top_5": above_threshold[:5],
            })
            _info(f"  Epoch {epoch}: {len(above_threshold)} concepts above {THRESHOLD} "
                  f"| top: {[c[0] for c in above_threshold[:3]]}")

        result_s3[seed_concept] = epoch_traces

    return result_s3


# ── S4: Percolation Analysis ──────────────────────────────────────────────────

def phase_s4_percolation(mgr: SocietyManager) -> Dict[str, Any]:
    _hdr("Phase S4 — Percolation Analysis (Bond-Strength Sweep)")

    _sub("Computing giant component fraction vs bond-strength threshold")
    n = len(mgr)
    sweep_results = []
    thresholds = np.linspace(0.0, 1.0, 21)

    for thr in thresholds:
        # Count edges above threshold
        edges_above = sum(
            1 for lhv in mgr._concepts.values()
            for bond in lhv._bonds.values()
            if bond.strength >= thr
        ) // 2  # undirected

        # Build simple adjacency for BFS
        adj: Dict[str, List[str]] = {cid: [] for cid in mgr._concepts}
        for cid, lhv in mgr._concepts.items():
            for bonded_id, bond in lhv._bonds.items():
                if bond.strength >= thr:
                    adj[cid].append(bonded_id)

        # BFS for giant component
        visited = set()
        max_comp = 0
        for start in adj:
            if start in visited:
                continue
            comp = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in comp:
                    continue
                comp.add(node)
                queue.extend(v for v in adj.get(node, []) if v not in comp)
            visited.update(comp)
            max_comp = max(max_comp, len(comp))

        frac = max_comp / n if n > 0 else 0.0
        sweep_results.append({
            "threshold": round(float(thr), 3),
            "edges_above": edges_above,
            "giant_component_size": max_comp,
            "giant_component_fraction": round(frac, 4),
        })

    # ASCII percolation curve
    _sub("ASCII percolation curve (giant component fraction vs threshold)")
    print(f"  {'Thr':>5s} {'GCF':>6s} {'Bar'}")
    perc_threshold = 1.0
    for r in sweep_results[::2]:  # every other point
        thr = r["threshold"]
        frac = r["giant_component_fraction"]
        bar = "#" * int(frac * 40)
        print(f"  {thr:5.2f} {frac:6.3f} {bar}")
        if frac < 0.5 and thr > 0:
            perc_threshold = min(perc_threshold, thr)

    # Official threshold from SocietyManager
    official_thr = mgr.percolation_threshold()
    _ok(f"Percolation threshold (official): {official_thr:.4f}")
    _ok(f"Percolation threshold (sweep):    {perc_threshold:.3f}")

    return {
        "official_threshold": round(official_thr, 4),
        "sweep_threshold_estimate": round(perc_threshold, 3),
        "sweep": sweep_results,
    }


# ── S5: Rust vs Python Speed ──────────────────────────────────────────────────

def phase_s5_rust_vs_python(mgr: SocietyManager) -> Dict[str, Any]:
    _hdr("Phase S5 — Rust vs Python Speed Comparison")
    result = {}
    N = 500

    # auto_bond timing
    _sub(f"auto_bond() timing (N={N} calls benchmark not feasible; timing 1 call×{N})")

    # HV bundle: Rust
    t0 = time.perf_counter()
    hv1, hv2 = hv_mod.HyperVector(42), hv_mod.HyperVector(99)
    for _ in range(N):
        hv1.bundle(hv2)
    bundle_us_rust = (time.perf_counter() - t0) / N * 1e6
    _ok(f"Rust bundle:     {bundle_us_rust:.2f} µs/op")
    result["rust_bundle_us"] = round(bundle_us_rust, 3)

    # HV similarity: Rust
    t0 = time.perf_counter()
    for _ in range(N):
        hv1.similarity(hv2)
    sim_us_rust = (time.perf_counter() - t0) / N * 1e6
    _ok(f"Rust similarity: {sim_us_rust:.2f} µs/op")
    result["rust_similarity_us"] = round(sim_us_rust, 3)

    # Python fallback
    try:
        from python.core.vsa.hypervec_py import HyperVectorPy as PyHV
        hv1p, hv2p = PyHV(42), PyHV(99)
        t0 = time.perf_counter()
        for _ in range(min(N, 100)):
            hv1p.bundle(hv2p)
        bundle_us_py = (time.perf_counter() - t0) / min(N, 100) * 1e6
        _ok(f"Python bundle:   {bundle_us_py:.2f} µs/op")
        bundle_speedup = bundle_us_py / (bundle_us_rust + 1e-9)
        _ok(f"Bundle speedup:  {bundle_speedup:.1f}×")
        result["python_bundle_us"] = round(bundle_us_py, 3)
        result["bundle_speedup"] = round(bundle_speedup, 2)

        t0 = time.perf_counter()
        for _ in range(min(N, 100)):
            hv1p.similarity(hv2p)
        sim_us_py = (time.perf_counter() - t0) / min(N, 100) * 1e6
        sim_speedup = sim_us_py / (sim_us_rust + 1e-9)
        _ok(f"Similarity speedup: {sim_speedup:.1f}×")
        result["python_similarity_us"] = round(sim_us_py, 3)
        result["similarity_speedup"] = round(sim_speedup, 2)
    except Exception as exc:
        _warn(f"Python HV benchmark skipped: {exc}")

    # spread_activation timing
    _sub("spread_activation() Rust path (via SemanticMemory)")
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig
    sub = NSCKSubstrate(NSCKConfig.v30())
    sub.register_task("perf_test")
    # Seed some concepts
    for i in range(50):
        sub.engine.semantic_memory.add_concept(f"perf_{i}", {"x": i})
    t0 = time.perf_counter()
    for _ in range(10):
        sub.engine.semantic_memory.query(
            hv_mod.HyperVector(42), k=10
        )
    spread_ms = _ms(t0) / 10
    _ok(f"Semantic search (50 concepts, k=10): {spread_ms:.2f}ms/query")
    result["semantic_search_ms"] = round(spread_ms, 3)

    return result


# ── S6: Helping or Holding Back? ──────────────────────────────────────────────

def phase_s6_helping_or_hindering(mgr: SocietyManager) -> Dict[str, Any]:
    _hdr("Phase S6 — Helping or Holding Back? (10 Reasoning Queries)")
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig

    TEST_QUERIES = [
        "What is photosynthesis?",
        "How does DNA work?",
        "What causes rain?",
        "Explain machine learning",
        "What is democracy?",
        "Why do species evolve?",
        "What if gravity did not exist?",
        "How does a neuron fire?",
        "What is the relationship between energy and entropy?",
        "How does consciousness relate to brain activity?",
    ]
    GROUND_TRUTH_KEYWORDS = [
        {"sunlight", "glucose", "plant", "chlorophyll"},
        {"genetic", "nucleotide", "sequence", "protein"},
        {"water", "evaporation", "cloud", "precipitation"},
        {"data", "model", "training", "prediction"},
        {"vote", "government", "citizen", "representative"},
        {"mutation", "selection", "adaptation", "population"},
        {"mass", "orbit", "fall", "float"},
        {"signal", "synapse", "action", "potential"},
        {"thermodynamics", "heat", "disorder", "work"},
        {"brain", "neuron", "experience", "subjective"},
    ]

    def keyword_overlap(response: str, keywords: set) -> float:
        words = set(response.lower().split())
        return len(words & keywords) / max(len(keywords), 1)

    # Run without societal
    _sub("Baseline (enable_societal=False)")
    cfg_base = NSCKConfig()
    cfg_base.enable_societal = False
    cfg_base.enable_transparency = True
    sub_base = NSCKSubstrate(cfg_base)
    sub_base.register_task("reasoning")

    base_results = []
    for query, keywords in zip(TEST_QUERIES, GROUND_TRUTH_KEYWORDS):
        try:
            t0 = time.perf_counter()
            res = sub_base.process(query, "reasoning")
            lat = _ms(t0)
            # Use trace to assess richness
            trace = res.trace
            n_causal = len(trace.get("causal_chains", []))
            overlap = keyword_overlap(res.chosen_action or "", keywords)
            base_results.append({
                "query": query,
                "confidence": res.confidence,
                "latency_ms": round(lat, 2),
                "n_causal": n_causal,
                "keyword_overlap": round(overlap, 4),
                "societal": False,
            })
        except Exception as exc:
            base_results.append({"query": query, "error": str(exc), "societal": False})

    # Run with societal
    _sub("With societal (enable_societal=True)")
    cfg_soc = NSCKConfig.societal()
    cfg_soc.enable_transparency = True
    sub_soc = NSCKSubstrate(cfg_soc)
    sub_soc.register_task("reasoning")

    # Populate society with relevant concepts
    concepts_to_add = [
        {"concept_id": "photosynthesis", "domain_path": ["science", "biology"]},
        {"concept_id": "dna", "domain_path": ["science", "biology"]},
        {"concept_id": "democracy", "domain_path": ["history"]},
        {"concept_id": "machine_learning", "domain_path": ["technology", "ml"]},
        {"concept_id": "consciousness", "domain_path": ["philosophy"]},
        {"concept_id": "evolution", "domain_path": ["science", "biology"]},
        {"concept_id": "thermodynamics", "domain_path": ["science", "physics"]},
        {"concept_id": "neuron", "domain_path": ["science", "biology"]},
    ]
    sub_soc.init_societal_world(concepts_to_add)

    soc_results = []
    for query, keywords in zip(TEST_QUERIES, GROUND_TRUTH_KEYWORDS):
        try:
            t0 = time.perf_counter()
            res = sub_soc.process(query, "reasoning")
            lat = _ms(t0)
            trace = res.trace
            n_causal = len(trace.get("causal_chains", []))
            overlap = keyword_overlap(res.chosen_action or "", keywords)
            soc_context = res.societal_context or {}
            soc_results.append({
                "query": query,
                "confidence": res.confidence,
                "latency_ms": round(lat, 2),
                "n_causal": n_causal,
                "keyword_overlap": round(overlap, 4),
                "societal_concepts_activated": len(soc_context.get("active_concepts", [])),
                "societal": True,
            })
        except Exception as exc:
            soc_results.append({"query": query, "error": str(exc), "societal": True})

    # Comparison
    _sub("Comparison: Base vs Societal")
    base_conf = np.mean([r.get("confidence", 0) for r in base_results])
    soc_conf  = np.mean([r.get("confidence", 0) for r in soc_results])
    base_lat  = np.mean([r.get("latency_ms", 0) for r in base_results])
    soc_lat   = np.mean([r.get("latency_ms", 0) for r in soc_results])
    base_over = np.mean([r.get("keyword_overlap", 0) for r in base_results])
    soc_over  = np.mean([r.get("keyword_overlap", 0) for r in soc_results])

    _info(f"  {'Metric':<30} {'Base':>10} {'Societal':>10} {'Delta':>10}")
    _info(f"  {'-'*62}")
    for label, b, s in [
        ("Mean confidence",    base_conf, soc_conf),
        ("Mean latency (ms)",  base_lat,  soc_lat),
        ("Keyword overlap",    base_over, soc_over),
    ]:
        delta = s - b
        sign = "+" if delta >= 0 else ""
        _info(f"  {label:<30} {b:>10.4f} {s:>10.4f} {sign}{delta:>9.4f}")

    conf_delta = soc_conf - base_conf
    lat_delta  = soc_lat - base_lat
    over_delta = soc_over - base_over

    # Verdict
    helping_score = 0
    if conf_delta > 0.01:
        helping_score += 1
    if lat_delta < 50:  # acceptable overhead
        helping_score += 1
    if over_delta >= 0:
        helping_score += 1

    if helping_score >= 2:
        verdict = "HELPING — societal context improves reasoning quality"
    elif helping_score == 1:
        verdict = "NEUTRAL — marginal effect, benefit depends on query type"
    else:
        verdict = "HOLDING BACK — overhead exceeds benefit at current config"

    _ok(f"VERDICT: {verdict}")

    return {
        "base_results": base_results,
        "soc_results": soc_results,
        "comparison": {
            "base_confidence": round(float(base_conf), 4),
            "soc_confidence":  round(float(soc_conf), 4),
            "base_latency_ms": round(float(base_lat), 2),
            "soc_latency_ms":  round(float(soc_lat), 2),
            "base_keyword_overlap": round(float(base_over), 4),
            "soc_keyword_overlap":  round(float(soc_over), 4),
            "verdict": verdict,
            "helping_score": helping_score,
        },
    }


# ── S7: Full Societal Report ──────────────────────────────────────────────────

def phase_s7_report(
    s1: Dict, s2: Dict, s3: Dict, s4: Dict, s5: Dict, s6: Dict
) -> Dict[str, Any]:
    _hdr("Phase S7 — Full Societal Observation Report")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rust_backend": hv_mod.__backend__,
        "s1_bootstrap": s1,
        "s2_community_analysis": s2,
        "s3_activation_dynamics": s3,
        "s4_percolation": s4,
        "s5_rust_speed": s5,
        "s6_helping_verdict": s6.get("comparison", {}),
    }

    _sub("Executive Summary")
    verdict = s6.get("comparison", {}).get("verdict", "unknown")
    _info(f"  Bootstrap:           {s1['n_concepts']} concepts, {s1['n_bonds_final']} bonds")
    _info(f"  Communities (res=1): {s2.get('res_1.0', {}).get('n_communities', '?')}")
    _info(f"  Modularity (res=1):  {s2.get('res_1.0', {}).get('modularity', '?'):.4f}")
    _info(f"  Percolation thr:     {s4.get('official_threshold', '?')}")
    if "bundle_speedup" in s5:
        _info(f"  Bundle speedup:      {s5['bundle_speedup']}×")
    _info(f"  Verdict:             {verdict}")

    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  NSCK V30 — Societal Living HyperVector Observation                 ║")
    print("╚" + "═" * 70 + "╝")

    mgr, s1 = phase_s1_bootstrap()
    s2 = phase_s2_community_analysis(mgr)
    s3 = phase_s3_activation_dynamics(mgr)
    s4 = phase_s4_percolation(mgr)
    s5 = phase_s5_rust_vs_python(mgr)
    s6 = phase_s6_helping_or_hindering(mgr)
    full_report = phase_s7_report(s1, s2, s3, s4, s5, s6)

    out_dir = os.path.join(_ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "societal_report.json")
    with open(out_path, "w") as fh:
        json.dump(full_report, fh, indent=2, default=str)
    _ok(f"Saved: {out_path}")
    print("\n" + "╚" + "═" * 70 + "╝")


if __name__ == "__main__":
    main()
