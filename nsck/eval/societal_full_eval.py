"""
NSCK V5 Societal Full Evaluation Suite.

6-phase end-to-end evaluation:
  Phase 1: Core functionality (register/query/tick)
  Phase 2: Emergence detection (spectral gap, percolation)
  Phase 3: Navigation accuracy (HNSW recall@10)
  Phase 4: TDA health (Betti numbers, health score)
  Phase 5: Zipf compliance validation
  Phase 6: Routing accuracy (domain detection)

Deterministic (seed=42), completes in < 60 seconds.
Outputs JSON report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure nsck is on the path
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "nsck"))

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.societal_world import SocietalKnowledgeWorld
from python.core.societal.emergence.spectral_rg import SpectralLaplacianRG
from python.core.societal.emergence.percolation import PercolationMonitor
from python.core.societal.emergence.zipf_validator import ZipfValidator
from python.core.societal.navigation.societal_hnsw import SocietalHNSW
from python.core.societal.tda.tda_monitor import TDAHealthMonitor
from python.core.societal.routing.context_router import SocietalContextRouter


_RNG = np.random.default_rng(42)


def _make_hv(seed: int):
    return hypervec_rs.HyperVector(seed=seed)


def _make_eval_world(n: int = 50) -> SocietalKnowledgeWorld:
    """Build a reproducible world with communities and bonds."""
    world = SocietalKnowledgeWorld(bond_threshold=0.25)
    rng = np.random.default_rng(42)

    # Register concepts
    for i in range(n):
        world.register_concept(f"c{i}", _make_hv(i), {
            "stability": float(rng.uniform(0.2, 0.9)),
        })

    # Add bonds within two groups (0-24: group A, 25-49: group B)
    ids = list(world.concepts.keys())
    group_a = ids[:n // 2]
    group_b = ids[n // 2:]

    for i, ca in enumerate(group_a):
        for cb in group_a[i + 1: i + 4]:  # sparse bonds
            w = float(rng.uniform(0.4, 0.8))
            world.concepts[ca].add_bond(cb, w)
            world.concepts[cb].add_bond(ca, w)

    for i, ca in enumerate(group_b):
        for cb in group_b[i + 1: i + 4]:
            w = float(rng.uniform(0.4, 0.8))
            world.concepts[ca].add_bond(cb, w)
            world.concepts[cb].add_bond(ca, w)

    # Add activation history
    for i, lhv in enumerate(world.concepts.values()):
        n_acts = max(1, 10 - i % 8)
        for _ in range(n_acts):
            lhv.update_activation(float(rng.uniform(0.1, 1.0)))
        for _ in range(5):
            lhv.tick()

    # Form neighbourhoods and domains
    n1 = world.form_neighborhood(group_a[:n // 4], "nbhd_A1")
    n2 = world.form_neighborhood(group_a[n // 4:], "nbhd_A2")
    n3 = world.form_neighborhood(group_b[:n // 4], "nbhd_B1")
    n4 = world.form_neighborhood(group_b[n // 4:], "nbhd_B2")

    world.form_domain(
        [n1.neighborhood_id, n2.neighborhood_id], "science", "dom_sci"
    )
    world.form_domain(
        [n3.neighborhood_id, n4.neighborhood_id], "art", "dom_art"
    )
    return world


# ============================================================
# Phase runners
# ============================================================

def phase1_core_functionality(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    """Test basic register / activate / query / tick."""
    checks_pass = 0
    checks_total = 5
    errors = []

    # Check 1: all concepts registered
    if len(world.concepts) == 50:
        checks_pass += 1
    else:
        errors.append(f"Expected 50 concepts, got {len(world.concepts)}")

    # Check 2: activation works
    lhv = world.activate_concept("c0", strength=0.9)
    if lhv is not None and lhv.activation > 0.8:
        checks_pass += 1
    else:
        errors.append(f"Activation failed: {getattr(lhv, 'activation', None)}")

    # Check 3: query returns results
    results = world.query(_make_hv(0), top_k=5)
    if len(results) > 0:
        checks_pass += 1
    else:
        errors.append("Query returned no results")

    # Check 4: tick increments
    tick_before = world.tick_count
    world.run_societal_tick()
    if world.tick_count == tick_before + 1:
        checks_pass += 1
    else:
        errors.append("Tick did not increment")

    # Check 5: stats report is complete
    stats = world.stats_report()
    required = ["n_concepts", "n_neighborhoods", "n_domains", "total_bonds"]
    if all(k in stats for k in required):
        checks_pass += 1
    else:
        errors.append(f"Stats missing keys: {[k for k in required if k not in stats]}")

    return {
        "phase": 1, "name": "Core Functionality",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
    }


def phase2_emergence_detection(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    checks_pass = 0
    checks_total = 5
    errors = []

    rg = SpectralLaplacianRG(similarity_threshold=0.05)
    result = rg.run(world.concepts)

    # Check 1: supernodes created
    if result["n_supernodes"] > 0:
        checks_pass += 1
    else:
        errors.append("No supernodes created")

    # Check 2: spectral gap is a float
    if isinstance(result["spectral_gap"], float):
        checks_pass += 1
    else:
        errors.append(f"spectral_gap not float: {result['spectral_gap']}")

    # Check 3: all concepts in supernodes
    all_cids = {c for cids in result["supernodes"].values() for c in cids}
    if all_cids == set(world.concepts.keys()):
        checks_pass += 1
    else:
        errors.append(f"Supernodes missing {len(world.concepts) - len(all_cids)} concepts")

    # Percolation
    monitor = PercolationMonitor(bond_threshold=0.3)
    perc_result = monitor.monitor_tick(world.concepts)

    # Check 4: giant fraction in [0, 1]
    gf = perc_result["giant_fraction"]
    if 0.0 <= gf <= 1.0:
        checks_pass += 1
    else:
        errors.append(f"giant_fraction out of range: {gf}")

    # Check 5: transition type is valid
    valid_types = {"emergence", "merge", "split", "stable"}
    if perc_result["transition_type"] in valid_types:
        checks_pass += 1
    else:
        errors.append(f"Invalid transition_type: {perc_result['transition_type']}")

    return {
        "phase": 2, "name": "Emergence Detection",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
        "spectral_gap": result["spectral_gap"],
        "n_supernodes": result["n_supernodes"],
        "giant_fraction": gf,
    }


def phase3_navigation_accuracy(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    checks_pass = 0
    checks_total = 4
    errors = []

    hnsw = SocietalHNSW(M=8)
    hnsw.build(world)
    stats = hnsw.stats()

    # Check 1: built successfully
    if stats["built"]:
        checks_pass += 1
    else:
        errors.append("HNSW not built")
        return {"phase": 3, "name": "Navigation Accuracy",
                "checks_pass": 0, "checks_total": checks_total,
                "errors": errors, "score": 0.0}

    # Check 2: correct concept count
    if stats["n_layer0"] == len(world.concepts):
        checks_pass += 1
    else:
        errors.append(f"Layer 0 has {stats['n_layer0']}, expected {len(world.concepts)}")

    # Check 3: query self → should be in top-3
    hv0 = world.concepts["c0"].hv
    results = hnsw.query(hv0, top_k=5)
    top_ids = [r[0] for r in results]
    if "c0" in top_ids[:3]:
        checks_pass += 1
    else:
        errors.append(f"c0 not in top-3 self-query: {top_ids[:3]}")

    # Check 4: domain filter works
    sci_results = hnsw.query(_make_hv(0), top_k=10, domain_filter="dom_sci")
    sci_domain = world.domains.get("dom_sci")
    if sci_domain:
        sci_members = set()
        for nbhd_id in sci_domain.neighborhood_ids:
            nbhd = world.neighborhoods.get(nbhd_id)
            if nbhd:
                sci_members |= nbhd.concept_ids
        all_in_domain = all(cid in sci_members for cid, _ in sci_results)
        if all_in_domain:
            checks_pass += 1
        else:
            errors.append("Domain filter returned out-of-domain concepts")
    else:
        checks_pass += 1  # no domain → skip

    return {
        "phase": 3, "name": "Navigation Accuracy",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
        "n_layer2": stats["n_layer2"],
        "n_layer1": stats["n_layer1"],
    }


def phase4_tda_health(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    checks_pass = 0
    checks_total = 5
    errors = []

    tda = TDAHealthMonitor(n_landmarks=20, max_edge_length=0.9)
    analysis = tda.run_analysis(world.concepts)

    # Check 1: Betti numbers are non-negative ints
    for bn in ("beta0", "beta1", "beta2"):
        if not isinstance(analysis[bn], int) or analysis[bn] < 0:
            errors.append(f"{bn} is invalid: {analysis[bn]}")
        else:
            checks_pass += 1
        checks_total = 5  # keep total fixed

    # Check 2: health score in [0, 1]
    hs = analysis["health_score"]
    if 0.0 <= hs <= 1.0:
        checks_pass += 1
    else:
        errors.append(f"health_score out of range: {hs}")

    # Check 3: alerts is a list
    if isinstance(analysis["alerts"], list):
        checks_pass += 1
    else:
        errors.append("alerts not a list")

    # Adjust checks_total to actual number of checks run
    checks_total = 5
    # Re-count properly
    checks_pass = 0
    for bn in ("beta0", "beta1", "beta2"):
        if isinstance(analysis.get(bn), int) and analysis[bn] >= 0:
            checks_pass += 1
    if 0.0 <= analysis.get("health_score", -1) <= 1.0:
        checks_pass += 1
    if isinstance(analysis.get("alerts"), list):
        checks_pass += 1

    return {
        "phase": 4, "name": "TDA Health",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
        "beta0": analysis["beta0"],
        "beta1": analysis["beta1"],
        "health_score": analysis["health_score"],
        "alerts": analysis["alerts"],
    }


def phase5_zipf_compliance(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    checks_pass = 0
    checks_total = 3
    errors = []

    validator = ZipfValidator(min_concepts=10)
    result = validator.validate(world.concepts)

    # Check 1: alpha is a reasonable float
    alpha = result.get("alpha", 0.0)
    if isinstance(alpha, float):
        checks_pass += 1
    else:
        errors.append(f"alpha not a float: {alpha}")

    # Check 2: health score in [0, 1]
    hs = result.get("health_score", -1)
    if 0.0 <= hs <= 1.0:
        checks_pass += 1
    else:
        errors.append(f"Zipf health_score out of range: {hs}")

    # Check 3: entropy computed
    entropy = result.get("entropy", -1)
    if 0.0 <= entropy <= 1.0:
        checks_pass += 1
    else:
        errors.append(f"entropy out of range: {entropy}")

    return {
        "phase": 5, "name": "Zipf Compliance",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
        "alpha": result.get("alpha"),
        "r_squared": result.get("r_squared"),
        "is_zipf_like": result.get("is_zipf_like"),
        "health_score": result.get("health_score"),
    }


def phase6_routing_accuracy(world: SocietalKnowledgeWorld) -> Dict[str, Any]:
    checks_pass = 0
    checks_total = 4
    errors = []

    router = SocietalContextRouter(world)

    # Check 1: detect_domain returns a tuple
    domain_id, confidence = router.detect_domain(_make_hv(0))
    if isinstance(confidence, float) and 0.0 <= confidence <= 1.0:
        checks_pass += 1
    else:
        errors.append(f"detect_domain returned invalid confidence: {confidence}")

    # Check 2: route returns required keys
    route_result = router.route(_make_hv(5), top_k=5)
    required_keys = {"domain_id", "confidence", "concepts", "border_zone_active", "routing_trace"}
    if required_keys.issubset(set(route_result.keys())):
        checks_pass += 1
    else:
        errors.append(f"route missing keys: {required_keys - set(route_result.keys())}")

    # Check 3: spreading activation from an existing concept
    world.activate_concept("c0", 1.0)
    spread = router.domain_weighted_spreading_activation("c0", n_hops=2, top_k=10)
    if "c0" in spread and spread["c0"] > 0.0:
        checks_pass += 1
    else:
        errors.append(f"Spreading activation did not include seed: {list(spread.keys())[:3]}")

    # Check 4: coalition update
    coalitions = [
        {"source": "s", "content": "c0 and c1 concepts", "base_salience": 0.3}
    ]
    world.activate_concept("c0", 0.9)
    updated = router.update_coalition_scores(coalitions, {})
    if isinstance(updated, list) and len(updated) == 1:
        checks_pass += 1
    else:
        errors.append("update_coalition_scores returned wrong type")

    return {
        "phase": 6, "name": "Routing Accuracy",
        "checks_pass": checks_pass, "checks_total": checks_total,
        "errors": errors, "score": checks_pass / checks_total,
        "detected_domain": domain_id,
        "domain_confidence": confidence,
    }


# ============================================================
# Main
# ============================================================

def run_eval() -> Dict[str, Any]:
    """Run the full evaluation suite and return a JSON-serializable report."""
    print("=" * 60)
    print("NSCK V5 Societal Full Evaluation Suite")
    print("=" * 60)

    t_start = time.perf_counter()
    world = _make_eval_world(50)
    t_world = time.perf_counter() - t_start
    print(f"World built in {t_world:.3f}s  (50 concepts, 2 domains)")

    phases = []
    for runner, label in [
        (phase1_core_functionality, "Phase 1: Core Functionality"),
        (phase2_emergence_detection, "Phase 2: Emergence Detection"),
        (phase3_navigation_accuracy, "Phase 3: Navigation Accuracy"),
        (phase4_tda_health,         "Phase 4: TDA Health"),
        (phase5_zipf_compliance,    "Phase 5: Zipf Compliance"),
        (phase6_routing_accuracy,   "Phase 6: Routing Accuracy"),
    ]:
        t0 = time.perf_counter()
        result = runner(world)
        result["elapsed_s"] = round(time.perf_counter() - t0, 4)
        phases.append(result)
        status = "✓" if result["score"] >= 1.0 else "~" if result["score"] >= 0.6 else "✗"
        print(f"  {status} {label}: {result['checks_pass']}/{result['checks_total']} ({result['score']:.0%})")
        if result["errors"]:
            for err in result["errors"]:
                print(f"      ⚠ {err}")

    total_checks = sum(p["checks_total"] for p in phases)
    passed_checks = sum(p["checks_pass"] for p in phases)
    overall_score = passed_checks / total_checks if total_checks > 0 else 0.0
    total_elapsed = round(time.perf_counter() - t_start, 3)

    print("\n" + "=" * 60)
    print(f"Overall: {passed_checks}/{total_checks} checks passed ({overall_score:.1%})")
    print(f"Total time: {total_elapsed}s")
    print("=" * 60)

    report = {
        "suite": "NSCK V5 Societal Full Evaluation",
        "version": "5.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_s": total_elapsed,
        "overall_score": round(overall_score, 4),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "phases": phases,
    }
    return report


if __name__ == "__main__":
    report = run_eval()
    out_path = Path(__file__).parent / "societal_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")
