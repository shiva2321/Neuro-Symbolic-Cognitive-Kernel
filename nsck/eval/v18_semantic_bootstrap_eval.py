#!/usr/bin/env python3
"""
NSCK V18 Semantic Bootstrap End-to-End Evaluation
==================================================
Measures semantic similarity improvement from V18 bootstrap vs baseline
(hash-seeded co-occurrence codebook) and exercises downstream systems.

Run from repository root::

    PYTHONPATH=nsck python3 nsck/eval/v18_semantic_bootstrap_eval.py

Outputs a formatted report to stdout and saves JSON metrics to
``nsck/eval/results/v18_eval_report.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

# ── ensure NSCK on path ──────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ── Rust backend detection ───────────────────────────────────────────────────
import python.core.vsa.hypervec_shim as _hv_mod
USE_RUST = getattr(_hv_mod, "_USE_RUST", False)

# ── benchmark pairs ──────────────────────────────────────────────────────────
PAIRS: List[Tuple[str, str]] = [
    ("brain", "memory"),
    ("king", "queen"),
    ("cat", "dog"),
    ("hot", "cold"),
    ("fast", "slow"),
    ("doctor", "hospital"),
    ("code", "program"),
    ("learn", "knowledge"),
]

TRAINING_SENTENCES: List[str] = [
    # Cognition
    "The brain stores memories through neural connections.",
    "Memory consolidation happens during sleep.",
    "Learning changes behaviour through experience.",
    "Attention focuses cognitive resources on a task.",
    "Perception interprets signals from the senses.",
    # Science
    "Neural networks are inspired by the brain.",
    "Machine learning identifies patterns in data.",
    "Algorithms are step-by-step instructions for computers.",
    "Data structures organise information efficiently.",
    "Artificial intelligence simulates human reasoning.",
    # Biology
    "Neurons transmit electrical signals through synapses.",
    "DNA carries the genetic blueprint of organisms.",
    "Cells are the basic unit of life.",
    "Evolution explains the diversity of species.",
    "Proteins are built from amino acids.",
    # Technology
    "Code is a set of instructions executed by a computer.",
    "Programs transform input data into output results.",
    "Software controls hardware through instructions.",
    "The internet connects computers worldwide.",
    "Cloud computing provides remote processing power.",
]


def _col(text: str, code: str) -> str:
    """Simple ANSI colour helper."""
    return f"\033[{code}m{text}\033[0m"


def run_eval() -> Dict:
    from python.core.language.semantic_bootstrap import SemanticBootstrapper
    from python.core.language.distributional_semantics import DistributionalCodebook

    print(_col("═" * 60, "1"))
    print(_col("  NSCK V18 Semantic Bootstrap Evaluation", "1"))
    print(_col("═" * 60, "1"))
    print(f"  Rust Backend:  {'ENABLED ✓' if USE_RUST else 'disabled (Python fallback)'}")
    print()

    # ── Baseline (corpus strategy) ───────────────────────────────────────────
    t0 = time.perf_counter()
    baseline_cb = SemanticBootstrapper.build_codebook(strategy="corpus")
    t_baseline = time.perf_counter() - t0

    # ── Bootstrap (auto strategy: bridge → corpus fallback) ──────────────────
    t0 = time.perf_counter()
    boot_cb = SemanticBootstrapper.build_codebook(strategy="auto")
    t_bootstrap = time.perf_counter() - t0

    # Detect which tier was actually used
    boot_words = len(boot_cb._codebook)
    from python.core.language.cognitive_vocabulary import COGNITIVE_VOCABULARY
    bridge_words_in_cb = sum(1 for w in COGNITIVE_VOCABULARY if boot_cb.get_hv(w) is not None)
    used_bridge = bridge_words_in_cb > len(baseline_cb._codebook)
    strategy_label = "bridge (Tier 1)" if used_bridge else "corpus fallback (Tier 3)"

    print(f"  Strategy:      auto → {strategy_label}")
    print(f"  Baseline words: {len(baseline_cb._codebook)}")
    print(f"  Bootstrap words: {boot_words}")
    print(f"  Baseline build time:  {t_baseline:.3f}s")
    print(f"  Bootstrap build time: {t_bootstrap:.3f}s")
    print()

    # ── Semantic similarity benchmarks ───────────────────────────────────────
    print(_col("  SEMANTIC SIMILARITY BENCHMARKS", "1"))
    print("  " + "─" * 52)
    print(f"  {'Pair':<24} {'Baseline':>8}  {'Bootstrap':>9}  {'Delta':>6}")
    print("  " + "─" * 52)

    sim_results = {}
    for w1, w2 in PAIRS:
        base_sim = baseline_cb.similarity(w1, w2)
        boot_sim = boot_cb.similarity(w1, w2)
        delta = boot_sim - base_sim
        check = "✓" if boot_sim > base_sim else "~"
        pair_label = f"{w1} ↔ {w2}"
        print(f"  {pair_label:<24} {base_sim:>8.3f}  {boot_sim:>9.3f}  "
              f"{delta:>+6.3f}  {check}")
        sim_results[f"{w1}_{w2}"] = {"baseline": base_sim, "bootstrap": boot_sim, "delta": delta}

    print()

    # ── Downstream system tests ───────────────────────────────────────────────
    print(_col("  DOWNSTREAM SYSTEM TESTS", "1"))
    print("  " + "─" * 52)

    downstream_pass = True

    # TextKnowledgeLearner + SemanticMemory
    try:
        from python.core.integration.config import NSCKConfig
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        from python.core.memory.semantic_memory import SemanticMemory

        cfg = NSCKConfig.semantic()
        sm = SemanticMemory()
        learner = TextKnowledgeLearner(semantic_memory=sm, config=cfg)
        for sent in TRAINING_SENTENCES:
            learner.learn_from_text(sent)

        # Query SemanticMemory with brain HV
        brain_hv = sm.concept_hvs.get("Brain") or sm.concept_hvs.get("brain")
        if brain_hv is not None:
            results = sm.query(brain_hv, k=5)
            result_names = [r[0] if isinstance(r, (list, tuple)) else str(r) for r in results]
            print(f"  SemanticMemory.query(brain_hv):  {result_names[:5]}  ✓")
        else:
            print("  SemanticMemory.query(brain_hv):  brain HV not found  ~")

        # Spread activation
        activation = sm.spread_activation(["Brain", "brain", "memory"], steps=3)
        if activation:
            top = sorted(activation.items(), key=lambda x: -x[1])[:4]
            top_str = ", ".join(f"{k}:{v:.2f}" for k, v in top)
            print(f"  spread_activation([brain]):  {top_str}  ✓")
        else:
            print("  spread_activation([brain]):  no activation (expected without edges)  ~")

    except Exception as exc:
        print(f"  TextKnowledgeLearner test FAILED: {exc}")
        downstream_pass = False

    # CognitiveEngine end-to-end
    try:
        from python.core.substrate import NSCKSubstrate
        cfg2 = NSCKConfig.semantic()
        substrate = NSCKSubstrate(cfg2)
        resp = substrate.process("The brain stores memories through neural connections.")
        has_response = bool(resp and str(resp).strip())
        print(f"  CognitiveEngine.process():  {'PASS ✓' if has_response else 'empty response ~'}")
    except Exception as exc:
        print(f"  CognitiveEngine.process() FAILED: {exc}")
        downstream_pass = False

    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    avg_delta = sum(v["delta"] for v in sim_results.values()) / max(len(sim_results), 1)
    verdict = "PASSED" if avg_delta >= 0 else "NEUTRAL"
    print(_col(f"  VERDICT: V18 Bootstrap {verdict} — avg Δsim = {avg_delta:+.3f}", "1"))
    print(_col("═" * 60, "1"))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "rust_backend": USE_RUST,
        "strategy": strategy_label,
        "baseline_words": len(baseline_cb._codebook),
        "bootstrap_words": boot_words,
        "similarities": sim_results,
        "avg_delta": avg_delta,
        "downstream_pass": downstream_pass,
        "verdict": verdict,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v18_eval_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Report saved → {out_path}")

    return report


if __name__ == "__main__":
    run_eval()
