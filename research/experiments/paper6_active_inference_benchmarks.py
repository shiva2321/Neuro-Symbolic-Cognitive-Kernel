"""
Paper 6 Experiments: Active Inference + Global Workspace Theory
===============================================================
Run from: cd nsck && python ../research/experiments/paper6_active_inference_benchmarks.py

Writes results to research/results/paper6_results.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve()
_nsck_dir = _here.parent.parent.parent / "nsck"
if str(_nsck_dir) not in sys.path:
    sys.path.insert(0, str(_nsck_dir))

RESULTS_DIR = _here.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "paper6_results.json"

rust_vsa_active = False
try:
    import hypervec_rs  # noqa: F401
    rust_vsa_active = True
except ImportError:
    pass
print(f"[Paper 6] Rust VSA backend: {'ACTIVE' if rust_vsa_active else 'Python fallback'}")

try:
    from python.core.learning.active_inference import ActiveInferenceLearner
    from python.core.learning.curiosity import CuriosityModule
    from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition, WorkspaceModule
    from python.core.vsa.hypervec_shim import HyperVector
    print("[Paper 6] Modules imported OK")
except ImportError as e:
    print(f"[Paper 6] ERROR: {e}")
    sys.exit(1)

results = {"backend": {"rust_vsa": rust_vsa_active}}


class _DummyModule(WorkspaceModule):
    """Minimal workspace module for testing."""
    def __init__(self):
        self.received = []
    def receive_broadcast(self, content):
        self.received.append(content)


# ── Experiment 6.1 — Free Energy Computation ─────────────────────────────────
def exp_6_1():
    print("\n[Exp 6.1] Free Energy Computation...")
    rng = np.random.default_rng(42)
    curiosity = CuriosityModule(novelty_threshold=0.7)
    ai = ActiveInferenceLearner(curiosity_module=curiosity, safety_threshold=0.9)

    n_steps = 200
    actions = ["explore", "exploit", "rest"]
    fe_trace = []

    for step in range(n_steps):
        state_hv = HyperVector(seed=rng.integers(0, 1000))
        action = actions[step % len(actions)]
        fe = ai.free_energy(action, state_hv)
        fe_trace.append({"step": step, "action": action, "free_energy": fe})

        # Update world model occasionally
        if step % 5 == 0:
            next_hv = HyperVector(seed=rng.integers(0, 1000))
            ai.update_world_model(state_hv, action, next_hv)

    # Sample every 10 steps for the report
    sampled = fe_trace[::20]
    mean_fe = float(np.mean([r["free_energy"] for r in fe_trace]))
    print(f"  mean_free_energy={mean_fe:.4f}  n_steps={n_steps}")
    return {
        "free_energy_trace": sampled,
        "mean_free_energy": mean_fe,
        "n_steps": n_steps,
    }


# ── Experiment 6.2 — Exploration-Exploitation Transition ─────────────────────
def exp_6_2():
    print("\n[Exp 6.2] Exploration-Exploitation Transition...")
    curiosity = CuriosityModule(novelty_threshold=0.7)
    n_steps = 200
    explore_decisions = []

    for step in range(n_steps):
        # Use a fixed set of HVs to simulate increasing familiarity
        hv_seed = step % 20  # 20 distinct situations, revisited
        state_hv = HyperVector(seed=hv_seed + 1)
        decision = curiosity.should_explore(
            situation_hv=state_hv,
            task_tag="test_task",
            confidence=0.5,
        )
        explore_decisions.append({
            "step": step,
            "should_explore": decision.should_explore,
            "novelty_score": decision.novelty_score,
        })
        # Record visit to build familiarity
        curiosity.record_visit(state_hv, "test_task")

    # Compute rolling explore rate
    window = 20
    explore_rate_trace = []
    for i in range(0, n_steps, window):
        chunk = explore_decisions[i:i + window]
        rate = float(np.mean([d["should_explore"] for d in chunk]))
        explore_rate_trace.append({"step": i, "explore_rate": rate})

    print(f"  initial_rate={explore_rate_trace[0]['explore_rate']:.3f}  final_rate={explore_rate_trace[-1]['explore_rate']:.3f}")
    return {
        "explore_rate_trace": explore_rate_trace,
        "total_explorations": sum(1 for d in explore_decisions if d["should_explore"]),
        "n_steps": n_steps,
    }


# ── Experiment 6.3 — World Model Learning ─────────────────────────────────────
def exp_6_3():
    print("\n[Exp 6.3] World Model Learning...")
    rng = np.random.default_rng(7)
    ai = ActiveInferenceLearner(safety_threshold=0.9)

    # 10 distinct state→action→next_state triples
    n_transitions = 10
    states = [HyperVector(seed=i + 100) for i in range(n_transitions)]
    next_states = [HyperVector(seed=i + 200) for i in range(n_transitions)]
    action = "move"

    update_checkpoints = [1, 5, 10, 25, 50, 100]
    n_updates = 0
    pe_trace = []

    max_updates = max(update_checkpoints)
    while n_updates < max_updates:
        idx = n_updates % n_transitions
        ai.update_world_model(states[idx], action, next_states[idx])
        n_updates += 1

        if n_updates in update_checkpoints:
            # Measure prediction error for known transitions
            errors = []
            for i in range(n_transitions):
                ai._last_states[ai._hv_key(states[i])] = next_states[i]
                err = ai.prediction_error(action, states[i])
                errors.append(err)
            mean_pe = float(np.mean(errors))
            pe_trace.append({"n_updates": n_updates, "mean_prediction_error": mean_pe})
            print(f"  n_updates={n_updates}  mean_pe={mean_pe:.4f}")

    return {"world_model_learning": pe_trace}


# ── Experiment 6.4 — GWT Coalition Competition ───────────────────────────────
def exp_6_4():
    print("\n[Exp 6.4] GWT Coalition Competition...")
    gw = GlobalWorkspace(attention_threshold=0.3)
    m1, m2, m3 = _DummyModule(), _DummyModule(), _DummyModule()
    gw.register_module("vision", m1)
    gw.register_module("language", m2)
    gw.register_module("motor", m3)

    proposals = [
        Coalition(source="vision", content="see_object", base_salience=0.9, relevance=0.2),
        Coalition(source="language", content="say_word", base_salience=0.5, relevance=0.4),
        Coalition(source="motor", content="move_arm", base_salience=0.3, relevance=0.1),
    ]
    winner = gw.compete(proposals)

    result = {
        "winner_source": winner.source if winner else None,
        "winner_activation": float(winner.activation) if winner else None,
        "coalition_scores": [
            {"source": c.source, "activation": float(c.activation)}
            for c in proposals
        ],
        "kle_uncertainty": float(gw._kle_uncertainty),
    }
    print(f"  winner={result['winner_source']}  activation={result['winner_activation']:.3f}")
    return {"gwt_competition": result}


# ── Experiment 6.5 — Mental Rehearsal Veto ────────────────────────────────────
def exp_6_5():
    print("\n[Exp 6.5] Mental Rehearsal Veto...")
    gw = GlobalWorkspace(attention_threshold=0.1)
    gw.veto_threshold = 0.5  # Lower threshold for testing

    # Register danger vectors
    danger_hvs = [HyperVector(seed=i + 500) for i in range(5)]
    for hv in danger_hvs:
        gw._danger_vectors.append(hv)

    n_dangerous = 5
    n_safe = 5
    n_vetoed = 0

    class _MockWorldModel:
        def __init__(self, dangerous_hvs):
            self._dangerous = dangerous_hvs
            self._step = 0
        def imagine(self, state_hv, action_hv):
            # Returns (pred_state_bits, pred_reward) as required by compete_with_rehearsal
            self._step += 1
            if self._step % 2 == 0:
                # Return bits of a dangerous HV → triggers veto
                dv = self._dangerous[self._step % len(self._dangerous)]
                bits = np.array(dv.bits, dtype=np.float32)
            else:
                # Return bits of a safe (random) HV
                bits = np.array(HyperVector(seed=self._step + 9999).bits, dtype=np.float32)
            return bits, 0.0

    wm = _MockWorldModel(danger_hvs)
    current_state = HyperVector(seed=1)

    for trial in range(n_dangerous + n_safe):
        props = [Coalition(source=f"src{trial}", content=f"act{trial}",
                           base_salience=0.8, relevance=0.1)]
        winner = gw.compete_with_rehearsal(
            proposals=props,
            current_state_hv=current_state,
            world_model=wm,
            get_action_hv_fn=lambda act: HyperVector(seed=hash(act) % 10000),
            n_cycles=1,
        )
        # Count vetoes: either None winner or emergency fallback
        if winner is None or getattr(winner, "source", None) == "EMERGENCY":
            n_vetoed += 1

    result = {
        "n_proposals": n_dangerous + n_safe,
        "n_vetoed": n_vetoed,
        "n_rehearsal_events": len(gw.rehearsal_log),
    }
    print(f"  n_vetoed={n_vetoed}/{n_dangerous + n_safe}  rehearsal_events={len(gw.rehearsal_log)}")
    return {"mental_rehearsal_veto": result}


# ── Experiment 6.6 — Free Energy Surprise ────────────────────────────────────
def exp_6_6():
    print("\n[Exp 6.6] Free Energy Surprise...")
    ai = ActiveInferenceLearner(safety_threshold=0.9)
    rng = np.random.default_rng(11)

    # Vary how well the world model was trained
    n_train_levels = [0, 1, 5, 20, 50]
    results_table = []

    for n_train in n_train_levels:
        state_hv = HyperVector(seed=100)
        next_hv = HyperVector(seed=200)

        for _ in range(n_train):
            ai.update_world_model(state_hv, "action_test", next_hv)

        # Measure prediction error (≈ surprise)
        ai._last_states[ai._hv_key(state_hv)] = next_hv
        surprise = ai.prediction_error("action_test", state_hv)
        results_table.append({
            "n_training_updates": n_train,
            "surprise_score": surprise,
        })
        print(f"  n_train={n_train}  surprise={surprise:.4f}")

    return {"free_energy_surprise": results_table}


# ── Run all experiments ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, fn in [
        ("exp_6_1", exp_6_1),
        ("exp_6_2", exp_6_2),
        ("exp_6_3", exp_6_3),
        ("exp_6_4", exp_6_4),
        ("exp_6_5", exp_6_5),
        ("exp_6_6", exp_6_6),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"[{name}] FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Paper 6] Results written to {RESULTS_FILE}")
