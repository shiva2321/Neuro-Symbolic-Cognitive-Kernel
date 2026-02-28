"""
Comprehensive test suite for the snn_rs Rust extension.
Tests every class: LIFLayer, StdpEngine, SnnCore, HebbianMatrix, ConceptMapper, RateCoder.
Also benchmarks Rust vs Python (SNNPerceptionModule) for perceive() speed.
"""

import sys, time, math
import numpy as np
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]  # nsck/rust_snn/tests/ -> repo root
_nsck_root = _repo_root / "nsck"
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_nsck_root))

# ── import Rust extension ────────────────────────────────────────────────────
import snn_rs
from snn_rs import LIFLayer, StdpEngine, SnnCore, HebbianMatrix, ConceptMapper, RateCoder

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SEP   = "─" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(label, cond, detail=""):
    tag = PASS if cond else FAIL
    msg = f"  {tag}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return cond

results = []

# ════════════════════════════════════════════════════════════
# 1. LIFLayer
# ════════════════════════════════════════════════════════════
section("1. LIFLayer — Leaky Integrate-and-Fire")

lif = LIFLayer(
    n_neurons=64, tau=20.0, v_rest=-70.0,
    v_reset=-75.0, v_thresh=-55.0,
    refractory_period=2.0, dt=1.0
)

results.append(check("n_neurons property",  lif.n_neurons == 64))
results.append(check("dt property",         lif.dt        == 1.0))
results.append(check("repr",                "LIFLayer" in repr(lif)))

# Step with zero current → no spikes
spikes = lif.step([0.0] * 64)
results.append(check("step() zero current → no spikes", sum(spikes) == 0,
                      f"got {sum(spikes)} spikes"))

# Step with large current → should spike.
# Threshold equation: v_rest + (dt/tau)*I >= v_thresh
# -70 + (1/20)*I >= -55  =>  I >= 300. Use 1500 to be safe.
LARGE = 1500.0
spikes_big = lif.step([LARGE] * 64)
results.append(check("step() large current → spikes", sum(spikes_big) > 0,
                      f"got {sum(spikes_big)}/64 spikes"))

# spike_history length matches steps called
lif.reset()
for _ in range(10):
    lif.step([50.0] * 64)
history = lif.get_spike_train()
results.append(check("get_spike_train() length == n_steps", len(history) == 10,
                      f"len={len(history)}"))
results.append(check("get_spike_train() inner width == n_neurons", len(history[0]) == 64))

# reset clears history
lif.reset()
results.append(check("reset() clears spike_history", len(lif.get_spike_train()) == 0))

# ════════════════════════════════════════════════════════════
# 2. StdpEngine
# ════════════════════════════════════════════════════════════
section("2. StdpEngine — Spike-Timing-Dependent Plasticity")

stdp = StdpEngine(
    input_dim=32, snn_size=16,
    stdp_lr=0.001, tau_stdp=20.0,
    a_plus=0.01, a_minus=0.005
)

results.append(check("initial updates == 0", stdp.updates == 0))
results.append(check("initial time == 0.0",  stdp.current_time == 0.0))

# advance time
stdp.advance_time(1.0)
results.append(check("advance_time() works", abs(stdp.current_time - 1.0) < 1e-9))

# apply with all-zero spikes → delta_w all zero
dw = stdp.apply([0.0]*32, [0.0]*16)
results.append(check("apply() all-zero spikes → zero delta_w",
                     all(abs(x) < 1e-12 for x in dw),
                     f"len={len(dw)}"))
results.append(check("updates incremented after apply", stdp.updates == 1))

# apply with pre spike then post spike → LTP
stdp.reset()
stdp.apply([1.0]*32, [0.0]*16)   # record pre spikes at t=0
stdp.advance_time(5.0)
dw_ltp = stdp.apply([0.0]*32, [1.0]*16)  # post fires 5ms later → LTP
results.append(check("LTP: post-after-pre → positive delta_w",
                     any(x > 0 for x in dw_ltp),
                     f"max={max(dw_ltp):.6f}"))

# reset clears state
stdp.reset()
results.append(check("reset() clears current_time", stdp.current_time == 0.0))

# ════════════════════════════════════════════════════════════
# 3. HebbianMatrix
# ════════════════════════════════════════════════════════════
section("3. HebbianMatrix — Oja's Rule")

hebb = HebbianMatrix(
    in_features=16, out_features=8,
    learning_rate=0.01, decay=0.001, normalize=True
)

results.append(check("repr contains HebbianMatrix", "HebbianMatrix" in repr(hebb)))
results.append(check("initial update_count == 0", hebb.update_count == 0))

# forward pass shape
y = hebb.forward([1.0] * 16)
results.append(check("forward() output length == out_features", len(y) == 8,
                      f"len={len(y)}"))

# wrong input length
try:
    hebb.forward([1.0] * 5)
    results.append(check("forward() rejects wrong input size", False))
except Exception:
    results.append(check("forward() rejects wrong input size", True))

# hebbian update changes weights
w_before = hebb.get_weights()
hebb.hebbian_update([1.0]*16, [1.0]*8)
w_after = hebb.get_weights()
results.append(check("hebbian_update() changes weights",
                     any(abs(a-b) > 1e-10 for a,b in zip(w_before, w_after))))
results.append(check("update_count incremented", hebb.update_count == 1))

# set/get roundtrip
new_w = [0.5] * (16 * 8)
hebb.set_weights(new_w)
got = hebb.get_weights()
results.append(check("set/get weights roundtrip",
                     all(abs(g - 0.5) < 1e-9 for g in got)))

# wrong weight count
try:
    hebb.set_weights([0.0] * 5)
    results.append(check("set_weights() rejects wrong count", False))
except Exception:
    results.append(check("set_weights() rejects wrong count", True))

# ════════════════════════════════════════════════════════════
# 4. ConceptMapper
# ════════════════════════════════════════════════════════════
section("4. ConceptMapper — Jaccard Concept Recognition")

cm = ConceptMapper()
results.append(check("initial n_concepts == 0", cm.n_concepts() == 0))

# recognize on empty mapper → (-1, 0.0)
cid, sim = cm.recognize_pattern([1, 2, 3], threshold=0.5)
results.append(check("recognize on empty → -1", cid == -1))

# register concept
id0 = cm.register_concept([1, 2, 3, 4, 5])
results.append(check("register_concept() returns 0", id0 == 0))
results.append(check("n_concepts == 1 after register", cm.n_concepts() == 1))

# recognize exact match → similarity 1.0
cid, sim = cm.recognize_pattern([1, 2, 3, 4, 5], threshold=0.5)
results.append(check("exact match found", cid == 0, f"cid={cid}"))
results.append(check("exact match similarity == 1.0", abs(sim - 1.0) < 1e-9,
                      f"sim={sim:.4f}"))

# register a second concept
id1 = cm.register_concept([10, 20, 30])
results.append(check("second concept gets id 1", id1 == 1))

# partial match (overlap [1,2,3] with [1,2,3,4,5] → J=3/5=0.6)
cid2, sim2 = cm.recognize_pattern([1, 2, 3], threshold=0.5)
results.append(check("partial match above threshold found", cid2 == 0,
                      f"cid={cid2}, sim={sim2:.3f}"))

# no match below threshold
cid3, sim3 = cm.recognize_pattern([99, 98], threshold=0.5)
results.append(check("no match below threshold → -1", cid3 == -1,
                      f"sim={sim3:.3f}"))

# concept_hv_seed
seed = ConceptMapper.concept_hv_seed(0)
results.append(check("concept_hv_seed(0) == 1000", seed == 1000))

# ════════════════════════════════════════════════════════════
# 5. RateCoder
# ════════════════════════════════════════════════════════════
section("5. RateCoder — Spike Train → Firing Rates")

rc = RateCoder(n_neurons=8, rate_threshold=0.0)
results.append(check("n_neurons property", rc.n_neurons == 8))

# flat spike train: 10 steps × 8 neurons, all ones → all fire
flat = [1.0] * (10 * 8)
active, rates = rc.encode(flat, time_window_ms=10.0)
results.append(check("all-ones spike train → all 8 active", len(active) == 8,
                      f"active={len(active)}"))
results.append(check("spike_rates length == n_neurons", len(rates) == 8))
results.append(check("all rates > 0", all(r > 0 for r in rates)))

# all-zeros → no active neurons
rc_thresh = RateCoder(n_neurons=8, rate_threshold=0.01)
flat_zero = [0.0] * (10 * 8)
active_z, _ = rc_thresh.encode(flat_zero, 10.0)
results.append(check("all-zeros spike train → 0 active", len(active_z) == 0))

# wrong length
try:
    rc.encode([1.0] * 5, 10.0)
    results.append(check("encode() rejects non-divisible length", False))
except Exception:
    results.append(check("encode() rejects non-divisible length", True))

# ════════════════════════════════════════════════════════════
# 6. SnnCore — full perceive() inner loop
# ════════════════════════════════════════════════════════════
section("6. SnnCore — Full Loop (LIF + Weight Proj + STDP)")

# v_thresh=-64.0 (easy to reach with normalised weights + sensory * 2.0 amplification)
# steady-state: v_ss = v_rest + I; need v_ss >= -64 => I >= 6 (tau=10 scales dv slowly but
# over 50 steps the neuron integrates well above threshold)
core = SnnCore(
    input_dim=32, snn_size=64,
    tau=10.0, v_rest=-70.0, v_reset=-75.0,
    v_thresh=-64.0, refractory_period=2.0, dt=1.0,
    stdp_enabled=True, stdp_lr=0.0001,
    tau_stdp=20.0, a_plus=0.001, a_minus=0.0005,
    seed=42
)

results.append(check("repr", "SnnCore" in repr(core)))
results.append(check("input_dim property", core.input_dim == 32))
results.append(check("snn_size property",  core.snn_size  == 64))

# sensory=1.0 maximises dot-product magnitude with random normalised weights
sensory = [1.0] * 32

# simulate — returns list of lists (n_steps × snn_size)
spike_train = core.simulate(sensory, n_steps=50, learn=True)
results.append(check("simulate() returns 50 steps", len(spike_train) == 50,
                      f"len={len(spike_train)}"))
results.append(check("each step has snn_size spikes", len(spike_train[0]) == 64,
                      f"len={len(spike_train[0])}"))

# With L2-normalised random weights the dot-product is ~N(0,1) — many neurons
# never reach threshold.  Explicitly set weights to 0.5 per synapse
# so current = dot(w, 1)*2 = 32*2 = 64 >> threshold delta of 6 mV → fires in step 2.
core.set_weights([0.5] * (64 * 32))
spike_check = core.simulate([1.0] * 32, n_steps=50, learn=False)
total_spikes = sum(sum(row) for row in spike_check)
results.append(check("network fires spikes with explicit weights", total_spikes > 0,
                      f"total={total_spikes:.0f}"))

# STDP updates incremented
results.append(check("stdp_updates > 0 after learning", core.stdp_updates > 0,
                      f"updates={core.stdp_updates}"))

# weight roundtrip
w = core.get_weights()
results.append(check("get_weights() length == snn_size×input_dim",
                     len(w) == 64 * 32, f"len={len(w)}"))

# set_weights
new_w = [0.1] * (64 * 32)
core.set_weights(new_w)
w2 = core.get_weights()
results.append(check("set_weights() roundtrip",
                     all(abs(x - 0.1) < 1e-9 for x in w2)))

# wrong weight count
try:
    core.set_weights([0.0] * 5)
    results.append(check("set_weights() rejects wrong count", False))
except Exception:
    results.append(check("set_weights() rejects wrong count", True))

# simulate without learning
st2 = core.simulate(sensory, n_steps=10, learn=False)
results.append(check("simulate() learn=False still returns spike train",
                     len(st2) == 10))

# ════════════════════════════════════════════════════════════
# 7. Benchmark — Rust SnnCore vs Python SNNPerceptionModule
# ════════════════════════════════════════════════════════════
section("7. Benchmark — Rust vs Python perceive() speed")

N_RUNS  = 20
INPUT   = np.random.randn(32).astype(np.float64)

# ── Rust ─────────────────────────────────────────────────────
rust_core = SnnCore(
    input_dim=32, snn_size=64,
    tau=10.0, v_rest=-70.0, v_reset=-75.0,
    v_thresh=-64.0, refractory_period=2.0, dt=1.0,
    stdp_enabled=True, stdp_lr=0.0001,
    tau_stdp=20.0, a_plus=0.001, a_minus=0.0005,
    seed=0
)

t0 = time.perf_counter()
for _ in range(N_RUNS):
    rust_core.simulate([1.0]*32, n_steps=50, learn=True)
rust_ms = (time.perf_counter() - t0) / N_RUNS * 1000

print(f"\n  Rust  SnnCore.simulate()     avg {rust_ms:.3f} ms / perceive()")

# ── Python ───────────────────────────────────────────────────
try:
    from python.core.perception.snn_perception import SNNPerceptionModule
    py_module = SNNPerceptionModule(
        input_dim=32, snn_size=64, hv_dimension=1024,
        n_concepts=20, simulation_time_ms=50.0
    )
    # warm-up
    py_module.perceive(INPUT.astype(np.float32), learn=False)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        py_module.perceive(INPUT.astype(np.float32), learn=True)
    py_ms = (time.perf_counter() - t0) / N_RUNS * 1000

    print(f"  Python SNNPerceptionModule avg {py_ms:.3f} ms / perceive()")
    speedup = py_ms / rust_ms if rust_ms > 0 else float("inf")
    print(f"\n  Speedup:  {speedup:.1f}×  faster in Rust")
    results.append(check("Rust faster than Python", speedup > 1.0,
                         f"{speedup:.1f}×"))
except Exception as e:
    print(f"  Python benchmark skipped: {e}")

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
section("SUMMARY")
passed = sum(results)
total  = len(results)
failed = total - passed
print(f"  Passed: {passed}/{total}")
if failed:
    print(f"  FAILED: {failed} tests")
print()
sys.exit(0 if failed == 0 else 1)
