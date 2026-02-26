"""
NSCK V11 Report Generator
=========================
Run: python nsck/scripts/generate_v11_report.py
"""
from __future__ import annotations

import os
import sys
import time

_nsck_root = os.path.join(os.path.dirname(__file__), '..')
if _nsck_root not in sys.path:
    sys.path.insert(0, _nsck_root)


def _benchmark_vsa():
    import python.core.vsa.hypervec_shim as shim
    hv1 = shim.HyperVector(1)
    hv2 = shim.HyperVector(2)
    # Warm up
    for _ in range(100):
        hv1.similarity(hv2)
    n = 10_000
    start = time.perf_counter()
    for _ in range(n):
        hv1.similarity(hv2)
    elapsed = time.perf_counter() - start
    return n / elapsed


def _run_tests():
    """Run pytest and capture results."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--no-header"],
        capture_output=True, text=True, cwd=_nsck_root, timeout=300
    )
    output = result.stdout + result.stderr
    # Parse last line for counts
    lines = output.strip().split('\n')
    summary = lines[-1] if lines else "unknown"
    return summary, result.returncode


def main():
    import python.core.vsa.hypervec_shim as shim
    info = shim.get_backend_info()

    print("Generating NSCK V11 Report...")
    vsa_ops = _benchmark_vsa()
    test_summary, test_code = _run_tests()

    report_lines = [
        "# NSCK V11 Implementation Report",
        "",
        "## What Was Implemented",
        "",
        "### Phase 1: Foundation Fixes",
        "- **1.1** LSH epoch-based index rebuilding in `episodic_memory.py`",
        "  - Added `REBUILD_INTERVAL = 500` and `_lsh_epoch_counter`",
        "  - Stale LSH entries are cleared every 500 store operations",
        "- **1.2** Continuous Generalization wired into `decide()` loop",
        "  - Config: `generalization_interval=100`, `enable_continuous_generalization=True`",
        "  - `_incremental_generalize()` runs automatically without manual `sleep()` call",
        "- **1.3** NgramNLU wired into CognitiveEngine text processing path",
        "  - Config: `enable_ngram_nlu=True`",
        "  - Intent detection adds `INTENT_*` predicates to decision loop",
        "",
        "### Phase 2: Stream & Time-Series Support",
        "- **2.1** `nsck/python/core/perception/stream_encoder.py`",
        "  - `TimeSeriesEncoder`: FPE-based HV encoding for numerical sequences",
        "  - `StreamBuffer`: Sliding window buffer emitting PerceptPackets",
        "- **2.2** `UniversalInput.encode_numeric_sequence()` added",
        "- **2.3** `NumericSequenceAdapter` in `adapters/numeric_sequence_adapter.py`",
        "  - `CognitiveEngine.decide()` auto-routes list/ndarray via NumericSequenceAdapter",
        "",
        "### Phase 3: Cross-Modal Correlation Learning",
        "- **3.1** `nsck/python/core/learning/cross_modal.py`",
        "  - `CrossModalCorrelationLearner` with Hebbian-style VSA binding",
        "  - Config: `enable_cross_modal_learning=False` (opt-in)",
        "  - Wired into `CognitiveEngine` and `NSCKSubstrate.process_multimodal()`",
        "",
        "### Phase 4: Substrate API",
        "- **4.1** `nsck/python/core/substrate.py`",
        "  - `NSCKSubstrate` wraps CognitiveEngine with clean public API",
        "  - `SubstrateResult` dataclass with action/confidence/predicates/trace",
        "  - `register_encoder()` for custom modalities",
        "  - `process()`, `process_multimodal()`, `learn()`, `sleep()`, `remember()`",
        "",
        "### Phase 5: Rust Backend",
        f"- VSA Backend: **{info['vsa_backend']}**",
        f"- SNN Backend: **{info['snn_backend']}**",
        "- `get_backend_info()` function added to `hypervec_shim.py`",
        "- `nsck/scripts/verify_rust.py` script created",
        "",
        "### Phase 6: Integration Tests",
        "- `nsck/tests/integration/test_e2e_substrate.py` — 13 end-to-end tests",
        "- `nsck/tests/integration/test_e2e_all_input_types.py` — 18 input type tests",
        "",
        "### Phase 7: Report",
        "- `nsck/scripts/generate_v11_report.py`",
        "- `NSCK_V11_REPORT.md`",
        "",
        "## Test Results",
        f"```",
        f"{test_summary}",
        f"```",
        f"Exit code: {test_code}",
        "",
        "## Performance Benchmarks",
        f"- VSA similarity ops/s: **{vsa_ops:,.0f}**",
        f"- Backend: {info['vsa_backend']}",
        "",
        "## Input Type Coverage Matrix",
        "",
        "| Input Type | Supported | Method |",
        "|---|---|---|",
        "| Plain text (str) | ✅ | DictStateAdapter + NgramNLU |",
        "| Numeric scalar (float) | ✅ | DictStateAdapter |",
        "| Numeric list | ✅ | NumericSequenceAdapter (FPE) |",
        "| NumPy 1D array | ✅ | NumericSequenceAdapter (FPE) |",
        "| NumPy 2D/3D array (image) | ✅ | Basic feature extraction |",
        "| Dict state | ✅ | DictStateAdapter |",
        "| Streaming sensor data | ✅ | StreamBuffer + TimeSeriesEncoder |",
        "| Multimodal (any combo) | ✅ | MultimodalFuser |",
        "| Custom modality | ✅ | register_encoder() |",
        "| PerceptPacket (pre-encoded) | ✅ | Direct path |",
    ]

    report = "\n".join(report_lines)
    report_path = os.path.join(_nsck_root, '..', 'NSCK_V11_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report written to {report_path}")
    print(f"VSA ops/s: {vsa_ops:,.0f}")
    print(f"Tests: {test_summary}")


if __name__ == "__main__":
    main()
