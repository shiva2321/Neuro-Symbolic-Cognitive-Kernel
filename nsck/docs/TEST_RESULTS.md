# NSCK V5 Test Results

> **Test suite:** 2088 test functions across 127 files
> **Run command:** `python -m pytest nsck/tests/ -q`

## Suite Summary

| Metric | Value |
|--------|-------|
| Total collected | 2088 |
| Passed | 1572 (unit) + 355 (integration) |
| Skipped | 94 (unit) + 67 (integration) |
| xfailed | 0 (unit) + 2 (integration — known open issues) |
| Failed | 0 |
| Run time | ~4–5 min total; unit only ~3 min |

## Integration Test Breakdown

Integration tests live in `nsck/tests/integration/`. They now pass cleanly as part of the standard suite:

| Test group | Passed | Skipped | Notes |
|-----------|--------|---------|-------|
| `test_system_capabilities.py` | ✅ | — | Rotation invariance now works (Gabor+FFT features) |
| `societal/test_transplant_societal.py` | skipped | 1 | Skipped when `transformers` not installed |
| All other integration | 355 total | 66 | Optional deps skip gracefully |

## Rust Backend

Rust backends (`hypervec_rs`, `snn_rs`, `societal_rs`) are optional. Tests degrade gracefully to the Python/NumPy fallback when Rust crates are not compiled. Tests that specifically require Rust are skipped automatically when the `.so` files are absent.

## Skipped Tests

Skips are primarily due to:
- Optional Rust backends not compiled (most common)
- Optional Python packages not installed (`torch`, `hnswlib`, `transformers`)
- Platform-specific tests

## xfailed Tests (2 in integration)

Two tests remain `xfail` (non-strict) to document known open research items:
- Open-domain NLU coverage — NgramNLU is fast but shallow
- Lifelong forgetting without SoCL — vanilla EWC shows ~16% drift at task boundaries

## Running Tests

```bash
# Standard run (full suite)
python -m pytest nsck/tests/ -q

# Unit tests only (fast, ~3 min)
python -m pytest nsck/tests/unit/ -q

# Integration tests only
python -m pytest nsck/tests/integration/ -q

# With Rust backends (if compiled — auto-detected)
python -m pytest nsck/tests/ -q
```
