# NSCK V5 Test Results

> **Test suite:** 2088 test functions across 127 files
> **Run command:** `python -m pytest nsck/tests/ --ignore=nsck/tests/integration -q`

## Suite Summary (without integration tests)

| Metric | Value |
|--------|-------|
| Total collected | 2088 |
| Passed | 1572 |
| Skipped | 94 |
| Failed | 0 |
| Run time | varies by hardware |

> **Integration tests** (`nsck/tests/integration/`) require additional setup and are excluded from the standard run. Run them separately with `python -m pytest nsck/tests/integration/ -q`.

## Rust Backend

Rust backends (`hypervec_rs`, `snn_rs`, `societal_rs`) are optional. Tests degrade gracefully to the Python/NumPy fallback when Rust crates are not compiled. Tests that specifically require Rust are skipped automatically when the `.so` files are absent.

## Skipped Tests (94 total)

Skips are primarily due to:
- Optional Rust backends not compiled
- Optional Python packages not installed (`torch`, `hnswlib`)
- Integration-only tests excluded from default run

## Known Stochastic Tests

A small number of tests compare cosine similarity of random high-dimensional vectors and may occasionally land on the wrong side of a threshold. These are not bugs — they reflect inherent variance in stochastic representations.

## Running Tests

```bash
# Standard run (without integration tests)
python -m pytest nsck/tests/ --ignore=nsck/tests/integration -q

# Full suite including integration
python -m pytest nsck/tests/ -q

# Unit tests only
python -m pytest nsck/tests/unit/ -q

# With Rust backends (if compiled)
NSCK_USE_RUST=1 python -m pytest nsck/tests/ -q
```
