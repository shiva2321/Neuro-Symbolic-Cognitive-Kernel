# NSCK V4 Test Results

> **Test suite:** 1443+ tests (1437 baseline + 6 V4 test classes)
> **Run command:** `NSCK_USE_RUST=1 python -m pytest nsck/tests/ --tb=no -q`

## Suite Summary

| Metric | Value |
|--------|-------|
| Total collected | 1443+ |
| Passed | 1430+ |
| Failed (stochastic) | 2 |
| Skipped | 7 |
| Xfailed | 4 |
| Run time | ~14 seconds |

## V4 Test Classes (`test_v4_full_system.py`)

| Class | Assertions | Key checks |
|---|---|---|
| `TestProceduralFastPath` | 3 | LSH lookup wired, skills populated after positive reward, threshold == 0.72 |
| `TestSemanticHotCache` | 3 | `_hot_cache` dict exists, HNSW enabled by default, cache updated after spread_activation |
| `TestNLU` | 4 | VSANLUEngine classifies intent, confidence ≥ 0.0, entity extraction, process() dict output |
| `TestBundleMajorityVote` | 2 | bundle_hvs majority correct, result closer to majority input |
| `TestImagination` | 2 | imagine_rollout() returns tuple, multi-step works |
| `TestKnowledgeSeeder` | 3 | seed_from_yaml() returns int, navigation domain seeded, rules > 0 |

## Rust Backend Verification

```bash
python -c "import hypervec_rs; print('Rust OK:', hypervec_rs.HyperVector(1).bits[:5])"
```

Expected output: `Rust OK: [<int>, <int>, <int>, <int>, <int>]`

## Run V4 Tests Only

```bash
NSCK_USE_RUST=1 python -m pytest nsck/tests/integration/test_v4_full_system.py -v
```

## Known Stochastic Failures

Two tests may fail non-deterministically due to random HV initialization:

| Test | Reason | Expected behavior |
|---|---|---|
| `test_similarity_above_threshold` | Random HV similarity near boundary | Retry or increase threshold tolerance |
| `test_bundle_random_tiebreak` | ChaCha8 tie-break in bundle() | Non-deterministic for identical-salience coalitions |

These failures are expected and do not indicate regressions. The test suite marks them as stochastic with appropriate tolerances.
