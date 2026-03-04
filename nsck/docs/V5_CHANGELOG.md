# NSCK V5 Changelog

## V5.0.0 (2026-03-04)

Complete architecture consolidation. All modules wired into the main decision loop.

### WP-1: Codebase Cleanup
- Deleted `nsck/archive/` (superseded legacy code)
- Removed `.venv/` and `.idea/` from version control
- Updated `.gitignore` with standard Python/IDE exclusions

### WP-2: CognitiveEngine Bug Fixes
- Removed duplicate `UniversalInput()` initialization
- Removed dead `return best_action` statement in `_get_best_action_from_q()`
- Fixed hardcoded `"maze_navigation"` in `_get_state_key()` — now uses generic position-aware routing
- Added L2 prototype normalization in `sleep()` to prevent catastrophic forgetting drift

### WP-3: Module Wiring
- **SpatialReasoner**: Wired into `decide()` coalition building for spatial predicate detection
- **BeliefRevisionEngine (BeliefScorer)**: Wired into `decide()` post-decision for low-confidence revision
- **ContextEngine**: Wired into `decide()` text processing path for word disambiguation
- **EmotionSystem**: Wired into `learn()` with reward and novelty signals
- **ContinualLearner**: Explicit per-task consolidation added to `sleep()`
- **MetaLearner**: Strategy selection wired into `register_task()`
- Added `SpatialReasoner.infer()` method for coalition-compatible interface

### WP-4: Duplicate System Consolidation
- `VSANLUEngine` established as primary NLU; `NgramNLU` explicitly as fallback
- `CrossDomainTransferPipeline` now delegates to `TransferEngine`

### WP-5: Substrate Pipeline Hardening
- Memory instances shared between substrate and engine (no duplication)
- Drift detector wired into `sleep()` — snapshots top-50 concept HVs
- Conformal calibration uses actual engine confidence

### WP-6: Integration Tests
- Created `nsck/tests/integration/test_v5_end_to_end.py` with 6 test classes
- All tests pass in ~13 seconds

### WP-7: Benchmark Harness
- Created `nsck/benchmarks/realworld_harness.py` with Iris + text classification
- Created `nsck/benchmarks/run_realworld.py` as CLI entry point

### WP-8: Documentation
- Updated REPOMAP.md, MODULE_REFERENCE.md, WIRING_MAP.md, WORKFLOWS.md, FORMULAS.md, TESTING.md, BENCHMARKS.md, GOAL_TRACKER.md
- Created new: QUICKSTART.md, API_REFERENCE.md, V5_CHANGELOG.md, V5_VALIDATION_REPORT.md
- Updated ARCHITECTURE.md with V5 mermaid diagram
- Updated README.md with V5 version, new doc links

### WP-9: Repository Hygiene
- Version bumped to 5.0.0 in `pyproject.toml`
- Makefile updated with `test-v5`, `benchmark-realworld`, `docs-check` targets
- `help` target updated to say "NSCK V5"

### WP-10: Validation
- Created `docs/V5_VALIDATION_REPORT.md`

---

## Previous Releases

See [V4_CHANGELOG.md](V4_CHANGELOG.md), [V17_CHANGELOG.md](V17_CHANGELOG.md),
[V15_CHANGELOG.md](V15_CHANGELOG.md), and [V14_CHANGELOG.md](V14_CHANGELOG.md)
for prior release notes.
