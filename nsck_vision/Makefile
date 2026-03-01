PYTHON := python
PYTEST := python -m pytest

test:
	NSCK_USE_RUST=1 $(PYTEST) nsck_vision/tests/ -q --tb=short

test-e2e:
	NSCK_USE_RUST=1 $(PYTEST) nsck_vision/tests/test_system_e2e.py -v

test-fast:
	NSCK_USE_RUST=1 $(PYTEST) nsck_vision/tests/ -q --tb=short -m "not slow"

bench:
	NSCK_USE_RUST=1 $(PYTHON) nsck_vision/benchmarks/run_all_benchmarks.py

dashboard:
	NSCK_USE_RUST=1 $(PYTHON) -m nsck_vision.dashboard.app --port 5091

.PHONY: test test-e2e test-fast bench dashboard
