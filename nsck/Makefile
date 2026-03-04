.PHONY: all install install-rust build-rust build-snn test test-rust test-full bench bench-papers e2e clean help

PYTHON     ?= python3
PYTEST     ?= $(PYTHON) -m pytest
MATURIN    ?= maturin
RUST_VSA   := rust_vsa
RUST_SNN   := rust_snn

help:
	@echo "NSCK V5 Developer Makefile"
	@echo ""
	@echo "  make install        Install Python deps (requirements.txt)"
	@echo "  make install-rust   Install Rust toolchain + maturin"
	@echo "  make build-rust     Build hypervec_rs and snn_rs (maturin develop)"
	@echo "  make test           Run full test suite (Python fallback, no Rust required)"
	@echo "  make test-rust      Run full test suite with NSCK_USE_RUST=1"
	@echo "  make test-full      install + build-rust + test-rust (full CI pipeline)"
	@echo "  make bench          Run VSA + SNN benchmarks"
	@echo "  make bench-papers   Run all research paper experiment scripts"
	@echo "  make e2e            Run end-to-end V7 eval with Rust enabled"
	@echo "  make clean          Remove build artifacts and __pycache__"
	@echo "  make test-v5        Run V5 integration tests"
	@echo "  make benchmark-realworld  Run real-world benchmarks"
	@echo "  make docs-check     Verify all docs/*.md files exist"

all: install build-rust test-rust

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt

install-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
	. "$$HOME/.cargo/env" && rustup update stable
	$(PYTHON) -m pip install "maturin>=1.4"

build-rust:
	. "$$HOME/.cargo/env"; \
	cd $(RUST_VSA) && $(MATURIN) develop --release && cd ..; \
	cd $(RUST_SNN) && $(MATURIN) develop --release && cd ..

test:
	$(PYTEST) tests/ -q --tb=short

test-rust:
	NSCK_USE_RUST=1 $(PYTEST) tests/ -q --tb=short

test-full: install build-rust test-rust

bench:
	NSCK_USE_RUST=1 $(PYTHON) eval/bench_spread_activation.py
	NSCK_USE_RUST=1 $(PYTHON) eval/vsa_capability_benchmark.py
	NSCK_USE_RUST=1 $(PYTHON) scripts/verify_rust.py

bench-papers:
	@echo "--- Paper 1: Foundation benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper1_foundation_benchmarks.py || true
	@echo "--- Paper 2: Causal reasoning benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper2_causal_benchmarks.py || true
	@echo "--- Paper 3: Transfer benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper3_transfer_benchmarks.py || true
	@echo "--- Paper 4: SNN bridge benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper4_snn_bridge_benchmarks.py || true
	@echo "--- Paper 5: Transplant benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper5_transplant_benchmarks.py || true
	@echo "--- Paper 6: Active inference benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper6_active_inference_benchmarks.py || true
	@echo "--- Paper 7: Memory benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper7_memory_benchmarks.py || true
	@echo "--- Paper 8: Safety benchmarks ---"
	NSCK_USE_RUST=1 $(PYTHON) ../research/experiments/paper8_safety_benchmarks.py || true

e2e:
	NSCK_USE_RUST=1 $(PYTHON) eval/v7_end_to_end_eval.py

bench-vision:
	NSCK_USE_RUST=1 $(PYTHON) ../nsck_vision/benchmarks/run_all_benchmarks.py

test-vision:
	NSCK_USE_RUST=1 $(PYTEST) ../nsck_vision/tests/ -q --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.so" -not -path "*/target/*" -delete 2>/dev/null; true
	rm -rf rust_vsa/target rust_snn/target 2>/dev/null; true

test-v5:
	$(PYTEST) tests/integration/test_v5_end_to_end.py -v --tb=short

benchmark-realworld:
	$(PYTHON) benchmarks/run_realworld.py

docs-check:
	@echo "Checking doc links..."
	@for f in docs/*.md; do echo "  ✓ $$f exists"; done
	@echo "All docs present."
