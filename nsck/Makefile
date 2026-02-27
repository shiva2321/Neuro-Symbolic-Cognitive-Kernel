.PHONY: test build-rust bench clean

test:
	python3 -m pytest tests/ -q

build-rust:
	cd rust_vsa && cargo build --release

bench:
	python3 eval/bench_spread_activation.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
