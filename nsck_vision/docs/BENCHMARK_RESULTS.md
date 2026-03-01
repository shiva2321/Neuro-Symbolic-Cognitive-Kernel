# NSCK-UPMA Benchmark Results

## Methodology

All benchmarks use **synthetic data** by default to ensure reproducibility without requiring
licensed datasets. Real-dataset benchmarks can be substituted by providing a `dataset_iter`.

Random seed: `42` for all experiments.  
Hardware: TBD (run `python benchmarks/bench_latency.py` to populate with your hardware).

---

## How to Run

```bash
# All benchmarks (writes JSON results to benchmarks/results/)
cd nsck_vision
make bench

# Individual benchmarks
python benchmarks/bench_absorption.py    # absorption speed & Spearman ρ
python benchmarks/bench_latency.py       # inference latency (p50/p95/p99)
python benchmarks/bench_crossdomain.py  # cross-domain transfer accuracy
python benchmarks/bench_fusion.py        # fusion accuracy vs reference model

# Research benchmark (Paper 6)
python research/experiments/paper6_vision_absorber_benchmarks.py
```

Results are saved to `research/results/paper6_results.json`.

---

## Absorption Benchmarks

| Model | Domain | Dim | Samples | Concepts | Spearman ρ | HV/s | Time (s) |
|-------|--------|-----|---------|----------|------------|------|----------|
| ResNet-18 (synthetic) | imagenet | 512 | 100 | TBD | TBD | TBD | TBD |
| ViT-B/16 (synthetic) | imagenet | 768 | 100 | TBD | TBD | TBD | TBD |
| EfficientNet-B0 (synthetic) | imagenet | 1280 | 100 | TBD | TBD | TBD | TBD |

> Run `python benchmarks/bench_absorption.py` to populate this table.

---

## Inference Latency Benchmarks

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Target |
|---------|----------|----------|----------|--------|
| Rust (hypervec_rs) | TBD | TBD | TBD | < 100ms |
| Python fallback | TBD | TBD | TBD | < 500ms |

> Run `python benchmarks/bench_latency.py` to populate this table.

---

## Cross-Domain Transfer

| Source Domain | Target Domain | Recall@10 | ARI |
|---------------|---------------|-----------|-----|
| imagenet | medical | TBD | TBD |
| imagenet | satellite | TBD | TBD |
| medical | traffic | TBD | TBD |

> Run `python benchmarks/bench_crossdomain.py` to populate this table.

---

## Fusion Accuracy vs Reference Model

| Scenario | NSCK Accuracy | Reference Accuracy | NSCK Override Rate |
|----------|--------------|-------------------|-------------------|
| High NSCK conf | TBD | TBD | TBD |
| Low NSCK conf | TBD | TBD | TBD |
| Agreement | TBD | TBD | N/A |

> Run `python benchmarks/bench_fusion.py` to populate this table.

---

## How to Reproduce

1. Install dependencies: `pip install -e nsck_vision/ numpy scipy`
2. Run from repo root:
   ```bash
   cd /path/to/Neuro-Symbolic-Cognitive-Kernel
   python research/experiments/paper6_vision_absorber_benchmarks.py
   ```
3. Results are written to `research/results/paper6_results.json`
4. For real-dataset benchmarks, set `IMAGENET_VAL_PATH` environment variable and
   re-run with `--real-data` flag (see each benchmark script's `--help`).

---

## Notes

- Spearman ρ measures how well pairwise cosine similarities in embedding space
  are preserved as Hamming similarities in HV space (JL guarantee).
- Recall@10 measures what fraction of queries find their true class in the
  top-10 nearest HVs.
- ARI (Adjusted Rand Index) measures clustering quality after absorption.
- ECE (Expected Calibration Error) measures confidence calibration.
