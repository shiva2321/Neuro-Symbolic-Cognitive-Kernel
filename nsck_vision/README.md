# NSCK-UPMA: Universal Pretrained Model Absorber

> Absorb any pretrained vision model into a Neuro-Symbolic Cognitive Kernel via Hyperdimensional Vector Symbolic Architectures.

## Overview

NSCK-UPMA lets you take an existing pretrained model (ResNet, ViT, CLIP, any callable) and "absorb" its knowledge into NSCK's 10,240-bit binary HyperVector memory.  Once absorbed, the system can:

- **Classify images** without running the original model at inference time
- **Explain decisions** via causal chains and cross-domain analogies
- **Combine knowledge** from multiple models across different domains
- **Run a real-time dashboard** showing accuracy, provenance, and reasoning traces

## Quick Start

```python
from nsck_vision.system import NSCKVisionSystem

system = NSCKVisionSystem()
system.absorb("resnet18", domain="imagenet", max_samples=500)
response = system.analyze("path/to/image.jpg")
print(response.label, f"{response.accuracy_rating:.1%}")
```

## Architecture

```
Pretrained Model (ResNet / ViT / CLIP / callable)
        │
        ▼
PretrainedModelAdapter   ← unified feature extraction interface
        │
        ▼
VSAProjector             ← JL random projection → 10,240-bit HV
  (SVDFactoredProjector or RandomProjector)
        │
        ▼
AbsorptionMemory         ← persistent HV store with provenance
        │
        ▼
NSCKVisionFusion         ← neuro-symbolic fusion + accuracy estimator
        │
        ▼
VisionResponse           ← label, confidence, accuracy_rating, causal_chain, …
```

The entire pipeline is deterministic: the same model_id always produces the same seed matrix, ensuring reproducibility across runs.

## Installation

```bash
pip install -e nsck_vision/
# Optional: for torchvision / HuggingFace adapters
pip install torch torchvision transformers
```

## Absorbing Models

### torchvision

```python
report = system.absorb("resnet18", domain="imagenet", max_samples=1000)
print(f"Absorbed {report.n_concepts_absorbed} concepts, ρ={report.spearman_rho:.3f}")
```

### HuggingFace ViT

```python
report = system.absorb(
    "google/vit-base-patch16-224",
    domain="imagenet_vit",
    max_samples=500,
)
```

### Custom callable

```python
import torch, torchvision

model = torchvision.models.efficientnet_b0(pretrained=True).eval()

def feature_extractor(x):
    with torch.no_grad():
        return model.features(torch.tensor(x).unsqueeze(0)).mean([2, 3]).numpy()

report = system.absorb(feature_extractor, domain="custom", max_samples=200, model_id="effnet_b0")
```

### Multiple models in parallel

```python
specs = [
    {"model": "resnet50",  "domain": "imagenet",  "model_id": "rn50",  "max_samples": 500},
    {"model": "vgg16",     "domain": "imagenet",  "model_id": "vgg16", "max_samples": 500},
]
reports = system.absorb_parallel(specs)
```

## Analyzing Images

```python
# Single image
response = system.analyze("cat.jpg")

# With reference model comparison
result = system.compare("cat.jpg")
print(result.nsck_label, result.accuracy_rating)

# Batch
responses = system.analyze_batch(["a.jpg", "b.jpg"])
```

## VisionResponse Fields

| Field | Type | Description |
|-------|------|-------------|
| `label` | str | Top predicted label |
| `confidence` | float | NSCK similarity score [0,1] |
| `accuracy_rating` | float | Calibrated accuracy estimate [0,1] |
| `top5` | list | Top-5 (label, score) pairs |
| `causal_chain` | list | Symbolic reasoning chain |
| `cross_domain_analogies` | list | Related concepts from other domains |
| `source_model_provenance` | list | Which absorbed models contributed |
| `nsck_overrode_reference` | bool | Whether NSCK overrode the reference model |
| `latency_ms` | float | End-to-end inference latency |

## Running the Dashboard

```bash
cd nsck_vision
python dashboard/app.py
# → http://localhost:5000
```

The dashboard provides:
- Live image upload and analysis
- Absorbed model registry browser
- Real-time accuracy and latency metrics
- Causal chain visualization

## Running Benchmarks

```bash
# All benchmarks
cd nsck_vision
make bench

# Individual benchmarks
python benchmarks/bench_absorption.py
python benchmarks/bench_latency.py
python benchmarks/bench_crossdomain.py
python benchmarks/bench_fusion.py
```

## Running Tests

```bash
# Fast tests (no slow mark)
pytest nsck_vision/tests/ -m "not slow" -q

# All tests including slow
pytest nsck_vision/tests/ -q
```

## Benchmark Results

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for detailed results.

| Metric | Target | Status |
|--------|--------|--------|
| Absorption latency | < 500ms/model | ✓ |
| Inference latency | < 100ms/image | ✓ |
| Spearman ρ (JL preservation) | > 0.3 | ✓ |
| Recall@10 | > 0.7 | ✓ |

## Contributing

1. Fork the repository
2. Add tests under `nsck_vision/tests/`
3. Ensure `pytest nsck_vision/tests/ -m "not slow"` passes
4. Submit a PR with description of what model/domain you added

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full details.

## License

Apache 2.0 — see [LICENSE](../LICENSE).
