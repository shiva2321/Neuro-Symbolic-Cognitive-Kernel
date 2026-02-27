# NSCK V15 Transplant Guide

## Overview

The **Model Transplantation Pipeline** (V15) enables NSCK to absorb learned knowledge from any pretrained neural network — BERT, GPT, ViT, Whisper, CLIP, etc. — by converting their embedding spaces into NSCK's native 10,240-bit binary hypervector space.

## Quick Start

```python
from python.core.substrate import NSCKSubstrate
from python.core.integration.config import NSCKConfig

# Enable transplantation
config = NSCKConfig.transplant()          # or NSCKConfig(enable_transplant=True)
substrate = NSCKSubstrate(config)

# Transplant a model (PyTorch or duck-typed)
import torch
bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
report = substrate.transplant(model=bert, domain_name="language")

print(f"Transplant passed: {report.passed}")
print(f"Spearman ρ: {report.spearman_rho:.3f}")
print(f"Recall@10: {report.recall_at_10:.3f}")
```

## Architecture

```
External Model (BERT/GPT/ViT/...)
           │
    ┌──────▼──────┐
    │  Harvester  │  ← extract embedding matrix
    └──────┬──────┘
           │ HarvestResult(embeddings, vocab_mapping, ...)
    ┌──────▼──────┐
    │  Projector  │  ← convert float→binary HVs
    └──────┬──────┘
           │ Dict[token → HyperVector]
    ┌──────▼──────┐
    │  Calibrator │  ← STDP SNN fine-tuning (optional)
    └──────┬──────┘
           │ CalibratedResult
    ┌──────▼──────┐
    │  Validator  │  ← measure quality metrics
    └──────┬──────┘
           │ TransplantReport
    ┌──────▼──────┐
    │  Integrate  │  ← inject into SemanticMemory
    └─────────────┘
```

## Projection Strategies

### `"random"` — RandomProjector (baseline)

Johnson–Lindenstrauss random projection:

```
hv = sign(e · P)  where P ∈ ℝ^(d × 10240), P_ij ~ N(0,1)
```

**Pros**: Fast, deterministic, guaranteed JL similarity preservation.  
**Cons**: No domain-specific adaptation.

### `"learned"` — LearnedProjector

Gradient-descent projection trained on cosine similarity preservation:

```
L = Σ (cosine(e_a, e_b) − hamming_sim(sign(e_a·P), sign(e_b·P)))²
```

**Pros**: Adapts to the embedding space structure.  
**Cons**: Requires training time.

### `"svd_factored"` — SVDFactoredProjector (default, recommended)

SVD dimensionality reduction + FPE codebook encoding:

```
E_c = E − mean(E)
U, Σ, Vᵀ = SVD(E_c)
z_i = Vᵀ[:k] · e_i           # reduce to k components
hv_i = FPE_bundle(z_i)        # encode via FPE codebooks
```

**Pros**: Matches NSCK's FPE pattern, efficient, preserves coarse cluster structure.  
**Cons**: FPE is discrete — fine-grained similarity ordering not fully preserved.

## Configuration

```python
NSCKConfig(
    enable_transplant=True,
    transplant_strategy="svd_factored",   # "random" | "learned" | "svd_factored"
    transplant_calibration_epochs=10,     # 0 = skip calibration
    transplant_validation_threshold=0.80, # minimum ρ for pass
    transplant_svd_components=128,        # SVD components
    transplant_fpe_bins=256,              # FPE quantization bins
    transplant_batch_size=512,
    transplant_sample_pairs=10000,
    transplant_auto_live_encoding=True,
)
```

Or use the preset:

```python
config = NSCKConfig.transplant()
```

## Quality Metrics

| Metric | Description | Minimum |
|--------|-------------|---------|
| Spearman ρ | Rank correlation: original cosine vs HV Hamming similarity | 0.80 |
| Recall@10 | % of true top-10 neighbours also found in HV space | 0.70 |
| Recall@50 | % of true top-50 neighbours also found in HV space | 0.60 |
| ARI | Adjusted Rand Index between k-means clusters | 0.65 |

## Manual Pipeline Usage

```python
from python.core.transplant.pipeline import TransplantPipeline

pipe = TransplantPipeline(config)
report = pipe.run(
    model=my_model,
    domain_name="vision",
    strategy="svd_factored",
    calibration_epochs=5,
    save_pack_path="/tmp/vision_knowledge.kp",
    cognitive_engine=my_engine,          # optional: inject into SemanticMemory
)
```

## Live Encoding

After transplanting, future inputs can be encoded using the fitted projector:

```python
# After transplant
projector = substrate._transplant_projectors["language"]
new_hv = projector.encode_new(my_embedding_vector)
```

## Feature Flag

The transplant feature is **disabled by default** (`enable_transplant=False`).  
All existing behaviour is unchanged when the flag is off.

## Module Reference

| Class | Location | Description |
|-------|----------|-------------|
| `ModelHarvester` | `transplant/harvester.py` | Extracts embeddings from neural models |
| `HarvestResult` | `transplant/harvester.py` | Result dataclass |
| `RandomProjector` | `transplant/projector.py` | JL random projection |
| `LearnedProjector` | `transplant/projector.py` | Gradient-descent projection |
| `SVDFactoredProjector` | `transplant/projector.py` | SVD+FPE projection |
| `STDPCalibrator` | `transplant/calibrator.py` | SNN+STDP refinement |
| `CalibratedResult` | `transplant/calibrator.py` | Calibration output |
| `TransplantValidator` | `transplant/validator.py` | Quality measurement |
| `TransplantReport` | `transplant/validator.py` | Quality report |
| `TransplantPipeline` | `transplant/pipeline.py` | Orchestrates full flow |
