# NSCK V15 Changelog

## V15.0.0 — Model Transplantation Pipeline (2026-02)

### New Feature: Model Transplantation Pipeline (WP-1 through WP-6)

NSCK V15 introduces the flagship **Model Transplantation Pipeline**, enabling any pretrained neural network (BERT, GPT, ViT, Whisper, CLIP, etc.) to have its learned knowledge absorbed into NSCK's native 10,240-bit binary hypervector space.

#### New Modules

| Module | Location | Description |
|--------|----------|-------------|
| `ModelHarvester` | `nsck/python/core/transplant/harvester.py` | Extracts embedding matrices from PyTorch models via 4 strategies |
| `HarvestResult` | `nsck/python/core/transplant/harvester.py` | Dataclass with embeddings, vocab_mapping, model_type, metadata |
| `RandomProjector` | `nsck/python/core/transplant/projector.py` | JL random projection: `hv = sign(e·P)` |
| `LearnedProjector` | `nsck/python/core/transplant/projector.py` | Gradient-descent cosine-preservation projection |
| `SVDFactoredProjector` | `nsck/python/core/transplant/projector.py` | SVD+FPE codebook encoding (default) |
| `STDPCalibrator` | `nsck/python/core/transplant/calibrator.py` | SNN+STDP fine-tuning of HV codebooks |
| `CalibratedResult` | `nsck/python/core/transplant/calibrator.py` | Calibration output with quality curve |
| `TransplantValidator` | `nsck/python/core/transplant/validator.py` | Quality metrics: Spearman ρ, Recall@k, ARI |
| `TransplantReport` | `nsck/python/core/transplant/validator.py` | Complete quality report |
| `TransplantPipeline` | `nsck/python/core/transplant/pipeline.py` | End-to-end orchestration |

#### Configuration Changes

9 new fields added to `NSCKConfig`:

```python
enable_transplant: bool = False                # feature flag (off by default)
transplant_strategy: str = "svd_factored"      # projection strategy
transplant_calibration_epochs: int = 10        # STDP refinement epochs
transplant_validation_threshold: float = 0.80  # minimum ρ for pass
transplant_svd_components: int = 128           # SVD components
transplant_fpe_bins: int = 256                 # FPE quantization bins
transplant_batch_size: int = 512
transplant_sample_pairs: int = 10000
transplant_auto_live_encoding: bool = True
```

New config factory:

```python
NSCKConfig.transplant()   # preset with all transplant flags enabled
```

#### Substrate Changes

New `NSCKSubstrate.transplant()` method:

```python
report = substrate.transplant(
    model=bert,           # any PyTorch model or duck-typed object
    domain_name="nlp",    # label for the knowledge domain
    strategy="svd_factored",
    calibration_epochs=10,
    save_pack="/tmp/nlp.kp",
)
```

New `_transplant_projectors: Dict[str, BaseProjector]` field for live encoding.

New `get_stats()` field: `"transplant_domains"`.

#### New Tests

| Test File | Tests |
|-----------|-------|
| `nsck/tests/unit/transplant/test_harvester.py` | 11 tests |
| `nsck/tests/unit/transplant/test_projector.py` | 14 tests |
| `nsck/tests/unit/transplant/test_calibrator.py` | 9 tests |
| `nsck/tests/unit/transplant/test_validator.py` | 9 tests |
| `nsck/tests/unit/transplant/test_pipeline.py` | 14 tests |
| `nsck/tests/integration/test_transplant_e2e.py` | 7 tests |
| **Total new** | **64 tests** |

#### New Evaluation

- `nsck/eval/bench_transplant.py` — benchmarks all 3 strategies at vocab scales 100/1K/10K

#### New Documentation

- `nsck/docs/TRANSPLANT_GUIDE.md` — user-facing guide with examples
- `nsck/docs/TRANSPLANT_REPORT.md` — quality metrics and benchmark results
- `nsck/docs/V15_CHANGELOG.md` — this file

### Technical Notes

- **Feature-flagged**: `enable_transplant=False` (default) → zero existing behaviour changes
- **PyTorch is optional**: all code paths work without PyTorch installed; harvester fails gracefully
- **Reuses existing patterns**: FPE codebook (from `image_adapter.py`/`audio_adapter.py`), `PythonSnnCore`/`PythonStdpEngine`, `KnowledgePack`, `SemanticMemory`
- **SVD capped at 10,000 rows** for large-vocabulary performance

### Previous Versions

See `nsck/docs/V14_CHANGELOG.md` for V14 changes.
