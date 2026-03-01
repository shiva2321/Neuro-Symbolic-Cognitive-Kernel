"""Unit tests for FeatureAbsorber."""
import os
import sys
import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_absorber():
    """Create a minimal FeatureAbsorber for testing."""
    from python.core.vision.absorption_memory import AbsorptionMemory
    from python.core.vision.domain_tagger import DomainTagger
    from python.core.vision.feature_absorber import FeatureAbsorber
    mem = AbsorptionMemory()
    tagger = DomainTagger()
    return FeatureAbsorber(absorption_memory=mem, domain_tagger=tagger), mem, tagger


def test_absorb_with_labels():
    """Labeled absorption produces correct concept→HV mapping."""
    absorber, mem, _ = _make_absorber()
    rng = np.random.default_rng(42)
    n = 10
    dataset_iter = [(rng.standard_normal(64).astype(np.float32), f"cat_{i%3}") for i in range(n)]

    def identity_model(x): return x

    report = absorber.absorb(
        model=identity_model,
        model_id="test_labeled",
        domain="test_domain",
        dataset_iter=dataset_iter,
        max_samples=n,
    )
    assert report.n_concepts_absorbed > 0
    assert report.passed


def test_absorb_unlabeled():
    """Unlabeled absorption clusters by similarity."""
    absorber, mem, _ = _make_absorber()
    rng = np.random.default_rng(55)
    n = 10
    dataset_iter = [(rng.standard_normal(32).astype(np.float32), None) for _ in range(n)]

    def identity_model(x): return x

    report = absorber.absorb(
        model=identity_model,
        model_id="test_unlabeled",
        domain="unlabeled_domain",
        dataset_iter=dataset_iter,
        max_samples=n,
    )
    assert report.n_concepts_absorbed >= 0
    assert report.model_id == "test_unlabeled"


def test_absorb_parallel():
    """Two models absorbed simultaneously, both appear in AbsorptionMemory."""
    from python.core.vision.absorption_memory import AbsorptionMemory
    from python.core.vision.domain_tagger import DomainTagger
    from python.core.vision.feature_absorber import FeatureAbsorber
    mem = AbsorptionMemory()
    tagger = DomainTagger()
    absorber = FeatureAbsorber(absorption_memory=mem, domain_tagger=tagger)

    rng = np.random.default_rng(77)

    def identity_model(x): return x

    specs = [
        {
            "model": identity_model,
            "model_id": "parallel_model_1",
            "domain": "domain_1",
            "dataset_iter": [(rng.standard_normal(32).astype(np.float32), f"c{i}") for i in range(5)],
            "max_samples": 5,
        },
        {
            "model": identity_model,
            "model_id": "parallel_model_2",
            "domain": "domain_2",
            "dataset_iter": [(rng.standard_normal(32).astype(np.float32), f"c{i}") for i in range(5)],
            "max_samples": 5,
        },
    ]

    reports = absorber.absorb_batch(specs)
    assert len(reports) == 2
    model_ids = {r.model_id for r in reports}
    assert "parallel_model_1" in model_ids
    assert "parallel_model_2" in model_ids


def test_absorption_memory_persistence(tmp_path):
    """Save + load AbsorptionMemory, record count matches."""
    absorber, mem, _ = _make_absorber()
    rng = np.random.default_rng(88)
    n = 5
    dataset_iter = [(rng.standard_normal(32).astype(np.float32), f"cls_{i}") for i in range(n)]

    def identity_model(x): return x

    absorber.absorb(
        model=identity_model,
        model_id="persist_test",
        domain="persist_domain",
        dataset_iter=dataset_iter,
        max_samples=n,
    )

    original_count = len(mem._records)
    save_path = str(tmp_path / "absorption_memory.pkl")
    mem.save(save_path)

    from python.core.vision.absorption_memory import AbsorptionMemory
    new_mem = AbsorptionMemory()
    new_mem.load(save_path)
    assert len(new_mem._records) == original_count
