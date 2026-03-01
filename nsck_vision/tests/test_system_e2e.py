"""End-to-end tests for NSCK-UPMA with Rust backend."""
import os
import sys
import numpy as np
import pytest

os.environ["NSCK_USE_RUST"] = "1"

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def vision_system():
    """Create NSCKVisionSystem for tests."""
    from nsck_vision.system import NSCKVisionSystem
    return NSCKVisionSystem()


def test_full_pipeline_basic(vision_system):
    """Basic pipeline: absorb → analyze → VisionResponse."""
    rng = np.random.default_rng(42)
    n_train = 10
    dim = 64

    dataset = [(rng.standard_normal(dim).astype(np.float32), f"class_{i%5}") for i in range(n_train)]

    def identity_model(x): return x

    report = vision_system.absorb(
        model_or_name=identity_model,
        domain="test_general",
        dataset=dataset,
        max_samples=n_train,
        model_id="e2e_test_model",
    )
    assert report.n_concepts_absorbed >= 0

    test_img = rng.standard_normal(dim).astype(np.float32)
    response = vision_system.analyze(test_img)

    assert hasattr(response, "query_id")
    assert hasattr(response, "label")
    assert hasattr(response, "confidence")
    assert hasattr(response, "accuracy_rating")
    assert hasattr(response, "causal_chain")
    assert hasattr(response, "cross_domain_analogies")
    assert hasattr(response, "source_model_provenance")
    assert hasattr(response, "latency_ms")
    assert response.latency_ms < 5000, f"Latency {response.latency_ms:.1f}ms > 5000ms"


@pytest.mark.slow
def test_full_pipeline_with_rust(vision_system):
    """Full pipeline with Rust backend: absorb → analyze → verify."""
    rng = np.random.default_rng(42)
    dim = 64
    n_samples = 10

    synthetic_images = [rng.standard_normal(dim).astype(np.float32) for _ in range(n_samples)]
    labels = [f"class_{i%5}" for i in range(n_samples)]
    dataset = list(zip(synthetic_images, labels))

    def identity_model(x): return x

    report = vision_system.absorb(
        model_or_name=identity_model,
        domain="rust_test",
        dataset=dataset,
        max_samples=n_samples,
        model_id="rust_e2e_model",
    )

    assert report.model_id == "rust_e2e_model"
    assert report.passed
    assert isinstance(report.rust_backend_active, bool)

    for img in synthetic_images[:5]:
        response = vision_system.analyze(img)
        assert hasattr(response, "label")
        assert hasattr(response, "accuracy_rating")
        assert 0.0 <= response.confidence <= 1.0
        assert response.latency_ms < 5000


@pytest.mark.slow
def test_multi_model_absorption_e2e(vision_system):
    """Absorb 2 models sequentially → both in registry."""
    rng = np.random.default_rng(10)
    dim = 32

    def identity_model(x): return x

    for i, domain in enumerate(["domain_x", "domain_y"]):
        dataset = [(rng.standard_normal(dim).astype(np.float32), f"cls_{j}") for j in range(5)]
        vision_system.absorb(
            model_or_name=identity_model,
            domain=domain,
            dataset=dataset,
            max_samples=5,
            model_id=f"multi_model_{i}",
        )

    all_models = vision_system.registry.all_models()
    model_ids = {m["model_id"] for m in all_models}
    assert "multi_model_0" in model_ids
    assert "multi_model_1" in model_ids


@pytest.mark.slow
def test_parallel_absorption_e2e(vision_system):
    """Absorb 2 models simultaneously → both in registry, no race conditions."""
    rng = np.random.default_rng(20)
    dim = 32

    def identity_model(x): return x

    specs = [
        {
            "model": identity_model,
            "model_id": "parallel_e2e_1",
            "domain": "parallel_domain_a",
            "dataset_iter": [(rng.standard_normal(dim).astype(np.float32), f"c{j}") for j in range(5)],
            "max_samples": 5,
        },
        {
            "model": identity_model,
            "model_id": "parallel_e2e_2",
            "domain": "parallel_domain_b",
            "dataset_iter": [(rng.standard_normal(dim).astype(np.float32), f"c{j}") for j in range(5)],
            "max_samples": 5,
        },
    ]

    reports = vision_system.absorb_parallel(specs)
    assert len(reports) == 2
    ids = {r.model_id for r in reports}
    assert "parallel_e2e_1" in ids
    assert "parallel_e2e_2" in ids


def test_reference_vs_nsck_comparison(vision_system):
    """compare() returns ComparisonResult with required fields."""
    from nsck_vision.system import ComparisonResult
    rng = np.random.default_rng(30)

    def identity_model(x): return x

    dataset = [(rng.standard_normal(32).astype(np.float32), f"cls_{i%3}") for i in range(5)]
    vision_system.absorb(
        model_or_name=identity_model,
        domain="compare_domain",
        dataset=dataset,
        max_samples=5,
        model_id="compare_model",
    )

    test_img = rng.standard_normal(32).astype(np.float32)
    result = vision_system.compare(test_img)

    assert isinstance(result, ComparisonResult)
    assert hasattr(result, "nsck_label")
    assert hasattr(result, "accuracy_rating")
    assert hasattr(result, "latency_ms")
    assert result.latency_ms < 5000


def test_get_stats(vision_system):
    """get_stats() returns valid statistics dict with version field."""
    stats = vision_system.get_stats()
    assert isinstance(stats, dict)
    assert "version" in stats
    assert stats["version"] == "1.0.0"
