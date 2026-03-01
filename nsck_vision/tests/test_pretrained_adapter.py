"""Unit tests for PretrainedModelAdapter."""
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


def test_numpy_adapter():
    """Load numpy fallback adapter, extract features from random arrays."""
    from python.core.vision.pretrained_adapter import PretrainedModelAdapter

    adapter = PretrainedModelAdapter.load(None, source="numpy")
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((2, 64)).astype(np.float32)
    features = adapter.extract_features(arr)
    assert features.shape[0] == 2
    assert features.dtype == np.float32


def test_callable_adapter():
    """Custom callable adapter works correctly."""
    from python.core.vision.pretrained_adapter import PretrainedModelAdapter

    def my_model(x):
        return np.asarray(x, dtype=np.float32) * 2.0

    adapter = PretrainedModelAdapter.load(my_model, source="callable")
    arr = np.ones((3, 32), dtype=np.float32)
    features = adapter.extract_features(arr)
    assert features.shape[0] == 3


@pytest.mark.slow
def test_torchvision_adapter():
    """Load resnet18, extract features from random image tensor."""
    try:
        import torch
        import torchvision
    except ImportError:
        pytest.skip("torch/torchvision not available")

    from python.core.vision.pretrained_adapter import PretrainedModelAdapter

    adapter = PretrainedModelAdapter.load("resnet18", source="torchvision")
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
    features = adapter.extract_features(arr)
    assert features.shape[0] == 2
    assert features.ndim == 2


def test_layer_names():
    """get_layer_names() returns a list."""
    from python.core.vision.pretrained_adapter import PretrainedModelAdapter

    adapter = PretrainedModelAdapter.load(None, source="numpy")
    names = adapter.get_layer_names()
    assert isinstance(names, list)


def test_predictions_schema():
    """get_predictions() returns PredictionResult with required fields."""
    from python.core.vision.pretrained_adapter import PretrainedModelAdapter, PredictionResult

    adapter = PretrainedModelAdapter.load(None, source="numpy")
    arr = np.random.default_rng(10).standard_normal((1, 64)).astype(np.float32)
    preds = adapter.get_predictions(arr)
    assert isinstance(preds, list)
    if preds:
        p = preds[0]
        assert hasattr(p, "label")
        assert hasattr(p, "confidence")
        assert hasattr(p, "top5")
        assert hasattr(p, "features")
        assert 0.0 <= p.confidence <= 1.0
