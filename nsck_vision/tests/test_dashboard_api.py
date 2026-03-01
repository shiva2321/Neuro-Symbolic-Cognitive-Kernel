"""Integration tests for the Flask dashboard API."""
import os
import sys
import json
import base64
import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


try:
    from flask import Flask
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FLASK_AVAILABLE, reason="Flask not available")


@pytest.fixture
def client():
    """Create a test Flask client."""
    import nsck_vision.dashboard.api as api_module
    api_module._vision_system = None
    from nsck_vision.dashboard.api import create_api
    app = create_api()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _make_dummy_image_b64() -> str:
    """Create a small random raw byte array encoded as base64."""
    rng = np.random.default_rng(42)
    arr = (rng.integers(0, 255, (3 * 224 * 224,), dtype=np.uint8)).tobytes()
    return base64.b64encode(arr).decode()


def test_health_endpoint(client):
    """GET /api/health returns correct schema."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "status" in data
    assert "rust_active" in data
    assert "absorbed_models" in data
    assert "version" in data
    assert data["status"] == "ok"


def test_analyze_endpoint(client):
    """POST /api/vision/analyze → valid VisionResponse JSON."""
    img_b64 = _make_dummy_image_b64()
    resp = client.post(
        "/api/vision/analyze",
        json={"image_b64": img_b64},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "label" in data
    assert "confidence" in data
    assert "accuracy_rating" in data


def test_registry_endpoint(client):
    """GET /api/vision/registry returns list."""
    resp = client.get("/api/vision/registry")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)


def test_stats_endpoint(client):
    """GET /api/vision/stats returns stats dict."""
    resp = client.get("/api/vision/stats")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, dict)


def test_absorb_endpoint_missing_model(client):
    """POST /api/vision/absorb without model_id returns 400."""
    resp = client.post(
        "/api/vision/absorb",
        json={"domain": "test"},
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_compare_endpoint(client):
    """POST /api/vision/compare → ComparisonResult with required fields."""
    img_b64 = _make_dummy_image_b64()
    resp = client.post(
        "/api/vision/compare",
        json={"image_b64": img_b64},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "nsck_label" in data
    assert "accuracy_rating" in data
