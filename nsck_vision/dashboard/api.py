"""NSCK Vision Dashboard REST API."""
from __future__ import annotations

import os
import sys
import json
import time
import base64
import logging
from io import BytesIO
from typing import Any, Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

import numpy as np

logger = logging.getLogger("nsck_vision.api")

# Global system instance (lazy init)
_vision_system = None


def get_system():
    """Lazily initialize the vision system."""
    global _vision_system
    if _vision_system is None:
        from nsck_vision.system import NSCKVisionSystem
        _vision_system = NSCKVisionSystem()
    return _vision_system


def _decode_image(b64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        from PIL import Image
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes[:224*224*3], dtype=np.uint8).reshape(-1).astype(np.float32) / 255.0
        return arr[:3*224*224].reshape(3, 224, 224) if len(arr) >= 3*224*224 else np.zeros((3, 224, 224), dtype=np.float32)
    except Exception:
        return np.zeros((3, 224, 224), dtype=np.float32)


def _vision_response_to_dict(response) -> dict:
    """Convert VisionResponse to JSON-serializable dict."""
    return {
        "query_id": response.query_id,
        "label": response.label,
        "confidence": response.confidence,
        "accuracy_rating": response.accuracy_rating,
        "top5": [{"label": l, "confidence": c} for l, c in response.top5],
        "causal_chain": response.causal_chain,
        "cross_domain_analogies": response.cross_domain_analogies,
        "source_model_provenance": response.source_model_provenance,
        "reference_label": response.reference_label,
        "reference_confidence": response.reference_confidence,
        "nsck_overrode_reference": response.nsck_overrode_reference,
        "latency_ms": response.latency_ms,
        "timestamp": response.timestamp,
        "accuracy_breakdown": response.accuracy_breakdown,
    }


def create_api(app: "Flask" = None) -> "Flask":
    """Create and configure the Flask API app.
    
    Parameters
    ----------
    app : optional existing Flask app to register routes on
    
    Returns
    -------
    Flask app with all API routes registered
    """
    if not _FLASK_AVAILABLE:
        raise ImportError("Flask is required. Install with: pip install flask")
    
    if app is None:
        app = Flask(__name__)

    @app.route("/api/health", methods=["GET"])
    def health():
        import python.core.vsa.hypervec_shim as hypervec_rs
        rust_active = hasattr(hypervec_rs, "_rust_active") and hypervec_rs._rust_active
        try:
            sys_stats = get_system().get_stats()
            absorbed = len(sys_stats.get("registry", []))
        except Exception:
            absorbed = 0
        return jsonify({
            "status": "ok",
            "rust_active": rust_active,
            "absorbed_models": absorbed,
            "version": "1.0.0",
        })

    @app.route("/api/vision/analyze", methods=["POST"])
    def analyze():
        data = request.get_json()
        if not data or "image_b64" not in data:
            return jsonify({"error": "image_b64 required"}), 400
        img = _decode_image(data["image_b64"])
        try:
            response = get_system().analyze(img)
            return jsonify(_vision_response_to_dict(response))
        except Exception as e:
            logger.error("analyze error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/analyze_batch", methods=["POST"])
    def analyze_batch():
        data = request.get_json()
        if not data or "images_b64" not in data:
            return jsonify({"error": "images_b64 required"}), 400
        images = [_decode_image(b) for b in data["images_b64"]]
        try:
            responses = get_system().analyze_batch(images)
            return jsonify([_vision_response_to_dict(r) for r in responses])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/absorb", methods=["POST"])
    def absorb():
        data = request.get_json()
        if not data or "model_id" not in data:
            return jsonify({"error": "model_id required"}), 400
        model_id = data["model_id"]
        domain = data.get("domain", "general")
        max_samples = int(data.get("max_samples", 100))
        try:
            report = get_system().absorb(
                model_or_name=model_id,
                domain=domain,
                max_samples=max_samples,
                model_id=model_id,
            )
            return jsonify({
                "model_id": report.model_id,
                "domain": report.domain,
                "n_concepts_absorbed": report.n_concepts_absorbed,
                "spearman_rho": report.spearman_rho,
                "rust_backend_active": report.rust_backend_active,
                "passed": report.passed,
            })
        except Exception as e:
            logger.error("absorb error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/registry", methods=["GET"])
    def registry():
        try:
            return jsonify(get_system().registry.all_models())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/stats", methods=["GET"])
    def stats():
        try:
            return jsonify(get_system().get_stats())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/benchmarks", methods=["GET"])
    def benchmarks():
        results_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
        summary_path = os.path.join(results_dir, "benchmark_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                return jsonify(json.load(f))
        return jsonify({"status": "no benchmarks run yet"})

    @app.route("/api/vision/compare", methods=["POST"])
    def compare():
        data = request.get_json()
        if not data or "image_b64" not in data:
            return jsonify({"error": "image_b64 required"}), 400
        img = _decode_image(data["image_b64"])
        try:
            result = get_system().compare(img)
            return jsonify({
                "query_id": result.query_id,
                "reference_label": result.reference_label,
                "reference_confidence": result.reference_confidence,
                "nsck_label": result.nsck_label,
                "nsck_confidence": result.nsck_confidence,
                "nsck_overrode_reference": result.nsck_overrode_reference,
                "accuracy_rating": result.accuracy_rating,
                "causal_chain": result.causal_chain,
                "cross_domain_analogies": result.cross_domain_analogies,
                "source_model_provenance": result.source_model_provenance,
                "latency_ms": result.latency_ms,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/vision/absorb/stream", methods=["GET"])
    def absorb_stream():
        def generate():
            yield "data: {\"status\": \"ready\", \"message\": \"Absorption SSE stream ready\"}\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    return app
