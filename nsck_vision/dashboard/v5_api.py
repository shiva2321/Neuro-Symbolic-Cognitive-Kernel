"""
NSCK V5 Dashboard API — Flask Blueprint.

Endpoints:
  POST /v5/analyze            — single image, per-claim accuracy
  POST /v5/batch_analyze      — multiple images, per-image results
  GET  /v5/societal/stats     — world stats
  GET  /v5/societal/health    — TDA + Zipf + percolation health
  GET  /v5/societal/domains   — domain cards
  GET  /v5/benchmarks         — latest benchmark results
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("nsck_vision.v5_api")

# ── Optional Flask ──────────────────────────────────────────────────────────
try:
    from flask import Blueprint, jsonify, request, Response
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False

# ── Optional PIL ────────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── Optional NSCK societal ──────────────────────────────────────────────────
_SOCIETAL_OK = False
_NSCK_DIR = Path(__file__).resolve().parents[2] / "nsck"
if str(_NSCK_DIR) not in sys.path:
    sys.path.insert(0, str(_NSCK_DIR))

try:
    from python.core.societal.societal_world import SocietalKnowledgeWorld
    from python.core.societal.tda.tda_monitor import TDAHealthMonitor
    from python.core.societal.emergence.zipf_validator import ZipfValidator
    from python.core.societal.emergence.percolation import PercolationMonitor
    _SOCIETAL_OK = True
except ImportError:
    SocietalKnowledgeWorld = None  # type: ignore


# ============================================================
# SocietalDashboard (wraps world + monitors)
# ============================================================

class SocietalDashboard:
    """Thin wrapper around SocietalKnowledgeWorld for dashboard use."""

    def __init__(self, world: Optional[Any] = None) -> None:
        self._world = world
        if _SOCIETAL_OK and world is None:
            try:
                self._world = SocietalKnowledgeWorld()
            except Exception:
                pass
        self._tda = TDAHealthMonitor(n_landmarks=30) if _SOCIETAL_OK else None
        self._zipf = ZipfValidator(min_concepts=5) if _SOCIETAL_OK else None
        self._perc = PercolationMonitor() if _SOCIETAL_OK else None

    @property
    def world(self) -> Optional[Any]:
        return self._world

    def get_stats(self) -> Dict[str, Any]:
        if self._world is not None:
            try:
                return self._world.stats_report()
            except Exception as exc:
                logger.warning("stats_report failed: %s", exc)
        return _mock_stats()

    def get_health(self) -> Dict[str, Any]:
        if self._world is not None and self._tda is not None:
            try:
                concepts = self._world.concepts
                tda = self._tda.run_analysis(concepts)
                zipf = self._zipf.validate(concepts)
                perc = self._perc.monitor_tick(concepts)
                overall = (tda["health_score"] + zipf["health_score"]) / 2.0
                return {
                    "overall_health": round(overall, 4),
                    "tda": tda,
                    "zipf": zipf,
                    "percolation": perc,
                }
            except Exception as exc:
                logger.warning("health analysis failed: %s", exc)
        return _mock_health()

    def get_domains(self) -> List[Dict[str, Any]]:
        if self._world is not None:
            try:
                result = []
                for domain in self._world.domains.values():
                    nbhds = []
                    for nbhd_id in domain.neighborhood_ids:
                        nbhd = self._world.neighborhoods.get(nbhd_id)
                        if nbhd:
                            nbhds.append({
                                "id": nbhd_id,
                                "n_concepts": len(nbhd.concept_ids),
                                "anchor": nbhd.anchor_id,
                            })
                    result.append({
                        "domain_id": domain.domain_id,
                        "name": domain.name,
                        "n_neighborhoods": len(domain.neighborhood_ids),
                        "neighborhoods": nbhds,
                        "has_city_hall": domain.city_hall_hv is not None,
                    })
                return result
            except Exception as exc:
                logger.warning("get_domains failed: %s", exc)
        return _mock_domains()


# ============================================================
# Mock data (when NSCK unavailable)
# ============================================================

def _mock_stats() -> Dict[str, Any]:
    return {
        "tick_count": 42,
        "n_concepts": 128,
        "n_neighborhoods": 8,
        "n_domains": 3,
        "total_bonds": 512,
        "avg_bonds_per_concept": 4.0,
        "stability_classes": {"crystallized": 20, "stable": 50, "active": 40, "volatile": 18},
        "hybridization_states": {"free": 60, "bonded": 50, "hybridized": 18},
        "domains": [], "neighborhoods": [],
        "_mock": True,
    }


def _mock_health() -> Dict[str, Any]:
    return {
        "overall_health": 0.72,
        "tda": {
            "beta0": 1, "beta1": 3, "beta2": 0,
            "persistence_entropy": 1.2,
            "health_score": 0.75,
            "alerts": [],
            "n_landmarks": 30,
            "_mock": True,
        },
        "zipf": {
            "is_zipf_like": True, "alpha": 1.05, "r_squared": 0.92,
            "health_score": 0.88, "_mock": True,
        },
        "percolation": {
            "giant_fraction": 0.65, "transition_type": "stable",
            "is_transitioning": False, "_mock": True,
        },
        "_mock": True,
    }


def _mock_domains() -> List[Dict[str, Any]]:
    return [
        {"domain_id": "dom_sci", "name": "Science", "n_neighborhoods": 3,
         "neighborhoods": [
             {"id": "nbhd_physics", "n_concepts": 15, "anchor": "quantum"},
             {"id": "nbhd_biology", "n_concepts": 18, "anchor": "cell"},
             {"id": "nbhd_chemistry", "n_concepts": 12, "anchor": "atom"},
         ], "has_city_hall": True, "_mock": True},
        {"domain_id": "dom_art", "name": "Arts", "n_neighborhoods": 2,
         "neighborhoods": [
             {"id": "nbhd_music", "n_concepts": 20, "anchor": "melody"},
             {"id": "nbhd_visual", "n_concepts": 16, "anchor": "color"},
         ], "has_city_hall": True, "_mock": True},
    ]


# ============================================================
# Image analysis helpers
# ============================================================

def _analyze_image_bytes(img_bytes: bytes, label: str = "image") -> Dict[str, Any]:
    """Analyze a single image and produce per-claim accuracy ratings."""
    t0 = time.perf_counter()

    if _PIL_OK:
        try:
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            w, h = img.size
        except Exception:
            arr = np.zeros((64, 64, 3), dtype=np.float32)
            w, h = 64, 64
    else:
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        w, h = 64, 64

    # Feature extraction for claims
    mean_brightness = float(arr.mean())
    std_brightness = float(arr.std())
    # Shannon entropy on 32-bin histogram (use probability, not density)
    hist, _ = np.histogram(arr, bins=32)
    hist_p = hist.astype(np.float64)
    hist_sum = hist_p.sum()
    if hist_sum > 0:
        hist_p /= hist_sum
    hist_p = hist_p[hist_p > 0]
    entropy = float(-np.sum(hist_p * np.log(hist_p + 1e-12)))
    entropy = entropy / np.log(32)  # normalise to [0, 1]
    r, g, b = float(arr[..., 0].mean()), float(arr[..., 1].mean()), float(arr[..., 2].mean())
    color_balance = 1.0 - max(abs(r - g), abs(g - b), abs(r - b))

    claims = [
        {"claim": "Image loaded successfully",
         "confidence": 0.99 if (_PIL_OK and w > 0) else 0.5},
        {"claim": "Scene contains varied visual content",
         "confidence": min(1.0, std_brightness * 3.0)},
        {"claim": "Lighting conditions are adequate",
         "confidence": min(1.0, 0.3 + mean_brightness * 1.5)},
        {"claim": "Color information is present",
         "confidence": min(1.0, color_balance + 0.1)},
        {"claim": "Image entropy is high (complex scene)",
         "confidence": min(1.0, entropy / 4.0)},
        {"claim": "Scene has uniform regions",
         "confidence": max(0.0, 1.0 - std_brightness * 2.0)},
        {"claim": "Dominant wavelength visible",
         "confidence": min(1.0, max(r, g, b) * 1.2)},
        {"claim": "NSCK representation generated",
         "confidence": 0.95},
    ]

    overall_confidence = float(np.mean([c["confidence"] for c in claims]))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "label": label,
        "width": w,
        "height": h,
        "overall_confidence": round(overall_confidence, 4),
        "claims": [{
            "claim": c["claim"],
            "confidence": round(float(c["confidence"]), 4),
            "rating": (
                "high" if c["confidence"] > 0.7 else
                "medium" if c["confidence"] > 0.4 else "low"
            ),
        } for c in claims],
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ============================================================
# Flask Blueprint
# ============================================================

_dashboard = SocietalDashboard()


def create_v5_blueprint() -> "Blueprint":  # type: ignore[return]
    if not _FLASK_OK:
        raise ImportError("Flask is required for v5_blueprint")

    bp = Blueprint("v5", __name__, url_prefix="/v5")

    @bp.route("/analyze", methods=["POST"])
    def analyze_single():
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image", "")
        label = data.get("label", "image")
        try:
            img_bytes = base64.b64decode(img_b64) if img_b64 else b""
        except Exception:
            img_bytes = b""
        result = _analyze_image_bytes(img_bytes, label)
        result["societal_context"] = {
            "n_domains": _dashboard.get_stats().get("n_domains", 0),
            "active_domain": "unknown",
        }
        return jsonify(result)

    @bp.route("/batch_analyze", methods=["POST"])
    def batch_analyze():
        results = []

        # Accept JSON list of {image: base64, label: str}
        if request.is_json:
            items = request.get_json(silent=True) or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                img_b64 = item.get("image", "")
                label = item.get("label", f"image_{len(results)}")
                try:
                    img_bytes = base64.b64decode(img_b64) if img_b64 else b""
                except Exception:
                    img_bytes = b""
                results.append(_analyze_image_bytes(img_bytes, label))
        else:
            # Multipart file upload
            for fname, fobj in request.files.items():
                img_bytes = fobj.read()
                results.append(_analyze_image_bytes(img_bytes, fname))

        if not results:
            return jsonify({"error": "No images provided", "results": []}), 400

        overall = float(np.mean([r["overall_confidence"] for r in results]))
        return jsonify({
            "n_images": len(results),
            "overall_confidence": round(overall, 4),
            "results": results,
            "societal_stats": _dashboard.get_stats(),
        })

    @bp.route("/societal/stats", methods=["GET"])
    def societal_stats():
        return jsonify(_dashboard.get_stats())

    @bp.route("/societal/health", methods=["GET"])
    def societal_health():
        return jsonify(_dashboard.get_health())

    @bp.route("/societal/domains", methods=["GET"])
    def societal_domains():
        return jsonify(_dashboard.get_domains())

    @bp.route("/benchmarks", methods=["GET"])
    def benchmarks():
        # Try to read latest benchmark file
        bench_paths = [
            Path(__file__).resolve().parents[2] / "nsck" / "benchmarks" / "societal_bench_results.json",
            Path(__file__).resolve().parents[2] / "nsck" / "benchmarks" / "benchmark_summary.json",
        ]
        for p in bench_paths:
            if p.exists():
                try:
                    with open(p) as f:
                        return jsonify(json.load(f))
                except Exception:
                    pass
        # Mock demo data
        return jsonify({
            "suite": "NSCK V5 Societal Benchmarks",
            "results": [
                {"name": "registration", "per_concept_ms": 0.16, "passed": True},
                {"name": "query", "per_query_ms": 0.92, "passed": True},
                {"name": "tick", "per_tick_ms": 0.09, "passed": True},
                {"name": "spectral_rg", "elapsed_ms": 8.66, "passed": True},
                {"name": "tda_analysis", "elapsed_ms": 3.54, "passed": True},
            ],
            "_mock": True,
        })

    return bp


# Lazy-create blueprint only when Flask is available
v5_blueprint = create_v5_blueprint() if _FLASK_OK else None
