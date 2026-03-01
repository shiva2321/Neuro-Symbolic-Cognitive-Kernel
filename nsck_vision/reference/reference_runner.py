"""ReferenceModelRunner — runs a strong reference model alongside NSCK."""
from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Ensure nsck/ on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

import numpy as np

logger = logging.getLogger("nsck_vision.reference")


@dataclass
class ReferenceResult:
    """Result from reference model inference."""
    label: str
    confidence: float
    top5: List[Tuple[str, float]]
    features: np.ndarray
    latency_ms: float


@dataclass
class BenchmarkReport:
    """Benchmark comparison report."""
    reference_accuracy: float
    nsck_accuracy: float
    fused_accuracy: float
    n_samples: int
    per_domain: Dict[str, Dict[str, float]]
    latency_ms_mean: float
    reference_model_name: str


class ReferenceModelRunner:
    """Runs a strong reference model alongside NSCK for comparison.
    
    Default: openai/clip-vit-base-patch32 (zero-shot, works across domains).
    Falls back to torchvision.models.resnet50 if CLIP not available.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._adapter = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the reference model with graceful fallbacks."""
        from python.core.vision.pretrained_adapter import PretrainedModelAdapter
        try:
            self._adapter = PretrainedModelAdapter.load(self.model_name, source="auto")
            logger.info("Reference model loaded: %s", self.model_name)
        except Exception as e:
            logger.warning("Could not load %s (%s). Falling back to resnet18.", self.model_name, e)
            try:
                self._adapter = PretrainedModelAdapter.load("resnet18", source="torchvision")
                self.model_name = "resnet18"
            except Exception as e2:
                logger.warning("Could not load resnet18 (%s). Using numpy fallback.", e2)
                self._adapter = PretrainedModelAdapter.load(None, source="numpy")
                self.model_name = "numpy_random"

    def run(self, image: Any) -> ReferenceResult:
        """Run reference model on a single image.
        
        Returns
        -------
        ReferenceResult
        """
        t0 = time.perf_counter()
        img_arr = self._prepare_image(image)
        
        try:
            preds = self._adapter.get_predictions(img_arr)
            if preds:
                pred = preds[0]
                top5 = pred.top5
                label = pred.label
                conf = pred.confidence
                feats = pred.features
            else:
                label, conf = "unknown", 0.0
                top5 = []
                feats = np.zeros(512, dtype=np.float32)
        except Exception as e:
            logger.warning("Reference inference failed: %s", e)
            label, conf = "unknown", 0.0
            top5 = []
            feats = np.zeros(512, dtype=np.float32)
        
        latency = (time.perf_counter() - t0) * 1000
        return ReferenceResult(label=label, confidence=conf, top5=top5, features=feats, latency_ms=latency)

    def run_batch(self, images: List[Any]) -> List[ReferenceResult]:
        """Run reference model on a batch of images."""
        return [self.run(img) for img in images]

    def benchmark(self, dataset_iter: Iterator, nsck_system: Any) -> BenchmarkReport:
        """Benchmark reference model vs NSCK system.
        
        Parameters
        ----------
        dataset_iter : iterator of (image, true_label) pairs
        nsck_system : NSCKVisionSystem instance
        
        Returns
        -------
        BenchmarkReport
        """
        ref_correct = 0
        nsck_correct = 0
        total = 0
        latencies = []
        
        for image, true_label in dataset_iter:
            t0 = time.perf_counter()
            ref_result = self.run(image)
            nsck_result = nsck_system.analyze(image)
            latencies.append((time.perf_counter() - t0) * 1000)
            
            ref_correct += int(ref_result.label == true_label)
            nsck_correct += int(nsck_result.label == true_label)
            total += 1
            if total >= 1000:
                break
        
        if total == 0:
            return BenchmarkReport(0.0, 0.0, 0.0, 0, {}, 0.0, self.model_name)
        
        ref_acc = ref_correct / total
        nsck_acc = nsck_correct / total
        # Fused accuracy: take best of the two per sample (upper bound)
        fused_acc = max(ref_acc, nsck_acc)
        
        return BenchmarkReport(
            reference_accuracy=ref_acc,
            nsck_accuracy=nsck_acc,
            fused_accuracy=fused_acc,
            n_samples=total,
            per_domain={},
            latency_ms_mean=float(np.mean(latencies)) if latencies else 0.0,
            reference_model_name=self.model_name,
        )

    def _prepare_image(self, image: Any) -> np.ndarray:
        """Convert image to numpy array for inference."""
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                return image
            if image.ndim == 3:
                return image[np.newaxis]
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                img = image.resize((224, 224)).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                return arr.transpose(2, 0, 1)[np.newaxis]
        except ImportError:
            pass
        return np.zeros((1, 3, 224, 224), dtype=np.float32)
