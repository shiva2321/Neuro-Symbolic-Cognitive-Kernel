"""NSCKVisionSystem — high-level user-facing class for NSCK-UPMA."""
from __future__ import annotations

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

# Ensure nsck/ is on path for python.core imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

logger = logging.getLogger("nsck_vision.system")


@dataclass
class ComparisonResult:
    """Side-by-side comparison of reference model vs NSCK prediction."""
    query_id: str
    reference_label: Optional[str]
    reference_confidence: Optional[float]
    nsck_label: str
    nsck_confidence: float
    nsck_overrode_reference: bool
    accuracy_rating: float
    causal_chain: List[str]
    cross_domain_analogies: List[str]
    source_model_provenance: List[str]
    latency_ms: float
    timestamp: str


class NSCKVisionSystem:
    """High-level interface for NSCK Universal Pretrained Model Absorption.
    
    Example::
    
        system = NSCKVisionSystem()
        system.absorb("resnet18", domain="general")
        response = system.analyze("path/to/image.jpg")
        print(response.label, response.accuracy_rating)
    """

    def __init__(self, config=None) -> None:
        from python.core.integration.config import NSCKConfig
        from python.core.substrate import NSCKSubstrate
        
        self._config = config or NSCKConfig.vision()
        self._substrate = NSCKSubstrate(config=self._config)
        self._registry = None
        self._reference_runner = None
        logger.info("NSCKVisionSystem initialized (version 1.0.0)")

    @property
    def registry(self):
        if self._registry is None:
            from nsck_vision.registry.model_registry import ModelRegistry
            self._registry = ModelRegistry()
        return self._registry

    def absorb(
        self,
        model_or_name: Any,
        domain: str,
        dataset=None,
        max_samples: int = 500,
        model_id: Optional[str] = None,
        layer_names: Optional[List[str]] = None,
        strategy: str = "svd_factored",
    ):
        """Absorb a single pretrained model into NSCK's HV space.
        
        Parameters
        ----------
        model_or_name : model object or HuggingFace/torchvision name string
        domain : semantic domain label
        dataset : optional iterable of (image_tensor, label) pairs
        max_samples : max samples to absorb (default 500)
        
        Returns
        -------
        AbsorptionReport
        """
        report = self._substrate.absorb_vision_model(
            model_or_name=model_or_name,
            domain=domain,
            dataset_iter=dataset,
            layer_names=layer_names,
            model_id=model_id,
            max_samples=max_samples,
            strategy=strategy,
        )
        # Register in model registry
        eff_id = model_id or str(model_or_name)
        self.registry.register(
            model_id=eff_id,
            domain=domain,
            metadata={
                "n_concepts": report.n_concepts_absorbed,
                "spearman_rho": report.spearman_rho,
                "strategy": strategy,
                "passed": report.passed,
            },
        )
        return report

    def absorb_parallel(self, model_specs: List[Dict[str, Any]]):
        """Absorb multiple models simultaneously.
        
        Parameters
        ----------
        model_specs : list of dicts with keys: model (or model_or_name), domain, 
                      optionally model_id, layer_names, max_samples, strategy
        
        Returns
        -------
        List[AbsorptionReport]
        """
        reports = self._substrate.absorb_vision_models_parallel(model_specs)
        for spec, report in zip(model_specs, reports):
            eff_id = spec.get("model_id") or str(spec.get("model") or spec.get("model_or_name", ""))
            domain = spec.get("domain", "general")
            self.registry.register(
                model_id=eff_id,
                domain=domain,
                metadata={"n_concepts": report.n_concepts_absorbed, "passed": report.passed},
            )
        return reports

    def analyze(self, image_or_path: Any, reference_model=None):
        """Full analysis of an image using absorbed knowledge + optional reference model.
        
        Returns
        -------
        VisionResponse
        """
        img = self._load_image(image_or_path)
        return self._substrate.analyze_image(img, reference_model=reference_model)

    def analyze_batch(self, images: List[Any], reference_model=None):
        """Analyze a batch of images."""
        loaded = [self._load_image(img) for img in images]
        return self._substrate.analyze_images(loaded, reference_model=reference_model)

    def compare(self, image_or_path: Any) -> ComparisonResult:
        """Side-by-side comparison: reference model vs NSCK prediction.
        
        Returns
        -------
        ComparisonResult
        """
        import uuid
        from python.core.vision.pretrained_adapter import PretrainedModelAdapter
        
        ref_model_name = getattr(self._config, "vision_reference_model", "openai/clip-vit-base-patch32")
        try:
            ref_model = PretrainedModelAdapter.load(ref_model_name)
        except Exception:
            ref_model = None
        
        response = self.analyze(image_or_path, reference_model=ref_model)
        
        return ComparisonResult(
            query_id=response.query_id,
            reference_label=response.reference_label,
            reference_confidence=response.reference_confidence,
            nsck_label=response.label,
            nsck_confidence=response.confidence,
            nsck_overrode_reference=response.nsck_overrode_reference,
            accuracy_rating=response.accuracy_rating,
            causal_chain=response.causal_chain,
            cross_domain_analogies=response.cross_domain_analogies,
            source_model_provenance=response.source_model_provenance,
            latency_ms=response.latency_ms,
            timestamp=response.timestamp,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self._substrate.get_absorption_stats()
        stats["registry"] = self.registry.all_models()
        stats["version"] = "1.0.0"
        return stats

    def save_state(self, path: str) -> None:
        """Save absorption memory and registry to path."""
        os.makedirs(path, exist_ok=True)
        # Save registry
        self.registry.save(os.path.join(path, "registry.json"))
        # Save absorption memory if available
        if self._substrate._absorption_memory is not None:
            self._substrate._absorption_memory.save(os.path.join(path, "absorption_memory.json"))
        logger.info("State saved to %s", path)

    def load_state(self, path: str) -> None:
        """Load state from path."""
        reg_path = os.path.join(path, "registry.json")
        if os.path.exists(reg_path):
            self.registry.load(reg_path)
        mem_path = os.path.join(path, "absorption_memory.json")
        if os.path.exists(mem_path):
            self._substrate._ensure_vision_components()
            self._substrate._absorption_memory.load(mem_path)
        logger.info("State loaded from %s", path)

    def _load_image(self, image_or_path: Any):
        """Load image from path or return as-is."""
        if isinstance(image_or_path, str) and os.path.exists(image_or_path):
            try:
                from PIL import Image
                return Image.open(image_or_path).convert("RGB")
            except ImportError:
                import numpy as np
                return np.zeros((224, 224, 3), dtype=np.float32)
        return image_or_path
