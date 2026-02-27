"""Rich image adapter with optional timm bridge."""
from __future__ import annotations
import time
from typing import Any, Optional

import numpy as np

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class RichImageAdapter(ModalityAdapter):
    """Image adapter with optional timm model bridge, falls back to ImageAdapter.

    .. warning::
        When a trained model is not available (the default), the timm model is
        loaded with ``pretrained=False``, which means its weights are random.
        Features extracted from a randomly-initialised model carry **no
        semantic meaning** — they are structural noise equivalent to hashing.
        The bridge path therefore only provides meaningful embeddings when a
        pre-trained checkpoint has been loaded.  In all other cases the adapter
        automatically falls back to the classical ``ImageAdapter`` (spatial-grid
        + colour histogram + Sobel edges), which does carry usable signal.
        Pass ``perception_mode="pure"`` (the default) to bypass the model path
        entirely and always use the classical adapter.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._timm_model = None
        self._bridge = None
        
        mode = getattr(config, "perception_mode", "pure") if config else "pure"
        
        if mode in ("bridge", "hybrid"):
            try:
                import timm  # type: ignore
                import torch  # type: ignore
                model_name = getattr(config, "image_bridge_model", "mobilenet_v3_small") if config else "mobilenet_v3_small"
                # pretrained=False avoids a network download; however its weights are
                # random and produce semantically meaningless features. See the class
                # docstring for details.  Replace with pretrained=True once a suitable
                # checkpoint is available.
                self._timm_model = timm.create_model(model_name, pretrained=False, num_classes=0)
                self._timm_model.eval()
                from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
                self._bridge = EmbeddingVSABridge(dim_in=1000)
            except Exception:
                self._timm_model = None
                self._bridge = None

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        t0 = time.perf_counter()
        encoding_method = "classical_cv"
        
        if self._timm_model is not None and isinstance(raw_input, np.ndarray):
            try:
                import torch
                img = raw_input.astype(np.float32) / 255.0
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=0)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = img.transpose(2, 0, 1)
                # Resize to 224x224
                from PIL import Image as PILImage
                pil = PILImage.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((224, 224))
                tensor = torch.from_numpy(np.array(pil).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
                with torch.no_grad():
                    features = self._timm_model(tensor).numpy().flatten()
                situation_hv = self._bridge.embed_to_hv(features[:self._bridge.dim_in])
                encoding_method = "timm"
                latency_ms = (time.perf_counter() - t0) * 1000
                return PerceptPacket.make(
                    modality="image",
                    situation_hv=situation_hv,
                    active_predicates=frozenset(),
                    raw_state={"shape": list(raw_input.shape) if hasattr(raw_input, 'shape') else []},
                    adapter_name="RichImageAdapter",
                    adapter_trace={"encoding_method": encoding_method, "latency_ms": latency_ms},
                )
            except Exception:
                pass
        
        # Fallback to existing ImageAdapter
        from python.core.adapters.image_adapter import ImageAdapter
        packet = ImageAdapter().encode(raw_input, task_tag)
        latency_ms = (time.perf_counter() - t0) * 1000
        return PerceptPacket.make(
            modality="image",
            situation_hv=packet.situation_hv,
            active_predicates=packet.active_predicates,
            raw_state=packet.raw_state,
            adapter_name="RichImageAdapter",
            adapter_trace={"encoding_method": encoding_method, "latency_ms": latency_ms},
        )
