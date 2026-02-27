"""Rich text adapter with optional sentence-transformer bridge."""
from __future__ import annotations
import time
from typing import Any, Optional

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class RichTextAdapter(ModalityAdapter):
    """Text adapter that uses sentence-transformers when available, falls back to char-ngrams."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._bridge: Optional[Any] = None
        
        mode = getattr(config, "perception_mode", "pure") if config else "pure"
        bridge_dim = getattr(config, "bridge_dim", 384) if config else 384
        model_name = getattr(config, "text_bridge_model", "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2"
        
        if mode in ("bridge", "hybrid"):
            try:
                from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
                self._bridge = EmbeddingVSABridge(dim_in=bridge_dim)
                self._bridge.try_load_sentence_transformer(model_name)
            except Exception:
                self._bridge = None
        
        # Always init distributional codebook fallback
        try:
            from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
            self._fallback_bridge = EmbeddingVSABridge(dim_in=bridge_dim)
        except Exception:
            self._fallback_bridge = None

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        text = str(raw_input) if not isinstance(raw_input, dict) else raw_input.get("text", str(raw_input))
        t0 = time.perf_counter()
        
        encoding_method = "hash"
        situation_hv = None
        
        if self._bridge is not None:
            try:
                if hasattr(self._bridge, 'encode_text'):
                    situation_hv = self._bridge.encode_text(text)
                elif self._fallback_bridge is not None:
                    situation_hv = self._fallback_bridge.encode_text(text)
                else:
                    situation_hv = None
                encoding_method = "sentence_transformer"
            except Exception:
                situation_hv = None
        
        if situation_hv is None:
            if self._fallback_bridge is not None:
                try:
                    situation_hv = self._fallback_bridge.encode_text(text)
                    encoding_method = "char_ngram"
                except Exception:
                    situation_hv = None
        
        if situation_hv is None:
            import python.core.vsa.hypervec_shim as hv_mod
            situation_hv = hv_mod.HyperVector(hash(text) % (2**32))
            encoding_method = "hash"
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return PerceptPacket.make(
            modality="text",
            situation_hv=situation_hv,
            active_predicates=frozenset(),
            raw_state={"text": text},
            adapter_name="RichTextAdapter",
            adapter_trace={"encoding_method": encoding_method, "latency_ms": latency_ms},
        )
