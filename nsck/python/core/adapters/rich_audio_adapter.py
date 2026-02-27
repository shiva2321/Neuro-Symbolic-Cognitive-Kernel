"""Rich audio adapter with optional Whisper bridge."""
from __future__ import annotations
import time
from typing import Any, Optional

import numpy as np

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class RichAudioAdapter(ModalityAdapter):
    """Audio adapter with optional Whisper bridge, falls back to AudioAdapter."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._whisper_model = None
        self._bridge = None
        
        mode = getattr(config, "perception_mode", "pure") if config else "pure"
        
        if mode in ("bridge", "hybrid"):
            try:
                import whisper  # type: ignore
                model_name = getattr(config, "audio_bridge_model", "whisper-tiny") if config else "whisper-tiny"
                # Map "whisper-tiny" → "tiny" for whisper.load_model
                wm_name = model_name.replace("whisper-", "")
                self._whisper_model = whisper.load_model(wm_name)
                from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
                self._bridge = EmbeddingVSABridge(dim_in=512)
            except Exception:
                self._whisper_model = None
                self._bridge = None

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        t0 = time.perf_counter()
        encoding_method = "classical_dsp"
        
        if self._whisper_model is not None and isinstance(raw_input, np.ndarray):
            try:
                audio = raw_input.astype(np.float32)
                # Whisper feature extraction
                import whisper
                mel = whisper.log_mel_spectrogram(audio)
                features = mel.mean(dim=-1).numpy()[:self._bridge.dim_in]
                if len(features) < self._bridge.dim_in:
                    features = np.pad(features, (0, self._bridge.dim_in - len(features)))
                situation_hv = self._bridge.embed_to_hv(features)
                encoding_method = "whisper"
                latency_ms = (time.perf_counter() - t0) * 1000
                return PerceptPacket.make(
                    modality="audio",
                    situation_hv=situation_hv,
                    active_predicates=frozenset(),
                    raw_state={"length": len(raw_input)},
                    adapter_name="RichAudioAdapter",
                    adapter_trace={"encoding_method": encoding_method, "latency_ms": latency_ms},
                )
            except Exception:
                pass
        
        # Fallback to existing AudioAdapter
        from python.core.adapters.audio_adapter import AudioAdapter
        packet = AudioAdapter().encode(raw_input, task_tag)
        latency_ms = (time.perf_counter() - t0) * 1000
        return PerceptPacket.make(
            modality="audio",
            situation_hv=packet.situation_hv,
            active_predicates=packet.active_predicates,
            raw_state=packet.raw_state,
            adapter_name="RichAudioAdapter",
            adapter_trace={"encoding_method": encoding_method, "latency_ms": latency_ms},
        )
