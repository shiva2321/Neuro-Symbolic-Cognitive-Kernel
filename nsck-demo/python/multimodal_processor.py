"""
NSCK Multimodal Processor
=========================
Unified input handler that accepts text, images, audio, and structured data,
converts them into a common VSA HyperVector representation, and routes them
through the cognitive pipeline.

Capabilities:
- Text → tokenise + embed → HV
- Image → resize + flatten → HV
- Audio → spectrogram features → HV
- Structured dict → predicate encoding → HV
- Any combination → fused multimodal HV
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import hashlib

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV

    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModalityResult:
    """Result from processing a single modality."""
    modality: str          # "text", "image", "audio", "structured"
    hv: Any                # HyperVector representation
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_summary: str = ""  # human-readable summary of what was extracted


@dataclass
class MultimodalInput:
    """Container for a multi-modal input bundle."""
    text: Optional[str] = None
    image: Optional[np.ndarray] = None    # H×W or H×W×C uint8
    audio: Optional[np.ndarray] = None    # 1-D float waveform
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedInput:
    """Fully processed multimodal input."""
    fused_hv: Any                                 # combined HyperVector
    modality_results: List[ModalityResult]         # per-modality details
    extracted_concepts: List[str]                  # human-readable concepts
    context_cues: Dict[str, Any]                   # cues for ContextEngine
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class MultimodalProcessor:
    """
    Converts any supported input modality into a unified HyperVector
    representation suitable for the NSCK cognitive pipeline.
    """

    # Role HVs for binding modalities (deterministic seeds)
    _ROLE_SEEDS = {
        "text": 55501,
        "image": 55502,
        "audio": 55503,
        "structured": 55504,
    }

    def __init__(self, context_engine=None, semantic_memory=None):
        self.context_engine = context_engine
        self.semantic_memory = semantic_memory

        # Create role HVs
        self.roles: Dict[str, Any] = {
            name: hypervec_rs.HyperVector(seed)
            for name, seed in self._ROLE_SEEDS.items()
        }

        # Word-level HV cache (deterministic)
        self._word_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, inp: MultimodalInput) -> ProcessedInput:
        """Process a multimodal input into a unified representation."""
        results: List[ModalityResult] = []
        concepts: List[str] = []
        cues: Dict[str, Any] = dict(inp.metadata)

        if inp.text is not None:
            r = self._process_text(inp.text)
            results.append(r)
            concepts.extend(r.features.get("tokens", []))
            cues["text_tokens"] = r.features.get("tokens", [])

        if inp.image is not None:
            r = self._process_image(inp.image)
            results.append(r)
            concepts.extend(r.features.get("descriptors", []))
            cues["image_stats"] = r.features.get("stats", {})

        if inp.audio is not None:
            r = self._process_audio(inp.audio)
            results.append(r)
            concepts.extend(r.features.get("descriptors", []))
            cues["audio_stats"] = r.features.get("stats", {})

        if inp.structured is not None:
            r = self._process_structured(inp.structured)
            results.append(r)
            concepts.extend(r.features.get("predicates", []))
            cues.update(inp.structured)

        if not results:
            # Empty input → zero HV
            fused = hypervec_rs.HyperVector(0)
            return ProcessedInput(
                fused_hv=fused,
                modality_results=[],
                extracted_concepts=[],
                context_cues=cues,
                confidence=0.0,
            )

        # Fuse modality HVs with role binding
        fused = self._fuse_modalities(results)
        avg_conf = sum(r.confidence for r in results) / len(results)

        return ProcessedInput(
            fused_hv=fused,
            modality_results=results,
            extracted_concepts=concepts,
            context_cues=cues,
            confidence=avg_conf,
        )

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def _process_text(self, text: str) -> ModalityResult:
        """Convert text into an HV by tokenising and bundling word HVs."""
        tokens = self._tokenise(text)
        if not tokens:
            hv = hypervec_rs.HyperVector(hash("EMPTY_TEXT") % (2**32))
            return ModalityResult(
                modality="text", hv=hv, confidence=0.3,
                raw_summary="(empty text)"
            )

        word_hvs = [self._word_hv(t) for t in tokens]

        # Positional binding: HV_i XOR position_role_i
        bound = []
        for i, whv in enumerate(word_hvs):
            pos_role = hypervec_rs.HyperVector((7919 + i) % (2**32))
            bound.append(whv.xor(pos_role))

        # Bundle all position-bound word HVs
        result = bound[0]
        for b in bound[1:]:
            result = result.bundle(b)

        return ModalityResult(
            modality="text",
            hv=result,
            features={"tokens": tokens, "n_tokens": len(tokens)},
            confidence=min(1.0, 0.5 + 0.05 * len(tokens)),
            raw_summary=f"Text({len(tokens)} tokens): {text[:80]}",
        )

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def _process_image(self, image: np.ndarray) -> ModalityResult:
        """
        Convert an image (H×W or H×W×C) into an HV.

        Strategy: downsample to 8×8 greyscale, threshold to binary,
        use as deterministic seed for an HV, then encode spatial statistics.
        """
        img = np.asarray(image, dtype=np.float32)

        # Convert to greyscale if colour
        if img.ndim == 3:
            # Average across channels
            img = img.mean(axis=-1)

        h, w = img.shape[:2]

        # Compute basic statistics
        stats = {
            "height": int(h), "width": int(w),
            "mean": float(np.mean(img)),
            "std": float(np.std(img)),
        }

        # Downsample to 8×8
        from_h = np.linspace(0, h - 1, 8, dtype=int)
        from_w = np.linspace(0, w - 1, 8, dtype=int)
        thumb = img[np.ix_(from_h, from_w)]

        # Threshold → binary fingerprint
        median = float(np.median(thumb))
        fingerprint = (thumb > median).astype(np.uint8).flatten()  # 64 bits

        # Derive seed from fingerprint
        fp_bytes = fingerprint.tobytes()
        seed = int(hashlib.sha256(fp_bytes).hexdigest()[:8], 16)
        hv = hypervec_rs.HyperVector(seed)

        # Descriptors based on simple statistics
        descriptors = []
        if stats["mean"] > 200:
            descriptors.append("bright")
        elif stats["mean"] < 55:
            descriptors.append("dark")
        if stats["std"] > 60:
            descriptors.append("high_contrast")
        elif stats["std"] < 20:
            descriptors.append("uniform")

        return ModalityResult(
            modality="image",
            hv=hv,
            features={"stats": stats, "descriptors": descriptors},
            confidence=0.7,
            raw_summary=f"Image({h}×{w}): mean={stats['mean']:.1f}",
        )

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def _process_audio(self, audio: np.ndarray) -> ModalityResult:
        """
        Convert a 1-D audio waveform into an HV.

        Strategy: compute simple spectral features (energy, zero-crossing
        rate, spectral centroid approximation) and encode via HV.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        n = len(audio)
        if n == 0:
            return ModalityResult(
                modality="audio",
                hv=hypervec_rs.HyperVector(0),
                confidence=0.2,
                raw_summary="(empty audio)",
            )

        # Basic features
        energy = float(np.mean(audio ** 2))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0)) if n > 1 else 0.0
        peak = float(np.max(np.abs(audio)))

        stats = {"energy": energy, "zcr": zcr, "peak": peak, "length": n}

        # Quantise features into a fingerprint
        fp = f"E{energy:.4f}_Z{zcr:.4f}_P{peak:.4f}_N{n}"
        seed = int(hashlib.sha256(fp.encode()).hexdigest()[:8], 16)
        hv = hypervec_rs.HyperVector(seed)

        descriptors = []
        if energy > 0.1:
            descriptors.append("loud")
        elif energy < 0.001:
            descriptors.append("quiet")
        if zcr > 0.3:
            descriptors.append("noisy")
        elif zcr < 0.05:
            descriptors.append("tonal")

        return ModalityResult(
            modality="audio",
            hv=hv,
            features={"stats": stats, "descriptors": descriptors},
            confidence=0.65,
            raw_summary=f"Audio({n} samples): energy={energy:.4f}",
        )

    # ------------------------------------------------------------------
    # Structured data processing
    # ------------------------------------------------------------------

    def _process_structured(self, data: Dict[str, Any]) -> ModalityResult:
        """
        Convert a structured dict into an HV by encoding key-value pairs
        as role-filler bindings and bundling.
        """
        if not data:
            return ModalityResult(
                modality="structured",
                hv=hypervec_rs.HyperVector(0),
                confidence=0.3,
                raw_summary="(empty dict)",
            )

        predicates = []
        bound_hvs = []
        for key, value in data.items():
            key_hv = self._word_hv(str(key))
            val_hv = self._word_hv(str(value))
            bound_hvs.append(key_hv.xor(val_hv))
            predicates.append(f"{key}={value}")

        result = bound_hvs[0]
        for b in bound_hvs[1:]:
            result = result.bundle(b)

        return ModalityResult(
            modality="structured",
            hv=result,
            features={"predicates": predicates, "n_fields": len(data)},
            confidence=0.85,
            raw_summary=f"Structured({len(data)} fields)",
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_modalities(self, results: List[ModalityResult]) -> Any:
        """
        Fuse multiple modality HVs into one using role binding + bundling.
        Each modality is bound with its role HV, then all are bundled.
        """
        bound = []
        for r in results:
            role = self.roles.get(r.modality)
            if role is not None:
                bound.append(r.hv.xor(role))
            else:
                bound.append(r.hv)

        fused = bound[0]
        for b in bound[1:]:
            fused = fused.bundle(b)
        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokeniser."""
        import re
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?", text.lower())
        return tokens

    def _word_hv(self, word: str) -> Any:
        """Get or create a deterministic HV for a word."""
        if word not in self._word_cache:
            seed = hash(f"word_{word}") % (2**32)
            self._word_cache[word] = hypervec_rs.HyperVector(seed)
        return self._word_cache[word]
