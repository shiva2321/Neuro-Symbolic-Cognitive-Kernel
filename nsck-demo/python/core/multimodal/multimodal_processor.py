"""
NSCK Multimodal Processor
=========================
Unified input handler that accepts text, images, audio, video, and structured
data, converts them into a common VSA HyperVector representation, and routes
them through the cognitive pipeline.

Classical Signal Processing (no neural networks):
- Text → tokenise + positional bundling → HV
- Image → HOG + color histogram + edge density + LBP texture + spatial quadrants → HV
- Audio → MFCC + spectral centroid + spectral rolloff + ZCR + energy bands → HV
- Video → per-frame image HVs + optical-flow motion descriptors → temporal bundle
- Structured dict → predicate encoding → HV
- Any combination → fused multimodal HV via role binding
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
    video: Optional[List[np.ndarray]] = None  # List of frames (H×W or H×W×C)
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
        "video": 55505,
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

        if inp.video is not None:
            r = self._process_video(inp.video)
            results.append(r)
            concepts.extend(r.features.get("descriptors", []))
            cues["video_stats"] = r.features.get("stats", {})

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
    # Image processing (classical CV features — no neural nets)
    # ------------------------------------------------------------------

    def _process_image(self, image: np.ndarray) -> ModalityResult:
        """
        Convert an image (H×W or H×W×C) into an HV using classical
        computer-vision features:

        1. **Gradient histogram (HOG-lite)**: Compute Sobel gradients
           in x/y, quantise orientations into 8 bins, histogram per
           spatial cell (4×4 grid = 16 cells × 8 bins = 128 features).
        2. **Color histogram**: 8-bin histogram per channel (max 3 ch = 24).
        3. **Edge density**: Fraction of high-gradient pixels.
        4. **LBP-lite texture**: Local Binary Pattern on 8 neighbours,
           histogram of LBP codes in 4 quadrants (4 × 10 bins = 40).
        5. **Spatial quadrant statistics**: Mean/std per quadrant (8 features).

        All features are deterministically quantised into a fingerprint
        string → SHA-256 seed → HV.  No matrix multiplications, no trained
        weights.
        """
        img = np.asarray(image, dtype=np.float64)

        # Keep colour channels if available
        is_color = img.ndim == 3 and img.shape[2] >= 3
        if img.ndim == 3:
            grey = img.mean(axis=-1)
        else:
            grey = img.copy()

        h, w = grey.shape[:2]

        # ─── 1. HOG-lite: Sobel gradients + orientation histogram ────
        # Pad by 1 for Sobel kernel
        padded = np.pad(grey, 1, mode="edge")
        gx = padded[1:-1, 2:] - padded[1:-1, :-2]  # horizontal
        gy = padded[2:, 1:-1] - padded[:-2, 1:-1]  # vertical
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)  # -π to π

        # 8 orientation bins, 4×4 spatial grid
        n_orient_bins = 8
        grid_r, grid_c = 4, 4
        hog_features = np.zeros(grid_r * grid_c * n_orient_bins)
        cell_h = max(1, h // grid_r)
        cell_w = max(1, w // grid_c)
        bin_edges = np.linspace(-np.pi, np.pi, n_orient_bins + 1)
        for r in range(grid_r):
            for c in range(grid_c):
                r0, r1 = r * cell_h, min((r + 1) * cell_h, h)
                c0, c1 = c * cell_w, min((c + 1) * cell_w, w)
                cell_mag = magnitude[r0:r1, c0:c1].ravel()
                cell_ori = orientation[r0:r1, c0:c1].ravel()
                hist, _ = np.histogram(cell_ori, bins=bin_edges, weights=cell_mag)
                offset = (r * grid_c + c) * n_orient_bins
                hog_features[offset:offset + n_orient_bins] = hist
        # L2-normalise
        hog_norm = np.linalg.norm(hog_features)
        if hog_norm > 1e-8:
            hog_features /= hog_norm

        # ─── 2. Color histogram (8 bins per channel) ────────────────
        color_features = np.zeros(24)  # max 3 channels × 8 bins
        if is_color:
            for ch in range(min(3, img.shape[2])):
                hist, _ = np.histogram(img[:, :, ch].ravel(), bins=8, range=(0, 256))
                color_features[ch * 8:(ch + 1) * 8] = hist / max(1, hist.sum())
        else:
            hist, _ = np.histogram(grey.ravel(), bins=8, range=(0, 256))
            color_features[:8] = hist / max(1, hist.sum())

        # ─── 3. Edge density ────────────────────────────────────────
        threshold = np.mean(magnitude) + np.std(magnitude)
        edge_density = float(np.mean(magnitude > threshold))

        # ─── 4. LBP-lite texture (simplified) ───────────────────────
        # Compare each pixel to its 8 neighbours → 8-bit code
        # To keep it fast, downsample to max 32×32
        ds_h = min(32, h)
        ds_w = min(32, w)
        rows = np.linspace(1, h - 2, ds_h, dtype=int) if h > 2 else np.array([0])
        cols = np.linspace(1, w - 2, ds_w, dtype=int) if w > 2 else np.array([0])
        lbp_codes = np.zeros((ds_h, ds_w), dtype=np.int32)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for ri, r in enumerate(rows):
            for ci, c in enumerate(cols):
                center = grey[r, c]
                code = 0
                for bit_idx, (dr, dc) in enumerate(offsets):
                    if grey[r + dr, c + dc] >= center:
                        code |= (1 << bit_idx)
                lbp_codes[ri, ci] = code

        # Histogram of LBP codes per quadrant (4 quadrants × 10 bins)
        lbp_features = np.zeros(40)
        mid_r = ds_h // 2
        mid_c = ds_w // 2
        for qi, (rs, re, cs, ce) in enumerate([
            (0, mid_r, 0, mid_c), (0, mid_r, mid_c, ds_w),
            (mid_r, ds_h, 0, mid_c), (mid_r, ds_h, mid_c, ds_w),
        ]):
            quad = lbp_codes[rs:re, cs:ce].ravel()
            if len(quad) > 0:
                hist, _ = np.histogram(quad, bins=10, range=(0, 256))
                lbp_features[qi * 10:(qi + 1) * 10] = hist / max(1, hist.sum())

        # ─── 5. Spatial quadrant stats ──────────────────────────────
        quad_stats = np.zeros(8)
        mid_h = h // 2
        mid_w = w // 2
        for qi, (rs, re, cs, ce) in enumerate([
            (0, mid_h, 0, mid_w), (0, mid_h, mid_w, w),
            (mid_h, h, 0, mid_w), (mid_h, h, mid_w, w),
        ]):
            quad = grey[rs:re, cs:ce]
            if quad.size > 0:
                quad_stats[qi * 2] = np.mean(quad) / 255.0
                quad_stats[qi * 2 + 1] = np.std(quad) / 128.0

        # ─── Assemble fingerprint → HV ──────────────────────────────
        # Quantise all features to 2 decimal places for determinism
        all_feats = np.concatenate([
            hog_features, color_features, [edge_density],
            lbp_features, quad_stats,
        ])
        fp_str = "_".join(f"{v:.2f}" for v in all_feats)
        seed = int(hashlib.sha256(fp_str.encode()).hexdigest()[:8], 16)
        result_hv = hypervec_rs.HyperVector(seed)

        # Build stats and descriptors
        stats = {
            "height": int(h), "width": int(w),
            "mean": float(np.mean(grey)),
            "std": float(np.std(grey)),
            "edge_density": edge_density,
            "is_color": is_color,
            "hog_energy": float(np.sum(magnitude)),
        }

        descriptors = []
        if stats["mean"] > 200:
            descriptors.append("bright")
        elif stats["mean"] < 55:
            descriptors.append("dark")
        if stats["std"] > 60:
            descriptors.append("high_contrast")
        elif stats["std"] < 20:
            descriptors.append("uniform")
        if stats["mean"] < 1:  # Absolute black/empty
            descriptors.append("black")
            return ModalityResult(
                modality="image",
                hv=result_hv,
                features={
                    "stats": stats,
                    "descriptors": descriptors,
                    "hog_bins": 0,
                    "color_bins": 0,
                    "lbp_bins": 0,
                },
                confidence=1.0,
                raw_summary=f"Image({h}×{w}): Black/Empty",
            )

        if edge_density > 0.3:
            descriptors.append("detailed")
        elif edge_density < 0.05:
            descriptors.append("smooth")
            
        # [Phase 3] Texture detection via LBP variance/entropy
        lbp_variance = float(np.var(lbp_features))
        if lbp_variance > 0.005:  # High variance in LBP codes = textured
            descriptors.append("textured")
        else:
            descriptors.append("smooth_texture")

        # [Phase 3] Circularity detection via HOG orientation uniformity
        # A circle has gradients in all directions. 
        # Sum of HOG bins across all cells should be relatively uniform.
        hog_sums = np.sum(hog_features.reshape(-1, n_orient_bins), axis=0)
        hog_uniformity = float(np.std(hog_sums) / (np.mean(hog_sums) + 1e-8))
        if hog_uniformity < 0.4 and edge_density > 0.01:
            descriptors.append("circular")

        if is_color:
            descriptors.append("color")
        else:
            descriptors.append("greyscale")
        # Dominant orientation
        dom_bin = int(np.argmax(hog_sums))
        orient_names = ["R", "UR", "U", "UL", "L", "DL", "D", "DR"]
        descriptors.append(f"orient:{orient_names[dom_bin]}")

        return ModalityResult(
            modality="image",
            hv=result_hv,
            features={
                "stats": stats,
                "descriptors": descriptors,
                "hog_bins": int(len(hog_features)),
                "color_bins": int(np.count_nonzero(color_features)),
                "lbp_bins": int(np.count_nonzero(lbp_features)),
            },
            confidence=0.80,
            raw_summary=f"Image({h}×{w}): HOG+color+LBP+edge",
        )

    # ------------------------------------------------------------------
    # Audio processing (classical DSP features — no neural nets)
    # ------------------------------------------------------------------

    def _process_audio(self, audio: np.ndarray) -> ModalityResult:
        """
        Convert a 1-D audio waveform into an HV using classical DSP features:

        1. **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients
           computed via FFT → mel filterbank → log → DCT.  Standard in
           speech/audio processing since the 1980s.
        2. **Energy bands**: Split spectrum into 4 bands and compute
           relative energy per band.
        3. **Zero-crossing rate (ZCR)**: Simple time-domain feature.
        4. **Spectral centroid**: Centre of mass of the spectrum.
        5. **Spectral rolloff**: Frequency below which 85% of energy lies.
        6. **Peak / RMS energy**: Amplitude statistics.

        All pure numpy — no librosa or trained models needed.
        """
        audio = np.asarray(audio, dtype=np.float64).ravel()
        n = len(audio)
        if n == 0:
            return ModalityResult(
                modality="audio",
                hv=hypervec_rs.HyperVector(0),
                confidence=0.2,
                raw_summary="(empty audio)",
            )

        # Basic amplitude features
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0)) if n > 1 else 0.0

        # ─── FFT magnitude spectrum ──────────────────────────────────
        # Use the full signal (or first 8192 samples for efficiency)
        chunk = audio[:min(n, 8192)]
        # Apply Hann window to reduce spectral leakage
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(chunk)) / len(chunk)))
        windowed = chunk * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.arange(len(spectrum))
        spec_sum = spectrum.sum()

        # ─── Spectral centroid ───────────────────────────────────────
        if spec_sum > 1e-10:
            spectral_centroid = float(np.sum(freqs * spectrum) / spec_sum)
        else:
            spectral_centroid = 0.0

        # ─── Spectral rolloff (85% energy) ──────────────────────────
        cumsum = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * spec_sum
        rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
        spectral_rolloff = float(rolloff_idx) / max(1, len(spectrum))

        # ─── Energy bands (4 quartile bands) ────────────────────────
        n_bins = len(spectrum)
        band_size = max(1, n_bins // 4)
        band_energies = np.zeros(4)
        for b in range(4):
            start = b * band_size
            end = min(start + band_size, n_bins)
            band_energies[b] = np.sum(spectrum[start:end] ** 2)
        total_band = band_energies.sum()
        if total_band > 1e-10:
            band_energies /= total_band

        # ─── MFCC (13 coefficients) ─────────────────────────────────
        # Mel filterbank: 26 triangular filters on mel scale
        n_mfcc = 13
        n_filters = 26
        low_freq_mel = 0
        # Assume 16 kHz sample rate for mel scaling (reasonable default)
        sample_rate = 16000
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((len(chunk) + 1) * hz_points / sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, n_bins - 1)

        filterbank = np.zeros((n_filters, n_bins))
        for m in range(n_filters):
            f_start = bin_points[m]
            f_center = bin_points[m + 1]
            f_end = bin_points[m + 2]
            # Rising slope
            if f_center > f_start:
                filterbank[m, f_start:f_center] = (
                    np.arange(f_start, f_center) - f_start
                ) / max(1, f_center - f_start)
            # Falling slope
            if f_end > f_center:
                filterbank[m, f_center:f_end] = (
                    f_end - np.arange(f_center, f_end)
                ) / max(1, f_end - f_center)

        # Apply filterbank → log → DCT
        power_spectrum = spectrum ** 2
        mel_energies = filterbank @ power_spectrum  # (n_filters,)
        mel_energies = np.log(mel_energies + 1e-10)

        # Type-II DCT (manual, no scipy needed)
        mfcc = np.zeros(n_mfcc)
        for k in range(n_mfcc):
            mfcc[k] = np.sum(
                mel_energies * np.cos(np.pi * k * (2 * np.arange(n_filters) + 1) / (2 * n_filters))
            )

        # ─── Assemble fingerprint → HV ──────────────────────────────
        all_feats = np.concatenate([
            mfcc,
            band_energies,
            [zcr, spectral_centroid / max(1, n_bins),
             spectral_rolloff, rms_energy, peak],
        ])
        fp_str = "_".join(f"{v:.3f}" for v in all_feats)
        seed = int(hashlib.sha256(fp_str.encode()).hexdigest()[:8], 16)
        result_hv = hypervec_rs.HyperVector(seed)

        stats = {
            "energy": float(rms_energy ** 2),
            "rms": rms_energy,
            "zcr": zcr,
            "peak": peak,
            "length": n,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "n_mfcc": n_mfcc,
            "band_energies": band_energies.tolist(),
        }

        descriptors = []
        if rms_energy > 0.3:
            descriptors.append("loud")
        elif rms_energy < 0.01:
            descriptors.append("quiet")
        if zcr > 0.3:
            descriptors.append("noisy")
        elif zcr < 0.05:
            descriptors.append("tonal")
        if band_energies[3] > 0.4:
            descriptors.append("high_freq")
        elif band_energies[0] > 0.6:
            descriptors.append("low_freq")
        if spectral_rolloff < 0.3:
            descriptors.append("narrow_band")
        elif spectral_rolloff > 0.7:
            descriptors.append("wide_band")

        return ModalityResult(
            modality="audio",
            hv=result_hv,
            features={"stats": stats, "descriptors": descriptors,
                       "mfcc": mfcc.tolist()},
            confidence=0.75,
            raw_summary=f"Audio({n} samples): MFCC+spectral+energy",
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
    # Video processing (frame analysis + optical flow — no neural nets)
    # ------------------------------------------------------------------

    def _process_video(self, frames: List[np.ndarray]) -> ModalityResult:
        """
        Convert a list of video frames into an HV.

        Pipeline:
        1. Sample up to 8 frames evenly across the clip.
        2. Process each frame through the full image pipeline (HOG, etc.).
        3. Compute **block-matching optical flow** between consecutive
           sampled frames — gives motion magnitude and direction histograms.
        4. Bundle frame HVs with temporal position encoding.
        5. XOR-bind a motion descriptor HV into the result.

        This captures both appearance and motion without neural nets.
        """
        if not frames:
            return ModalityResult(
                modality="video",
                hv=hypervec_rs.HyperVector(0),
                confidence=0.2,
                raw_summary="(empty video)",
            )

        # Sample up to 8 frames evenly
        n_frames = len(frames)
        indices = np.linspace(0, n_frames - 1, min(8, n_frames), dtype=int)
        sampled = [frames[i] for i in indices]

        frame_hvs = []
        frame_greys = []
        for i, frame in enumerate(sampled):
            img_result = self._process_image(frame)
            # Bind with temporal position
            pos_role = hypervec_rs.HyperVector((66601 + i) % (2**32))
            frame_hvs.append(img_result.hv.xor(pos_role))

            # Store greyscale for optical flow
            f = np.asarray(frame, dtype=np.float64)
            if f.ndim == 3:
                f = f.mean(axis=-1)
            frame_greys.append(f)

        result = frame_hvs[0]
        for fhv in frame_hvs[1:]:
            result = result.bundle(fhv)

        # ─── Block-matching optical flow ─────────────────────────────
        # For each consecutive pair, compute motion vectors via block
        # matching on 8×8 blocks with a 4-pixel search radius.
        total_motion_mag = 0.0
        motion_angle_hist = np.zeros(8)  # 8 direction bins
        n_flow_pairs = 0

        for fi in range(len(frame_greys) - 1):
            prev_g = frame_greys[fi]
            next_g = frame_greys[fi + 1]
            # Downsample to max 64×64 for speed
            def _downsample(img, target=64):
                h, w = img.shape
                if h <= target and w <= target:
                    return img
                rr = np.linspace(0, h - 1, min(target, h), dtype=int)
                cc = np.linspace(0, w - 1, min(target, w), dtype=int)
                return img[np.ix_(rr, cc)]

            p = _downsample(prev_g)
            q = _downsample(next_g)
            bh, bw = p.shape
            block = 8
            search = 4

            for by in range(0, bh - block, block):
                for bx in range(0, bw - block, block):
                    ref = p[by:by + block, bx:bx + block]
                    best_sad = 1e30
                    best_dy, best_dx = 0, 0
                    for dy in range(-search, search + 1):
                        for dx in range(-search, search + 1):
                            cy = by + dy
                            cx = bx + dx
                            if cy < 0 or cy + block > bh or cx < 0 or cx + block > bw:
                                continue
                            cand = q[cy:cy + block, cx:cx + block]
                            sad = np.sum(np.abs(ref - cand))
                            if sad < best_sad:
                                best_sad = sad
                                best_dy, best_dx = dy, dx
                    mag = np.sqrt(best_dy**2 + best_dx**2)
                    total_motion_mag += mag
                    if mag > 0.5:
                        angle = np.arctan2(best_dy, best_dx)
                        bin_idx = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                        motion_angle_hist[bin_idx] += 1.0
            n_flow_pairs += 1

        avg_motion = total_motion_mag / max(1, n_flow_pairs)
        # Normalise angle histogram
        angle_sum = motion_angle_hist.sum()
        if angle_sum > 0:
            motion_angle_hist /= angle_sum

        # Encode motion as a separate HV and bind into result
        motion_fp = "_".join(f"{v:.3f}" for v in np.concatenate([
            [avg_motion], motion_angle_hist
        ]))
        motion_seed = int(hashlib.sha256(motion_fp.encode()).hexdigest()[:8], 16)
        motion_hv = hypervec_rs.HyperVector(motion_seed)
        motion_role = hypervec_rs.HyperVector(77701)
        result = result.bundle(motion_hv.xor(motion_role))

        descriptors = [f"{n_frames}_frames"]
        if avg_motion > 20:
            descriptors.append("high_motion")
        elif avg_motion < 2:
            descriptors.append("static")
        else:
            descriptors.append("moderate_motion")

        # Dominant motion direction
        if angle_sum > 0:
            dom_dir = int(np.argmax(motion_angle_hist))
            dir_names = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
            descriptors.append(f"motion:{dir_names[dom_dir]}")

        stats = {
            "n_frames": n_frames,
            "sampled": len(sampled),
            "avg_motion": float(avg_motion),
            "motion_direction_hist": motion_angle_hist.tolist(),
            "flow_pairs": n_flow_pairs,
        }

        return ModalityResult(
            modality="video",
            hv=result,
            features={"stats": stats, "descriptors": descriptors},
            confidence=0.70,
            raw_summary=f"Video({n_frames}f, motion={avg_motion:.1f})",
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
            seed = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
            self._word_cache[word] = hypervec_rs.HyperVector(seed)
        return self._word_cache[word]

    @property
    def supported_modalities(self) -> List[str]:
        """Return list of supported input modalities."""
        return ["text", "image", "audio", "video", "structured"]
