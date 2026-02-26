"""
ImageAdapter — encodes image arrays (2D/3D numpy arrays) as PerceptPackets.

Extracts classical computer-vision features (spatial-grid statistics,
colour histograms, edge density) and encodes the resulting feature vector
with Fractional Power Encoding (FPE) so that visually similar images map
to similar HyperVectors.

No neural networks, no trained weights — pure classical CV + VSA math.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter

# Number of quantisation bins for FPE
_N_BINS: int = 256


def _build_codebook(n_bins: int) -> List[hypervec_rs.HyperVector]:
    """Build a codebook of n_bins HVs for FPE quantisation."""
    return [hypervec_rs.HyperVector(i * 31 + 7777) for i in range(n_bins)]


# Module-level codebook (shared, built once)
_CODEBOOK: List[hypervec_rs.HyperVector] = _build_codebook(_N_BINS)


def _features_to_hv(feature_vec: np.ndarray) -> hypervec_rs.HyperVector:
    """
    Encode a continuous feature vector into an HV using FPE.

    Each dimension d with normalised value v_d:
      1. Quantise v_d → bin index b_d in [0, N_BINS)
      2. Get codebook HV for b_d
      3. Bind with a deterministic role HV for dimension d
      4. Bundle all bound HVs

    Similar feature vectors → many agreeing dimensions → high cosine similarity.
    """
    feats = np.asarray(feature_vec, dtype=np.float64)
    if len(feats) == 0:
        return hypervec_rs.HyperVector(0)

    # Normalise each dimension independently to [0, 1] then quantise
    f_min = feats.min()
    f_max = feats.max()
    span = f_max - f_min
    if span < 1e-10:
        return hypervec_rs.HyperVector(int(abs(f_min * 1000)) % (2 ** 32))

    normalised = (feats - f_min) / span
    bin_indices = np.clip(
        (normalised * (_N_BINS - 1)).astype(int), 0, _N_BINS - 1
    )

    acc = None
    for d, b in enumerate(bin_indices):
        val_hv = _CODEBOOK[b]
        role_hv = hypervec_rs.HyperVector((d * 1009 + 3001) % (2 ** 32))
        bound = val_hv.xor(role_hv)
        acc = bound if acc is None else acc.bundle(bound)

    return acc if acc is not None else hypervec_rs.HyperVector(0)


def _extract_image_features(img: np.ndarray) -> np.ndarray:
    """
    Extract a 65-dimensional feature vector from an image using only numpy.

    Features (all normalised to a comparable scale):
      - Spatial 4×4 grid: mean per cell (16 features)
      - Spatial 4×4 grid: std per cell  (16 features)
      - Colour channel histograms: 8 bins per channel, up to 3 channels
        (only first 3 channels used; grayscale padded with zeros) → 24 features
      - Sobel edge density (fraction of pixels above gradient threshold) → 1 feature
      - Global mean, global std, channel count → 3 features
      - Quadrant mean (2×2 spatial quadrants) → 4 features
      - Image aspect ratio → 1 feature
      Total: 65 features
    """
    img_f = np.asarray(img, dtype=np.float64)

    # Ensure 2D grayscale is available
    if img_f.ndim == 3:
        grey = img_f.mean(axis=-1)
        n_channels = img_f.shape[2]
    else:
        grey = img_f.copy()
        n_channels = 1

    h, w = grey.shape[:2]

    # ── 1. Spatial 4×4 grid statistics (32 features) ─────────────────
    grid_r, grid_c = 4, 4
    cell_h = max(1, h // grid_r)
    cell_w = max(1, w // grid_c)
    grid_means = np.zeros(grid_r * grid_c)
    grid_stds = np.zeros(grid_r * grid_c)
    for r in range(grid_r):
        for c in range(grid_c):
            r0, r1 = r * cell_h, min((r + 1) * cell_h, h)
            c0, c1 = c * cell_w, min((c + 1) * cell_w, w)
            cell = grey[r0:r1, c0:c1]
            if cell.size > 0:
                grid_means[r * grid_c + c] = cell.mean() / 255.0
                grid_stds[r * grid_c + c] = cell.std() / 128.0

    # ── 2. Colour histogram (24 features, 8 bins × up to 3 channels) ─
    color_hist = np.zeros(24)
    source = img_f if img_f.ndim == 3 else img_f[:, :, np.newaxis]
    for ch in range(min(3, source.shape[2] if source.ndim == 3 else 1)):
        data = source[:, :, ch].ravel() if source.ndim == 3 else source.ravel()
        hist, _ = np.histogram(data, bins=8, range=(0.0, 255.0))
        total = hist.sum()
        color_hist[ch * 8:(ch + 1) * 8] = hist / max(1, total)

    # ── 3. Sobel edge density (1 feature) ────────────────────────────
    padded = np.pad(grey, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = magnitude.mean() + magnitude.std()
    edge_density = float(np.mean(magnitude > threshold))

    # ── 4. Global stats (3 features) ─────────────────────────────────
    global_mean = float(grey.mean()) / 255.0
    global_std = float(grey.std()) / 128.0
    channel_norm = float(n_channels) / 3.0

    # ── 5. Quadrant means (4 features) ───────────────────────────────
    mid_h, mid_w = h // 2, w // 2
    quads = [
        grey[:mid_h, :mid_w],
        grey[:mid_h, mid_w:],
        grey[mid_h:, :mid_w],
        grey[mid_h:, mid_w:],
    ]
    quad_means = np.array([
        q.mean() / 255.0 if q.size > 0 else 0.0 for q in quads
    ])

    # ── 6. Aspect ratio (1 feature) ──────────────────────────────────
    aspect = float(w) / max(1, h)
    aspect_norm = min(aspect, 3.0) / 3.0  # cap at 3:1

    return np.concatenate([
        grid_means, grid_stds, color_hist,
        [edge_density, global_mean, global_std, channel_norm],
        quad_means,
        [aspect_norm],
    ])


def _get_descriptors(img: np.ndarray, feature_vec: np.ndarray) -> List[str]:
    """Derive human-readable descriptor strings from features."""
    img_f = np.asarray(img, dtype=np.float64)
    grey = img_f.mean(axis=-1) if img_f.ndim == 3 else img_f

    global_mean = float(grey.mean())
    global_std = float(grey.std())
    is_color = img_f.ndim == 3 and img_f.shape[2] >= 3

    # edge_density is at index 32 (after grid_means[16] + grid_stds[16])
    edge_density = float(feature_vec[32]) if len(feature_vec) > 32 else 0.0

    descs: List[str] = []
    if global_mean > 200:
        descs.append("bright")
    elif global_mean < 55:
        descs.append("dark")
    if global_std > 60:
        descs.append("high_contrast")
    elif global_std < 20:
        descs.append("uniform")
    if edge_density > 0.3:
        descs.append("detailed")
    elif edge_density < 0.05:
        descs.append("smooth")
    descs.append("color" if is_color else "grayscale")
    return descs


class ImageAdapter(ModalityAdapter):
    """
    Modality adapter for 2D/3D image arrays.

    Extracts classical CV features (spatial grid statistics, colour
    histograms, Sobel edge density, quadrant means) and encodes them via
    FPE into a similarity-preserving HyperVector.

    Similar images (same scene, slight perturbations) → similar HVs.
    Deep-learning-level rotation/scale invariance is NOT provided; that
    requires a CNN feature extractor paired via EmbeddingVSABridge.
    """

    def encode(self, image: Any, task_tag: str) -> PerceptPacket:
        """
        Encode an image array as a PerceptPacket.

        Parameters
        ----------
        image : np.ndarray
            2D (H×W) or 3D (H×W×C) uint8 or float array.
        task_tag : str
            Task domain identifier.

        Returns
        -------
        PerceptPacket with modality="image".
        """
        img = np.asarray(image)
        if img.ndim < 2:
            return PerceptPacket.make(
                modality="image",
                situation_hv=hypervec_rs.HyperVector(0),
                active_predicates=frozenset(["IMAGE_DEGENERATE"]),
                raw_state={"shape": list(img.shape)},
            )

        feature_vec = _extract_image_features(img)
        situation_hv = _features_to_hv(feature_vec)
        descriptors = _get_descriptors(img, feature_vec)

        preds = frozenset(f"IMAGE_{d.upper()}" for d in descriptors)
        img_arr = np.asarray(img, dtype=np.float64)
        grey = img_arr.mean(axis=-1) if img_arr.ndim == 3 else img_arr
        edge_density = float(feature_vec[32]) if len(feature_vec) > 32 else 0.0

        raw_state = {
            "shape": list(img.shape),
            "image_mean": float(grey.mean()),
            "image_std": float(grey.std()),
            "edge_density": edge_density,
            "is_color": img_arr.ndim == 3 and img_arr.shape[2] >= 3,
            "descriptors": descriptors,
        }

        return PerceptPacket.make(
            modality="image",
            situation_hv=situation_hv,
            active_predicates=preds,
            raw_state=raw_state,
            adapter_name="ImageAdapter",
            adapter_trace={
                "n_feature_dims": len(feature_vec),
                "descriptors": descriptors,
            },
        )
