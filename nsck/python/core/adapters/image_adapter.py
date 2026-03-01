"""
ImageAdapter — encodes image arrays (2D/3D numpy arrays) as PerceptPackets.

Extracts classical computer-vision features (spatial-grid statistics,
colour histograms, edge density) and encodes the resulting feature vector
with Fractional Power Encoding (FPE) so that visually similar images map
to similar HyperVectors.

No neural networks, no trained weights — pure classical CV + VSA math.

V22: Added HOG features, enhanced feature extraction for small images,
and fixed per-feature normalization in FPE encoding.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter

# Number of quantisation bins for FPE
_N_BINS: int = 256

# Pre-built role HVs for up to 1024 feature dimensions (avoids per-call construction)
_MAX_ROLE_HVS: int = 1024
_ROLE_HVS: Optional[List] = None


def _get_role_hvs() -> List:
    """Lazily build and cache role HVs for all feature dimensions."""
    global _ROLE_HVS
    if _ROLE_HVS is None:
        _ROLE_HVS = [
            hypervec_rs.HyperVector((d * 1009 + 3001) % (2 ** 32))
            for d in range(_MAX_ROLE_HVS)
        ]
    return _ROLE_HVS


def _build_codebook(n_bins: int) -> List[hypervec_rs.HyperVector]:
    """Build a codebook of n_bins HVs for FPE quantisation."""
    return [hypervec_rs.HyperVector(i * 31 + 7777) for i in range(n_bins)]


# Module-level codebook (shared, built once)
_CODEBOOK: List[hypervec_rs.HyperVector] = _build_codebook(_N_BINS)


def _features_to_hv(feature_vec: np.ndarray) -> hypervec_rs.HyperVector:
    """
    Encode a continuous feature vector into an HV using FPE.

    V22: Features are assumed to be pre-normalised to [0, 1] per-dimension.
    Each dimension d with value v_d in [0, 1]:
      1. Quantise v_d → bin index b_d in [0, N_BINS)
      2. Get codebook HV for b_d
      3. Bind with a deterministic role HV for dimension d
      4. Bundle all bound HVs

    Similar feature vectors → many agreeing dimensions → high cosine similarity.
    """
    feats = np.asarray(feature_vec, dtype=np.float64)
    if len(feats) == 0:
        return hypervec_rs.HyperVector(0)

    # Detect if features already span [0,1] per-dimension (new path) or need
    # global normalization (legacy fallback for non-normalized inputs)
    f_max = feats.max()
    f_min = feats.min()
    span = f_max - f_min
    if span < 1e-10:
        return hypervec_rs.HyperVector(int(abs(f_min * 1000)) % (2 ** 32))

    # Per-feature clamped binning: features assumed already in [0, 1] by
    # _extract_image_features() contract. Clamp to be safe.
    clamped = np.clip(feats, 0.0, 1.0)
    bin_indices = np.minimum((clamped * (_N_BINS - 1)).astype(int), _N_BINS - 1)

    role_hvs = _get_role_hvs()
    acc = None
    for d, b in enumerate(bin_indices):
        val_hv = _CODEBOOK[b]
        role_hv = role_hvs[d] if d < _MAX_ROLE_HVS else hypervec_rs.HyperVector(
            (d * 1009 + 3001) % (2 ** 32)
        )
        bound = val_hv.xor(role_hv)
        acc = bound if acc is None else acc.bundle(bound)

    return acc if acc is not None else hypervec_rs.HyperVector(0)


def _compute_hog(grey: np.ndarray, cells_y: int = 4, cells_x: int = 4,
                 n_bins: int = 8) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) features using pure numpy.

    Uses unsigned gradients (angles in [0, π)) and L2-normalised cell histograms.
    Each cell produces n_bins features → total cells_y × cells_x × n_bins features.

    Parameters
    ----------
    grey : np.ndarray  (H, W) float64, values in any range
    cells_y, cells_x : grid of cells
    n_bins : orientation bins

    Returns
    -------
    np.ndarray of shape (cells_y * cells_x * n_bins,) with values in [0, 1].
    """
    h, w = grey.shape

    # Finite-difference gradients (avoid border artifacts)
    gx = np.zeros_like(grey, dtype=np.float64)
    gy = np.zeros_like(grey, dtype=np.float64)
    gx[:, 1:-1] = grey[:, 2:].astype(np.float64) - grey[:, :-2].astype(np.float64)
    gy[1:-1, :] = grey[2:, :].astype(np.float64) - grey[:-2, :].astype(np.float64)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # Unsigned orientations: fold atan2 output from [-π, π] into [0, π)
    angle = np.arctan2(gy, gx) % np.pi
    bin_map = np.minimum((angle / np.pi * n_bins).astype(int), n_bins - 1)

    cell_h = max(1, h // cells_y)
    cell_w = max(1, w // cells_x)

    hog = np.zeros(cells_y * cells_x * n_bins, dtype=np.float64)
    for cy in range(cells_y):
        for cx in range(cells_x):
            y0 = cy * cell_h
            y1 = min(y0 + cell_h, h)
            x0 = cx * cell_w
            x1 = min(x0 + cell_w, w)
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_bin = bin_map[y0:y1, x0:x1].ravel()
            hist = np.bincount(cell_bin, weights=cell_mag, minlength=n_bins).astype(
                np.float64
            )
            # L2-normalise cell histogram to [0, 1]
            norm = float(np.sqrt(hist.dot(hist) + 1e-6))
            hog[(cy * cells_x + cx) * n_bins:(cy * cells_x + cx + 1) * n_bins] = (
                hist / norm
            )

    return hog


def _extract_enhanced_features(img: np.ndarray) -> np.ndarray:
    """
    Extract a rich, type-agnostic feature vector from ANY image.

    Handles: greyscale, colour (RGB/BGR), small/large, high/low contrast.
    All output values are in [0, 1].

    Feature layout:
      A. Raw normalised pixels      — only for images ≤ 32×32 (e.g. 64 for 8×8)
      B. Multi-scale HOG            — coarse (2×2 cells) + fine (4×4 cells)
         shapes: 32 + 128 = 160
      C. LBP texture histogram      — 64 bins
      D. HSV colour histograms      — 48 features (only for 3-channel images)
      E. Per-channel spatial stats  — 4×4 grid mean+std, up to 3 channels
         shape: 96 (colour) or 32 (greyscale)
      F. Classic spatial/edge       — 65 features (unchanged for compatibility)

    Returns
    -------
    np.ndarray  — all values in [0, 1].
    """
    img_f = np.asarray(img, dtype=np.float64)
    is_color = img_f.ndim == 3 and img_f.shape[2] >= 3

    # ── Derive greyscale ─────────────────────────────────────────────────────
    if is_color:
        grey = (0.299 * img_f[:, :, 0]
                + 0.587 * img_f[:, :, 1]
                + 0.114 * img_f[:, :, 2])
    else:
        grey = img_f.copy() if img_f.ndim == 2 else img_f[:, :, 0]

    h, w = grey.shape
    parts: List[np.ndarray] = []

    # ── A. Raw normalised pixels (≤ 32×32 only) ──────────────────────────────
    if h * w <= 32 * 32:
        g_min, g_max = grey.min(), grey.max()
        span = g_max - g_min
        norm_pix = (grey - g_min) / max(1e-8, span)
        parts.append(norm_pix.ravel())

    # ── B. Multi-scale HOG ────────────────────────────────────────────────────
    g_range = grey.max() - grey.min()
    grey_255 = ((grey - grey.min()) / max(1e-8, g_range)) * 255.0
    parts.append(_compute_hog(grey_255, cells_y=2, cells_x=2, n_bins=8))   # 32 feats
    parts.append(_compute_hog(grey_255, cells_y=4, cells_x=4, n_bins=8))   # 128 feats

    # ── C. LBP texture histogram (64 bins) ───────────────────────────────────
    parts.append(_compute_lbp_histogram(grey_255, n_bins=64))

    # ── D. HSV colour histograms (colour images only, 16 bins × 3 channels) ─
    if is_color:
        parts.append(_extract_hsv_features(img_f, n_bins=16))   # 48 feats

    # ── E. Per-channel 4×4 spatial grid (mean + std) ─────────────────────────
    #    Up to 3 channels; greyscale uses 1 channel padded to preserve dim
    n_ch = min(3, img_f.shape[2]) if img_f.ndim == 3 else 1
    source = img_f if img_f.ndim == 3 else img_f[:, :, np.newaxis]
    grid_r, grid_c = 4, 4
    cell_h = max(1, h // grid_r)
    cell_w = max(1, w // grid_c)
    chan_grid = np.zeros(grid_r * grid_c * 3 * 2)  # always 96 slots (3 ch × 2 stats)
    for ch in range(min(n_ch, 3)):
        chan_img = source[:, :, ch] if source.ndim == 3 else source[:, :, 0]
        ch_min, ch_max = chan_img.min(), chan_img.max()
        ch_range = max(1e-8, ch_max - ch_min)
        for r in range(grid_r):
            for c in range(grid_c):
                r0, r1 = r * cell_h, min((r + 1) * cell_h, h)
                c0, c1 = c * cell_w, min((c + 1) * cell_w, w)
                cell = chan_img[r0:r1, c0:c1]
                if cell.size > 0:
                    idx = (ch * grid_r * grid_c + r * grid_c + c) * 2
                    chan_grid[idx] = float(
                        np.clip((cell.mean() - ch_min) / ch_range, 0, 1))
                    chan_grid[idx + 1] = float(
                        np.clip(cell.std() / max(1e-8, ch_range), 0, 1))
    parts.append(chan_grid)

    # ── F. Classic features (65, always included for compatibility) ──────────
    parts.append(_extract_image_features(img))

    return np.concatenate(parts)


def _extract_fixed_features(img: np.ndarray, thumb_size: int = 8) -> np.ndarray:
    """
    Fixed-length feature vector for ANY image, regardless of size or colour mode.

    Use this when images of mixed sizes/colour-modes must be compared together.

    Feature layout (all values in [0, 1]):
      A. Downsampled pixels  — always thumb_size × thumb_size = 64 features
      B. HOG 2×2 coarse      — 32 features
      C. HOG 4×4 fine        — 128 features
      D. LBP histogram       — 64 features
      E. HSV histograms      — always 48 features (greyscale: s=0 histogram)
      F. Per-channel grid    — always 96 features (greyscale: duplicated to 3ch)
      G. Classic features    — 65 features
    Total: 64 + 32 + 128 + 64 + 48 + 96 + 65 = 497 features (always fixed).

    Parameters
    ----------
    img : np.ndarray  2D or 3D, any dtype, any size
    thumb_size : int  Thumbnail dimension for raw-pixel component (default 8)

    Returns
    -------
    np.ndarray  shape (497,), all values in [0, 1].
    """
    img_f = np.asarray(img, dtype=np.float64)
    is_color = img_f.ndim == 3 and img_f.shape[2] >= 3

    # Greyscale
    if is_color:
        grey = (0.299 * img_f[:, :, 0]
                + 0.587 * img_f[:, :, 1]
                + 0.114 * img_f[:, :, 2])
    else:
        grey = img_f.copy() if img_f.ndim == 2 else img_f[:, :, 0]

    h, w = grey.shape
    parts: List[np.ndarray] = []

    # ── A. Downsampled thumbnail (always thumb_size × thumb_size = 64 feats) ──
    th = thumb_size
    # Average-pool to th × th
    ph, pw = max(1, h // th), max(1, w // th)
    # Crop to multiples of pool size
    g_crop = grey[: ph * th, : pw * th]
    thumb = g_crop.reshape(th, ph, th, pw).mean(axis=(1, 3))
    g_min, g_max = thumb.min(), thumb.max()
    thumb_norm = (thumb - g_min) / max(1e-8, g_max - g_min)
    parts.append(thumb_norm.ravel())

    # ── B & C. Multi-scale HOG ─────────────────────────────────────────────────
    g_range = grey.max() - grey.min()
    grey_255 = ((grey - grey.min()) / max(1e-8, g_range)) * 255.0
    parts.append(_compute_hog(grey_255, cells_y=2, cells_x=2, n_bins=8))    # 32
    parts.append(_compute_hog(grey_255, cells_y=4, cells_x=4, n_bins=8))    # 128

    # ── D. LBP ────────────────────────────────────────────────────────────────
    parts.append(_compute_lbp_histogram(grey_255, n_bins=64))

    # ── E. HSV histograms (always 48 features) ────────────────────────────────
    if is_color:
        parts.append(_extract_hsv_features(img_f, n_bins=16))
    else:
        # Greyscale: treat as HSV with h=0, s=0, v=intensity
        grey_01 = grey / max(1e-8, grey.max())
        h_hist = np.zeros(16)
        s_hist = np.zeros(16); s_hist[0] = 1.0   # all pixels have saturation 0
        v_hist, _ = np.histogram(grey_01.ravel(), bins=16, range=(0.0, 1.0))
        v_hist = v_hist.astype(np.float64) / max(1.0, v_hist.sum())
        parts.append(np.concatenate([h_hist, s_hist, v_hist]))

    # ── F. Per-channel spatial grid (always 96 features) ─────────────────────
    if is_color:
        source = img_f[:, :, :3]
        ch_list = [0, 1, 2]
    else:
        # Expand grey to 3 channels for consistent feature shape
        source_3ch = np.stack([grey, grey, grey], axis=-1)
        source = source_3ch
        ch_list = [0, 1, 2]

    grid_r, grid_c = 4, 4
    cell_h_g = max(1, h // grid_r)
    cell_w_g = max(1, w // grid_c)
    chan_grid = np.zeros(grid_r * grid_c * 3 * 2)
    for ch_idx, ch in enumerate(ch_list):
        chan_img = source[:, :, ch]
        ch_min, ch_max = chan_img.min(), chan_img.max()
        ch_range_val = max(1e-8, ch_max - ch_min)
        for r in range(grid_r):
            for c_idx in range(grid_c):
                r0, r1 = r * cell_h_g, min((r + 1) * cell_h_g, h)
                c0, c1 = c_idx * cell_w_g, min((c_idx + 1) * cell_w_g, w)
                cell = chan_img[r0:r1, c0:c1]
                if cell.size > 0:
                    gidx = (ch_idx * grid_r * grid_c + r * grid_c + c_idx) * 2
                    chan_grid[gidx] = float(
                        np.clip((cell.mean() - ch_min) / ch_range_val, 0, 1))
                    chan_grid[gidx + 1] = float(
                        np.clip(cell.std() / max(1e-8, ch_range_val), 0, 1))
    parts.append(chan_grid)

    # ── G. Classic features (65) ──────────────────────────────────────────────
    parts.append(_extract_image_features(img))

    result = np.concatenate(parts)
    assert len(result) == 64 + 32 + 128 + 64 + 48 + 96 + 65, \
        f"_extract_fixed_features: expected 497 features, got {len(result)}"
    return result


def _compute_lbp_histogram(grey: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    8-neighbor Local Binary Pattern histogram (pure numpy).

    Returns an n_bins-bin normalised histogram (all values in [0, 1]).
    LBP encodes local texture robustly under monotonic illumination changes.
    """
    # Compute 8-neighbor LBP codes
    patterns = np.zeros(grey.shape, dtype=np.int32)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(grey, dy, axis=0), dx, axis=1)
        patterns += ((shifted >= grey).astype(np.int32) << bit)

    # Compressed histogram (bin 256 LBP codes into n_bins)
    full_hist = np.bincount(patterns.ravel(), minlength=256)[:256].astype(
        np.float64
    )
    bin_size = max(1, 256 // n_bins)
    n = n_bins * bin_size
    compressed = full_hist[:n].reshape(n_bins, bin_size).sum(axis=1)
    total = compressed.sum()
    return compressed / max(1.0, total)


def _rgb_to_hsv(img_f: np.ndarray) -> np.ndarray:
    """Convert float RGB image to HSV using pure numpy. Returns H,S,V in [0,1]."""
    r_max = img_f.max()
    scale = 255.0 if r_max > 1.0 else 1.0
    rgb = img_f / scale  # [0, 1]
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(r)
    eps = 1e-8
    mask_r = (delta > eps) & (cmax == r)
    mask_g = (delta > eps) & (cmax == g)
    mask_b = (delta > eps) & (cmax == b)
    h[mask_r] = (((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps)) % 6) / 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + eps) + 2.0) / 6.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + eps) + 4.0) / 6.0

    s = np.where(cmax > eps, delta / (cmax + eps), 0.0)
    v = cmax

    return np.stack([h, s, v], axis=-1)  # each channel in [0, 1]


def _extract_hsv_features(img_f: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    Compute normalised HSV histograms for a colour image.

    Returns 3 * n_bins features, all in [0, 1].
    Robust colour descriptor invariant to illumination magnitude.
    """
    hsv = _rgb_to_hsv(img_f)
    features = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch].ravel(), bins=n_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        features.append(hist / max(1.0, total))
    return np.concatenate(features)


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
