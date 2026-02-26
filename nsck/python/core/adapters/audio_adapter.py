"""
AudioAdapter — encodes 1-D audio waveforms as PerceptPackets.

Extracts classical DSP features (MFCC, energy bands, ZCR, spectral
centroid/rolloff, RMS) using pure numpy and encodes the resulting
feature vector with Fractional Power Encoding (FPE) so that acoustically
similar audio signals map to similar HyperVectors.

No neural networks, no librosa, no trained weights.
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
    return [hypervec_rs.HyperVector(i * 37 + 8888) for i in range(n_bins)]


# Module-level codebook (shared, built once; separate from ImageAdapter's)
_AUDIO_CODEBOOK: List[hypervec_rs.HyperVector] = _build_codebook(_N_BINS)


def _features_to_hv(feature_vec: np.ndarray) -> hypervec_rs.HyperVector:
    """
    Encode a continuous feature vector into an HV using FPE.

    Similar feature vectors → many agreeing dimensions → high cosine similarity.
    """
    feats = np.asarray(feature_vec, dtype=np.float64)
    if len(feats) == 0:
        return hypervec_rs.HyperVector(0)

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
        val_hv = _AUDIO_CODEBOOK[b]
        role_hv = hypervec_rs.HyperVector((d * 1013 + 4001) % (2 ** 32))
        bound = val_hv.xor(role_hv)
        acc = bound if acc is None else acc.bundle(bound)

    return acc if acc is not None else hypervec_rs.HyperVector(0)


def _extract_audio_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract a 26-dimensional feature vector from a 1-D audio waveform.

    Features:
      - 13 MFCC coefficients (mel filterbank → log → DCT)
      - 4 normalised energy bands (spectral quartiles)
      - Zero-crossing rate
      - Normalised spectral centroid
      - Spectral rolloff
      - RMS energy
      - Peak amplitude
      - Log(n_samples + 1) normalised
      Total: 23 features
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    n = len(audio)
    if n == 0:
        return np.zeros(22)

    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0)) if n > 1 else 0.0

    # FFT on first 8192 samples (Hann-windowed)
    chunk = audio[:min(n, 8192)]
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(chunk)) / len(chunk)))
    spectrum = np.abs(np.fft.rfft(chunk * window))
    n_bins = len(spectrum)
    freqs = np.arange(n_bins)
    spec_sum = spectrum.sum()

    # Spectral centroid
    spectral_centroid = float(np.sum(freqs * spectrum) / spec_sum) / max(1, n_bins) \
        if spec_sum > 1e-10 else 0.0

    # Spectral rolloff (85% energy)
    cumsum = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * spec_sum))
    spectral_rolloff = rolloff_idx / max(1, n_bins)

    # 4 energy bands
    band_size = max(1, n_bins // 4)
    band_energies = np.zeros(4)
    for b in range(4):
        s, e = b * band_size, min((b + 1) * band_size, n_bins)
        band_energies[b] = float(np.sum(spectrum[s:e] ** 2))
    total_band = band_energies.sum()
    if total_band > 1e-10:
        band_energies /= total_band

    # MFCC (13 coefficients via mel filterbank)
    n_mfcc = 13
    n_filters = 26
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(0, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_pts = np.clip(
        np.floor((len(chunk) + 1) * hz_points / sample_rate).astype(int),
        0, n_bins - 1,
    )
    filterbank = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        f0, fc, f1 = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if fc > f0:
            filterbank[m, f0:fc] = (np.arange(f0, fc) - f0) / max(1, fc - f0)
        if f1 > fc:
            filterbank[m, fc:f1] = (f1 - np.arange(fc, f1)) / max(1, f1 - fc)

    power_spectrum = spectrum ** 2
    mel_energies = np.log(filterbank @ power_spectrum + 1e-10)
    mfcc = np.array([
        np.sum(mel_energies * np.cos(
            np.pi * k * (2 * np.arange(n_filters) + 1) / (2 * n_filters)
        ))
        for k in range(n_mfcc)
    ])

    log_len = np.log1p(n) / np.log1p(8192)  # normalised log length

    return np.concatenate([
        mfcc,
        band_energies,
        [zcr, spectral_centroid, spectral_rolloff, rms_energy, peak, log_len],
    ])


def _get_audio_descriptors(feature_vec: np.ndarray) -> List[str]:
    """Derive human-readable descriptor strings from the feature vector.

    Feature vector layout (23 dims):
      mfcc[0:13], bands[13:17], zcr[17], centroid[18], rolloff[19],
      rms[20], peak[21], log_len[22]
    """
    if len(feature_vec) < 23:
        return ["audio_unknown"]

    bands = feature_vec[13:17]
    zcr_v = float(feature_vec[17])
    rms_v = float(feature_vec[20])
    rolloff_v = float(feature_vec[19])

    descs: List[str] = []
    if rms_v > 0.3:
        descs.append("loud")
    elif rms_v < 0.01:
        descs.append("quiet")
    if zcr_v > 0.3:
        descs.append("noisy")
    elif zcr_v < 0.05:
        descs.append("tonal")
    if bands[3] > 0.4:
        descs.append("high_freq")
    elif bands[0] > 0.6:
        descs.append("low_freq")
    if rolloff_v < 0.3:
        descs.append("narrow_band")
    elif rolloff_v > 0.7:
        descs.append("wide_band")
    return descs if descs else ["audio_signal"]


class AudioAdapter(ModalityAdapter):
    """
    Modality adapter for 1-D audio waveforms.

    Extracts classical DSP features (MFCC, energy bands, ZCR, spectral
    centroid/rolloff, RMS energy) and encodes them via FPE into a
    similarity-preserving HyperVector.

    Acoustically similar signals → similar HVs.  Speaker-identity or
    phoneme-level recognition requires trained models (e.g. wav2vec2)
    paired via EmbeddingVSABridge.
    """

    SAMPLE_RATE: int = 16000  # assumed default sample rate

    def encode(self, audio: Any, task_tag: str) -> PerceptPacket:
        """
        Encode a 1-D audio waveform as a PerceptPacket.

        Parameters
        ----------
        audio : np.ndarray
            1-D float array of audio samples (assumed 16 kHz mono).
        task_tag : str
            Task domain identifier.

        Returns
        -------
        PerceptPacket with modality="audio".
        """
        arr = np.asarray(audio, dtype=np.float64).ravel()
        if len(arr) == 0:
            return PerceptPacket.make(
                modality="audio",
                situation_hv=hypervec_rs.HyperVector(0),
                active_predicates=frozenset(["AUDIO_SILENT"]),
                raw_state={"n_samples": 0},
            )

        feature_vec = _extract_audio_features(arr, sample_rate=self.SAMPLE_RATE)
        situation_hv = _features_to_hv(feature_vec)
        descriptors = _get_audio_descriptors(feature_vec)

        preds = frozenset(f"AUDIO_{d.upper()}" for d in descriptors)

        rms_energy = float(np.sqrt(np.mean(arr ** 2)))
        raw_state = {
            "n_samples": len(arr),
            "duration_s": len(arr) / self.SAMPLE_RATE,
            "rms_energy": rms_energy,
            "descriptors": descriptors,
        }

        return PerceptPacket.make(
            modality="audio",
            situation_hv=situation_hv,
            active_predicates=preds,
            raw_state=raw_state,
            adapter_name="AudioAdapter",
            adapter_trace={
                "n_feature_dims": len(feature_vec),
                "descriptors": descriptors,
                "sample_rate": self.SAMPLE_RATE,
            },
        )
