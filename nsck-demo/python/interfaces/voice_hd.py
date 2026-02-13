"""
NSCK VoiceHD Audio Encoder (Phase 2.1)
======================================
Converts raw audio into Hypervectors using custom MFCC extraction and VSA encoding.
Hypervector Dimension: 10,000 bits (Binary).
Operations: XOR (Binding), Cyclic Shift (Permutation), Majority Vote (Bundling).

No heavy dependencies (librosa). Uses numpy/scipy.
"""

import numpy as np
import scipy.fft
import scipy.signal
from scipy.fftpack import dct
from typing import List, Tuple, Optional

# --- CONSTANTS ---
HV_DIM = 10000
SAMPLE_RATE = 16000
N_MFCC = 13
N_FILTERS = 26
N_FFT = 512
FRAME_SIZE = 0.025  # 25ms
FRAME_STRIDE = 0.010 # 10ms
NGRAM_SIZE = 3      # Temporal window
QUANTIZATION_LEVELS = 20

class VoiceHDEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # 1. Generate ID Vectors (Role: MFCC Index 0..12)
        # Shape: (13, 10000)
        self.id_vectors = self._generate_random_hv(N_MFCC)
        
        # 2. Generate Level Vectors (Filler: Quantized Value)
        # "Thermometer" encoding: correlated levels
        self.level_vectors = self._generate_correlated_levels(QUANTIZATION_LEVELS, flip_rate=0.05)
        
        # Min/Max for quantization (will adapt or fix based on expected MFCC range)
        # MFCCs are typically negative (Log Energy). 
        # Range [-20, 20] was too wide. Values clustered around -5.
        # Adjusted to [-5, 5] for Delta MFCCs (centered at 0)
        self.min_val = -5.0 
        self.max_val = 5.0

    def _generate_random_hv(self, count: int) -> np.ndarray:
        """Generate random binary hypervectors (0/1)."""
        # Using 0/1 (bool) for efficient storage, though XOR works on 0/1 intuitively.
        return self.rng.randint(0, 2, size=(count, HV_DIM)).astype(bool)

    def _generate_correlated_levels(self, levels: int, flip_rate: float) -> np.ndarray:
        """
        Generate sequence of vectors where L_i is close to L_i+1.
        Method: Bit-flip drift.
        """
        vectors = np.zeros((levels, HV_DIM), dtype=bool)
        current = self.rng.randint(0, 2, size=HV_DIM).astype(bool)
        vectors[0] = current.copy()
        
        n_flip = int(HV_DIM * flip_rate)
        
        for i in range(1, levels):
            # Select random indices to flip from the PREVIOUS vector
            # ensuring we drift gradually
            indices = self.rng.choice(HV_DIM, size=n_flip, replace=False)
            current = current.copy()
            current[indices] = ~current[indices] # Flip
            vectors[i] = current
            
        return vectors

    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Custom MFCC implementation matching spec.
        Input: 1D array (samples).
        Output: (NumFrames, 13) array.
        """
        # 0. Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 1. Framing
        frame_len = int(FRAME_SIZE * SAMPLE_RATE)
        frame_step = int(FRAME_STRIDE * SAMPLE_RATE)
        sig_len = len(audio)
        
        if sig_len <= frame_len:
            num_frames = 1
        else:
            num_frames = 1 + int(np.ceil((1.0 * sig_len - frame_len) / frame_step))
            
        pad_len = int((num_frames - 1) * frame_step + frame_len)
        zeros = np.zeros((pad_len - sig_len,))
        pad_signal = np.append(audio, zeros)
        
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # 2. Window (Hamming)
        frames *= np.hamming(frame_len)
        
        # 3. FFT (Power Spectrum)
        mag_frames = np.absolute(scipy.fft.rfft(frames, N_FFT))
        pow_frames = ((1.0 / N_FFT) * ((mag_frames) ** 2))
        
        # 4. Mel Filterbank
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_FILTERS + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE)

        fbank = np.zeros((N_FILTERS, int(np.floor(N_FFT / 2 + 1))))
        for m in range(1, N_FILTERS + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
                
        filter_banks = np.dot(pow_frames, fbank.T)
        # Avoid log of zero
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = np.log(filter_banks)
        
        # 5. DCT
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :N_MFCC]
        
        return mfcc

    def _quantize(self, val: float) -> int:
        """Map value to level index 0..L-1."""
        # Clip and normalize
        norm = (val - self.min_val) / (self.max_val - self.min_val)
        norm = max(0.0, min(1.0, norm))
        idx = int(norm * (QUANTIZATION_LEVELS - 1))
        return idx
    
    def _majority_vote(self, vectors: np.ndarray) -> np.ndarray:
        """
        Bundling via Majority Vote.
        Input: (N, HV_DIM) bool array.
        Output: (HV_DIM,) bool array.
        """
        # Sum bits along axis 0
        sums = np.sum(vectors, axis=0)
        
        # Dynamic Thresholding for Sparse Vectors
        # If we use N/2, sparse vectors vanish.
        # If we use >0, they saturate.
        # Use Mean + k*Std to pick the "Significant" bits.
        mean_act = np.mean(sums)
        std_act = np.std(sums)
        threshold = mean_act # + 0.0 * std_act
        
        # Ensure at least 1 vote required
        threshold = max(0.5, threshold) 
        
        return sums > threshold

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Full Pipeline: Audio -> MFCC -> Deltas -> Frame Encoding -> Temporal -> Clip Vector.
        """
        # 1. Extract Features
        mfccs = self._compute_mfcc(audio) # (T, 13)
        
        
        # 2. Compute Deltas (Strided diff for larger magnitude)
        DELTA_STRIDE = 3
        if len(mfccs) < DELTA_STRIDE + 1:
             return np.zeros(HV_DIM, dtype=bool)
             
        deltas = mfccs[DELTA_STRIDE:] - mfccs[:-DELTA_STRIDE]
        
        # Normalize Deltas
        deltas = deltas * 2.0
        
        num_frames = deltas.shape[0]
        frame_hvs = []
        
        # 3. Frame Encoding
        for t in range(num_frames):
            frame_feats = deltas[t]  # (13,)
            bindings = []
            for k in range(N_MFCC):
                val = frame_feats[k]
                level_idx = self._quantize(val)
                
                # Roll-Filler Binding: ID_k XOR Level_val
                # XOR on bools is logical_xor
                bind = np.logical_xor(self.id_vectors[k], self.level_vectors[level_idx])
                bindings.append(bind)
            
            # Bundle frame: Majority Vote of the 13 bindings
            v_t = self._majority_vote(np.array(bindings))
            frame_hvs.append(v_t)
            
        frame_hvs = np.array(frame_hvs)
        
        if len(frame_hvs) == 0:
             return np.zeros(HV_DIM, dtype=bool)

        # 3. Global Temporal Encoding with Differential (Edge) Detection
        # V_clip = MajorityVote( sum( Perm^t( V_t XOR V_{t-1} ) ) )
        # This removes the static background and encodes "Change Direction".
        
        temporal_hvs = []
        if len(frame_hvs) < NGRAM_SIZE:
             # Fallback
             return self._majority_vote(frame_hvs)

        # 3. Temporal Encoding (N-gram Sliding Window)
        # N-gram captures Direction: (A, B) != (B, A)
        # Gram = V_t XOR Perm(V_{t-1}) XOR Perm^2(V_{t-2})
        window_hvs = []
        
        for t in range(NGRAM_SIZE-1, num_frames):
            # Construct Gram from t, t-1, t-2...
            # Start with V_t
            gram = frame_hvs[t]
            
            for k in range(1, NGRAM_SIZE):
                v_prev = frame_hvs[t-k]
                v_perm = np.roll(v_prev, k) # Permute by lag k
                gram = np.logical_xor(gram, v_perm)
            
            window_hvs.append(gram)
            
        if not window_hvs:
             return np.zeros(HV_DIM, dtype=bool)

        # 4. Sequence Bundling
        v_clip = self._majority_vote(np.array(window_hvs))
        return v_clip

    @staticmethod
    def hamming_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
        """1.0 - Hamming Distance (Normalized)"""
        diffs = np.logical_xor(hv1, hv2)
        dist = np.mean(diffs) # 0.0 = Identical, 0.5 = Random, 1.0 = Inverse
        return 1.0 - dist

