"""
NSCK Voice Interface Module (Phase 1.4)
=======================================
Roles:
1. Speech-to-Text (ASR) Interface
2. Text-to-Speech (TTS) Interface
3. Prosody Analysis (Pitch/Energy/Emotion)
4. Audio-to-HV Encoding (VoiceHD)
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Optional
import os

from voice_hd import VoiceHDEngine

class ProsodyAnalyzer:
    """
    Extracts acoustic features (Pitch, Energy, Tempo) to infer emotional state.
    Uses basic Signal Processing (No Deep Learning).
    """
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def analyze(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Returns:
            - pitch: Fundamental frequency (Hz)
            - energy: RMS power
            - tempo: Estimated speaking rate (simplified)
        """
        if len(audio) == 0:
            return {"pitch": 0, "energy": 0, "tempo": 0}

        # 1. Energy (RMS)
        energy = np.sqrt(np.mean(audio**2))

        # 2. Pitch (Fundamental Frequency via Autocorrelation)
        # Focus on human speech range: 50Hz - 500Hz
        corr = scipy.signal.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        
        # Define search range for pitch
        d_min = self.sr // 500
        d_max = self.sr // 50
        
        # Find peak in autocorrelation
        if len(corr) > d_max:
            # Look for max in range [d_min, d_max]
            peak = np.argmax(corr[d_min:d_max]) + d_min
            pitch = self.sr / peak
        else:
            pitch = 0

        # 3. Tempo (Simplified: Number of bursts/syllables)
        # Using Zero-Crossing Rate as a proxy for activity
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        
        return {
            "pitch": float(pitch),
            "energy": float(energy),
            "tempo": float(zcr)
        }

class VoiceInterface:
    def __init__(self):
        # Engines
        self.voice_hd = VoiceHDEngine()
        self.prosody = ProsodyAnalyzer()
        
        # Mocks for heavy components (Whisper / Coqui)
        # In a full install, these would be initialized here.
        self.asr_available = False
        self.tts_available = False
        
        print("VoiceInterface Initialized (Prosody + VoiceHD)")

    def listen(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transforms raw audio into:
        1. Text (via ASR)
        2. Acoustic Features (Prosody)
        3. Identity/Grounded Vector (VoiceHD)
        """
        # 1. Prosody Analysis
        features = self.prosody.analyze(audio)
        
        # 2. VoiceHD Encoding
        # This is the "Sub-symbolic" grounding
        audio_hv = self.voice_hd.encode(audio)
        
        # 3. ASR (Mocked for now)
        # In production: text = self.whisper.transcribe(audio)
        text = self._mock_asr(audio, features)
        
        # 4. Infer Emotion from Prosody
        emotion = self.infer_emotion(features)
        
        return {
            "text": text,
            "prosody": features,
            "audio_hv": audio_hv,
            "emotion": emotion
        }

    def speak(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Generates audio from text, modulated by emotion.
        """
        print(f"[Agent Speaking - {emotion}]: {text}")
        
        # 1. Modulation Logic (Affective TTS)
        # In a real system, we would pass these to Coqui TTS.
        params = self._get_tts_params(emotion)
        
        # 2. TTS Generation (Mocked: Returns a sine wave or silence)
        # In production: audio = self.tts.synthesize(text, **params)
        audio = self._mock_tts(text, params)
        
        return audio

    def infer_emotion(self, features: Dict[str, float]) -> str:
        """Heuristic mapping from Prosody -> Emotion."""
        p = features["pitch"]
        e = features["energy"]
        
        # Values relative to typical speech (Calibrated for the mock)
        if e > 0.1: # High energy
            if p > 200: return "excited"
            else: return "angry"
        elif e < 0.01: # Low energy
            if p <= 110: return "sad"
            else: return "calm"
        else:
            return "neutral"

    def _get_tts_params(self, emotion: str) -> Dict[str, Any]:
        """Map emotion to TTS engine parameters."""
        if emotion == "excited":
            return {"pitch": 1.2, "speed": 1.2}
        elif emotion == "sad":
            return {"pitch": 0.8, "speed": 0.8}
        elif emotion == "angry":
            return {"pitch": 1.1, "speed": 1.1, "volume": 1.5}
        return {"pitch": 1.0, "speed": 1.0}

    # --- PRIVACY & MOCK HELPERS ---
    def _mock_asr(self, audio: np.ndarray, prosody: Dict) -> str:
        """Fallback ASR for dev environment."""
        # Heuristic: If energy is very low, assume silence.
        if prosody["energy"] < 0.001:
            return ""
        return "I am speaking to you." # Placeholder

    def _mock_tts(self, text: str, params: Dict) -> np.ndarray:
        """Generates a dummy audio buffer."""
        # 1 second of audio (16kHz)
        t = np.linspace(0, 1, 16000)
        # Frequency modulated by pitch param
        freq = 440 * params.get("pitch", 1.0)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        return audio
