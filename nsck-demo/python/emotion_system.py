"""
NSCK Emotion System (Phase 2.1)
===============================
Implement's Affective Computing based on Plutchik's Wheel and Russell's Circumplex Model.
Maps physiological drives (Homeostasis) -> Emotional States (Valence/Arousal) -> Behaviors (VSA Bias).

Theory:
- Plutchik's Wheel: 8 basic emotions.
- Russell's Circumplex: Valence (Positive/Negative) vs Arousal (Calm/Excited).
"""

import math
from typing import Dict, List, Any
import numpy as np

# Use our lightweight VSA shim (Auto-selects Rust or Python)
try:
    from hypervec_py import HyperVector
except ImportError:
    # Fallback if module not in path
    print("WARNING: hypervec_py not found. Using minimal mocks.")
    class HyperVector:
        def __init__(self, seed=None): 
            self.mock = True
        def __repr__(self): return "<MockHV>"

class EmotionSystem:
    """
    Emotion generator and recognizer.
    """
    def __init__(self):
        # Emotion space: 2D (valence, arousal)
        self.valence = 0.0  # -1.0 (negative) to +1.0 (positive)
        self.arousal = 0.0  # 0.0 (calm) to 1.0 (excited)
        
        # 8 basic emotions (Plutchik)
        self.basic_emotions = [
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation"
        ]
        
        # Current emotional state
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.0
        
        # Emotion-to-hypervector mapping
        self.emotion_codebook = self._build_emotion_codebook()
    
    def _build_emotion_codebook(self) -> Dict[str, Any]:
        """Create VSA representations for emotions."""
        codebook = {}
        for emotion in self.basic_emotions + ["neutral"]:
            # Generate random hypervector representing this concept
            codebook[emotion] = HyperVector()
        return codebook
    
    def update_from_drives(self, drives: Dict[str, float], reward: float):
        """
        Map homeostatic drives to emotions.
        Drives: {'hunger': 0.0..1.0, 'pain': 0.0..1.0, ...}
        Reward: -1.0..1.0 (External feedback)
        """
        # Valence: driven by reward and drive satisfaction
        # High 'pain' drive pushes valence negative.
        # Positive reward pushes valence positive.
        
        # Decay valence towards 0 (homeostasis)
        self.valence *= 0.95
        
        # Impact of Reward
        if reward > 0.1:
            self.valence = min(1.0, self.valence + 0.2)
        elif reward < -0.1:
            self.valence = max(-1.0, self.valence - 0.2)
            
        # Impact of High Drives (Unmet needs = Negative Valence)
        avg_drive = np.mean(list(drives.values())) if drives else 0
        if avg_drive > 0.5:
             # Suffering
             self.valence = max(-1.0, self.valence - 0.05)
        
        # Arousal: driven by drive urgency (Max drive)
        # High hunger/pain = High arousal
        max_drive = max(drives.values()) if drives else 0
        
        # Arousal also decays but is propped up by drives
        self.arousal = max_drive
        
        # Map (valence, arousal) -> discrete emotion
        self.current_emotion = self._map_to_basic_emotion()
        self.emotion_intensity = math.sqrt(self.valence**2 + self.arousal**2)
    
    def _map_to_basic_emotion(self) -> str:
        """Convert (valence, arousal) to basic emotion."""
        v, a = self.valence, self.arousal
        
        # Circumplex Model Mapping chunks
        if abs(v) < 0.2 and a < 0.3:
            return "neutral"
        elif v > 0.3 and a > 0.5:
            return "joy"
        elif v > 0.2 and a < 0.4:
            return "trust"
        elif v < -0.5 and a > 0.6:
            return "fear" # Passive avoidance? Or panic.
        elif v < -0.3 and a < 0.4:
            return "sadness"
        elif v < -0.2 and a > 0.5:
            return "anger" # Active aggression
        elif a > 0.7:
             return "surprise" # High arousal, neutral/ambiguous valence
        else:
            return "anticipation" # Default active state?
    
    def recognize_emotion_from_text(self, text: str) -> str:
        """Detect emotion in user's text."""
        # Simple keyword-based classifier (replace with ML model later)
        emotion_keywords = {
            "joy": ["happy", "glad", "great", "wonderful", "excellent", "good"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "bad"],
            "anger": ["angry", "furious", "mad", "irritated", "hate"],
            "fear": ["afraid", "scared", "worried", "anxious"],
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"
    
    def get_emotion_info(self) -> Dict[str, Any]:
        """Telemetry."""
        return {
            "name": self.current_emotion,
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "intensity": round(self.emotion_intensity, 2)
        }
    
    def get_emotion_hypervector(self):
        """Return the VSA vector for the current emotion."""
        return self.emotion_codebook.get(self.current_emotion)
