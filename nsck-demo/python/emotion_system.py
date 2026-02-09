"""
NSCK Emotion System (Phase 2.1 + 2.2)
======================================
Implement's Affective Computing based on Plutchik's Wheel and Russell's Circumplex Model.
Maps physiological drives (Homeostasis) -> Emotional States (Valence/Arousal) -> Behaviors (VSA Bias).

Phase 2.2 Enhancement:
- Emotion blending: Produces weighted blend of multiple active emotions
  instead of a single hard label (e.g., 60 % joy + 30 % anticipation).
- Emotion history: Tracks emotional trajectory over time for mood
  analysis and emotional-trend detection.

Theory:
- Plutchik's Wheel: 8 basic emotions.
- Russell's Circumplex: Valence (Positive/Negative) vs Arousal (Calm/Excited).
"""

import math
from typing import Dict, List, Any, Tuple
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


# Circumplex prototypes: (valence, arousal) centres for each basic emotion
_EMOTION_PROTOTYPES: Dict[str, Tuple[float, float]] = {
    "joy":          ( 0.8,  0.7),
    "trust":        ( 0.5,  0.2),
    "fear":         (-0.7,  0.8),
    "surprise":     ( 0.0,  0.9),
    "sadness":      (-0.6,  0.2),
    "disgust":      (-0.5,  0.4),
    "anger":        (-0.5,  0.8),
    "anticipation": ( 0.3,  0.5),
    "neutral":      ( 0.0,  0.0),
}


class EmotionSystem:
    """
    Emotion generator and recognizer.
    """

    HISTORY_LIMIT = 200  # Max entries kept in emotion history

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

        # Phase 2.2: Emotion blend (weighted mix of active emotions)
        self.emotion_blend: Dict[str, float] = {"neutral": 1.0}

        # Phase 2.2: Emotion history for trajectory / mood analysis
        self.emotion_history: List[Dict[str, Any]] = []
        self._step = 0
        
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
        
        # Arousal blends towards max_drive with smooth EMA; decays when drives drop
        blend = 0.3  # how fast arousal tracks the current drive level
        self.arousal = (1.0 - blend) * self.arousal + blend * max_drive
        
        # Map (valence, arousal) -> discrete emotion + blend
        self.current_emotion = self._map_to_basic_emotion()
        self.emotion_intensity = math.sqrt(self.valence**2 + self.arousal**2)
        self.emotion_blend = self._compute_emotion_blend()

        # Record in history
        self._step += 1
        self._record_history()
    
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

    # ------------------------------------------------------------------
    # Phase 2.2: Emotion blending
    # ------------------------------------------------------------------

    def _compute_emotion_blend(self) -> Dict[str, float]:
        """
        Produce a soft blend of active emotions based on distance in
        (valence, arousal) space to each emotion prototype.

        Returns a dict emotion_name -> weight (weights sum to 1.0).
        """
        v, a = self.valence, self.arousal
        inv_distances: Dict[str, float] = {}
        for emo, (pv, pa) in _EMOTION_PROTOTYPES.items():
            dist = math.sqrt((v - pv) ** 2 + (a - pa) ** 2)
            inv_distances[emo] = 1.0 / (dist + 0.01)  # avoid div-by-zero

        total = sum(inv_distances.values())
        blend = {e: w / total for e, w in inv_distances.items()}

        # Keep only emotions with weight >= 5 %
        blend = {e: w for e, w in blend.items() if w >= 0.05}
        total2 = sum(blend.values())
        if total2 > 0:
            blend = {e: w / total2 for e, w in blend.items()}
        else:
            blend = {"neutral": 1.0}
        return blend

    def get_emotion_blend(self) -> Dict[str, float]:
        """Return the current emotion blend (emotion -> weight)."""
        return dict(self.emotion_blend)

    # ------------------------------------------------------------------
    # Phase 2.2: Emotion history & mood analysis
    # ------------------------------------------------------------------

    def _record_history(self):
        """Append current state to emotion history (bounded)."""
        self.emotion_history.append({
            "step": self._step,
            "emotion": self.current_emotion,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "intensity": round(self.emotion_intensity, 4),
        })
        if len(self.emotion_history) > self.HISTORY_LIMIT:
            self.emotion_history.pop(0)

    def get_mood(self, window: int = 10) -> Dict[str, Any]:
        """
        Compute a *mood* — the average emotional state over the last
        *window* steps.  Mood is a slow-moving summary unlike the fast
        per-step emotion.

        Returns:
            Dict with 'avg_valence', 'avg_arousal', 'dominant_emotion',
            and 'stability' (std-dev of valence, lower = more stable).
        """
        recent = self.emotion_history[-window:] if self.emotion_history else []
        if not recent:
            return {
                "avg_valence": 0.0,
                "avg_arousal": 0.0,
                "dominant_emotion": "neutral",
                "stability": 1.0,
            }
        vals = [r["valence"] for r in recent]
        aros = [r["arousal"] for r in recent]
        avg_v = sum(vals) / len(vals)
        avg_a = sum(aros) / len(aros)
        stability = float(np.std(vals)) if len(vals) > 1 else 0.0

        # Dominant emotion = most frequent in window
        from collections import Counter
        counts = Counter(r["emotion"] for r in recent)
        dominant = counts.most_common(1)[0][0]

        return {
            "avg_valence": round(avg_v, 3),
            "avg_arousal": round(avg_a, 3),
            "dominant_emotion": dominant,
            "stability": round(stability, 3),
        }

    def get_emotional_trajectory(self) -> List[Dict[str, Any]]:
        """Return the full emotion history (up to HISTORY_LIMIT entries)."""
        return list(self.emotion_history)
    
    def recognize_emotion_from_text(self, text: str) -> str:
        """Detect emotion in user's text."""
        # Simple keyword-based classifier (replace with ML model later)
        emotion_keywords = {
            "joy": ["happy", "glad", "great", "wonderful", "excellent", "good"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "bad"],
            "anger": ["angry", "furious", "mad", "irritated", "hate"],
            "fear": ["afraid", "scared", "worried", "anxious"],
            "surprise": ["surprised", "shocked", "amazed", "astonished"],
            "trust": ["trust", "reliable", "safe", "confident", "loyal"],
            "disgust": ["disgusting", "gross", "revolting", "nasty"],
            "anticipation": ["excited", "eager", "looking forward", "can't wait"],
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
            "intensity": round(self.emotion_intensity, 2),
            "blend": self.get_emotion_blend(),
        }
    
    def get_emotion_hypervector(self):
        """Return the VSA vector for the current emotion."""
        return self.emotion_codebook.get(self.current_emotion)
    
    def get_emotion_vector(self, emotion_name: str):
        """Get the VSA vector for a specific emotion."""
        return self.emotion_codebook.get(emotion_name, self.emotion_codebook.get("neutral"))
