"""
NSCK Empathy Module (Phase 2.3)
===============================
Implements emotional resonance (Mirror Neurons) and compassion.
Uses Theory of Mind to infer others' states, and EmotionSystem to "feel" them.
"""

from typing import Dict, Any, Optional
from python.core.cognitive.emotion_system import EmotionSystem
from python.core.cognitive.theory_of_mind import TheoryOfMind, MentalStateModel

class EmpathyModule:
    """
    Emotional resonance and compassionate responses.
    """
    def __init__(self, emotion_system: EmotionSystem, theory_of_mind: TheoryOfMind):
        self.emotions = emotion_system
        self.tom = theory_of_mind
        
        # Empathy strength (How much we "feel" others' emotions 0.0 to 1.0)
        self.empathy_coefficient = 0.7 
        print("EmpathyModule Initialized.")

    def empathize(self, other_agent_id: str) -> Dict[str, Any]:
        """
        Experience emotional resonance with another agent.
        Returns details of the interaction.
        """
        # 1. Infer their emotional state from behavior/expression
        other_emotion = self._infer_emotion(other_agent_id)
        
        # 2. Partially adopt their emotional state (Mirror Neuron)
        valence_shift = self._emotional_contagion(other_emotion)
        
        # 3. Generate compassionate response (Social behavior)
        response = self._generate_compassionate_response(other_emotion)
        
        return {
            "target": other_agent_id,
            "inferred_emotion": other_emotion,
            "self_valence_shift": valence_shift,
            "response": response
        }
    
    def _infer_emotion(self, agent_id: str) -> str:
        """
        Infer agent's emotion from their mental state.
        Heuristic: Are their desires satisfied?
        """
        model = self.tom.agent_models.get(agent_id)
        if not model:
            return "neutral"
        
        # Simple heuristic eval based on desires
        # We need to know if the agent THINKS its desires are met.
        satisfied_count = 0
        blocked_count = 0
        
        for desire in model.desires:
            # Assume desire is a string "action_target" or "have_object"
            # This is very simplified logic for Phase 2.3
            if desire.startswith("have_"):
                obj = desire.replace("have_", "")
                # Check if they believe they have it (e.g. location match)
                # For this demo, we'll check if belief 'holding' == obj
                if model.get_belief("holding") == obj:
                    satisfied_count += 1
                else:
                    blocked_count += 1
            
            # Special case for Sally-Anne: "find_ball"
            elif desire == "find_ball":
                # If they believe they know where it is, they are hopeful/anticipating. 
                # If they looked and failed, they are sad.
                # For now, let's assume if they have a location belief, they are content/anticipating.
                loc = model.get_belief("ball_location")
                if loc:
                    satisfied_count += 0.5 # Anticipation
                else:
                    blocked_count += 1 # Frustration
        
        # Map satisfaction to basic emotions
        if satisfied_count > blocked_count:
            return "joy"
        elif blocked_count > satisfied_count:
            return "sadness" # or anger
        else:
            return "neutral"
    
    def _emotional_contagion(self, other_emotion: str) -> float:
        """
        Partially adopt the other's emotional state.
        Modulates the self-agent's Valence.
        """
        # Map emotion to valence impact
        impact_map = {
            "joy": +0.5,
            "trust": +0.2,
            "sadness": -0.5,
            "anger": -0.3,
            "fear": -0.4,
            "neutral": 0.0
        }
        
        raw_impact = impact_map.get(other_emotion, 0.0)
        
        # Apply coefficient
        weighted_impact = raw_impact * self.empathy_coefficient
        
        # Update self-valence
        # We clamp it to -1.0..1.0 inside EmotionSystem usually, 
        # but here we manually modify the float attribute.
        original_valence = self.emotions.valence
        self.emotions.valence = max(-1.0, min(1.0, original_valence + weighted_impact))
        
        # Return the delta for logging
        return weighted_impact
    
    def _generate_compassionate_response(self, other_emotion: str) -> str:
        """Generate a social response based on the emotion."""
        if other_emotion == "sadness":
            return "I see you are sad. Can I help you?"
        elif other_emotion == "joy":
            return "I am happy that you are happy!"
        elif other_emotion == "anger":
            return "Please calm down, I wish you no harm."
        elif other_emotion == "fear":
            return "Do not be afraid, you are safe."
        else:
            return "Hello friend."
