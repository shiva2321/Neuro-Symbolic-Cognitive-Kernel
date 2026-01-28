from typing import Dict, Any, Tuple
from ncgn.control.act_inf import ActiveInferenceController

class Decider:
    """
    Arbitrates between System 1 (Reflex) and System 2 (Deliberation).
    """
    
    def __init__(self, act_inf: ActiveInferenceController):
        self.ai = act_inf
        self.label_to_id: Dict[str, int] = {"None": 0}
        self.next_id = 1
        self.threshold = 80.0 # Z-score threshold
        
    def get_observation_id(self, label: str) -> int:
        """Dynamically map string concepts to integer IDs for POMDP."""
        if label not in self.label_to_id:
            # If we run out of state space in fixed POMDP, this is a problem.
            # ideally POMDP dims are large enough.
            # For this prototype, we assume small concept space.
            self.label_to_id[label] = self.next_id
            self.next_id += 1
            
        return self.label_to_id[label]

    def decide(self, perception_result: Dict[str, Any]) -> str:
        """
        Main Decision Loop.
        
        Args:
            perception_result: Output from Brain.process_perception()
            
        Returns:
            str: Description of action taken.
        """
        label = perception_result.get("label", "None")
        z_score = perception_result.get("confidence_z", 0.0)
        
        # System 1 Check
        if z_score > self.threshold:
            return f"SYSTEM_1_REFLEX: Confident recognition of '{label}' (Z={z_score:.1f}). Proceeding with graph propagation."
            
        # System 2 Engagement (Active Inference)
        # 1. Map observation
        obs_id = self.get_observation_id(label)
        
        # 2. Step Agent
        # We model the "Cognitive State" as the observation
        action_idx = self.ai.step([obs_id], z_score)
        
        # 3. Interpret Action
        # (Placeholder: Action 0=Wait, 1=LookCloser, 2=AskUser)
        actions = ["WAIT", "LOOK_CLOSER", "ASK_USER"]
        action_name = actions[action_idx % len(actions)]
        
        return f"SYSTEM_2_ACT_INF: Ambiguous input '{label}' (Z={z_score:.1f}). Agent chose: {action_name}"
