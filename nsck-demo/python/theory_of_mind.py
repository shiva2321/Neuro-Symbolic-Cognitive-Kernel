"""
NSCK Theory of Mind Module (Phase 2.2)
======================================
This module enables the agent to model the mental states (beliefs, desires, intentions) of OTHER agents.
It supports:
1.  **Perspective Taking:** Updating an agent's beliefs based on *their* observation, not the global truth.
2.  **False Belief Detection:** Identifying when another agent's belief diverges from reality (The Sally-Anne Test).
3.  **Behavior Prediction:** Predicting an agent's next action based on their mental state.
"""

from typing import Dict, List, Any, Optional

class MentalStateModel:
    """Mental model of another agent."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.beliefs = {}  # What they believe about the world {key: value}
        self.desires = []  # Their goals (Simplified as strings for now)
        self.intentions = []  # Their planned actions
        self.position = None  # Their logical or spatial location
        
    def update_beliefs(self, observation: Dict[str, Any]):
        """
        Update beliefs based on what THIS agent observed.
        We assume if they see it, they believe it.
        """
        for key, value in observation.items():
            self.beliefs[key] = value

    def set_desire(self, desire: str):
        """Add a high-level goal."""
        if desire not in self.desires:
            self.desires.append(desire)
            
    def get_belief(self, key: str) -> Any:
        return self.beliefs.get(key)


class TheoryOfMind:
    """
    Cognitive system for tracking and predicting other agents.
    """
    def __init__(self):
        # Track multiple agents by ID
        self.agent_models: Dict[str, MentalStateModel] = {} 
        print("TheoryOfMind Initialized.")
    
    def get_or_create_model(self, agent_id: str) -> MentalStateModel:
        if agent_id not in self.agent_models:
            self.agent_models[agent_id] = MentalStateModel(agent_id)
        return self.agent_models[agent_id]
    
    def update_agent_perspective(self, agent_id: str, agent_loc: str, observable_world: Dict[str, Any]):
        """
        Update what we think `agent_id` knows.
        
        Args:
            agent_id: The other agent.
            agent_loc: Where the other agent is (room/context).
            observable_world: The state of the world *visible from agent_loc*.
        """
        model = self.get_or_create_model(agent_id)
        model.position = agent_loc
        
        # Simple Logic: The agent sees everything in their current location.
        # So we update their beliefs with 'observable_world'.
        model.update_beliefs(observable_world)
        
    def predict_action(self, agent_id: str) -> str:
        """
        Predict what the agent will do next based on their *Beliefs* and *Desires*.
        Simple heuristic implementation for Sally-Anne.
        """
        model = self.get_or_create_model(agent_id)
        
        # Heuristic 1: If they want an object, and believe it is in X, they go to X.
        for desire in model.desires:
            # Example Desire: "find_ball"
            if desire.startswith("find_"):
                target_obj = desire.replace("find_", "")
                
                # Check where they BELIEVE it is
                believed_loc = model.get_belief(f"{target_obj}_location")
                
                if believed_loc:
                    return f"search_{believed_loc}"
                else:
                    return "search_random"
                    
        return "idle"
    
    def detect_false_belief(self, agent_id: str, reality: Dict[str, Any]) -> List[str]:
        """
        Sally-Anne Test Core Logic.
        Compare Agent's Beliefs vs Reality.
        
        Returns:
            List of keys where the agent holds a false belief.
        """
        false_beliefs = []
        model = self.get_or_create_model(agent_id)
        
        for key, true_val in reality.items():
            agent_val = model.get_belief(key)
            if agent_val is not None:
                if agent_val != true_val:
                    # Mismatch!
                    false_beliefs.append(key)
                    
        return false_beliefs
