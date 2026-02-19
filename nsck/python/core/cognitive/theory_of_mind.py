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
    Supports level-1 (Sally-Anne) and level-2 recursive reasoning
    ("I think agent A thinks agent B thinks …").
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

    # ----------------------------------------------------------------
    # Level-2 Recursive Theory of Mind
    # ----------------------------------------------------------------
    def recursive_belief(
        self,
        observer: str,
        target: str,
        key: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Recursive belief reasoning up to *depth* levels.

        Level 1: "What does *observer* believe about *key*?"
        Level 2: "What does *observer* think *target* believes about *key*?"

        Returns a dict::

            {
                "depth": <int>,
                "chain": ["observer", "target", ...],
                "belief_value": <value or None>,
                "is_accurate": <bool>,     # compared to target's actual belief
                "explanation": <str>,
            }
        """
        obs_model = self.get_or_create_model(observer)
        tgt_model = self.get_or_create_model(target)

        if depth <= 1:
            # Level 1: direct belief
            val = obs_model.get_belief(key)
            return {
                "depth": 1,
                "chain": [observer],
                "belief_value": val,
                "is_accurate": True,  # trivially accurate for own belief
                "explanation": f"{observer} believes {key} = {val}",
            }

        # Level 2+:  observer's *model* of target's belief
        # Heuristic — observer projects own knowledge state onto target,
        # then applies any false-belief adjustments they know about.
        #
        # Key insight: if observer saw target observe X, observer knows
        # target believes X.  If observer saw the world change AFTER
        # target left, observer can infer target still holds old belief.

        # Step 1: check if observer has an explicit record of target's
        # belief (set via update_agent_perspective).
        meta_key = f"__tom_{target}_believes_{key}"
        observer_thinks_target_believes = obs_model.get_belief(meta_key)

        if observer_thinks_target_believes is not None:
            # Observer has an explicit model
            actual_target_belief = tgt_model.get_belief(key)
            accurate = observer_thinks_target_believes == actual_target_belief
            return {
                "depth": depth,
                "chain": [observer, target],
                "belief_value": observer_thinks_target_believes,
                "is_accurate": accurate,
                "explanation": (
                    f"{observer} thinks {target} believes {key} = "
                    f"{observer_thinks_target_believes} "
                    f"(actually {actual_target_belief}, "
                    f"{'correct' if accurate else 'WRONG'})"
                ),
            }

        # Fallback: observer projects own belief onto target (simulation theory)
        projected = obs_model.get_belief(key)
        actual_target_belief = tgt_model.get_belief(key)
        accurate = projected == actual_target_belief

        return {
            "depth": depth,
            "chain": [observer, target],
            "belief_value": projected,
            "is_accurate": accurate,
            "explanation": (
                f"{observer} projects own belief onto {target}: "
                f"{key} = {projected} "
                f"(target actually believes {actual_target_belief}, "
                f"{'correct' if accurate else 'WRONG'})"
            ),
        }

    def update_observer_model_of_target(
        self,
        observer: str,
        target: str,
        key: str,
        value: Any,
    ):
        """Record that *observer* knows *target* believes *key* = *value*.

        Call this when observer witnesses target observe something.
        """
        meta_key = f"__tom_{target}_believes_{key}"
        obs_model = self.get_or_create_model(observer)
        obs_model.update_beliefs({meta_key: value})

    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Return a human-readable summary of an agent's mental model."""
        model = self.get_or_create_model(agent_id)
        return {
            "agent_id": agent_id,
            "position": model.position,
            "beliefs": dict(model.beliefs),
            "desires": list(model.desires),
            "intentions": list(model.intentions),
        }

    # ----------------------------------------------------------------
    # Level-2+ Belief Simulation (replaces pure projection heuristic)
    # ----------------------------------------------------------------
    def simulate_belief(
        self,
        observer: str,
        target: str,
        key: str,
        observation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Simulate what *target* believes by replaying their observation history.

        Instead of the projection heuristic (observer projects own belief
        onto target), this method *simulates* the target's belief formation
        by iterating over the observations that the target actually made.

        Parameters
        ----------
        observer : Agent doing the reasoning.
        target : Agent whose beliefs are being inferred.
        key : The belief key to infer.
        observation_history : List of observation dicts that *target* received,
            in chronological order. If ``None``, falls back to projection.

        Returns
        -------
        Dict with *depth*, *chain*, *belief_value*, *is_accurate*, *explanation*.
        """
        tgt_model = self.get_or_create_model(target)

        if not observation_history:
            # No history — fall back to recursive_belief with projection
            return self.recursive_belief(observer, target, key, depth=2)

        # Replay observations chronologically to compute target's latest belief
        simulated_beliefs: Dict[str, Any] = {}
        for obs in observation_history:
            simulated_beliefs.update(obs)

        simulated_value = simulated_beliefs.get(key)
        actual_value = tgt_model.get_belief(key)
        accurate = simulated_value == actual_value

        return {
            "depth": 2,
            "chain": [observer, target],
            "belief_value": simulated_value,
            "is_accurate": accurate,
            "method": "simulation",
            "explanation": (
                f"{observer} simulated {target}'s belief formation: "
                f"{key} = {simulated_value} after {len(observation_history)} observations "
                f"(actual: {actual_value}, {'correct' if accurate else 'WRONG'})"
            ),
        }

    def predict_action_from_simulation(
        self,
        agent_id: str,
        world_state: Dict[str, Any],
    ) -> str:
        """Predict action by simulating the agent's decision process.

        Constructs what the agent *believes* the world looks like,
        then applies the same desire-based heuristics the agent would use.
        """
        model = self.get_or_create_model(agent_id)

        # Build agent's subjective world view
        subjective_world = dict(model.beliefs)
        # Override with ground truth only for keys the agent has observed
        # (agent's beliefs may be stale)

        # Apply desire-based reasoning on subjective world
        for desire in model.desires:
            if desire.startswith("find_"):
                target_obj = desire.replace("find_", "")
                # Where does the AGENT think the object is?
                believed_loc = model.get_belief(f"{target_obj}_location")
                if believed_loc:
                    return f"search_{believed_loc}"
                else:
                    return "search_random"

        return self.predict_action(agent_id)
