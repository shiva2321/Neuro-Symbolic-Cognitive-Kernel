import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from typing import List, Optional, Any

class ActiveInferenceController:
    """
    Active Inference Controller (System 2 Control).
    
    This agent uses the Free Energy Principle to select actions that:
    1. Minimize Surprise (Pragmatic Value): Reach goals (preferences).
    2. Minimize Ambiguity (Epistemic Value): Explore unknown states.
    
    Crucially, it is modulated by a context-sensitive 'Precision' ($\gamma$)
    derived from the VSA Memory's confidence.
    """
    
    def __init__(
        self, 
        num_obs: List[int], 
        num_states: List[int], 
        num_controls: List[int],
        initial_precision: float = 16.0
    ):
        """
        Initialize the POMDP Agent.
        
        Args:
            num_obs: Dimensions of observation modalities [ObsDim1, ObsDim2...]
            num_states: Dimensions of hidden state factors [StateDim1, StateDim2...]
            num_controls: Dimensions of control factors [ActionDim1, ActionDim2...]
        """
        self.num_obs = num_obs
        self.num_states = num_states
        self.num_controls = num_controls
        
        # Initialize generative model matrices (A, B, C, D)
        # Default initialization (uniform/random - typically needs setup)
        self.A = utils.random_A_matrix(num_obs, num_states)
        self.B = utils.random_B_matrix(num_states, num_controls)
        self.C = utils.obj_array_zeros(num_obs) # Preferences (Goals)
        self.D = utils.obj_array_uniform(num_states) # Prior beliefs
        
        self.agent = Agent(
            A=self.A, 
            B=self.B, 
            C=self.C, 
            D=self.D,
            inference_horizon=1,
            policy_len=1
        )
        
        self.gamma = initial_precision
        
    def set_goal(self, modality_idx: int, obs_idx: int, value: float = 1.0):
        """
        Set a preference (Goal) for a specific observation.
        Basically setting C vector values.
        """
        # pymdp C vector is log-preferences. 
        # Higher value = more preferred.
        self.agent.C[modality_idx][obs_idx] = value

    def step(self, observations: List[int], confidence_z: float) -> int:
        """
        Perform one cognitive cycle: Perception -> Inference -> Action.
        
        Args:
            observations: List of indices for each modality (e.g. [ObservationID])
            confidence_z: Z-score from VSA memory indicating signal quality.
            
        Returns:
            int: Action index
        """
        
        # 1. Modulate Precision (Gamma) based on Confidence
        # If low confidence (High uncertainty), reduce Expected Free Energy precision?
        # Actually, in ActInf, usually precision is learned.
        # Here we hard-code the "Tri-Cognitive Split":
        # Low VSA Confidence -> High Epistemic Value weighting or Trigger Explore?
        
        # We can modulate the softmax temperature of action selection.
        # High Confidence -> Deterministic (Exploit)
        # Low Confidence -> Stochastic (Explore)
        
        # Map Z-score (0 to 100) to simple inverse temp
        # Z < 2.0 -> Entropy High
        if confidence_z < 2.0:
            # High ambiguity. 
            # We might want to boost epistemic value, but simpler:
            # Flatten action probability (high entropy)
            pass 
        else:
            pass

        # 2. State Inference (Perception)
        # pymdp expects list of numpy arrays or ints? ints usually fine for 1D obs
        qs = self.agent.infer_states(observations)
        
        # 3. Policy Inference (Decision)
        q_pi, efe = self.agent.infer_policies()
        
        # 4. Action Selection
        action = self.agent.sample_action()
        
        # Just return first control factor action for now
        return int(action[0])

    def reset(self):
        """Reset beliefs to priors."""
        self.agent.reset()
