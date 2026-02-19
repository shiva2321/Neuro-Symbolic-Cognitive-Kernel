import numpy as np
from pymdp.agent import Agent
from pymdp import utils

class SnakeAI:
    def __init__(self):
        """
        Active Inference Agent for Snake.
        
        States (Hidden):
        - Snake Position (Not fully tracked here, simplistic model)
        - Apple Direction (North, South, East, West)
        - Immediate Obstacle (None, Wall_North, Wall_South, etc.)
        
        Observations:
        - Visual Classification from SNN (Apple_Front, Wall_Front, Empty_Front)
        
        Actions:
        - Move North, South, East, West
        """
        
        # Simple Model: 
        # Observations: [Apple, Wall, Empty] (3)
        # Actions: [Move Forward, Turn Left, Turn Right] or absolute [N, S, E, W]
        # Let's stick to absolute for simplicity in mapping.
        
        # A_matrix (Likelihood): Maps Hidden State -> Observation
        # B_matrix (Transition): Maps State(t) + Action -> State(t+1)
        # C_matrix (Prior Preferences): Desired Observations
        
        # Let's simplify: The agent wants to observe "Eating Apple".
        # But SNN gives us [Apple, Wall, Empty] relative to head? 
        # Or relative to grid?
        # Let's assume the SNN validates "Is there an apple in the direction I am facing?"
        
        # Constructing a minimal ActInf agent
        num_obs = [3] # 0:Apple, 1:Wall, 2:Empty
        num_states = [4] # Context: Apple is N, S, E, W
        num_controls = [4] # Move N, S, E, W
        
        # Likelihood mapping (approximate)
        # If State is Apple_N and I Move N -> Obs Apple?
        # This requires more complex state mapping. 
        # Let's use a very high level "Reflexive" ActInf for the prototype.
        # State: [Safe, Danger]
        # Control: [Continue, Turn]
        
        # Reverting to user requirement: "Minimize Expected Free Energy"
        # We need a proper C matrix.
        
        self.num_obs = num_obs
        self.num_states = num_states
        self.num_controls = num_controls
        
        # Initialize Agent
        # A: Random initialization then sculpted? Or fixed?
        # Let's make it flat for now, as we don't have a full world model training loop in the prompt.
        
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        
        # Preference: Highly prefer Apple (index 0), Dislike Wall (index 1)
        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 5.0  # Prefer Apple
        C[0][1] = -5.0 # Avoid Wall
        C[0][2] = 0.0  # Indifferent to Empty space (explore)
        
        self.agent = Agent(A=A, B=B, C=C, policy_len=1)
        
    def step(self, observation_idx, precision_signal=1.0):
        """
        Step function.
        observation_idx: 0=Apple, 1=Wall, 2=Empty
        precision_signal: Derived from VSA confidence (0.0 to 1.0)
        """
        
        # We can modulate the updating of beliefs (gamma/alpha) or influence the softmax temp.
        # In pymdp, precision is usually handled internally or via `gamma`.
        # Here we will perform a standard step but conceptually "weigh" it.
        # If precision is low (VSA unclean), we might stick to prior.
        
        # Since pymdp Agent.step() is a black box, we can just pass the observation.
        # To use `precision`, we might modify the `inference_horizon` or just rely on the fact 
        # that high uncertainty in observation (e.g. flat probability) would naturally occur 
        # if we passed a distribution.
        # But here we pass a hard index.
        
        # Let's just run the standard ActInf loop
        obs = [observation_idx]
        qs = self.agent.infer_states(obs)
        
        # Plan policy
        # If precision is high, we trust our model likelihood.
        q_pi, neg_efe = self.agent.infer_policies()
        
        # Sample action
        action = self.agent.sample_action()
        
        return int(action[0])

# Standalone check
if __name__ == "__main__":
    ai = SnakeAI()
    print("SnakeAI Initialized")
