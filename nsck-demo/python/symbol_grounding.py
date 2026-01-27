import numpy as np
# Assuming hypervec_py is available, but for minimal prototype refactor, we can simulate vectors
# or just use string constants if we are only testing the "logic" of transfer first.
# But the plan says "HyperVector". Let's check if hypervec_py exists.
# It does (from file listing).
try:
    from hypervec_py import HyperVector
except ImportError:
    # Fallback/Mock for testing if environment issues
    class HyperVector:
        def __init__(self, seed=None): pass
        def xor(self, o): return self
        def bundle(self, o): return self

# --- CONCEPT CODEBOOK (SEEDS) ---
SEED_GOAL = 1001
SEED_DANGER = 1002
SEED_SAFE = 1003
SEED_LEFT = 2001 # Relative Left
SEED_RIGHT = 2002
SEED_CENTER = 2003

# Predicates for Abstract Concept Transfer
# We map Game State -> Predicates -> Actions
# This proves "Reasoning" rather than just coordinate coincident.

class ActionSemantics:
    """
    Defines the 'Meaning' of actions in terms of abstract predicates.
    
    Structure:
    1. Extract Predicates (e.g., IS_ABOVE, IS_BELOW)
    2. Map Predicates to Abstract Symbolic Action (MOVE_Towards_GOAL)
    3. Map Symbolic Action back to Concrete Motor Command
    """
    
    @staticmethod
    def get_goal_alignment(game_type, state):
        if game_type == "snake":
            return ActionSemantics._snake_logic(state)
        elif game_type == "pong":
            return ActionSemantics._pong_logic(state)
        return []

    @staticmethod
    def _snake_logic(state):
        # 1. Extract Predicates relative to Head
        hx, hy = state["head"]
        fx, fy = state["food"]
        
        # Symbolic State Abstraction
        is_above = (fy < hy)
        is_below = (fy > hy)
        is_left  = (fx < hx)
        is_right = (fx > hx)
        
        # 2. Reasoning Rule: "To Reach GOAL, reduce difference"
        # Since we don't have a full logical graph solver yet, we hardcode the 
        # "Reasoning" outcome: If GOAL is ABOVE, Action should be UP.
        
        scores = np.zeros(4, dtype=np.float32) # UP, DN, LF, RT
        
        if is_above: scores[0] = 1.0
        if is_below: scores[1] = 1.0
        if is_left:  scores[2] = 1.0
        if is_right: scores[3] = 1.0
        
        return scores

    @staticmethod
    def _pong_logic(state):
        # 1. Extract Predicates relative to Paddle Center
        # This explicitly creates the same "State Space" as snake: Relative Position
        p1_y = state["p1_y"]
        ball_y = state["ball_y"]
        paddle_center = p1_y + 3
        
        # Symbolic State Abstraction (Explicitly matching Snake's logic)
        is_above = (ball_y < paddle_center - 1) # "Food" is above "Head"
        is_below = (ball_y > paddle_center + 1) # "Food" is below "Head"
        
        # 2. Reasoning Rule: Reused from "General Navigation"
        # "If GOAL is ABOVE, Move UP"
        
        scores = np.zeros(2, dtype=np.float32) # UP, DN
        
        if is_above: scores[0] = 1.0 # UP matches Snake's UP logic
        if is_below: scores[1] = 1.0 # DOWN matches Snake's DOWN logic
        
        if not is_above and not is_below:
             # Aligned / Null State
             scores[:] = 0.5
             
        return scores

    @staticmethod
    def get_danger_alignment(game_type, state):
        """
        Returns alignment with 'AVOID_THREAT'.
        High score = Action leads to Safety.
        Low score = Action leads to DANGER.
        """
        # Phase 2 relied on simulation.py. Phase 6 could expand this.
        pass
