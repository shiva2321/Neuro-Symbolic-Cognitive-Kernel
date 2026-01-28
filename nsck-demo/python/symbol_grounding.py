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
        def similarity(self, o): return 0.5

# --- CONCEPT CODEBOOK (SEEDS) ---
SEED_GOAL = 1001
SEED_DANGER = 1002
SEED_SAFE = 1003
SEED_LEFT = 2001 # Relative Left
SEED_RIGHT = 2002
SEED_CENTER = 2003

# --- REASONING ENGINE (BITWISE VSA) ---
class ReasoningEngine:
    """
    Replaces coordinate math with Bitwise Hypervector logic.
    Instead of 'if x < y', we use 'State XOR Concept' similarity.
    """
    def __init__(self):
        # 1. Base Symbols (Random High-Dim Patterns)
        self.codebook = {
            "ACTION_UP": HyperVector(10),
            "ACTION_DN": HyperVector(20),
            "ACTION_LF": HyperVector(30),
            "ACTION_RT": HyperVector(40),
            "REL_ABOVE": HyperVector(101),
            "REL_BELOW": HyperVector(102),
            "REL_LEFT":  HyperVector(103),
            "REL_RIGHT": HyperVector(104)
        }
        
        # 2. Bindings: Mapping Relations to Actions (This is the 'Expert Knowledge')
        # Logic: (REL_ABOVE XOR ACTION_UP) creates a 'Concept Pair'
        self.rules = [
            self.codebook["REL_ABOVE"].xor(self.codebook["ACTION_UP"]),
            self.codebook["REL_BELOW"].xor(self.codebook["ACTION_DN"]),
            self.codebook["REL_LEFT"].xor(self.codebook["ACTION_LF"]),
            self.codebook["REL_RIGHT"].xor(self.codebook["ACTION_RT"])
        ]
        
        # 3. Consolidated Knowledge Base (Bundled Rules)
        self.knowledge_base = self.rules[0]
        for rule in self.rules[1:]:
            self.knowledge_base = self.knowledge_base.bundle(rule)

    def infer_navigation(self, is_above, is_below, is_left, is_right):
        # Logic: If ABOVE is true, we unbind REL_ABOVE from our Knowledge Base
        # to see what action it 'recommends'.
        
        active_rels = []
        if is_above: active_rels.append(self.codebook["REL_ABOVE"])
        if is_below: active_rels.append(self.codebook["REL_BELOW"])
        if is_left:  active_rels.append(self.codebook["REL_LEFT"])
        if is_right: active_rels.append(self.codebook["REL_RIGHT"])

        if not active_rels:
            return np.zeros(4, dtype=np.float32)

        # Query the Knowledge Base: For each active relation, what's the recommendation?
        # We collect all recommendations and bundle them.
        recommendations = []
        for rel in active_rels:
            # UNBIND: Rule XOR Rel = Action
            # (REL_A XOR ACTION_A) XOR REL_A = ACTION_A
            rec = self.knowledge_base.xor(rel)
            recommendations.append(rec)
            
        final_rec = recommendations[0]
        for r in recommendations[1:]:
            final_rec = final_rec.bundle(r)

        # Compare result against known Motor Actions
        results = [
            final_rec.similarity(self.codebook["ACTION_UP"]),
            final_rec.similarity(self.codebook["ACTION_DN"]),
            final_rec.similarity(self.codebook["ACTION_LF"]),
            final_rec.similarity(self.codebook["ACTION_RT"])
        ]
        return np.array(results, dtype=np.float32)

# Global engine instance
kernel_engine = ReasoningEngine()

class ActionSemantics:
    """
    Defines the 'Meaning' of actions in terms of abstract predicates.
    Now optimized to use Bitwise VSA reasoning.
    """
    
    @staticmethod
    def get_goal_alignment(game_type, state):
        if game_type == "snake":
            hx, hy = state["head"]
            fx, fy = state["food"]
            return kernel_engine.infer_navigation(fy < hy, fy > hy, fx < hx, fx > hx)
        
        elif game_type == "pong":
            paddle_center = state["p1_y"] + 3
            ball_y = state["ball_y"]
            # Map Pong to same bitwise navigation logic as Snake
            scores_2 = kernel_engine.infer_navigation(ball_y < paddle_center - 1, ball_y > paddle_center + 1, False, False)
            return scores_2[:2] # Only UP/DN
            
        return []

    @staticmethod
    def get_danger_alignment(game_type, state):
        # Expandable conceptual logic
        pass
