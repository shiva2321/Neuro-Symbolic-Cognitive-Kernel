import unittest
from brain_fusion import BrainFusion, TaskBrain, ConceptType

class TestForwardChaining(unittest.TestCase):
    def test_forward_chain(self):
        # 1. Setup Brain
        fusion = BrainFusion()
        brain = TaskBrain("logic_test")
        
        # 2. Add Rules: A -> B, B -> C
        brain.add_rule(frozenset(["FACT_A"]), "FACT_B", strength=1.0)
        brain.add_rule(frozenset(["FACT_B"]), "FACT_C", strength=1.0)
        
        # 3. Add Rules: X, Y -> Z
        brain.add_rule(frozenset(["FACT_X", "FACT_Y"]), "FACT_Z", strength=1.0)
        
        fusion.register_brain(brain)
        fused = fusion.fuse()
        
        # 4. Test Chain: A -> [A, B, C]
        facts, inferred = fused.forward_chain_multi({"FACT_A"}, max_steps=5)
        print(f"Chain A: {inferred}")
        self.assertIn("FACT_B", facts)
        self.assertIn("FACT_C", facts)
        
        # 5. Test Composite: X, Y -> Z
        facts2, inferred2 = fused.forward_chain_multi({"FACT_X", "FACT_Y"}, max_steps=5)
        print(f"Chain XY: {inferred2}")
        self.assertIn("FACT_Z", facts2)
        
    def test_action_exclusion(self):
        # Ensure ACTION_ rules are ignored by default in fact inference
        fusion = BrainFusion()
        brain = TaskBrain("action_test")
        brain.add_rule(frozenset(["DANGER"]), "ACTION_RUN", strength=1.0)
        fusion.register_brain(brain)
        fused = fusion.fuse()
        
        facts, _ = fused.forward_chain_multi({"DANGER"})
        self.assertNotIn("ACTION_RUN", facts)

if __name__ == '__main__':
    unittest.main()
