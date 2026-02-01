import unittest
from causal_reasoning import TheoryModule, CausalLink, CausalRelation

class TestTheoryFormation(unittest.TestCase):
    def setUp(self):
        self.theory_module = TheoryModule()
        
    def test_abstraction(self):
        """Verify that specific terms are correctly abstracted."""
        self.assertEqual(self.theory_module.abstract_term("ACTION_UP"), "MOVEMENT")
        self.assertEqual(self.theory_module.abstract_term("WALL_COLLISION"), "COLLIDER")
        self.assertEqual(self.theory_module.abstract_term("GAME_OVER"), "FAILURE")
        self.assertEqual(self.theory_module.abstract_term("REWARD_POS"), "SUCCESS")

    def test_theory_formation_logic(self):
        """Verify that multiple similar links form a general theory."""
        links = [
            CausalLink("ACTION_UP", "WALL_COLLISION", CausalRelation.CAUSES, context="snake"),
            CausalLink("ACTION_LEFT", "WALL_COLLISION", CausalRelation.CAUSES, context="snake"),
            CausalLink("ACTION_RIGHT", "BODY_COLLISION", CausalRelation.CAUSES, context="snake"),
            CausalLink("WALL_COLLISION", "DEATH", CausalRelation.CAUSES, context="snake"),
            CausalLink("BODY_COLLISION", "GAME_OVER", CausalRelation.CAUSES, context="snake")
        ]
        
        theories = self.theory_module.form_theories(links)
        
        # We expect:
        # MOVEMENT -> COLLIDER (from UP, LEFT, RIGHT)
        # COLLIDER -> FAILURE (from WALL, BODY)
        
        theory_templates = [t.template for t in self.theory_module.theories]
        self.assertIn("MOVEMENT leads to COLLIDER", theory_templates)
        self.assertIn("COLLIDER leads to FAILURE", theory_templates)
        
        # Verify examples are captured
        coll_theory = next(t for t in self.theory_module.theories if t.cause_type == "COLLIDER")
        self.assertGreaterEqual(len(coll_theory.examples), 2)

    def test_prediction_from_theory(self):
        """Verify using theory to predict in a vacuum."""
        # 1. Train theory
        links = [
            CausalLink("ACTION_UP", "WALL_COLLISION", CausalRelation.CAUSES),
            CausalLink("ACTION_DOWN", "WALL_COLLISION", CausalRelation.CAUSES)
        ]
        self.theory_module.form_theories(links)
        
        # 2. Predict for a NEW action that maps to MOVEMENT
        predictions = self.theory_module.predict_from_theory("ACTION_LEFT")
        self.assertIn("COLLIDER", predictions)

if __name__ == "__main__":
    unittest.main()
