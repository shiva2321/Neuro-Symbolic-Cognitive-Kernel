from python.core.perception.symbol_grounding import ActionSemantics
import unittest
import numpy as np

class TestSymbolGrounding(unittest.TestCase):
    def test_snake_goal_up(self):
        # Food is above Head. Expect UP alignment.
        state = {"head": (5, 5), "food": (5, 2)}
        alignment = ActionSemantics.get_goal_alignment("snake", state)
        # Snake Actions: UP, DOWN, LEFT, RIGHT
        # Index 0 is UP.
        self.assertEqual(alignment[0], 1.0)
        self.assertEqual(alignment[1], 0.0)

    def test_snake_goal_right(self):
        # Food is Right of Head. Expect RIGHT alignment.
        state = {"head": (5, 5), "food": (8, 5)}
        alignment = ActionSemantics.get_goal_alignment("snake", state)
        self.assertEqual(alignment[3], 1.0)

    def test_pong_goal_up(self):
        # Ball (y=5) is above Paddle Center (p1_y=10 -> center=13). Expect UP.
        state = {"p1_y": 10, "ball_y": 5}
        alignment = ActionSemantics.get_goal_alignment("pong", state)
        # Pong Actions: UP, DOWN
        self.assertEqual(alignment[0], 1.0)
        self.assertEqual(alignment[1], 0.0)

    def test_pong_goal_stay(self):
        # Ball (y=13) is at Paddle Center (center=13). Expect Split/Neutral.
        state = {"p1_y": 10, "ball_y": 13}
        alignment = ActionSemantics.get_goal_alignment("pong", state)
        self.assertEqual(alignment[0], 0.5)
        self.assertEqual(alignment[1], 0.5)

if __name__ == '__main__':
    unittest.main()
