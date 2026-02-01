from simulation import sim_snake, sim_pong
import unittest

class TestSimulation(unittest.TestCase):
    def test_snake_wraparound(self):
        # Head at (5, 0), moving UP -> should wrap to (5, 9)
        state = {"head": (5, 0), "body": [(5, 0), (5, 1)]}
        next_state, collision = sim_snake(state, "UP")
        self.assertEqual(next_state["head"], (5, 9))
        self.assertFalse(collision)

    def test_snake_collision(self):
        # Head at (5, 5), Body includes (5, 4). Moving UP to (5, 4) -> Collision
        state = {"head": (5, 5), "body": [(5, 5), (5, 4), (6, 4)]}
        next_state, collision = sim_snake(state, "UP")
        self.assertEqual(next_state["head"], (5, 4))
        self.assertTrue(collision)

    def test_pong_stay(self):
        state = {"p1_y": 10, "ball_y": 15, "ball_dy": 1, "ball_x": 15, "ball_dx": -1}
        next_state, miss = sim_pong(state, "STAY")
        self.assertEqual(next_state["p1_y"], 10) # Should not move

    def test_pong_miss(self):
        # Ball at x=2, dx=-1. Next x=1. Paddle Y range [10, 16]. Ball Y=5 (Miss)
        state = {"p1_y": 10, "ball_y": 5, "ball_dy": 0, "ball_x": 2, "ball_dx": -1}
        next_state, miss = sim_pong(state, "STAY")
        self.assertTrue(miss)
        
    def test_pong_hit(self):
        # Ball at x=2, dx=-1. Next x=1. Paddle Y range [10, 16]. Ball Y=12 (Hit)
        state = {"p1_y": 10, "ball_y": 12, "ball_dy": 0, "ball_x": 2, "ball_dx": -1}
        next_state, miss = sim_pong(state, "STAY")
        self.assertFalse(miss)

if __name__ == '__main__':
    unittest.main()
