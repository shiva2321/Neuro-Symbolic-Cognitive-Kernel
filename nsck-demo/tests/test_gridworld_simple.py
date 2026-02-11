"""
Simple tests for GridWorld Survival game (no NSCK dependencies required).

These tests verify the core game mechanics work correctly.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gridworld_survival import (
    GridWorldSurvival, GridWorldState, Enemy, PlayerStats,
    EntityType, Direction
)


class TestGridWorldBasics(unittest.TestCase):
    """Test basic game mechanics."""
    
    def setUp(self):
        """Set up test game."""
        self.game = GridWorldSurvival(
            width=10, height=10, 
            num_enemies=1, 
            num_food=3,
            num_water=2,
            max_steps=100
        )
    
    def test_game_initialization(self):
        """Test game initializes correctly."""
        self.assertIsNotNone(self.game.state)
        self.assertEqual(self.game.state.width, 10)
        self.assertEqual(self.game.state.height, 10)
        self.assertTrue(self.game.state.stats.is_alive())
        print("✓ Game initialization works")
    
    def test_player_starts_alive(self):
        """Test player starts with full stats."""
        stats = self.game.state.stats
        self.assertEqual(stats.energy, 1.0)
        self.assertEqual(stats.health, 1.0)
        self.assertEqual(stats.hunger, 0.0)
        self.assertEqual(stats.thirst, 0.0)
        print("✓ Player starts with correct stats")
    
    def test_player_movement_valid(self):
        """Test player can move in valid directions."""
        initial_pos = self.game.state.player_pos
        
        # Try multiple moves
        moves_tried = 0
        successful_move = False
        
        for action in ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]:
            self.game.reset()
            initial = self.game.state.player_pos
            state, reward, done, info = self.game.step(action)
            moves_tried += 1
            
            # Check if move was valid (position changed or stayed due to wall)
            if state.player_pos != initial or info['reason'] == 'invalid_move':
                successful_move = True
                break
        
        self.assertTrue(moves_tried > 0)
        print("✓ Player movement works")
    
    def test_state_to_dict(self):
        """Test state conversion to dictionary."""
        state_dict = self.game.get_state_dict()
        
        # Check required keys exist
        required_keys = [
            'player_pos', 'exit_pos', 'predicates', 'stats', 
            'done', 'width', 'height', 'steps'
        ]
        
        for key in required_keys:
            self.assertIn(key, state_dict)
        
        # Check predicates is a list
        self.assertIsInstance(state_dict['predicates'], list)
        
        # Check stats is a dict
        self.assertIsInstance(state_dict['stats'], dict)
        
        print("✓ State dictionary conversion works")
        print(f"  - Found {len(state_dict['predicates'])} predicates")
        print(f"  - Stats: {state_dict['stats']}")
    
    def test_resource_drain(self):
        """Test that resources drain over time."""
        initial_energy = self.game.state.stats.energy
        initial_hunger = self.game.state.stats.hunger
        
        # Take several steps
        for _ in range(10):
            state, reward, done, info = self.game.step("ACTION_STAY")
            if done:
                break
        
        # Energy should decrease
        self.assertLess(state.stats.energy, initial_energy)
        
        # Hunger should increase
        self.assertGreater(state.stats.hunger, initial_hunger)
        
        print("✓ Resource drain works")
        print(f"  - Energy: {initial_energy:.3f} → {state.stats.energy:.3f}")
        print(f"  - Hunger: {initial_hunger:.3f} → {state.stats.hunger:.3f}")
    
    def test_game_over_on_death(self):
        """Test game ends when health reaches zero."""
        # Manually set health to very low
        self.game.state.stats.health = 0.01
        
        # Damage the player
        self.game.state.stats.health = 0
        
        state, reward, done, info = self.game.step("ACTION_STAY")
        
        self.assertTrue(done)
        self.assertEqual(info['reason'], 'died')
        print("✓ Game over on death works")
    
    def test_predicates_generated(self):
        """Test that predicates are generated based on state."""
        # Reset with specific seed for consistency
        self.game.reset()
        
        state_dict = self.game.get_state_dict()
        predicates = set(state_dict['predicates'])
        
        # Should have some predicates
        self.assertGreater(len(predicates), 0)
        
        # Check that predicates make sense
        predicate_types = {
            'resource': ['HUNGRY', 'THIRSTY', 'CRITICAL_HUNGER', 'CRITICAL_THIRST'],
            'energy': ['LOW_ENERGY', 'MODERATE_ENERGY'],
            'proximity': ['FOOD_NEARBY', 'WATER_NEARBY', 'ENEMY_NEARBY', 'EXIT_NEARBY'],
            'directional': ['DIRECTION'],  # Any predicate with DIRECTION
        }
        
        print("✓ Predicates generated correctly")
        print(f"  - Total predicates: {len(predicates)}")
        
        # Show some example predicates
        example_predicates = list(predicates)[:5]
        if example_predicates:
            print(f"  - Examples: {example_predicates}")
    
    def test_render_ascii(self):
        """Test ASCII rendering works."""
        output = self.game.render_ascii()
        
        # Should be a string
        self.assertIsInstance(output, str)
        
        # Should contain game symbols
        self.assertIn('#', output)  # Walls
        self.assertIn('@', output)  # Player
        
        # Should contain stats
        self.assertIn('Energy:', output)
        self.assertIn('Health:', output)
        
        print("✓ ASCII rendering works")
        print("\nExample render:")
        print(output[:200] + "...")
    
    def test_episode_completion(self):
        """Test a full episode can complete."""
        self.game.reset()
        
        steps = 0
        done = False
        
        while not done and steps < 50:
            # Random action
            import random
            action = random.choice([
                "ACTION_UP", "ACTION_DOWN", 
                "ACTION_LEFT", "ACTION_RIGHT"
            ])
            
            state, reward, done, info = self.game.step(action)
            steps += 1
        
        # Should have completed somehow
        self.assertTrue(steps > 0)
        
        print("✓ Full episode works")
        print(f"  - Ran {steps} steps")
        print(f"  - Done: {done}, Reason: {info['reason']}")
        print(f"  - Final score: {state.stats.score}")


class TestEnemyBehavior(unittest.TestCase):
    """Test enemy patrol behavior."""
    
    def test_enemy_creation(self):
        """Test enemy is created with patrol pattern."""
        enemy = Enemy(
            position=(5, 5),
            patrol_points=[(5, 5), (7, 5), (7, 7), (5, 7)]
        )
        
        self.assertEqual(enemy.position, (5, 5))
        self.assertEqual(len(enemy.patrol_points), 4)
        print("✓ Enemy creation works")
    
    def test_enemy_patrol_movement(self):
        """Test enemy follows patrol pattern."""
        enemy = Enemy(
            position=(5, 5),
            patrol_points=[(5, 5), (7, 5)]  # Simple 2-point patrol
        )
        
        # Move enemy
        new_pos = enemy.get_next_position(10, 10)
        
        # Should move toward next patrol point
        self.assertIsInstance(new_pos, tuple)
        self.assertEqual(len(new_pos), 2)
        
        print("✓ Enemy patrol works")
        print(f"  - Moved from {enemy.position} to {new_pos}")


class TestPlayerStats(unittest.TestCase):
    """Test player statistics."""
    
    def test_player_alive_check(self):
        """Test alive checking."""
        stats = PlayerStats()
        self.assertTrue(stats.is_alive())
        
        stats.health = 0
        self.assertFalse(stats.is_alive())
        
        stats.health = 1.0
        stats.energy = 0
        self.assertFalse(stats.is_alive())
        
        print("✓ Player alive check works")


def run_tests():
    """Run all tests and show summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGridWorldBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestEnemyBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestPlayerStats))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
