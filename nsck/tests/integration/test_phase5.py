"""
Tests for Phase 5: Generalization & Transfer
"""
import pytest
from python.core.reasoning.analogy import AnalogyEngine, ConceptMapping, Analogy


class TestAnalogy:
    """Test analogical reasoning and transfer."""
    
    def test_lift_and_ground(self):
        """Test lifting concepts to abstract and grounding back."""
        print("\n--- Test: Lift and Ground ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        # Lift Snake concept to abstract
        abstract = engine.lift_to_abstract("SNAKE_HEAD", "snake")
        assert abstract == "AGENT", f"SNAKE_HEAD should lift to AGENT, got {abstract}"
        
        # Ground abstract to Pong
        grounded = engine.ground_to_domain("AGENT", "pong")
        assert grounded == "PLAYER_PADDLE", f"AGENT should ground to PLAYER_PADDLE, got {grounded}"
        
        # Full round-trip: Snake -> Abstract -> Maze
        abstract_target = engine.lift_to_abstract("SNAKE_FOOD", "snake")
        maze_target = engine.ground_to_domain(abstract_target, "maze")
        assert maze_target == "MAZE_EXIT", f"SNAKE_FOOD should map to MAZE_EXIT, got {maze_target}"
        
        print(f"SNAKE_HEAD -> {abstract} -> {grounded}")
        print("Round-trip mapping successful!")
    
    def test_find_analogy(self):
        """Test building analogical mapping between domains."""
        print("\n--- Test: Find Analogy ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        analogy = engine.find_analogy("snake", "pong")
        
        assert analogy.source_domain == "snake"
        assert analogy.target_domain == "pong"
        assert len(analogy.mappings) >= 3, "Should have at least 3 mappings"
        assert analogy.overall_similarity > 0.5, "Domains should be reasonably similar"
        
        print(f"Found {len(analogy.mappings)} mappings:")
        for m in analogy.mappings:
            print(f"  {m.source_concept} ≈ {m.target_concept}")
        print(f"Overall similarity: {analogy.overall_similarity:.0%}")
    
    def test_transfer_rule(self):
        """Test transferring a rule between domains."""
        print("\n--- Test: Transfer Rule ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        # Snake rule: if food above, go up
        snake_rule_condition = {"REL_ABOVE"}
        snake_rule_action = "ACTION_UP"
        
        # Transfer to Pong
        pong_condition, pong_action = engine.transfer_rule(
            snake_rule_condition,
            snake_rule_action,
            "snake",
            "pong"
        )
        
        assert "BALL_ABOVE" in pong_condition, f"REL_ABOVE should transfer to BALL_ABOVE, got {pong_condition}"
        assert pong_action == "ACTION_UP", f"ACTION_UP should stay ACTION_UP, got {pong_action}"
        
        print(f"Rule transfer: {snake_rule_condition} → {snake_rule_action}")
        print(f"  Becomes: {pong_condition} → {pong_action}")
    
    def test_transfer_to_maze(self):
        """Test transferring Snake knowledge to a new Maze domain."""
        print("\n--- Test: Transfer to Maze (Zero-Shot) ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        # Snake rules learned from experience
        snake_rules = [
            ({"REL_ABOVE"}, "ACTION_UP"),
            ({"REL_BELOW"}, "ACTION_DOWN"),
        ]
        
        # Transfer to maze
        for condition, action in snake_rules:
            maze_condition, maze_action = engine.transfer_rule(
                condition, action, "snake", "maze"
            )
            print(f"  Snake: {condition} → {action}")
            print(f"  Maze:  {maze_condition} → {maze_action}")
        
        # Test zero-shot action
        maze_predicates = {"EXIT_ABOVE"}  # Exit is above player
        
        action = engine.zero_shot_action(
            state={},
            known_domain="snake",
            new_domain="maze",
            learned_rules=[({"REL_ABOVE"}, "ACTION_UP")],
            active_predicates=maze_predicates
        )
        
        assert action == "ACTION_UP", f"Should recommend UP when exit is above, got {action}"
        print(f"\nZero-shot: EXIT_ABOVE → {action} ✓")
    
    def test_get_explanation(self):
        """Test transfer explanation generation."""
        print("\n--- Test: Transfer Explanation ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        explanation = engine.get_transfer_explanation("snake", "maze")
        
        assert "snake" in explanation.lower()
        assert "maze" in explanation.lower()
        
        print(explanation)


class TestCrossTaskCapability:
    """End-to-end tests for cross-task learning."""
    
    def test_snake_to_pong_behavioral(self):
        """Test that Snake-trained agent can play Pong."""
        print("\n--- Test: Snake → Pong Behavioral ---")
        
        engine = AnalogyEngine(load_defaults=True)
        
        # Simulated Snake knowledge
        snake_rules = [
            ({"REL_ABOVE"}, "ACTION_UP"),
            ({"REL_BELOW"}, "ACTION_DOWN"),
            ({"DANGER_UP"}, "ACTION_DOWN"),  # Avoid danger
        ]
        
        # Pong scenarios
        scenarios = [
            ({"BALL_ABOVE"}, "ACTION_UP", "ball above paddle"),
            ({"BALL_BELOW"}, "ACTION_DOWN", "ball below paddle"),
        ]
        
        for pong_predicates, expected_action, description in scenarios:
            action = engine.zero_shot_action(
                state={},
                known_domain="snake",
                new_domain="pong",
                learned_rules=snake_rules,
                active_predicates=pong_predicates
            )
            
            assert action == expected_action, f"Failed for {description}: got {action}"
            print(f"  {description}: {action} ✓")
        
        print("Cross-task transfer successful!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
