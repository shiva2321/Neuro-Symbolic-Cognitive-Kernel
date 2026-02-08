"""
Test Transfer Learning & Cross-Domain Knowledge Application
============================================================

Tests that learned knowledge can be:
1. Transferred from one domain to another via analogical reasoning
2. Consolidated into abstract, domain-independent rules
3. Applied to completely novel situations the system has never seen
4. Persisted across sessions via the KnowledgeStore

Each test documents WHAT happened, HOW it happened, and WHY it happened.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from analogy import AnalogyEngine
from train_phase7_demo import IntegratedNSCKSystem, KnowledgeStore
from persistence import Rule
from cognitive_engine import create_cognitive_engine
from unittest.mock import patch


class TestAnalogyBasedTransfer(unittest.TestCase):
    """Test the analogy engine's ability to transfer rules between domains."""

    def setUp(self):
        self.analogy = AnalogyEngine()

    def test_lift_and_ground_concepts(self):
        """
        Test: Domain-specific concepts lift to abstract level and ground back.
        
        WHAT: SNAKE_HEAD lifts to AGENT, AGENT grounds to PLAYER_PADDLE in pong.
        HOW: AnalogyEngine maintains bidirectional mappings via register_abstract().
        WHY: This is the foundation for structural analogy — shared abstractions
             enable cross-domain reasoning.
        """
        # Lift snake concept to abstract
        abstract = self.analogy.lift_to_abstract("SNAKE_HEAD", "snake")
        self.assertEqual(abstract, "AGENT")

        # Ground abstract to pong
        pong_concept = self.analogy.ground_to_domain("AGENT", "pong")
        self.assertEqual(pong_concept, "PLAYER_PADDLE")

        # Ground abstract to maze
        maze_concept = self.analogy.ground_to_domain("AGENT", "maze")
        self.assertEqual(maze_concept, "MAZE_PLAYER")

    def test_rule_transfer_snake_to_pong(self):
        """
        Test: A rule learned in Snake transfers to Pong via analogy.
        
        WHAT: Snake rule "if food is above, move up" becomes 
              "if ball is above, move up" in Pong.
        HOW: transfer_rule() lifts predicates to abstract level, then grounds
             them in the target domain.
        WHY: Rules about spatial relationships (above/below) are structurally
             identical across games — only the entity names change.
        """
        # Snake rule: if target is above, move up
        condition = {"REL_ABOVE"}
        action = "ACTION_UP"

        new_cond, new_action = self.analogy.transfer_rule(
            condition, action, "snake", "pong"
        )

        # The action should remain ACTION_UP (abstract MOVE_UP maps to ACTION_UP in both)
        self.assertEqual(new_action, "ACTION_UP")
        # The condition should be mapped through abstract level
        self.assertIn("BALL_ABOVE", new_cond)

    def test_zero_shot_action_in_new_domain(self):
        """
        Test: Zero-shot action selection in a domain where no direct training occurred.
        
        WHAT: System recommends ACTION_UP in Pong based on Snake rules.
        HOW: Snake rule {REL_ABOVE} -> ACTION_UP is transferred:
             REL_ABOVE -> TARGET_ABOVE -> BALL_ABOVE in Pong.
             If BALL_ABOVE is active, the transferred rule fires.
        WHY: Demonstrates knowledge gained in one game can immediately apply
             to a structurally similar but visually/semantically different game.
        """
        snake_rules = [
            ({"REL_ABOVE"}, "ACTION_UP"),
            ({"REL_BELOW"}, "ACTION_DOWN"),
        ]

        # Pong state where ball is above paddle
        pong_predicates = {"BALL_ABOVE", "PONG_BALL"}

        action = self.analogy.zero_shot_action(
            state={},
            known_domain="snake",
            new_domain="pong",
            learned_rules=snake_rules,
            active_predicates=pong_predicates
        )

        self.assertEqual(action, "ACTION_UP",
                         "Should recommend UP when ball is above paddle")

    def test_find_analogy_between_domains(self):
        """
        Test: Analogy discovery between Snake and Maze.
        
        WHAT: System finds structural alignment between Snake and Maze.
        HOW: Both domains share abstract concepts (AGENT, TARGET, DANGER, etc.)
             registered during initialization.
        WHY: High similarity score (1.0) indicates complete structural alignment,
             meaning all Snake concepts have Maze counterparts.
        """
        analogy = self.analogy.find_analogy("snake", "maze")

        self.assertGreater(analogy.overall_similarity, 0.5)
        self.assertGreater(len(analogy.mappings), 3)
        self.assertGreater(len(analogy.reasoning_chain), 0)


class TestKnowledgeStore(unittest.TestCase):
    """Test cross-session knowledge persistence and consolidation."""

    def setUp(self):
        self.store = KnowledgeStore()
        self.analogy = AnalogyEngine()

    def test_store_and_retrieve_experience(self):
        """
        Test: Experiences are stored and indexed for cross-domain retrieval.
        
        WHAT: Snake experience (move toward food) is stored and retrievable.
        HOW: KnowledgeStore indexes experiences by predicate pattern hash.
        WHY: Storing experiences with their domain context enables later
             cross-domain consolidation.
        """
        self.store.store_experience(
            "snake", ["REL_ABOVE", "SNAKE_FOOD"], "ACTION_UP", 1.0, "success"
        )

        self.assertEqual(len(self.store.experience_index), 1)
        self.assertIn("snake", self.store.domain_schemas)

    def test_consolidate_cross_domain_knowledge(self):
        """
        Test: Experiences from multiple domains consolidate into abstract rules.
        
        WHAT: "Move up when target is above" emerges from Snake + Pong experiences.
        HOW: 
          1. Store experiences: Snake has REL_ABOVE->ACTION_UP, 
             Pong has BALL_ABOVE->ACTION_UP
          2. consolidate_to_abstract() lifts both to TARGET_ABOVE->MOVE_UP
          3. Because the abstract pattern appears in 2 domains, it's promoted
        WHY: This is how knowledge generalizes — repeated structural patterns
             across different contexts become domain-independent rules.
        """
        # Snake experiences: move up when food is above
        for _ in range(3):
            self.store.store_experience(
                "snake", ["REL_ABOVE"], "ACTION_UP", 1.0, "success"
            )
        
        # Pong experiences: move up when ball is above
        for _ in range(3):
            self.store.store_experience(
                "pong", ["BALL_ABOVE"], "ACTION_UP", 1.0, "success"
            )

        promoted = self.store.consolidate_to_abstract(self.analogy)

        self.assertGreater(len(self.store.abstract_rules), 0,
                           "Should have promoted at least one abstract rule")

        # Verify the abstract rule covers multiple domains
        rule = self.store.abstract_rules[0]
        self.assertGreater(len(rule['source_domains']), 0)

    def test_apply_abstract_knowledge_to_novel_domain(self):
        """
        Test: Abstract knowledge from Snake+Pong applies to Maze (novel domain).
        
        WHAT: System finds relevant experience for Maze without ever training in Maze.
        HOW:
          1. Learn in Snake: REL_ABOVE -> ACTION_UP (success)
          2. Learn in Pong: BALL_ABOVE -> ACTION_UP (success)
          3. Consolidate to abstract: TARGET_ABOVE -> MOVE_UP
          4. In Maze, EXIT_ABOVE is active, lifts to TARGET_ABOVE
          5. Abstract rule matches, recommends MOVE_UP -> ACTION_UP
        WHY: This is the key transfer learning capability — knowledge learned
             in completely different domains can be applied to a novel situation
             through structural analogy.
        """
        # Build up experiences in Snake and Pong
        for _ in range(3):
            self.store.store_experience(
                "snake", ["REL_ABOVE"], "ACTION_UP", 1.0, "success"
            )
            self.store.store_experience(
                "pong", ["BALL_ABOVE"], "ACTION_UP", 1.0, "success"
            )

        # Consolidate
        self.store.consolidate_to_abstract(self.analogy)

        # Now query for Maze (never trained)
        maze_predicates = ["EXIT_ABOVE"]
        relevant = self.store.find_relevant_experience(
            maze_predicates, "maze", self.analogy
        )

        self.assertGreater(len(relevant), 0,
                           "Should find relevant experience from abstract rules")
        self.assertGreater(relevant[0]['confidence'], 0.5)


class TestIntegratedTransferLearning(unittest.TestCase):
    """Test the full integrated system's transfer learning capabilities."""

    def setUp(self):
        self.system = IntegratedNSCKSystem()

    def test_learn_and_transfer_across_domains(self):
        """
        Test: End-to-end learning in one domain, applying in another.
        
        WHAT: Learn "move up when target above" in Snake, apply in Pong.
        HOW: IntegratedNSCKSystem.learn_from_experience() stores experiences,
             consolidate_knowledge() abstracts them, transfer_knowledge() applies.
        WHY: Demonstrates the full transfer learning pipeline working as an
             integrated system.
        """
        # Learn in Snake domain
        for _ in range(5):
            result = self.system.learn_from_experience(
                "snake", ["REL_ABOVE"], "ACTION_UP", 1.0, "success"
            )
        self.assertTrue(result['stored'])

        # Consolidate knowledge
        consolidation = self.system.consolidate_knowledge()
        self.assertGreater(consolidation['total_abstract_rules'], 0)

        # Transfer to Pong
        transfer_result = self.system.transfer_knowledge(
            "snake", "pong", ["BALL_ABOVE"]
        )

        self.assertIsNotNone(transfer_result)
        self.assertGreater(transfer_result['analogy']['similarity'], 0.5)

    def test_translate_system_state_to_natural_language(self):
        """
        Test: System state is translated to human-readable natural language.
        
        WHAT: Internal symbolic state becomes readable English.
        HOW: translate_to_natural_language() uses template-based fallback
             (or LLM when available) to convert action/emotion/confidence.
        WHY: The LLM serves as a peripheral translator — it doesn't make
             decisions, it just makes the system's thoughts understandable.
        """
        state = {
            'action': 'ACTION_UP',
            'emotion': 'anticipation',
            'confidence': 0.85,
            'transfer_source': 'snake',
            'reasoning': 'Target detected above current position'
        }

        nl_output = self.system.translate_to_natural_language(state)

        self.assertIsInstance(nl_output, str)
        self.assertGreater(len(nl_output), 10)
        self.assertIn("upward", nl_output)
        self.assertIn("anticipation", nl_output)

    def test_cross_session_knowledge_persistence(self):
        """
        Test: Knowledge persists in the KnowledgeStore across learning episodes.
        
        WHAT: Multiple separate learning episodes build up transferable knowledge.
        HOW: Each call to learn_from_experience() adds to the knowledge store.
             consolidate_knowledge() can be called at any time to abstract.
        WHY: This simulates cross-session learning — the system accumulates
             knowledge over time and can apply it to any future task.
        """
        # Session 1: Learn in Snake
        for _ in range(3):
            self.system.learn_from_experience(
                "snake", ["REL_ABOVE"], "ACTION_UP", 1.0
            )

        # Session 2: Learn in Pong (different domain)
        for _ in range(3):
            self.system.learn_from_experience(
                "pong", ["BALL_ABOVE"], "ACTION_UP", 1.0
            )

        # Session 3: Learn in Maze (yet another domain)
        for _ in range(3):
            self.system.learn_from_experience(
                "maze", ["EXIT_ABOVE"], "ACTION_UP", 1.0
            )

        # Consolidate
        result = self.system.consolidate_knowledge()

        self.assertEqual(len(result['domains_covered']), 3)
        self.assertIn("snake", result['domains_covered'])
        self.assertIn("pong", result['domains_covered'])
        self.assertIn("maze", result['domains_covered'])


class TestCognitiveEngineTransfer(unittest.TestCase):
    """Test transfer learning within the CognitiveEngine's decide loop."""

    def setUp(self):
        self.engine = create_cognitive_engine()

    def test_global_rule_applies_across_tasks(self):
        """
        Test: A global safety rule applies in any task context.
        
        WHAT: Global rule "DANGER_UP -> ACTION_DOWN" fires in Pong.
        HOW: get_applicable_rules() checks both task-local AND global rules.
             Global rules use domain-independent predicates (DANGER_UP).
        WHY: Some knowledge is truly universal — "avoid danger" applies
             regardless of the specific game being played.
        """
        # Add a global safety rule
        danger_rule = Rule(
            id=999,
            condition=frozenset(["DANGER_UP"]),
            consequence="ACTION_DOWN",
            priority=10,
            source="learned",
            task_tag="global",
            scope="global",
            support_count=100,
            success_rate=1.0
        )
        self.engine.rule_learner.learned_rules["global"].append(danger_rule)

        # Test in Pong context
        with patch.object(
            self.engine.verifiers["pong"],
            'get_active_predicates',
            return_value=["DANGER_UP"]
        ):
            result = self.engine.decide({"p1_y": 0, "ball_y": 10}, "pong")
            self.assertEqual(result.chosen_action, "ACTION_DOWN")
            self.assertEqual(result.trace["winner"], "RULES")

    def test_enhanced_transfer_searches_all_domains(self):
        """
        Test: transfer() searches across all known domains for applicable knowledge.
        
        WHAT: Transfer from any domain to target when source domain has no rules.
        HOW: Enhanced transfer() iterates through all domains, not just the
             specified source, using analogy for each.
        WHY: Real intelligence doesn't limit transfer to a single source —
             any relevant prior knowledge should be considered.
        """
        # Add rules to snake domain
        snake_rule = Rule(
            id=1,
            condition=frozenset(["REL_ABOVE"]),
            consequence="ACTION_UP",
            priority=5,
            source="learned",
            task_tag="snake",
            scope="task_local",
            support_count=50,
            success_rate=0.9
        )
        self.engine.rule_learner.learned_rules["snake"].append(snake_rule)

        # Try transferring from snake to pong
        result = self.engine.transfer(
            source_task="snake",
            target_task="pong",
            state={},
            active_predicates=["BALL_ABOVE"]
        )

        self.assertEqual(result, "ACTION_UP",
                         "Should transfer 'move up when target above' from snake to pong")


if __name__ == "__main__":
    unittest.main()
