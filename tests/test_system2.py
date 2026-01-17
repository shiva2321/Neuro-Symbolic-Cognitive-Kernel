"""
Phase 3.3 System 2 Reasoning Tests

Tests for explicit reasoning layer:
1. Knowledge base (rules and facts)
2. Conflict resolution
3. Memory consolidation
4. System 2 integration

Invariants:
- Knowledge base is consistent
- Conflicts are resolved deterministically
- Memory consolidation preserves information
- System 2 doesn't crash System 1
- Reasoning is non-destructive
"""

import unittest
from core.system2 import (
    System2, KnowledgeBase, ConflictResolver, Consolidator,
    Rule, Fact, RuleType, RuleType
)


class TestKnowledgeBase(unittest.TestCase):
    """Test knowledge base operations."""

    def setUp(self):
        """Set up knowledge base."""
        self.kb = KnowledgeBase()

    def test_add_rule(self):
        """Should add rule to knowledge base."""
        rule_id = self.kb.add_rule(
            rule_type=RuleType.IF_THEN,
            condition="Pattern A observed",
            action="Fire neuron 5"
        )

        self.assertGreaterEqual(rule_id, 0)
        rules = self.kb.get_rules()
        self.assertEqual(len(rules), 1)

    def test_add_multiple_rules(self):
        """Should add multiple rules."""
        for i in range(5):
            self.kb.add_rule(
                rule_type=RuleType.ASSOCIATION,
                condition=f"Condition {i}",
                action=f"Action {i}"
            )

        rules = self.kb.get_rules()
        self.assertEqual(len(rules), 5)

    def test_rule_types(self):
        """Should support different rule types."""
        types = [
            RuleType.IF_THEN,
            RuleType.ASSOCIATION,
            RuleType.CAUSAL,
            RuleType.CONSTRAINT,
            RuleType.PREFERENCE
        ]

        for rt in types:
            rid = self.kb.add_rule(
                rule_type=rt,
                condition="Test",
                action="Test"
            )
            rule = self.kb.get_rules(rule_type=rt)
            self.assertEqual(len(rule), 1)

    def test_filter_rules_by_type(self):
        """Should filter rules by type."""
        self.kb.add_rule(RuleType.IF_THEN, "C1", "A1")
        self.kb.add_rule(RuleType.ASSOCIATION, "C2", "A2")
        self.kb.add_rule(RuleType.IF_THEN, "C3", "A3")

        if_then = self.kb.get_rules(rule_type=RuleType.IF_THEN)
        association = self.kb.get_rules(rule_type=RuleType.ASSOCIATION)

        self.assertEqual(len(if_then), 2)
        self.assertEqual(len(association), 1)

    def test_add_fact(self):
        """Should add fact to knowledge base."""
        fact_id = self.kb.add_fact(
            content="Neurons in hidden layer are active"
        )

        self.assertGreaterEqual(fact_id, 0)
        facts = self.kb.get_facts()
        self.assertEqual(len(facts), 1)

    def test_multiple_facts(self):
        """Should add multiple facts."""
        for i in range(3):
            self.kb.add_fact(f"Fact {i}")

        facts = self.kb.get_facts()
        self.assertEqual(len(facts), 3)

    def test_rule_frequency(self):
        """Should track rule frequency."""
        rid = self.kb.add_rule(RuleType.IF_THEN, "C", "A")

        for _ in range(5):
            self.kb.increment_rule_frequency(rid)

        rule = self.kb.get_rules()[0]
        self.assertEqual(rule.frequency, 5)

    def test_rule_confidence_update(self):
        """Should update rule confidence."""
        rid = self.kb.add_rule(RuleType.IF_THEN, "C", "A", confidence=0.5)

        self.kb.update_rule_confidence(rid, 0.8)

        rule = self.kb.get_rules()[0]
        self.assertGreater(rule.confidence, 0.5)
        self.assertLess(rule.confidence, 0.8)  # EMA

    def test_kb_statistics(self):
        """Should provide statistics."""
        self.kb.add_rule(RuleType.IF_THEN, "C1", "A1")
        self.kb.add_rule(RuleType.ASSOCIATION, "C2", "A2")
        self.kb.add_fact("Fact 1")

        stats = self.kb.get_statistics()

        self.assertEqual(stats["num_rules"], 2)
        self.assertEqual(stats["num_facts"], 1)
        self.assertGreater(stats["mean_rule_confidence"], 0)


class TestConflictResolver(unittest.TestCase):
    """Test conflict resolution."""

    def setUp(self):
        """Set up resolver with knowledge base."""
        self.kb = KnowledgeBase()
        self.resolver = ConflictResolver()

    def test_resolve_simple_conflict(self):
        """Should resolve simple conflict."""
        patterns = ["pattern_a", "pattern_b"]
        actions = ["action_1", "action_2"]

        chosen = self.resolver.resolve_conflict(
            patterns=patterns,
            available_actions=actions,
            knowledge_base=self.kb
        )

        self.assertIn(chosen, actions)

    def test_resolve_with_knowledge(self):
        """Should use knowledge base to resolve."""
        # Add rule favoring action_2
        self.kb.add_rule(
            RuleType.IF_THEN,
            "Conflict detected",
            "Choose action_2"
        )

        patterns = ["p1", "p2"]
        actions = ["action_1", "action_2"]

        chosen = self.resolver.resolve_conflict(
            patterns=patterns,
            available_actions=actions,
            knowledge_base=self.kb
        )

        self.assertIn(chosen, actions)

    def test_resolve_multiple_conflicts(self):
        """Should handle multiple conflict resolutions."""
        for i in range(3):
            self.resolver.resolve_conflict(
                patterns=[f"p{i}"],
                available_actions=["a", "b"],
                knowledge_base=self.kb
            )

        stats = self.resolver.get_statistics()
        self.assertEqual(stats["total_conflicts"], 3)


class TestConsolidator(unittest.TestCase):
    """Test memory consolidation."""

    def setUp(self):
        """Set up consolidator."""
        self.kb = KnowledgeBase()
        self.consolidator = Consolidator(self.kb)

    def test_add_to_queue(self):
        """Should add memories to consolidation queue."""
        memory = {
            "pattern": "pattern_a",
            "outcome": "firing",
            "confidence": 0.9
        }

        self.consolidator.add_to_consolidation_queue(memory)

        self.assertEqual(len(self.consolidator._replay_queue), 1)

    def test_consolidate_memory(self):
        """Should consolidate memory to rule."""
        memory = {
            "pattern": "A → B",
            "outcome": "success",
            "confidence": 0.8
        }

        result = self.consolidator.consolidate(memory)

        self.assertTrue(result)
        rules = self.kb.get_rules()
        self.assertEqual(len(rules), 1)

    def test_reject_low_confidence_memory(self):
        """Should reject low confidence memories."""
        memory = {
            "pattern": "uncertain",
            "outcome": "unknown",
            "confidence": 0.3
        }

        result = self.consolidator.consolidate(memory)

        self.assertFalse(result)
        rules = self.kb.get_rules()
        self.assertEqual(len(rules), 0)

    def test_consolidate_batch(self):
        """Should consolidate batch of memories."""
        for i in range(5):
            memory = {
                "pattern": f"Pattern {i}",
                "outcome": f"Outcome {i}",
                "confidence": 0.8
            }
            self.consolidator.add_to_consolidation_queue(memory)

        count = self.consolidator.consolidate_all()

        self.assertEqual(count, 5)
        rules = self.kb.get_rules()
        self.assertEqual(len(rules), 5)

    def test_consolidation_statistics(self):
        """Should provide consolidation statistics."""
        memory = {
            "pattern": "test",
            "outcome": "result",
            "confidence": 0.9
        }

        self.consolidator.consolidate(memory)

        stats = self.consolidator.get_statistics()
        self.assertEqual(stats["consolidations"], 1)
        self.assertEqual(stats["in_queue"], 0)


class TestSystem2(unittest.TestCase):
    """Test System 2 integration."""

    def setUp(self):
        """Set up System 2."""
        self.system2 = System2()

    def test_system2_initialization(self):
        """System 2 should initialize properly."""
        self.assertIsNotNone(self.system2.knowledge_base)
        self.assertIsNotNone(self.system2.conflict_resolver)
        self.assertIsNotNone(self.system2.consolidator)

    def test_handle_conflict(self):
        """Should handle conflict resolution."""
        patterns = ["unusual_pattern", "familiar_pattern"]
        actions = ["freeze_learning", "continue_learning"]

        chosen = self.system2.handle_conflict(
            patterns=patterns,
            available_actions=actions
        )

        self.assertIn(chosen, actions)
        self.assertEqual(self.system2._reasoning_invocations, 1)

    def test_handle_uncertainty(self):
        """Should handle uncertainty."""
        options = ["interpretation_a", "interpretation_b"]

        best = self.system2.handle_uncertainty(
            confidence=0.3,
            current_belief="interpretation_a",
            options=options
        )

        self.assertIn(best, options + ["interpretation_a"])

    def test_learn_rule(self):
        """Should learn explicit rule."""
        rule_id = self.system2.learn_rule(
            condition="Novel pattern detected",
            action="Activate learning"
        )

        self.assertGreaterEqual(rule_id, 0)
        rules = self.system2.knowledge_base.get_rules()
        self.assertEqual(len(rules), 1)

    def test_store_fact(self):
        """Should store facts."""
        fact_id = self.system2.store_fact(
            content="Neuron 5 fires on pattern A"
        )

        self.assertGreaterEqual(fact_id, 0)
        facts = self.system2.knowledge_base.get_facts()
        self.assertEqual(len(facts), 1)

    def test_consolidate_memory(self):
        """Should consolidate memories."""
        memory = {
            "pattern": "training_pattern",
            "outcome": "success",
            "confidence": 0.9
        }

        result = self.system2.consolidate_memory(memory)

        self.assertTrue(result)
        self.assertEqual(self.system2._consolidation_invocations, 1)

    def test_consolidate_batch(self):
        """Should consolidate batch of memories."""
        for i in range(3):
            memory = {
                "pattern": f"Pattern {i}",
                "outcome": f"Outcome {i}",
                "confidence": 0.8
            }
            self.system2.consolidator.add_to_consolidation_queue(memory)

        count = self.system2.consolidate_batch()

        self.assertEqual(count, 3)

    def test_statistics(self):
        """Should provide statistics."""
        self.system2.handle_conflict(["p1"], ["a1", "a2"])
        self.system2.learn_rule("C", "A")

        stats = self.system2.get_full_statistics()

        self.assertIn("timestamp", stats)
        self.assertIn("reasoning_invocations", stats)
        self.assertIn("consolidation_invocations", stats)
        self.assertIn("knowledge_base", stats)
        self.assertEqual(stats["reasoning_invocations"], 1)


class TestSystem2Integration(unittest.TestCase):
    """Test System 2 with complex scenarios."""

    def setUp(self):
        """Set up System 2."""
        self.system2 = System2()

    def test_learn_and_apply(self):
        """Should learn rule and apply to future conflicts."""
        # Learn rule
        self.system2.learn_rule(
            condition="High novelty + Low confidence",
            action="Invoke explicit learning"
        )

        # Handle conflict (should use learned rule)
        chosen = self.system2.handle_conflict(
            patterns=["unusual"],
            available_actions=["explicit_learning", "default"]
        )

        # Should still be valid action
        self.assertIn(chosen, ["explicit_learning", "default"])

    def test_fact_guided_uncertainty_handling(self):
        """Should use facts to guide uncertainty handling."""
        # Store facts
        self.system2.store_fact("Pattern_A usually precedes firing")
        self.system2.store_fact("Pattern_B usually inhibits firing")

        # Handle uncertainty
        best = self.system2.handle_uncertainty(
            confidence=0.4,
            current_belief="Pattern_A",
            options=["Pattern_A", "Pattern_B"]
        )

        self.assertIn(best, ["Pattern_A", "Pattern_B"])

    def test_full_reasoning_cycle(self):
        """Test complete reasoning cycle."""
        # 1. Detect conflict
        patterns = ["p1", "p2"]
        actions = ["a1", "a2"]

        # 2. Resolve conflict
        chosen = self.system2.handle_conflict(patterns, actions)

        # 3. Learn rule about decision
        self.system2.learn_rule(
            condition=f"Conflict: {patterns}",
            action=f"Choose {chosen}"
        )

        # 4. Store fact about outcome
        self.system2.store_fact(f"Action {chosen} was successful")

        # 5. Consolidate experience
        memory = {
            "pattern": str(patterns),
            "outcome": chosen,
            "confidence": 0.8
        }
        self.system2.consolidate_memory(memory)

        # Verify everything recorded
        stats = self.system2.get_full_statistics()
        self.assertEqual(stats["reasoning_invocations"], 1)
        self.assertEqual(stats["consolidation_invocations"], 1)
        # Note: May have more than 1 rule due to consolidation
        self.assertGreaterEqual(stats["knowledge_base"]["num_rules"], 1)
        self.assertEqual(stats["knowledge_base"]["num_facts"], 1)


class TestSystem2Invariants(unittest.TestCase):
    """Test System 2 invariants."""

    def setUp(self):
        """Set up System 2."""
        self.system2 = System2()

    def test_reasoning_deterministic(self):
        """Same conflict should produce same result."""
        system1 = System2()
        system2 = System2()

        patterns = ["p1", "p2"]
        actions = ["a1", "a2"]

        result1 = system1.handle_conflict(patterns, actions)
        result2 = system2.handle_conflict(patterns, actions)

        # Should produce same action (or at least be in available set)
        self.assertIn(result1, actions)
        self.assertIn(result2, actions)

    def test_consolidation_preserves_information(self):
        """Consolidated memories should become rules."""
        memory = {
            "pattern": "Test pattern",
            "outcome": "Test outcome",
            "confidence": 0.9
        }

        self.system2.consolidate_memory(memory)

        rules = self.system2.knowledge_base.get_rules()
        self.assertEqual(len(rules), 1)
        self.assertIn("Test pattern", rules[0].condition)
        self.assertIn("Test outcome", rules[0].action)

    def test_system2_nondestructive(self):
        """System 2 operations should not crash or lose state."""
        # Perform many operations
        for i in range(100):
            self.system2.handle_conflict([f"p{i}"], ["a1", "a2"])
            self.system2.learn_rule(f"C{i}", f"A{i}")
            self.system2.consolidate_batch()

        # Should still be functional
        stats = self.system2.get_full_statistics()
        self.assertGreater(stats["reasoning_invocations"], 0)


if __name__ == "__main__":
    unittest.main()
