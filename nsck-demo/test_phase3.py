"""
Tests for Phase 3: Semantic Understanding
"""
import pytest
from python.causal_reasoning import CausalGraph, CausalReasoner, create_snake_causal_graph
from python.explanation import ExplanationGenerator, ExplanationType
from python.semantic_coherence import SemanticCoherence


class TestCausalReasoning:
    """Test causal graph and reasoning."""
    
    def test_forward_chain(self):
        """Test forward chaining through causal graph."""
        print("\n--- Test: Forward Chaining ---")
        
        graph = create_snake_causal_graph()
        
        # Find what ACTION_UP causes
        chains = graph.forward_chain("ACTION_UP", max_depth=3, context="snake")
        
        assert len(chains) >= 1, "Should find at least one causal chain"
        
        # Should find HEAD_MOVES_UP
        effects = [c.end for c in chains]
        assert "HEAD_MOVES_UP" in effects, "Should find HEAD_MOVES_UP as effect"
        
        print(f"Found {len(chains)} chains from ACTION_UP")
        for chain in chains[:3]:
            print(f"  → {chain.end} (strength={chain.total_strength:.2f})")
    
    def test_backward_chain(self):
        """Test backward chaining to find causes."""
        print("\n--- Test: Backward Chaining ---")
        
        graph = create_snake_causal_graph()
        
        # Find what causes DEATH
        chains = graph.backward_chain("DEATH", max_depth=4, context="snake")
        
        assert len(chains) >= 1, "Should find causes of DEATH"
        
        causes = [c.start for c in chains]
        print(f"Found {len(chains)} chains leading to DEATH")
        for chain in chains[:3]:
            print(f"  {chain.start} → DEATH")
    
    def test_counterfactual(self):
        """Test counterfactual simulation."""
        print("\n--- Test: Counterfactual ---")
        
        graph = create_snake_causal_graph()
        
        # Create reasoner with mock simulator
        def mock_sim(state, action):
            if action == "UP" and state.get("danger_up"):
                return state, True  # Terminal
            return {"head": (5, 4)}, False
        
        reasoner = CausalReasoner(graph, simulator=mock_sim)
        
        state = {"head": (5, 5), "danger_up": True}
        result = reasoner.counterfactual(
            "ACTION_UP", "ACTION_DOWN",
            state, "snake"
        )
        
        assert result.query is not None
        print(f"Counterfactual: {result.query}")
        print(f"  Original: {result.original_outcome}")
        print(f"  Alternative: {result.counterfactual_outcome}")


class TestExplanation:
    """Test explanation generation."""
    
    def test_action_explanation(self):
        """Test explaining an action."""
        print("\n--- Test: Action Explanation ---")
        
        gen = ExplanationGenerator()
        
        state = {"head": (5, 5), "food": (5, 3)}  # Food above
        trace = {"confidence": 0.8, "veto": False}
        
        explanation = gen.explain_action("ACTION_UP", state, "snake", trace)
        
        assert explanation.type == ExplanationType.ACTION
        assert explanation.confidence == 0.8
        assert "up" in explanation.summary.lower()
        
        print(f"Summary: {explanation.summary}")
        print(f"Details:\n{explanation.details}")
    
    def test_rejection_explanation(self):
        """Test explaining why action was rejected."""
        print("\n--- Test: Rejection Explanation ---")
        
        gen = ExplanationGenerator()
        
        explanation = gen.explain_rejection(
            "ACTION_UP", "ACTION_DOWN",
            state={"head": (5, 0)},
            task_tag="snake",
            reason="collision risk with wall"
        )
        
        assert explanation.type == ExplanationType.REJECTION
        assert "collision" in explanation.summary.lower()
        
        print(f"Summary: {explanation.summary}")
    
    def test_state_explanation(self):
        """Test describing perceived state."""
        print("\n--- Test: State Explanation ---")
        
        gen = ExplanationGenerator()
        
        state = {"head": (5, 5), "food": (5, 3), "body": [(5, 6), (5, 7)]}
        predicates = ["REL_ABOVE", "DANGER_DOWN"]
        
        explanation = gen.explain_state(state, "snake", predicates)
        
        assert explanation.type == ExplanationType.STATE
        assert len(explanation.supporting_facts) > 0
        
        print(f"Summary: {explanation.summary}")


class TestSemanticCoherence:
    """Test semantic coherence checking."""
    
    def test_mutual_exclusion(self):
        """Test detection of mutually exclusive predicates."""
        print("\n--- Test: Mutual Exclusion ---")
        
        checker = SemanticCoherence()
        
        # These should conflict
        predicates = {"REL_ABOVE", "REL_BELOW"}
        result = checker.check_predicates(predicates)
        
        assert not result.is_coherent, "Should detect contradiction"
        assert len(result.contradictions) >= 1
        
        print(f"Coherent: {result.is_coherent}")
        for c in result.contradictions:
            print(f"  Contradiction: {c.description}")
    
    def test_state_consistency(self):
        """Test state-predicate consistency."""
        print("\n--- Test: State Consistency ---")
        
        checker = SemanticCoherence()
        
        # Claim food is above, but it's actually below
        state = {"head": (5, 5), "food": (5, 8)}  # food.y > head.y = below
        predicates = {"REL_ABOVE"}  # Wrong!
        
        result = checker.check_state_consistency(state, predicates, "snake")
        
        assert not result.is_coherent, "Should detect state mismatch"
        
        print(f"Coherent: {result.is_coherent}")
        for c in result.contradictions:
            print(f"  Issue: {c.description}")
    
    def test_rule_conflict(self):
        """Test rule conflict detection."""
        print("\n--- Test: Rule Conflict ---")
        
        checker = SemanticCoherence()
        
        rules = [
            ({"REL_ABOVE"}, "ACTION_UP"),
            ({"REL_ABOVE"}, "ACTION_DOWN"),  # Conflict!
        ]
        predicates = {"REL_ABOVE"}
        
        result = checker.check_rule_conflict(rules, predicates)
        
        assert not result.is_coherent, "Should detect rule conflict"
        
        print(f"Coherent: {result.is_coherent}")
        for c in result.contradictions:
            print(f"  Conflict: {c.description}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
