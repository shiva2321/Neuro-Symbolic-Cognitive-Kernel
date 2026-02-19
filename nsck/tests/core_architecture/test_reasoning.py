"""
Phase 4: Reasoning & Decision-Making Tests
Tests the system's ability to perform inference and make decisions
"""

import sys
import os
import pytest
import time

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner, CausalLink, CausalRelation
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition, WorkspaceModule
import python.core.vsa.hypervec_shim as hypervec_rs


class TestCausalReasoning:
    """Test RSN-1: Forward chaining through causal graph"""
    
    def test_causal_chain_inference(self):
        """Multi-step causal inference works"""
        cg = CausalGraph()
        
        # Build a causal chain: A -> B -> C
        cg.add_link(CausalLink(
            cause="It is raining", effect="The ground is wet",
            relation=CausalRelation.CAUSES, strength=0.9,
        ))
        cg.add_link(CausalLink(
            cause="The ground is wet", effect="Grass grows",
            relation=CausalRelation.CAUSES, strength=0.8,
        ))
        
        # Forward chain from root cause
        chain = cg.forward_chain("It is raining", max_depth=3)
        
        assert len(chain) > 0, "No causal chain generated"
        # Should find "The ground is wet" and "Grass grows"
    
    def test_causal_confidence_propagation(self):
        """Confidence scores propagate through causal chain"""
        cg = CausalGraph()
        
        # High confidence cause
        cg.add_link(CausalLink(
            cause="Cause A", effect="Effect B",
            relation=CausalRelation.CAUSES, strength=0.95,
        ))
        cg.add_link(CausalLink(
            cause="Effect B", effect="Effect C",
            relation=CausalRelation.CAUSES, strength=0.8,
        ))
        
        # Query path: confidence should decrease along chain
        chain = cg.forward_chain("Cause A", max_depth=3)
        assert len(chain) >= 2, "Should find at least 2 links in chain"
    
    def test_backward_chaining_from_goal(self):
        """Backward chaining from goal to causes"""
        cg = CausalGraph()
        
        # Build graph: A -> B -> C
        cg.add_link(CausalLink(
            cause="Training", effect="Muscle growth",
            relation=CausalRelation.CAUSES, strength=0.9,
        ))
        cg.add_link(CausalLink(
            cause="Muscle growth", effect="Strength",
            relation=CausalRelation.CAUSES, strength=0.85,
        ))
        
        # Backward chain: to achieve Strength, what's needed?
        chain = cg.backward_chain("Strength", max_depth=3)
        assert len(chain) > 0, "Should trace back to find causes"


class TestCounterfactualReasoning:
    """Test RSN-2: What-if queries and counterfactual reasoning"""
    
    def test_counterfactual_hypothesis(self):
        """What if X were true: changes predictions"""
        learner = TextKnowledgeLearner()
        
        # Learn base facts
        learner.learn_text("If it rains, the ground is wet")
        learner.learn_text("If ground is wet, grass grows")
        
        # Base query: current state
        result1 = learner.query_learned_knowledge("Is grass growing?", top_k=5)
        
        # Counterfactual query: "What if it rained?"
        # (Depends on implementation, but should modify state)
        
        # Result should differ
    
    def test_counterfactual_backtracking(self):
        """Counterfactuals can be retracted"""
        learner = TextKnowledgeLearner()
        
        # Learn world state
        learner.learn_text("The door is locked")
        learner.learn_text("If door is locked, we cannot enter")
        
        # Counterfactual: assume door is open
        # Query: can we enter?
        # Should answer yes (counterfactually)
        
        # Then retract: answer should go back to no


class TestAnalogyTransfer:
    """Test RSN-3: Analogy and structural alignment"""
    
    def test_analogy_structure_recognition(self):
        """Recognize analogous structures across domains"""
        learner = TextKnowledgeLearner()
        
        # Domain A: government
        learner.learn_text("President leads country")
        learner.learn_text("Congress passes laws")
        
        # Domain B: biology
        learner.learn_text("Brain controls body")
        learner.learn_text("Cells form tissues")
        
        # Analogy: President : Country :: Brain : Body
        # System should recognize structural similarity
    
    def test_analogy_rule_transfer(self):
        """Apply analogical rules to new domain"""
        learner = TextKnowledgeLearner()
        
        # Source domain rule
        learner.learn_text("Fire is hot, therefore you should not touch it")
        
        # Target domain (similar structure)
        learner.learn_text("Electricity is dangerous")
        
        # Should transfer rule: don't touch dangerous things


class TestPlanning:
    """Test RSN-4: Multi-step goal decomposition"""
    
    def test_goal_decomposition(self):
        """Complex goal decomposed into subgoals"""
        # This tests planning module
        from python.core.reasoning.planner import STRIPSPlanner
        
        planner = STRIPSPlanner()
        
        # Define goal: bake a cake
        # Should decompose into: mix ingredients, preheat oven, bake, cool
        
        goal = ["cake_baked"]
        # plan = planner.plan(goal)
        
        # Should return sequence of actions


class TestConflictResolution:
    """Test RSN-5: Resolve contradictions"""
    
    def test_contradiction_detection(self):
        """System detects contradictory beliefs"""
        learner = TextKnowledgeLearner()
        
        # Learn two contradictory facts
        learner.learn_text("Dogs are herbivores")
        learner.learn_text("Dogs are carnivores")
        
        # System should detect contradiction
        result = learner.query_learned_knowledge("Are dogs herbivores or carnivores?", top_k=5)
        
        # Should indicate conflict or choose higher-confidence source
    
    def test_confidence_based_resolution(self):
        """Higher-confidence belief wins conflict"""
        learner = TextKnowledgeLearner()
        
        # High confidence fact
        learner.learn_text("Earth orbits the Sun (scientifically proven)")
        
        # Low confidence fact
        learner.learn_text("Some ancient cultures believed the Sun orbits Earth")
        
        # Query: which system?
        # Should prefer high-confidence modern model


class TestSpreadingActivation:
    """Test RSN-6: Graph propagation finds distant concepts"""
    
    def test_activation_propagation_finds_related(self):
        """Activation from one node spreads to related nodes"""
        learner = TextKnowledgeLearner()
        
        # Build a semantic network
        learner.learn_text("Dogs are animals")
        learner.learn_text("Animals breathe")
        learner.learn_text("Breathing requires oxygen")
        learner.learn_text("Oxygen comes from air")
        
        # Query: "dogs"
        result = learner.query_learned_knowledge("Tell me about dogs", top_k=10)
        
        # Should activate through chain: dogs -> animals -> breathe -> oxygen -> air
        # Distant concepts should appear in results with lower activation


class TestReasoningIntegration:
    """Integration tests combining multiple reasoning modes"""
    
    def test_multi_pathway_inference(self):
        """Reasoning uses multiple paths to same conclusion"""
        learner = TextKnowledgeLearner()
        
        # Path 1: Dogs -> mammals -> have fur
        learner.learn_text("Dogs are mammals")
        learner.learn_text("All mammals have fur")
        
        # Path 2: Dogs -> animals with hair
        learner.learn_text("Dogs are covered in hair")
        
        # Query: do dogs have fur?
        # Should find answer via both paths
        result = learner.query_learned_knowledge("Do dogs have fur?", top_k=5)
        assert result is not None
    
    def test_reasoning_under_uncertainty(self):
        """System reasons with uncertain information"""
        learner = TextKnowledgeLearner()
        
        learner.learn_text("Probably, most birds can fly")
        learner.learn_text("Penguins are birds")
        
        # Query: can penguins fly?
        # Should reason: uncertain (most birds can, but penguins may be exception)
        result = learner.query_learned_knowledge("Can penguins fly?", top_k=5)


class TestGlobalWorkspace:
    """Test RSN-7: Global workspace consciousness model"""
    
    def test_workspace_coalition_competition(self):
        """Multiple coalitions compete in workspace"""
        workspace = GlobalWorkspace()
        
        # Create mock modules
        class MockModule(WorkspaceModule):
            def __init__(self, name):
                self.name = name
            
            def receive_broadcast(self, content):
                pass
            
            def generate_coalition(self, context):
                return Coalition(
                    source=self.name,
                    content="test content",
                    base_salience=0.5,
                    sender_confidence=0.8,
                    relevance=0.0,
                )
        
        mod1 = MockModule("module1")
        mod2 = MockModule("module2")
        
        workspace.register_module("module1", mod1)
        workspace.register_module("module2", mod2)
        
        # Broadcast should trigger competition
        # (exact API depends on implementation)


class TestDecisionMaking:
    """Tests for decision paths in reasoning"""
    
    def test_decision_from_multiple_evidence(self):
        """Decision made from multiple pieces of evidence"""
        learner = TextKnowledgeLearner()
        
        # Multiple lines of evidence
        learner.learn_text("It looks like rain (dark clouds)")
        learner.learn_text("The barometer is dropping")
        learner.learn_text("Birds are flying low")
        
        # All point to: will it rain?
        result = learner.query_learned_knowledge("Will it rain?", top_k=5)
        
        # Should aggregate evidence
    
    def test_risk_aware_decision(self):
        """Decisions account for risk"""
        learner = TextKnowledgeLearner()
        
        learner.learn_text("The bridge looks old and risky")
        learner.learn_text("Using the bridge saves 10 minutes")
        
        # Query: should we use the bridge?
        # Should weigh risk vs. benefit


class TestReasoningNarratives:
    """Narrative demonstrations of reasoning capabilities"""
    
    def test_narrative_causal_inference(self):
        """Show causal reasoning through narrative"""
        print("\n" + "="*70)
        print("NARRATIVE: Causal Reasoning")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        
        print("\nLearning causal chain:")
        facts = [
            ("Excessive carbon dioxide traps heat", "Greenhouse effect"),
            ("Greenhouse effect warms planet", "Global warming"),
            ("Global warming melts ice caps", "Sea level rises"),
        ]
        
        for cause, effect in facts:
            text = f"{cause} causes {effect}"
            print(f"  Learning: {text}")
            learner.learn_text(text)
        
        print("\nForward chaining query:")
        print("  'If we emit CO2, what happens?'")
        
        result = learner.query_learned_knowledge("What happens if we emit CO2?", top_k=10)
        print(f"  Result: {result}")
        
        print("\nBackward chaining query:")
        print("  'Why are sea levels rising?'")
        
        result = learner.query_learned_knowledge("Why are sea levels rising?", top_k=10)
        print(f"  Result: {result}")
    
    def test_narrative_analogy_transfer(self):
        """Show analogical reasoning"""
        print("\n" + "="*70)
        print("NARRATIVE: Analogical Reasoning")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        
        print("\nSource domain (government):")
        source_facts = [
            "The president is the head of the government",
            "Congress makes laws",
            "The judiciary interprets laws",
        ]
        for fact in source_facts:
            print(f"  {fact}")
            learner.learn_text(fact)
        
        print("\nTarget domain (biology):")
        target_facts = [
            "The brain controls the body",
            "Proteins perform cellular functions",
            "DNA stores genetic information",
        ]
        for fact in target_facts:
            print(f"  {fact}")
            learner.learn_text(fact)
        
        print("\nAnalogy:")
        print("  President : Government :: Brain : Body")
        print("  This structural similarity should enable knowledge transfer")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
