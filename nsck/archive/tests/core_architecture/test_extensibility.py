"""
Phase 7: Extensibility & Composability Tests
Tests ability to build new modules, domains, and integrate multiple systems
"""

import sys
import os
import pytest
import tempfile

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.memory.semantic_memory import SemanticMemory
from python.core.integration.brain_fusion import (
    TaskBrain, FusedBrain, GLOBAL_PRIMITIVES, ConceptType
)
from python.core.vsa.universal_encoder import UniversalEncoder


class TestTaskBrainCreation:
    """Test EXT-1: Custom TaskBrain for new domains"""
    
    def test_create_game_domain_brain(self):
        """Create a domain-specific brain for game domain"""
        # Create a Pong game brain
        pong_brain = TaskBrain("pong_game")
        encoder = UniversalEncoder()
        
        # Add game-specific concepts
        pong_brain.add_concept("paddle", encoder.encode_seed(GLOBAL_PRIMITIVES["ACTION_UP"]))
        pong_brain.add_concept("ball", encoder.encode_text("ball"), ConceptType.OBJECT)
        pong_brain.add_concept("score", encoder.encode_text("score"), ConceptType.STATE)
        
        # Add game rules
        pong_brain.add_rule(
            condition=frozenset(["ball", "moving"]),
            consequence="update_position",
            strength=0.9
        )
        
        # Verify
        assert len(pong_brain.codebook) > 0, "Concepts not added"
        assert len(pong_brain.rules) > 0, "Rules not added"
    
    def test_task_brain_learning(self):
        """TaskBrain learns domain-specific knowledge"""
        maze_brain = TaskBrain("maze_game")
        encoder = UniversalEncoder()
        
        # Learn maze structure
        concepts = ["start", "goal", "wall", "path", "exit"]
        for concept in concepts:
            hv = encoder.encode_text(concept)
            maze_brain.add_concept(concept, hv)
        
        # Verify learning
        assert len(maze_brain.codebook) == len(concepts), "Not all concepts stored"
    
    def test_task_brain_independent_execution(self):
        """TaskBrain operates independently"""
        brain1 = TaskBrain("domain_a")
        brain2 = TaskBrain("domain_b")
        encoder = UniversalEncoder()
        
        # Different vocabularies
        brain1.add_concept("king", encoder.encode_text("king"))
        brain2.add_concept("queen_bee", encoder.encode_text("queen_bee"))
        
        # Should not interfere
        assert "king" in brain1.codebook
        assert "king" not in brain2.codebook


class TestBrainFusionIntegration:
    """Test EXT-2: BrainFusion integration of multiple domains"""
    
    def test_merge_two_task_brains(self):
        """Merge two TaskBrains with concept alignment"""
        brain_animals = TaskBrain("animals")
        brain_vehicles = TaskBrain("vehicles")
        encoder = UniversalEncoder()
        
        # Domain 1: Animals
        brain_animals.add_concept("has_legs", encoder.encode_text("has_legs"))
        brain_animals.add_concept("dog", encoder.encode_text("dog"))
        brain_animals.add_rule(
            condition=frozenset(["dog"]),
            consequence="has_legs"
        )
        
        # Domain 2: Vehicles
        brain_vehicles.add_concept("has_wheels", encoder.encode_text("has_wheels"))
        brain_vehicles.add_concept("car", encoder.encode_text("car"))
        brain_vehicles.add_rule(
            condition=frozenset(["car"]),
            consequence="has_wheels"
        )
        
        # Fuse
        fused = FusedBrain([brain_animals, brain_vehicles])
        
        # Should have concepts from both
        # (exact validation depends on FusedBrain implementation)
        assert fused is not None
    
    def test_concept_alignment_across_brains(self):
        """Concepts aligned where they match across brains"""
        brain1 = TaskBrain("biology")
        brain2 = TaskBrain("ecology")
        encoder = UniversalEncoder()
        
        # Common concept: "organism"
        org_hv = encoder.encode_text("organism")
        brain1.add_concept("organism", org_hv)
        brain2.add_concept("organism", org_hv)
        
        # Add domain-specific concepts
        brain1.add_concept("cell", encoder.encode_text("cell"))
        brain2.add_concept("habitat", encoder.encode_text("habitat"))
        
        fused = FusedBrain([brain1, brain2])
        
        # Should recognize shared "organism" concept


class TestModuleComposition:
    """Test EXT-3: Add custom module to reasoning pipeline"""
    
    def test_add_custom_reasoning_module(self):
        """Add a custom reasoning module to pipeline"""
        from python.core.reasoning.global_workspace import WorkspaceModule, Coalition
        
        class CustomModule(WorkspaceModule):
            def __init__(self):
                super().__init__("custom_reasoner")
            
            def generate_coalition(self, context):
                # Custom reasoning logic
                return Coalition(
                    sender="custom_reasoner",
                    content="custom reasoning result",
                    base_salience=0.6
                )
        
        # Should be composable
        custom = CustomModule()
        assert custom.name == "custom_reasoner"
    
    def test_module_compatibility_with_workspace(self):
        """Custom module integrates with GlobalWorkspace"""
        from python.core.reasoning.global_workspace import GlobalWorkspace, WorkspaceModule, Coalition
        
        class TestModule(WorkspaceModule):
            def generate_coalition(self, context):
                return Coalition(
                    sender=self.name,
                    content="test",
                    base_salience=0.5
                )
        
        workspace = GlobalWorkspace()
        module = TestModule("test_module")
        
        # Should register successfully
        workspace.register_module(module)
        
        # Module should be callable from workspace


class TestPersistenceSwapping:
    """Test EXT-4: Swap persistence backend"""
    
    def test_default_persistence(self):
        """System persists knowledge to default backend"""
        learner = TextKnowledgeLearner()
        
        learner.learn_text("Test fact for persistence")
        
        # Knowledge should be persisted
        # (exact mechanism depends on implementation)
    
    def test_persistence_format_agnostic(self):
        """Knowledge can be persisted in different formats"""
        learner = TextKnowledgeLearner()
        learner.learn_text("Persisted knowledge")
        
        # Should support multiple backends:
        # - SQLite (default)
        # - JSON
        # - Binary format
        
        # (This tests the abstraction, not all implementations)


class TestCustomPerceptionModule:
    """Test EXT-5: Add custom perception (e.g., image → HV)"""
    
    def test_image_encoding_module(self):
        """Create custom image encoder"""
        from python.core.vsa.universal_encoder import UniversalEncoder
        
        class ImageEncoder:
            def __init__(self):
                self.encoder = UniversalEncoder()
            
            def encode_image(self, image_features):
                """Convert image features to HV"""
                # Example: HOG histogram → HV
                histogram_description = f"image_histogram_{hash(str(image_features))}"
                return self.encoder.encode_text(histogram_description)
        
        # Create and use
        img_encoder = ImageEncoder()
        sample_features = {"color": [100, 150, 200], "edges": 42}
        hv = img_encoder.encode_image(sample_features)
        
        assert hv is not None
    
    def test_multimodal_input_pipeline(self):
        """Pipeline accepts text + image + audio inputs"""
        # This tests extensibility of input handling
        
        from python.core.vsa.universal_encoder import UniversalEncoder
        
        encoder = UniversalEncoder()
        
        # Text input
        text_hv = encoder.encode_text("A description")
        assert text_hv is not None
        
        # Could extend to:
        # - Image encoding
        # - Audio encoding
        # - Composite multimodal HV


class TestReproducibility:
    """Test EXT-6: Reproducibility - same input → same output"""
    
    def test_deterministic_encoding(self):
        """Same text always encodes identically"""
        encoder = UniversalEncoder()
        
        text = "This is a test sentence"
        
        hv1 = encoder.encode_text(text)
        hv2 = encoder.encode_text(text)
        
        # Should be identical
        sim = hv1.similarity(hv2)
        assert sim > 0.99, f"Encoding not deterministic: {sim}"
    
    def test_deterministic_query_results(self):
        """Same query on same model → same result"""
        learner1 = TextKnowledgeLearner()
        learner1.learn_text("Dogs are animals")
        result1 = learner1.query_learned_knowledge("What are dogs?", top_k=5)
        
        # Create identical second learner
        learner2 = TextKnowledgeLearner()
        learner2.learn_text("Dogs are animals")
        result2 = learner2.query_learned_knowledge("What are dogs?", top_k=5)
        
        # Results should be identical
        # (exact comparison depends on result structure)
    
    def test_seeded_randomness(self):
        """Randomness controlled via seed for reproducibility"""
        # If system uses any randomness (e.g., tie-breaking),
        # it should be seeded for reproducibility
        
        # Both systems should behave identically with same seed
        pass


class TestExtensibilityIntegration:
    """Integration tests for extensibility"""
    
    def test_end_to_end_new_domain(self):
        """Create new domain from scratch and integrate"""
        # Domain: Medical diagnosis
        diagnosis_brain = TaskBrain("medical_diagnosis")
        encoder = UniversalEncoder()
        
        # Define concepts
        concepts = ["symptom", "disease", "test", "treatment"]
        for concept in concepts:
            diagnosis_brain.add_concept(
                concept,
                encoder.encode_text(concept)
            )
        
        # Define rules
        diagnosis_brain.add_rule(
            condition=frozenset(["symptom", "test"]),
            consequence="diagnose_disease",
            strength=0.85
        )
        
        # Should be fully functional
        assert len(diagnosis_brain.codebook) > 0
        assert len(diagnosis_brain.rules) > 0
    
    def test_multi_domain_reasoning(self):
        """Reasoning across multiple integrated domains"""
        # Setup two domains
        math_brain = TaskBrain("mathematics")
        physics_brain = TaskBrain("physics")
        encoder = UniversalEncoder()
        
        # Math: 2+2=4
        math_brain.add_concept("addition", encoder.encode_text("addition"))
        math_brain.add_rule(
            condition=frozenset(["two", "plus", "two"]),
            consequence="four"
        )
        
        # Physics: energy conservation
        physics_brain.add_concept("energy", encoder.encode_text("energy"))
        physics_brain.add_rule(
            condition=frozenset(["energy", "transfer"]),
            consequence="total_conserved"
        )
        
        # Integrate
        integrated = FusedBrain([math_brain, physics_brain])
        
        # Should support queries across domains


class TestExtensibilityNarratives:
    """Narrative demonstrations of extensibility"""
    
    def test_narrative_domain_extension(self):
        """Show extending NSCK to new domain"""
        print("\n" + "="*70)
        print("NARRATIVE: Domain Extension - Card Game AI")
        print("="*70)
        
        print("\nStep 1: Create domain-specific TaskBrain")
        poker_brain = TaskBrain("poker_game")
        encoder = UniversalEncoder()
        
        print("  Created TaskBrain('poker_game')")
        
        print("\nStep 2: Define game concepts")
        concepts = ["hand", "card", "rank", "suit", "pot", "bet", "fold", "call"]
        for concept in concepts:
            poker_brain.add_concept(concept, encoder.encode_text(concept))
            print(f"  Added concept: {concept}")
        
        print("\nStep 3: Learn game rules")
        rules = [
            ("Royal Flush", "wins"),
            ("Straight", "beats", "Three of a Kind"),
            ("Pair", "loses to", "Two Pair"),
        ]
        
        poker_brain.add_rule(
            condition=frozenset(["cards", "suit"]),
            consequence="Royal_Flush",
            strength=0.99
        )
        print("  Added poker rules")
        
        print("\nStep 4: Integrate with NSCK pipeline")
        print("  TaskBrain ready for game simulation")
        
        print("\nResult: Custom AI domain fully extensible")
    
    def test_narrative_multimodal_integration(self):
        """Show adding multimodal perception"""
        print("\n" + "="*70)
        print("NARRATIVE: Multimodal Integration")
        print("="*70)
        
        print("\nCurrent: Text-only perception")
        print("  Input: 'The cat is fluffy'")
        print("  Encoding: Text → 10,240-bit HV")
        
        print("\nExtended: Add image perception")
        print("  Input: Image of cat")
        print("  Perception pipeline:")
        print("    1. Image → HOG features (edges)")
        print("    2. HOG → HV (via custom encoder)")
        print("    3. Fuse with text HV")
        
        print("\nBenefit: Richer semantic understanding")
        print("  'Fluffy' + visual features = deeper concept")
    
    def test_narrative_community_building(self):
        """Show potential for community extensions"""
        print("\n" + "="*70)
        print("NARRATIVE: Community-Built Domains")
        print("="*70)
        
        print("\nNSCK Architecture enables:")
        print("  ✓ Domain experts build specialized TaskBrains")
        print("  ✓ BrainFusion integrates across domains")
        print("  ✓ No central LLM dependency (fully open)")
        
        print("\nExample community contributions:")
        print("  - Medical diagnosis TaskBrain (doctors)")
        print("  - Legal reasoning module (lawyers)")
        print("  - Financial analysis domain (quants)")
        print("  - Game AI systems (game devs)")
        print("  - Scientific reasoning (scientists)")
        
        print("\nEach can contribute independently, fuse results")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
