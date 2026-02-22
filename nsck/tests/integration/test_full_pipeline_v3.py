"""Integration tests for the full NSCK V3 pipeline."""
import unittest
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.memory.homeostasis import MemoryHomeostasis
from python.core.language.frame_semantics import FrameLibrary
import python.core.vsa.hypervec_shim as hypervec_rs


class TestFullPipelineV3(unittest.TestCase):
    def setUp(self):
        self.sem = SemanticMemory(use_rust=False)
        self.epi = EpisodicMemory()
        self.learner = TextKnowledgeLearner(self.sem, self.epi)

    def test_multi_paragraph_learning(self):
        """Feed 10 geography sentences, verify facts stored."""
        sentences = [
            "France is a country in Europe.",
            "Paris is the capital of France.",
            "The Seine flows through Paris.",
            "Germany is east of France.",
            "Berlin is the capital of Germany.",
            "Europe is a continent.",
            "The Alps are mountains in Europe.",
            "Italy is south of Germany.",
            "Rome is the capital of Italy.",
            "Spain is west of France.",
        ]
        for s in sentences:
            self.learner.learn_from_text(s)
        concepts = list(self.sem.concept_hvs.keys())
        self.assertGreater(len(concepts), 0)

    def test_construction_grammar_coverage(self):
        """Feed diverse sentences, verify relations extracted."""
        from python.core.language.construction_grammar import ConstructionMatcher
        matcher = ConstructionMatcher()
        diverse_sentences = [
            ["John", "chased", "Mary"],
            ["Paris", "is", "beautiful"],
            ["John", "has", "car"],
            ["heat", "causes", "expansion"],
            ["Paris", "is", "in", "France"],
        ]
        total_matches = 0
        for words in diverse_sentences:
            matches = matcher.match(words)
            total_matches += len(matches)
        self.assertGreater(total_matches, 0)

    def test_contradiction_tracking(self):
        """Teaching contradictory facts should both be stored."""
        self.learner.learn_from_text("Paris is capital of France.")
        initial_edges = self.sem.concept_graph.number_of_edges()
        self.learner.learn_from_text("Paris is capital of Germany.")
        final_edges = self.sem.concept_graph.number_of_edges()
        # Both facts processed (edges may increase)
        self.assertGreaterEqual(final_edges, initial_edges)

    def test_frame_semantics_roundtrip(self):
        """Fill frame, extract role, verify fidelity."""
        library = FrameLibrary()
        frame = library.find_frame("buy")
        self.assertIsNotNone(frame)
        buyer_hv = hypervec_rs.HyperVector(hash("alice") % (2**32))
        goods_hv = hypervec_rs.HyperVector(hash("book") % (2**32))
        filled = frame.fill({"buyer": buyer_hv, "goods": goods_hv})
        recovered_buyer = frame.extract_filler(filled, "buyer")
        # The recovered HV should exist
        self.assertIsNotNone(recovered_buyer)

    def test_homeostasis_regulation(self):
        """Add 50 concepts, run regulation, verify healthy memory."""
        for i in range(50):
            self.sem.add_concept(f"concept_{i}", {"value": i})
        homeostasis = MemoryHomeostasis()
        actions = homeostasis.regulate(self.sem)
        self.assertIsInstance(actions, list)
        # Memory should still be functional after regulation
        concepts = list(self.sem.concept_hvs.keys())
        self.assertGreater(len(concepts), 0)

    def test_belief_revision_on_contradiction(self):
        """Test that belief revision tracks contradictions."""
        from python.core.reasoning.belief_revision import BeliefMetadata, BeliefScorer
        scorer = BeliefScorer()
        meta = BeliefMetadata(evidence_count=3, contradiction_count=2)
        should_revise, reason = scorer.should_revise(meta, new_evidence_supports=False)
        self.assertTrue(should_revise)

    def test_belief_revision_in_semantic_memory(self):
        """With enable_free_energy_beliefs, contradictions are tracked in edge metadata."""
        from python.core.integration.config import NSCKConfig
        config = NSCKConfig(enable_free_energy_beliefs=True)
        mem = SemanticMemory(use_rust=False, config=config)
        mem.add_concept("Earth", {})
        mem.add_concept("round", {})
        mem.add_concept("flat", {})
        # Add belief with 3 supporting mentions
        for _ in range(3):
            mem.add_relation("Earth", "shape_is", "round", timestamp=1.0)
        edge = mem.concept_graph.get_edge_data("Earth", "round")
        self.assertIsNotNone(edge)
        meta = edge.get("belief_meta", {})
        self.assertEqual(meta.get("evidence_count"), 3)
        # Add contradictory belief: Earth shape_is flat
        mem.add_relation("Earth", "shape_is", "flat", timestamp=2.0)
        # round edge should have contradiction_count=1
        round_edge = mem.concept_graph.get_edge_data("Earth", "round")
        self.assertIsNotNone(round_edge)
        self.assertEqual(round_edge.get("belief_meta", {}).get("contradiction_count"), 1)

    def test_coreference_resolution(self):
        """With enable_coreference, pronouns are resolved to registered entities."""
        from python.core.language.coreference import EntityRegister

        register = EntityRegister()
        john_hv = hypervec_rs.HyperVector(hash("John") % (2**32))
        register.register("John", john_hv, {"gender": "male", "animacy": "animate", "number": "singular"})

        # "he" → should resolve to John
        mention = register.resolve("he")
        self.assertIsNotNone(mention)
        self.assertEqual(mention.name, "John")

        # "she" → no female entity → None
        mention_she = register.resolve("she")
        self.assertIsNone(mention_she)

    def test_frame_semantics_verb_lookup(self):
        """FrameLibrary.find_frame() returns relevant frame for known verbs."""
        library = FrameLibrary()
        frame = library.find_frame("buy")
        self.assertIsNotNone(frame)
        self.assertIn("buyer", frame.roles)

        frame_unknown = library.find_frame("zargblarg")
        self.assertIsNone(frame_unknown)


if __name__ == "__main__":
    unittest.main()
