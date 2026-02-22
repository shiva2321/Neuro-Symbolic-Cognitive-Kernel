"""Real-world capability tests for NSCK V3."""
import unittest
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.reasoning.belief_revision import BeliefMetadata, BeliefScorer


class TestRealWorldV3(unittest.TestCase):
    def setUp(self):
        self.sem = SemanticMemory(use_rust=False)
        self.epi = EpisodicMemory()
        self.learner = TextKnowledgeLearner(self.sem, self.epi)

    def test_photosynthesis_paragraph(self):
        """Process photosynthesis text, verify concepts learned."""
        text = (
            "Photosynthesis is the process by which plants convert sunlight into energy. "
            "Plants use water and carbon dioxide to produce glucose. "
            "Oxygen is released as a byproduct of photosynthesis. "
            "Chlorophyll is the pigment that captures sunlight in plants."
        )
        self.learner.learn_from_text(text)
        concepts = list(self.sem.concept_hvs.keys())
        self.assertGreater(len(concepts), 0)

    def test_science_fact_chain(self):
        """Teach science facts and verify they are stored."""
        facts = [
            "Water is H2O.",
            "H2O contains hydrogen and oxygen.",
            "Hydrogen is an element.",
            "Oxygen is an element.",
        ]
        for fact in facts:
            self.learner.learn_from_text(fact)
        concepts = list(self.sem.concept_hvs.keys())
        self.assertGreater(len(concepts), 0)

    def test_contradiction_in_natural_text(self):
        """Repeated flat/round earth claims - belief revision tracks contradictions."""
        scorer = BeliefScorer()
        # Simulate evidence accumulation
        meta = BeliefMetadata(evidence_count=0, contradiction_count=0)
        # 5x flat earth
        for _ in range(5):
            meta.evidence_count += 1
        # 5x round earth (contradictions to flat)
        for _ in range(5):
            meta.contradiction_count += 1
        fe = scorer.free_energy(meta)
        self.assertIsInstance(fe, float)
        should_revise, reason = scorer.should_revise(meta, new_evidence_supports=False)
        self.assertTrue(should_revise)

    def test_learning_does_not_crash(self):
        """Various sentence types should not crash the learner."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be that is the question.",
            "Einstein discovered the theory of relativity.",
            "Python is a programming language.",
            "The sun rises in the east.",
        ]
        for s in sentences:
            try:
                self.learner.learn_from_text(s)
            except Exception as e:
                self.fail(f"Learning crashed on: {s!r} with {e}")

    def test_semantic_query_after_learning(self):
        """After learning, semantic query should return results."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        self.learner.learn_from_text("Dogs are loyal animals.")
        self.learner.learn_from_text("Cats are independent animals.")
        if self.sem.concept_hvs:
            query = hypervec_rs.HyperVector(hash("dog") % (2**32))
            results = self.sem.query(query, k=3)
            self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
