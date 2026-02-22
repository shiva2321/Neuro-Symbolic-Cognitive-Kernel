import unittest
import math
from python.core.reasoning.belief_revision import BeliefMetadata, BeliefScorer

class TestBeliefRevision(unittest.TestCase):
    def setUp(self):
        self.scorer = BeliefScorer()

    def test_free_energy_no_contradiction(self):
        meta = BeliefMetadata(evidence_count=5, contradiction_count=0, complexity=1.0)
        fe = self.scorer.free_energy(meta)
        self.assertIsInstance(fe, float)
        self.assertGreater(fe, 0.0)
        self.assertAlmostEqual(fe, -math.log(5/6) + 0.1, places=5)

    def test_free_energy_with_contradiction(self):
        meta_low = BeliefMetadata(evidence_count=5, contradiction_count=0)
        meta_high = BeliefMetadata(evidence_count=5, contradiction_count=4)
        fe_low = self.scorer.free_energy(meta_low)
        fe_high = self.scorer.free_energy(meta_high)
        self.assertGreater(fe_high, fe_low)

    def test_should_revise_when_contradictions_exceed_evidence(self):
        meta = BeliefMetadata(evidence_count=2, contradiction_count=2)
        should_revise, reason = self.scorer.should_revise(meta, new_evidence_supports=False)
        self.assertTrue(should_revise)
        self.assertEqual(reason, "contradictions_exceed_evidence")

    def test_mark_contested(self):
        meta = BeliefMetadata(evidence_count=5, contradiction_count=1)
        should_revise, reason = self.scorer.should_revise(meta, new_evidence_supports=False)
        self.assertFalse(should_revise)
        self.assertEqual(reason, "mark_contested")

    def test_supporting_evidence_keeps_belief(self):
        meta = BeliefMetadata(evidence_count=3, contradiction_count=0)
        should_revise, reason = self.scorer.should_revise(meta, new_evidence_supports=True)
        self.assertFalse(should_revise)
        self.assertEqual(reason, "evidence_supported")

if __name__ == "__main__":
    unittest.main()
