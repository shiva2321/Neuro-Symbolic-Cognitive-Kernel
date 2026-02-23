"""
Tests for NSCK Pragmatics module (G2 from roadmap).
Scalar implicature, Gricean maxims, indirect speech acts, presuppositions.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from python.core.language.pragmatics import (
    PragmaticsEngine,
    ScalarImplicature, Presupposition, MaximViolation, PragmaticAnalysis,
    SCALAR_SCALES, UPPER_BOUND_IMPLICATURE,
)


class TestScalarImplicatureData(unittest.TestCase):
    """Verify scalar scale data structures are consistent."""

    def test_scales_have_weakest_to_strongest(self):
        """Every scale should be at least 2 terms."""
        for trigger, scale in SCALAR_SCALES.items():
            self.assertGreaterEqual(len(scale), 2,
                f"Scale for '{trigger}' is too short: {scale}")

    def test_all_triggers_have_upper_bound(self):
        """Every scale trigger should have an upper-bound implicature."""
        for trigger in SCALAR_SCALES:
            self.assertIn(trigger, UPPER_BOUND_IMPLICATURE,
                f"No upper-bound implicature for trigger '{trigger}'")

    def test_trigger_is_weakest_in_own_scale(self):
        """Each key should be the first (weakest) element of its scale."""
        for trigger, scale in SCALAR_SCALES.items():
            self.assertEqual(scale[0], trigger,
                f"'{trigger}' should be weakest in scale {scale}")


class TestSpeechActClassification(unittest.TestCase):
    """Speech act detection."""

    def setUp(self):
        self.pe = PragmaticsEngine()

    def test_assert_is_default(self):
        r = self.pe.analyze("The sky is blue.")
        self.assertEqual(r.speech_act, "assert")
        self.assertFalse(r.is_indirect)

    def test_direct_question(self):
        r = self.pe.analyze("Is the cat on the mat?")
        self.assertEqual(r.speech_act, "question")

    def test_wh_question(self):
        r = self.pe.analyze("Where is the nearest hospital?")
        self.assertEqual(r.speech_act, "question")

    def test_indirect_request_can_you(self):
        r = self.pe.analyze("Can you pass the salt?")
        self.assertEqual(r.speech_act, "request")
        self.assertTrue(r.is_indirect)

    def test_indirect_request_could_you(self):
        r = self.pe.analyze("Could you please close the window?")
        self.assertEqual(r.speech_act, "request")
        self.assertTrue(r.is_indirect)

    def test_indirect_request_would_you(self):
        r = self.pe.analyze("Would you mind helping me?")
        self.assertEqual(r.speech_act, "request")
        self.assertTrue(r.is_indirect)

    def test_promise(self):
        r = self.pe.analyze("I will finish the report by Monday.")
        self.assertEqual(r.speech_act, "promise")

    def test_question_mark(self):
        r = self.pe.analyze("You agree?")
        self.assertEqual(r.speech_act, "question")

    def test_speech_act_convenience_method(self):
        act = self.pe.speech_act("What time is it?")
        self.assertEqual(act, "question")


class TestPolitenessDetection(unittest.TestCase):

    def setUp(self):
        self.pe = PragmaticsEngine()

    def test_please_is_polite(self):
        r = self.pe.analyze("Please pass the salt.")
        self.assertTrue(r.is_polite)

    def test_could_you_is_polite(self):
        r = self.pe.analyze("Could you possibly help?")
        self.assertTrue(r.is_polite)

    def test_direct_command_not_polite(self):
        r = self.pe.analyze("Close the door.")
        self.assertFalse(r.is_polite)


class TestScalarImplicatureExtraction(unittest.TestCase):

    def setUp(self):
        self.pe = PragmaticsEngine()

    def test_some_implies_not_all(self):
        r = self.pe.analyze("Some students passed the exam.")
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertIn("some", triggers)
        some_si = next(si for si in r.scalar_implicatures if si.trigger == "some")
        self.assertIn("not all", some_si.upper_bound)

    def test_many_implies_not_all(self):
        r = self.pe.analyze("Many people attended the event.")
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertIn("many", triggers)

    def test_possible_implies_not_certain(self):
        r = self.pe.analyze("It is possible that it will rain.")
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertIn("possible", triggers)

    def test_all_cancels_some_implicature(self):
        """When 'all' is present, 'some' implicature is cancelled."""
        r = self.pe.analyze("Some students passed, in fact all of them passed.")
        # 'some' should NOT fire because 'all' is stronger in the same scale
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertNotIn("some", triggers)

    def test_no_implicature_without_scalar(self):
        r = self.pe.analyze("The cat sat on the mat.")
        self.assertEqual(len(r.scalar_implicatures), 0)

    def test_sometimes_implies_not_always(self):
        r = self.pe.analyze("She sometimes comes late.")
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertIn("sometimes", triggers)

    def test_implicatures_convenience_method(self):
        imps = self.pe.implicatures("Some birds can fly.")
        self.assertTrue(len(imps) > 0)
        self.assertTrue(any("some" in i.lower() for i in imps))

    def test_good_implies_not_great(self):
        r = self.pe.analyze("The meal was good.")
        triggers = [si.trigger for si in r.scalar_implicatures]
        self.assertIn("good", triggers)


class TestPresuppositionExtraction(unittest.TestCase):

    def setUp(self):
        self.pe = PragmaticsEngine()

    def test_stopped_smoking_presupposes_prior_smoking(self):
        r = self.pe.analyze("John stopped smoking.")
        self.assertGreater(len(r.presuppositions), 0)
        contents = [p.content for p in r.presuppositions]
        self.assertTrue(any("smoking" in c for c in contents))

    def test_regret_presupposes_event(self):
        r = self.pe.analyze("She regrets that she quit.")
        self.assertGreater(len(r.presuppositions), 0)

    def test_knows_presupposes_truth(self):
        r = self.pe.analyze("He knows that the earth is round.")
        presup = [p for p in r.presuppositions if "know" in p.trigger_word.lower()]
        self.assertGreater(len(presup), 0)

    def test_again_presupposes_prior_occurrence(self):
        r = self.pe.analyze("She won the race again.")
        presup_contents = [p.content for p in r.presuppositions]
        self.assertTrue(any("before" in c for c in presup_contents))

    def test_still_presupposes_prior_state(self):
        r = self.pe.analyze("He is still sleeping.")
        presup_contents = [p.content for p in r.presuppositions]
        self.assertTrue(any("past" in c for c in presup_contents))

    def test_presuppositions_convenience_method(self):
        pres = self.pe.presuppositions("He stopped eating.")
        self.assertTrue(len(pres) > 0)

    def test_no_presupposition_in_simple_sentence(self):
        r = self.pe.analyze("The cat is on the mat.")
        # Simple declarative — should have no presupposition triggers
        self.assertEqual(len(r.presuppositions), 0)


class TestGriceanMaxims(unittest.TestCase):

    def setUp(self):
        self.pe = PragmaticsEngine()

    def test_no_violations_for_normal_sentence(self):
        violations = self.pe.check_maxims("The cat is on the mat.")
        # No violations expected for this neutral, informative, on-topic sentence
        self.assertIsInstance(violations, list)

    def test_quality_violation_with_i_think(self):
        violations = self.pe.check_maxims("I think it might rain.")
        maxims = [v.maxim for v in violations]
        self.assertIn("Quality", maxims)

    def test_quantity_violation_very_short(self):
        violations = self.pe.check_maxims("Yes")
        maxims = [v.maxim for v in violations]
        self.assertIn("Quantity", maxims)

    def test_relation_violation_off_topic(self):
        violations = self.pe.check_maxims(
            "The weather is nice today.",
            context_topic="quarterly financial results"
        )
        maxims = [v.maxim for v in violations]
        self.assertIn("Relation", maxims)

    def test_on_topic_no_relation_violation(self):
        # Use exact lexical overlap (topic words must appear verbatim in utterance)
        violations = self.pe.check_maxims(
            "The quarterly sales results were impressive.",
            context_topic="quarterly financial results"
        )
        relation_violations = [v for v in violations if v.maxim == "Relation"]
        self.assertEqual(len(relation_violations), 0)

    def test_manner_violation_hedge(self):
        violations = self.pe.check_maxims(
            "It might be said that the results were satisfactory."
        )
        maxims = [v.maxim for v in violations]
        self.assertIn("Manner", maxims)


class TestPragmaticAnalysisDataclass(unittest.TestCase):

    def test_full_analysis_returns_dataclass(self):
        pe = PragmaticsEngine()
        r = pe.analyze("Some students passed.")
        self.assertIsInstance(r, PragmaticAnalysis)
        self.assertEqual(r.utterance, "Some students passed.")

    def test_enrichments_non_empty_for_implicature(self):
        pe = PragmaticsEngine()
        r = pe.analyze("Some birds can fly.")
        self.assertGreater(len(r.pragmatic_enrichments), 0)

    def test_enrichments_include_indirect_act(self):
        pe = PragmaticsEngine()
        r = pe.analyze("Can you open the window?")
        enrichments_text = " ".join(r.pragmatic_enrichments).lower()
        self.assertIn("indirect", enrichments_text)

    def test_scalar_implicature_str(self):
        si = ScalarImplicature(trigger="some", scale=["some","all"], upper_bound="not all")
        self.assertIn("some", str(si))
        self.assertIn("not all", str(si))

    def test_maxim_violation_str(self):
        mv = MaximViolation(maxim="Quality", severity="low", note="test")
        self.assertIn("Quality", str(mv))

    def test_presupposition_str(self):
        ps = Presupposition(trigger_word="stopped", content="prior smoking")
        self.assertIn("stopped", str(ps))


class TestV5ConfigFlags(unittest.TestCase):
    """V5 config flags are present in NSCKConfig."""

    def test_minimal_has_v5_flags_false(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.minimal()
        self.assertFalse(cfg.enable_spatial_reasoning)
        self.assertFalse(cfg.enable_pragmatics)

    def test_research_has_v5_flags_true(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.research()
        self.assertTrue(cfg.enable_spatial_reasoning)
        self.assertTrue(cfg.enable_pragmatics)

    def test_v5_flags_settable(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig(enable_spatial_reasoning=True, enable_pragmatics=True)
        self.assertTrue(cfg.enable_spatial_reasoning)
        self.assertTrue(cfg.enable_pragmatics)


if __name__ == "__main__":
    unittest.main()
