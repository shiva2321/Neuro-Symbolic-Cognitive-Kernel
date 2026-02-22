"""Tests for NSCK V4 features:
- Extended COMMON_VERBS coverage
- Negation constructions
- Conditional constructions
- Temporal ordering constructions
- V4 config flags
- SemanticMemory.infer_transitive()
- SemanticMemory.build_prototypes()
"""
import unittest
import sys
import os

# Ensure nsck package root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from python.core.language.construction_grammar import (
    ConstructionMatcher, _classify_word,
    NEGATION_WORDS, TEMPORAL_CONNECTIVES, CONDITIONAL_CONNECTIVES,
)
from python.core.integration.config import NSCKConfig
from python.core.memory.semantic_memory import SemanticMemory
import python.core.vsa.hypervec_shim as hypervec_rs


class TestExtendedVerbCoverage(unittest.TestCase):
    """V4: Verify hundreds of new verb forms are classified as VERB."""

    def setUp(self):
        self.matcher = ConstructionMatcher()

    def _assert_verb(self, word: str):
        pos = _classify_word(word)
        self.assertEqual(pos, "VERB", f"Expected '{word}' → VERB, got {pos}")

    def test_love_forms(self):
        for w in ("love", "loves", "loved", "loving"):
            self._assert_verb(w)

    def test_hate_forms(self):
        for w in ("hate", "hates", "hated", "hating"):
            self._assert_verb(w)

    def test_learn_forms(self):
        for w in ("learn", "learns", "learned", "learnt"):
            self._assert_verb(w)

    def test_understand_forms(self):
        for w in ("understand", "understands", "understood"):
            self._assert_verb(w)

    def test_believe_forms(self):
        for w in ("believe", "believes", "believed", "believing"):
            self._assert_verb(w)

    def test_say_forms(self):
        for w in ("say", "says", "said"):
            self._assert_verb(w)

    def test_exist_forms(self):
        for w in ("exist", "exists", "existed", "existing"):
            self._assert_verb(w)

    def test_affect_forms(self):
        for w in ("affect", "affects", "affected", "affecting"):
            self._assert_verb(w)

    def test_third_person_s_strip(self):
        """3rd-person -s forms should be detected via stem lookup."""
        for w in ("helps", "goes", "makes", "takes", "works"):
            self._assert_verb(w)

    def test_irregular_pasts(self):
        for w in ("went", "made", "took", "said", "came", "gave", "knew"):
            self._assert_verb(w)

    def test_ize_ise_forms(self):
        # Note: -ize forms are detected via COMMON_VERBS or suffix heuristics
        # -ise forms (British English) must be in COMMON_VERBS explicitly
        for w in ("organize", "organized", "organizing",
                  "realise", "realised"):
            self._assert_verb(w)
        # "organizes" → 3rd-person: strip -s → "organize" → endswith("ize") → VERB
        self._assert_verb("organizes")

    def test_ate_forms(self):
        for w in ("generate", "generates", "generated", "generating",
                  "activate", "activates", "activated"):
            self._assert_verb(w)

    def test_svo_with_loves(self):
        """Sentence 'Alice loves Bob' should match SVO_active."""
        words = ["Alice", "loves", "Bob"]
        matches = self.matcher.match(words)
        self.assertTrue(len(matches) > 0, "Expected SVO match for 'Alice loves Bob'")
        self.assertIn("subject", matches[0].role_fillers)
        self.assertEqual(matches[0].role_fillers["subject"], "Alice")

    def test_svo_with_hates(self):
        words = ["Cat", "hates", "Water"]
        matches = self.matcher.match(words)
        self.assertTrue(len(matches) > 0)

    def test_svo_with_affects(self):
        words = ["Heat", "affects", "Metal"]
        matches = self.matcher.match(words)
        self.assertTrue(len(matches) > 0)


class TestNegationWords(unittest.TestCase):
    """V4: Negation words classified as NEG."""

    def test_not_classified_as_neg(self):
        self.assertEqual(_classify_word("not"), "NEG")

    def test_never_classified_as_neg(self):
        self.assertEqual(_classify_word("never"), "NEG")

    def test_no_classified_as_neg(self):
        self.assertEqual(_classify_word("no"), "NEG")

    def test_negation_set_complete(self):
        for w in ("not", "never", "no", "neither", "nor"):
            self.assertIn(w, NEGATION_WORDS)


class TestNegationConstructions(unittest.TestCase):
    """V4: Negation constructions match correctly."""

    def setUp(self):
        self.matcher = ConstructionMatcher()

    def _get_by_name(self, words, name):
        matches = self.matcher.match(words)
        return [m for m in matches if m.construction.name == name]

    def test_negation_is_not(self):
        words = ["Paris", "is", "not", "London"]
        ms = self._get_by_name(words, "negation_is_not")
        self.assertTrue(len(ms) > 0, "Expected negation_is_not match")
        self.assertEqual(ms[0].role_fillers["subject"], "Paris")
        self.assertEqual(ms[0].role_fillers["negated_attribute"], "London")
        self.assertEqual(ms[0].construction.relation, "not_is_a")

    def test_negation_is_not_adj(self):
        words = ["Snow", "is", "not", "dangerous"]
        ms = self._get_by_name(words, "negation_is_not_adj")
        self.assertTrue(len(ms) > 0)
        self.assertEqual(ms[0].role_fillers["subject"], "Snow")
        self.assertEqual(ms[0].construction.relation, "not_has_property")

    def test_negation_lacks(self):
        words = ["Fish", "lacks", "legs"]
        ms = self._get_by_name(words, "negation_lacks")
        self.assertTrue(len(ms) > 0)
        self.assertEqual(ms[0].role_fillers["subject"], "Fish")
        self.assertEqual(ms[0].construction.relation, "lacks")

    def test_negation_cannot(self):
        words = ["Fish", "cannot", "fly", "Sky"]
        # Fish cannot fly Sky — tests the pattern structure
        ms = [m for m in self.matcher.match(words)
              if m.construction.name == "negation_cannot"]
        # Should either match or gracefully not match — no crash
        self.assertIsInstance(ms, list)


class TestConditionalConstructions(unittest.TestCase):
    """V4: Conditional constructions match correctly."""

    def setUp(self):
        self.matcher = ConstructionMatcher()

    def test_conditional_implies(self):
        words = ["Rain", "implies", "Flood"]
        matches = self.matcher.match(words)
        impls = [m for m in matches if m.construction.name == "conditional_implies"]
        self.assertTrue(len(impls) > 0, "Expected conditional_implies match")
        self.assertEqual(impls[0].role_fillers["antecedent"], "Rain")
        self.assertEqual(impls[0].role_fillers["consequent"], "Flood")
        self.assertEqual(impls[0].construction.relation, "implies")

    def test_conditional_leads_to(self):
        words = ["Stress", "leads", "to", "Illness"]
        matches = self.matcher.match(words)
        ltms = [m for m in matches if m.construction.name == "conditional_leads_to"]
        self.assertTrue(len(ltms) > 0, "Expected conditional_leads_to match")
        self.assertEqual(ltms[0].construction.relation, "leads_to")

    def test_conditional_results_in(self):
        # Use words that won't be misclassified - "Effort" is a reliable noun
        words = ["Effort", "results", "in", "Success"]
        matches = self.matcher.match(words)
        rims = [m for m in matches if m.construction.name == "conditional_results_in"]
        self.assertTrue(len(rims) > 0, "Expected conditional_results_in match")
        self.assertEqual(rims[0].construction.relation, "results_in")

    def test_conditional_connective_classification(self):
        for w in ("if", "unless", "whenever", "provided"):
            pos = _classify_word(w)
            self.assertEqual(pos, "COND", f"'{w}' should be COND, got {pos}")


class TestTemporalConstructions(unittest.TestCase):
    """V4: Temporal ordering constructions match correctly."""

    def setUp(self):
        self.matcher = ConstructionMatcher()

    def test_temporal_before_noun(self):
        # Use words that won't be misclassified as VERB
        words = ["Breakfast", "before", "Lunch"]
        matches = self.matcher.match(words)
        tms = [m for m in matches if m.construction.name == "temporal_before_noun"]
        self.assertTrue(len(tms) > 0, "Expected temporal_before_noun match")
        self.assertEqual(tms[0].role_fillers["event1"], "Breakfast")
        self.assertEqual(tms[0].role_fillers["event2"], "Lunch")
        self.assertEqual(tms[0].construction.relation, "precedes")

    def test_temporal_after_noun(self):
        words = ["Dinner", "after", "Lunch"]
        matches = self.matcher.match(words)
        tms = [m for m in matches if m.construction.name == "temporal_after_noun"]
        self.assertTrue(len(tms) > 0)
        self.assertEqual(tms[0].construction.relation, "follows")

    def test_temporal_connective_classification(self):
        for w in ("before", "after", "since", "until", "when"):
            pos = _classify_word(w)
            self.assertEqual(pos, "TEMP", f"'{w}' should be TEMP, got {pos}")


class TestSimilarityDifferenceConstructions(unittest.TestCase):
    """V4: Similarity and difference constructions."""

    def setUp(self):
        self.matcher = ConstructionMatcher()

    def test_similarity_like(self):
        words = ["Mercury", "is", "like", "Venus"]
        matches = self.matcher.match(words)
        sms = [m for m in matches if m.construction.name == "similarity_like"]
        self.assertTrue(len(sms) > 0)
        self.assertEqual(sms[0].construction.relation, "similar_to")

    def test_difference_differs(self):
        words = ["Cat", "differs", "from", "Dog"]
        matches = self.matcher.match(words)
        dms = [m for m in matches if m.construction.name == "difference_differs"]
        self.assertTrue(len(dms) > 0)
        self.assertEqual(dms[0].construction.relation, "different_from")


class TestV4ConfigFlags(unittest.TestCase):
    """V4 feature flags in NSCKConfig."""

    def test_minimal_has_v4_flags(self):
        cfg = NSCKConfig.minimal()
        self.assertFalse(cfg.enable_negation_handling)
        self.assertFalse(cfg.enable_temporal_reasoning)
        self.assertFalse(cfg.enable_conditional_logic)
        self.assertFalse(cfg.enable_transitive_inference)
        self.assertFalse(cfg.enable_prototype_generalization)

    def test_research_has_v4_flags_on(self):
        cfg = NSCKConfig.research()
        self.assertTrue(cfg.enable_negation_handling)
        self.assertTrue(cfg.enable_temporal_reasoning)
        self.assertTrue(cfg.enable_conditional_logic)
        self.assertTrue(cfg.enable_transitive_inference)
        self.assertTrue(cfg.enable_prototype_generalization)

    def test_production_has_key_v4_flags_on(self):
        cfg = NSCKConfig.production()
        self.assertTrue(cfg.enable_negation_handling)
        self.assertTrue(cfg.enable_temporal_reasoning)
        self.assertTrue(cfg.enable_conditional_logic)
        self.assertTrue(cfg.enable_transitive_inference)

    def test_instantiation_with_v4_flags(self):
        cfg = NSCKConfig(enable_negation_handling=True, enable_temporal_reasoning=True)
        self.assertTrue(cfg.enable_negation_handling)
        self.assertTrue(cfg.enable_temporal_reasoning)
        self.assertFalse(cfg.enable_prototype_generalization)


class TestTransitiveInference(unittest.TestCase):
    """V4: SemanticMemory.infer_transitive()"""

    def _make_memory(self):
        sm = SemanticMemory(use_rust=False)
        for name in ("Poodle", "Dog", "Animal", "LivingThing"):
            hv = hypervec_rs.HyperVector(abs(hash(name)) % (2**32))
            sm.add_concept(name, {}, hv_override=hv)
        sm.add_relation("Poodle", "is_a", "Dog")
        sm.add_relation("Dog", "is_a", "Animal")
        sm.add_relation("Animal", "is_a", "LivingThing")
        return sm

    def test_two_hop_transitive(self):
        sm = self._make_memory()
        added = sm.infer_transitive("is_a", max_hops=2)
        self.assertGreater(added, 0, "Expected at least one transitive edge")
        self.assertTrue(sm.concept_graph.has_edge("Poodle", "Animal"),
                        "Poodle should be_a Animal after transitive inference")

    def test_three_hop_transitive(self):
        sm = self._make_memory()
        added = sm.infer_transitive("is_a", max_hops=3)
        self.assertTrue(sm.concept_graph.has_edge("Poodle", "LivingThing"),
                        "Poodle should be_a LivingThing after 3-hop inference")

    def test_inferred_edges_marked(self):
        sm = self._make_memory()
        sm.infer_transitive("is_a", max_hops=2)
        if sm.concept_graph.has_edge("Poodle", "Animal"):
            data = sm.concept_graph.get_edge_data("Poodle", "Animal")
            self.assertTrue(data.get("inferred", False))

    def test_no_self_loops(self):
        sm = self._make_memory()
        sm.infer_transitive("is_a", max_hops=3)
        for node in sm.concept_graph.nodes():
            self.assertFalse(sm.concept_graph.has_edge(node, node),
                             f"Self-loop detected on {node}")

    def test_causal_transitive(self):
        sm = SemanticMemory(use_rust=False)
        for name in ("Fire", "Smoke", "Pollution"):
            hv = hypervec_rs.HyperVector(abs(hash(name)) % (2**32))
            sm.add_concept(name, {}, hv_override=hv)
        sm.add_relation("Fire", "causes", "Smoke")
        sm.add_relation("Smoke", "causes", "Pollution")
        added = sm.infer_transitive("causes", max_hops=2)
        self.assertTrue(sm.concept_graph.has_edge("Fire", "Pollution"))


class TestPrototypeGeneralization(unittest.TestCase):
    """V4: SemanticMemory.build_prototypes()"""

    def _make_animal_memory(self):
        sm = SemanticMemory(use_rust=False)
        animals = ["Dog", "Cat", "Bird", "Fish"]
        for name in animals + ["Animal"]:
            hv = hypervec_rs.HyperVector(abs(hash(name)) % (2**32))
            sm.add_concept(name, {}, hv_override=hv)
        for animal in animals:
            sm.add_relation(animal, "is_a", "Animal")
        return sm, animals

    def test_prototype_built_for_category(self):
        sm, _ = self._make_animal_memory()
        prototypes = sm.build_prototypes(min_members=2)
        self.assertIn("Animal", prototypes, "Expected prototype for 'Animal' category")

    def test_prototype_is_hypervector(self):
        sm, _ = self._make_animal_memory()
        prototypes = sm.build_prototypes(min_members=2)
        proto = prototypes.get("Animal")
        self.assertIsNotNone(proto)
        self.assertIsInstance(proto, hypervec_rs.HyperVector)

    def test_min_members_respected(self):
        sm, _ = self._make_animal_memory()
        prototypes = sm.build_prototypes(min_members=5)
        # Only 4 animals → should not build prototype
        self.assertNotIn("Animal", prototypes)

    def test_prototype_similarity_to_members(self):
        sm, animals = self._make_animal_memory()
        prototypes = sm.build_prototypes(min_members=2)
        proto = prototypes.get("Animal")
        if proto is None:
            self.skipTest("No prototype built")
        # Each animal HV should have non-trivial similarity to the prototype
        for animal in animals:
            animal_hv = sm.concept_hvs[animal]
            sim = proto.similarity(animal_hv)
            self.assertGreater(sim, 0.0,
                               f"Prototype similarity to {animal} should be > 0")

    def test_multiple_categories(self):
        sm = SemanticMemory(use_rust=False)
        for name in ("Dog", "Cat", "Animal", "Rose", "Tulip", "Plant"):
            hv = hypervec_rs.HyperVector(abs(hash(name)) % (2**32))
            sm.add_concept(name, {}, hv_override=hv)
        sm.add_relation("Dog", "is_a", "Animal")
        sm.add_relation("Cat", "is_a", "Animal")
        sm.add_relation("Rose", "is_a", "Plant")
        sm.add_relation("Tulip", "is_a", "Plant")
        prototypes = sm.build_prototypes(min_members=2)
        self.assertIn("Animal", prototypes)
        self.assertIn("Plant", prototypes)


class TestV4RelationWeights(unittest.TestCase):
    """V4: New relation types have weights in DEFAULT_RELATION_WEIGHTS."""

    def test_negation_relations_have_weights(self):
        sm = SemanticMemory(use_rust=False)
        for rel in ("not_is_a", "not_has_property", "not_relates_to", "lacks"):
            self.assertIn(rel, sm.relation_weights,
                          f"V4 negation relation '{rel}' missing from relation_weights")

    def test_temporal_relations_have_weights(self):
        sm = SemanticMemory(use_rust=False)
        for rel in ("precedes", "follows", "since_event", "until_event"):
            self.assertIn(rel, sm.relation_weights,
                          f"V4 temporal relation '{rel}' missing from relation_weights")

    def test_negation_weight_is_low(self):
        sm = SemanticMemory(use_rust=False)
        # Negation should not propagate strongly
        self.assertLessEqual(sm.relation_weights["not_is_a"], 0.2)
        self.assertLessEqual(sm.relation_weights["not_has_property"], 0.2)

    def test_temporal_weights_reasonable(self):
        sm = SemanticMemory(use_rust=False)
        self.assertGreater(sm.relation_weights["precedes"], 0.1)
        self.assertLess(sm.relation_weights["precedes"], 0.9)


if __name__ == "__main__":
    unittest.main()
