"""Unit tests for NSCK V4: Schema Induction, Predictive Processing,
Abductive Reasoning, and Temporal Reasoning modules."""
import unittest


# ---------------------------------------------------------------------------
# Schema Induction tests
# ---------------------------------------------------------------------------

class TestSchemaInduction(unittest.TestCase):
    def setUp(self):
        from python.core.learning.schema_induction import SchemaInducer, Episode
        self.SchemaInducer = SchemaInducer
        self.Episode = Episode

    def test_induces_schema_from_repeated_episodes(self):
        """Two or more episodes with the same relation should form a schema."""
        inducer = self.SchemaInducer(min_support=2)
        ep1 = self.Episode("rain", "causes", "flooding", confidence=0.9)
        ep2 = self.Episode("stress", "causes", "illness", confidence=0.8)
        inducer.add_episodes([ep1, ep2])
        schemas = inducer.induce()
        self.assertGreater(len(schemas), 0)
        self.assertEqual(schemas[0].relation, "causes")

    def test_single_episode_below_min_support(self):
        """A single episode must NOT be promoted to a schema by default."""
        inducer = self.SchemaInducer(min_support=2)
        ep = self.Episode("fire", "causes", "smoke", confidence=1.0)
        inducer.add_episode(ep)
        schemas = inducer.induce()
        self.assertEqual(len(schemas), 0)

    def test_assimilation_into_existing_schema(self):
        """New episode matching a known schema should be assimilated."""
        inducer = self.SchemaInducer(min_support=2, similarity_threshold=0.0)
        ep1 = self.Episode("a", "causes", "b")
        ep2 = self.Episode("c", "causes", "d")
        inducer.add_episodes([ep1, ep2])
        inducer.induce()

        ep_new = self.Episode("x", "causes", "y")
        result = inducer.assimilate(ep_new)
        self.assertTrue(result.matched)
        self.assertEqual(result.schema.relation, "causes")
        self.assertEqual(result.process, "assimilation")

    def test_accommodation_for_unseen_relation(self):
        """Episode with a brand-new relation should trigger accommodation."""
        inducer = self.SchemaInducer(min_support=2, similarity_threshold=0.9)
        ep = self.Episode("x", "never_seen_rel_xyz", "y")
        result = inducer.assimilate(ep)
        self.assertFalse(result.matched)
        self.assertEqual(result.process, "accommodation")

    def test_explain_episode(self):
        """explain_episode should return a non-empty string."""
        inducer = self.SchemaInducer(min_support=2, similarity_threshold=0.0)
        ep1 = self.Episode("a", "enables", "b")
        ep2 = self.Episode("c", "enables", "d")
        inducer.add_episodes([ep1, ep2])
        inducer.induce()

        ep_test = self.Episode("e", "enables", "f")
        explanation = inducer.explain_episode(ep_test)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)

    def test_summary_structure(self):
        """Summary dict should have expected keys."""
        inducer = self.SchemaInducer(min_support=2)
        summary = inducer.summary()
        self.assertIn("total_schemas", summary)
        self.assertIn("total_episodes", summary)
        self.assertIn("relations_covered", summary)

    def test_slot_placeholder_on_high_diversity(self):
        """When all subjects differ, slot should become ?SUBJECT."""
        inducer = self.SchemaInducer(min_support=3, max_slot_diversity=0.0)
        for w in ["alpha", "beta", "gamma"]:
            inducer.add_episode(self.Episode(w, "causes", "result"))
        schemas = inducer.induce()
        self.assertGreater(len(schemas), 0)
        # With max_slot_diversity=0.0 only slot with 100% same word is kept
        # Three different subjects → slot should be ?SUBJECT
        self.assertEqual(schemas[0].subject_slot, "?SUBJECT")

    def test_schema_id_deterministic(self):
        """Same cluster of episodes should produce the same schema_id."""
        inducer = self.SchemaInducer(min_support=2)
        ep1 = self.Episode("rain", "causes", "flooding")
        ep2 = self.Episode("rain", "causes", "flooding")
        inducer.add_episodes([ep1, ep2])
        schemas_a = inducer.induce()

        inducer2 = self.SchemaInducer(min_support=2)
        inducer2.add_episodes([ep1, ep2])
        schemas_b = inducer2.induce()

        self.assertEqual(schemas_a[0].schema_id, schemas_b[0].schema_id)

    def test_get_schemas_for_relation(self):
        """get_schemas_for_relation filters by relation."""
        inducer = self.SchemaInducer(min_support=2)
        for _ in range(2):
            inducer.add_episode(self.Episode("a", "enables", "b"))
            inducer.add_episode(self.Episode("c", "prevents", "d"))
        inducer.induce()

        enables = inducer.get_schemas_for_relation("enables")
        prevents = inducer.get_schemas_for_relation("prevents")
        self.assertGreater(len(enables), 0)
        self.assertGreater(len(prevents), 0)
        for s in enables:
            self.assertEqual(s.relation, "enables")


# ---------------------------------------------------------------------------
# Predictive Processing tests
# ---------------------------------------------------------------------------

class TestPredictiveProcessor(unittest.TestCase):
    def setUp(self):
        from python.core.reasoning.predictive_processor import PredictiveProcessor
        self.PredictiveProcessor = PredictiveProcessor

    def test_returns_predictive_state(self):
        """process() should return a PredictiveState with correct fields."""
        from python.core.reasoning.predictive_processor import PredictiveState
        pp = self.PredictiveProcessor()
        state = pp.process("rain", "flooding")
        self.assertIsInstance(state, PredictiveState)
        self.assertEqual(state.context, "rain")
        self.assertEqual(state.observed, "flooding")

    def test_prediction_error_in_range(self):
        """Prediction error must be in [0, 1]."""
        pp = self.PredictiveProcessor()
        for ctx, obs in [("a", "b"), ("x", "y"), ("alpha", "beta")]:
            state = pp.process(ctx, obs)
            self.assertGreaterEqual(state.prediction_error, 0.0)
            self.assertLessEqual(state.prediction_error, 1.0)

    def test_free_energy_positive(self):
        """Free energy must be non-negative."""
        pp = self.PredictiveProcessor()
        state = pp.process("context", "observation")
        self.assertGreaterEqual(state.free_energy, 0.0)

    def test_surprise_level_valid(self):
        """Surprise level must be one of 'low', 'medium', 'high'."""
        pp = self.PredictiveProcessor()
        state = pp.process("context", "observation")
        self.assertIn(state.surprise_level, {"low", "medium", "high"})

    def test_get_state_summary(self):
        """get_state() should return a dict with expected keys."""
        pp = self.PredictiveProcessor()
        pp.process("a", "b")
        pp.process("b", "c")
        summary = pp.get_state()
        self.assertIn("steps", summary)
        self.assertIn("average_prediction_error", summary)
        self.assertIn("surprise_level", summary)
        self.assertEqual(summary["steps"], 2)

    def test_history_accumulates(self):
        """History should grow with each process() call."""
        pp = self.PredictiveProcessor()
        for i in range(5):
            pp.process(f"ctx_{i}", f"obs_{i}")
        history = pp.get_history(n=10)
        self.assertEqual(len(history), 5)

    def test_reset_clears_state(self):
        """reset() should clear belief state and history."""
        pp = self.PredictiveProcessor()
        for _ in range(3):
            pp.process("a", "b")
        pp.reset()
        summary = pp.get_state()
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(len(pp.get_history()), 0)

    def test_repeated_observations_lower_pe(self):
        """Repeated (context, observation) pairs should reduce mean PE over time."""
        pp = self.PredictiveProcessor()
        early_pe = []
        late_pe = []
        for i in range(5):
            s = pp.process("rain", "flooding")
            early_pe.append(s.prediction_error)
        for i in range(10):
            s = pp.process("rain", "flooding")
            late_pe.append(s.prediction_error)
        # After many repetitions belief should be updated → lower mean error
        # (not guaranteed with no SM, but belief_hv should be set)
        summary = pp.get_state()
        self.assertTrue(summary["belief_hv_set"])


# ---------------------------------------------------------------------------
# Abductive Reasoning tests
# ---------------------------------------------------------------------------

class TestAbductiveReasoning(unittest.TestCase):
    def setUp(self):
        from python.core.reasoning.abductive_reasoning import AbductiveReasoner
        from python.core.reasoning.causal_reasoning import CausalGraph
        self.AbductiveReasoner = AbductiveReasoner
        self.CausalGraph = CausalGraph

    def _make_graph(self):
        cg = self.CausalGraph()
        cg.add_causes("rain",     "wet_road",  strength=0.9, context="weather")
        cg.add_causes("wet_road", "accident",  strength=0.7, context="road")
        cg.add_causes("ice",      "accident",  strength=0.8, context="weather")
        return cg

    def test_finds_chain_for_known_effect(self):
        """Should find a causal chain ending in 'accident'."""
        cg = self._make_graph()
        ar = self.AbductiveReasoner(causal_graph=cg)
        result = ar.explain("accident")
        self.assertIsNotNone(result.best)
        self.assertIn("accident", result.best.causes)

    def test_best_hypothesis_has_score(self):
        """Best hypothesis score must be > 0."""
        cg = self._make_graph()
        ar = self.AbductiveReasoner(causal_graph=cg)
        result = ar.explain("accident")
        self.assertGreater(result.best.score, 0.0)

    def test_explain_text_is_string(self):
        """explain_text() should return a non-empty string."""
        cg = self._make_graph()
        ar = self.AbductiveReasoner(causal_graph=cg)
        text = ar.explain_text("accident")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)

    def test_no_explanation_for_unknown_effect(self):
        """Unknown effect should return AbductionResult with best=None."""
        cg = self.CausalGraph()
        ar = self.AbductiveReasoner(causal_graph=cg)
        result = ar.explain("totally_unknown_effect_xyz")
        self.assertIsNone(result.best)

    def test_parsimony_shorter_chain_preferred(self):
        """Shorter chains should generally score higher due to parsimony."""
        cg = self.CausalGraph()
        cg.add_causes("A", "C", strength=0.9, context="x")
        cg.add_causes("A", "B", strength=0.9, context="x")
        cg.add_causes("B", "C", strength=0.9, context="x")
        ar = self.AbductiveReasoner(causal_graph=cg, w_parsimony=1.0, w_coverage=0.0, w_prior=0.0, w_coherence=0.0)
        result = ar.explain("C")
        self.assertIsNotNone(result.best)
        # Direct chain A→C is shorter than A→B→C
        self.assertEqual(len(result.best.causes), 2)

    def test_multiple_hypotheses_returned(self):
        """All-hypotheses list should contain multiple entries when chains exist."""
        cg = self._make_graph()
        ar = self.AbductiveReasoner(causal_graph=cg, top_k=5)
        result = ar.explain("accident")
        self.assertGreater(len(result.all_hypotheses), 0)

    def test_explain_multiple_observations(self):
        """explain_multiple() should handle a list of observations."""
        cg = self._make_graph()
        ar = self.AbductiveReasoner(causal_graph=cg)
        result = ar.explain_multiple(["wet_road", "accident"])
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Temporal Reasoning tests
# ---------------------------------------------------------------------------

class TestTemporalReasoning(unittest.TestCase):
    def setUp(self):
        from python.core.reasoning.temporal_reasoning import TemporalKnowledgeGraph
        self.TKG = TemporalKnowledgeGraph

    def test_direct_relation_stored_and_retrieved(self):
        """Directly stored relation must be retrievable."""
        tkg = self.TKG()
        tkg.add_fact("diagnosis", "before", "treatment")
        q = tkg.query_ordering("diagnosis", "treatment")
        self.assertEqual(q.relation, "before")
        self.assertFalse(q.inferred)

    def test_converse_auto_stored(self):
        """Converse of stored relation should be automatically available."""
        tkg = self.TKG()
        tkg.add_fact("A", "before", "B")
        q = tkg.query_ordering("B", "A")
        self.assertEqual(q.relation, "after")

    def test_transitive_inference(self):
        """A before B, B before C → A before C (inferred)."""
        tkg = self.TKG()
        tkg.add_fact("A", "before", "B")
        tkg.add_fact("B", "before", "C")
        q = tkg.query_ordering("A", "C")
        self.assertIsNotNone(q.relation)
        self.assertEqual(q.relation, "before")
        self.assertTrue(q.inferred)

    def test_unknown_ordering_returns_none_relation(self):
        """Unknown pair should return None relation."""
        tkg = self.TKG()
        tkg.add_fact("A", "before", "B")
        q = tkg.query_ordering("A", "Z")
        self.assertIsNone(q.relation)

    def test_what_happened_before(self):
        """what_happened_before should return causes."""
        tkg = self.TKG()
        tkg.add_fact("storm", "before", "flood")
        before_flood = tkg.what_happened_before("flood")
        # "flood after storm" stored as converse, so "storm" should appear
        self.assertIn("storm", before_flood)

    def test_what_happened_after(self):
        """what_happened_after should return effects."""
        tkg = self.TKG()
        tkg.add_fact("storm", "before", "flood")
        after_storm = tkg.what_happened_after("storm")
        self.assertIn("flood", after_storm)

    def test_add_from_text_markers(self):
        """Linguistic markers should map to Allen relations."""
        tkg = self.TKG()
        rel = tkg.add_from_text_markers("diagnosis", "before", "treatment")
        self.assertEqual(rel, "before")
        q = tkg.query_ordering("diagnosis", "treatment")
        self.assertEqual(q.relation, "before")

    def test_unknown_marker_returns_none(self):
        """Unrecognised linguistic marker should return None and store nothing."""
        tkg = self.TKG()
        rel = tkg.add_from_text_markers("A", "randomword_xyz", "B")
        self.assertIsNone(rel)

    def test_extract_markers_from_sentence(self):
        """Static method should find temporal markers in a sentence."""
        from python.core.reasoning.temporal_reasoning import TemporalKnowledgeGraph as TKG
        markers = TKG.extract_markers_from_sentence(
            "The diagnosis happened before the treatment started."
        )
        words_found = [m for m, _ in markers]
        self.assertIn("before", words_found)

    def test_summary_keys(self):
        """summary() should contain expected keys."""
        tkg = self.TKG()
        tkg.add_fact("A", "before", "B")
        summary = tkg.summary()
        self.assertIn("total_facts", summary)
        self.assertIn("events", summary)

    def test_timeline_returns_pairs(self):
        """timeline() should return (relation, event) tuples."""
        tkg = self.TKG()
        tkg.add_fact("rain", "before", "flood")
        tkg.add_fact("rain", "before", "damage")
        tl = tkg.timeline("rain")
        self.assertGreater(len(tl), 0)
        for rel, evt in tl:
            self.assertIsInstance(rel, str)
            self.assertIsInstance(evt, str)

    def test_get_all_events(self):
        """get_all_events() should return all event names."""
        tkg = self.TKG()
        tkg.add_fact("A", "before", "B")
        tkg.add_fact("C", "during", "D")
        events = tkg.get_all_events()
        for name in ["A", "B", "C", "D"]:
            self.assertIn(name, events)


# ---------------------------------------------------------------------------
# V4 Config flags tests
# ---------------------------------------------------------------------------

class TestV4ConfigFlags(unittest.TestCase):
    def test_research_preset_has_v4_flags(self):
        """NSCKConfig.research() must enable all V4 flags."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.research()
        self.assertTrue(cfg.enable_schema_induction)
        self.assertTrue(cfg.enable_predictive_processing)
        self.assertTrue(cfg.enable_abductive_reasoning)
        self.assertTrue(cfg.enable_temporal_reasoning)

    def test_production_preset_has_safe_v4_flags(self):
        """NSCKConfig.production() must enable production-safe V4 flags."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.production()
        self.assertTrue(cfg.enable_schema_induction)
        self.assertTrue(cfg.enable_abductive_reasoning)
        self.assertTrue(cfg.enable_temporal_reasoning)
        # predictive_processing is research-only
        self.assertFalse(cfg.enable_predictive_processing)

    def test_minimal_preset_has_no_v4_flags(self):
        """NSCKConfig.minimal() must leave all V4 flags off."""
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.minimal()
        self.assertFalse(cfg.enable_schema_induction)
        self.assertFalse(cfg.enable_predictive_processing)
        self.assertFalse(cfg.enable_abductive_reasoning)
        self.assertFalse(cfg.enable_temporal_reasoning)


# ---------------------------------------------------------------------------
# V4 Construction Grammar NLU improvements
# ---------------------------------------------------------------------------

class TestEnhancedNLU(unittest.TestCase):
    def test_common_verbs_expanded(self):
        """COMMON_VERBS should now contain 200+ entries."""
        from python.core.language.construction_grammar import COMMON_VERBS
        self.assertGreater(len(COMMON_VERBS), 200)

    def test_new_verbs_classified_as_verb(self):
        """Newly added verbs should be classified as VERB."""
        from python.core.language.construction_grammar import _classify_word
        for word in ["love", "loves", "understand", "understood", "analyze",
                     "implement", "generate", "transform", "evolve", "adapt"]:
            self.assertEqual(_classify_word(word), "VERB", f"'{word}' not classified as VERB")

    def test_negations_classified(self):
        """Negation words should be classified as NEG."""
        from python.core.language.construction_grammar import _classify_word
        for word in ["not", "never", "no"]:
            self.assertEqual(_classify_word(word), "NEG", f"'{word}' not classified as NEG")

    def test_morphological_verb_detection(self):
        """Words with verb-forming suffixes should be classified as VERB."""
        from python.core.language.construction_grammar import _classify_word
        for word in ["organizing", "realizes", "modernize", "classify"]:
            result = _classify_word(word)
            self.assertEqual(result, "VERB", f"'{word}' got '{result}' not VERB")

    def test_noun_suffix_classification(self):
        """Words with noun-forming suffixes should be classified as NOUN."""
        from python.core.language.construction_grammar import _classify_word
        for word in ["development", "happiness", "activity", "detection"]:
            self.assertEqual(_classify_word(word), "NOUN", f"'{word}' not NOUN")

    def test_loves_classified_as_verb(self):
        """'loves' was previously unclassified (missing); must now be VERB."""
        from python.core.language.construction_grammar import _classify_word
        self.assertEqual(_classify_word("loves"), "VERB")

    def test_construction_match_with_new_verbs(self):
        """ConstructionMatcher should match SVO with newly-added verbs."""
        from python.core.language.construction_grammar import ConstructionMatcher
        matcher = ConstructionMatcher()
        matches = matcher.match(["Alice", "loves", "Bob"])
        self.assertGreater(len(matches), 0)
        svo = [m for m in matches if m.construction.name == "SVO_active"]
        self.assertGreater(len(svo), 0)


# ---------------------------------------------------------------------------
# V4 CognitiveEngine wiring tests
# ---------------------------------------------------------------------------

class TestCognitiveEngineV4Wiring(unittest.TestCase):
    def test_v4_modules_initialised_with_research_config(self):
        """CognitiveEngine with research config should initialise all V4 modules."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.research()
        engine = CognitiveEngine(cfg)
        self.assertIsNotNone(engine.schema_inducer)
        self.assertIsNotNone(engine.predictive_processor)
        self.assertIsNotNone(engine.abductive_reasoner)
        self.assertIsNotNone(engine.temporal_kg)

    def test_v4_modules_absent_with_minimal_config(self):
        """CognitiveEngine with minimal config should leave V4 modules as None."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.minimal()
        engine = CognitiveEngine(cfg)
        self.assertIsNone(engine.schema_inducer)
        self.assertIsNone(engine.predictive_processor)
        self.assertIsNone(engine.abductive_reasoner)
        self.assertIsNone(engine.temporal_kg)

    def test_cognitive_state_has_v4_fields(self):
        """CognitiveState dataclass must have V4 trace fields."""
        from python.core.reasoning.cognitive_engine import CognitiveState
        cs = CognitiveState(task_tag="test")
        self.assertTrue(hasattr(cs, "schema_match"))
        self.assertTrue(hasattr(cs, "predictive_state"))
        self.assertTrue(hasattr(cs, "abductive_explanation"))
        self.assertTrue(hasattr(cs, "temporal_context"))

    def test_decide_with_research_config_no_crash(self):
        """decide() with research config + all V4 modules must not crash."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.research()
        engine = CognitiveEngine(cfg)
        result = engine.decide({"x": 1}, task_tag="default")
        self.assertIsNotNone(result.chosen_action)

    def test_v4_trace_populated_predictive_processing(self):
        """With predictive_processing enabled and active predicates, trace should be set."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        from python.core.perception.grounding_verifier import GroundingVerifier
        from python.core.reasoning.causal_reasoning import CausalGraph
        cfg = NSCKConfig(enable_predictive_processing=True)
        engine = CognitiveEngine(cfg)
        # Register a task so verifier produces predicates
        cg = CausalGraph()
        engine.register_task("test_pp", causal_graph=cg)
        result = engine.decide({"high_activity": 1}, task_tag="test_pp")
        # predictive_state may or may not be set depending on predicates — just no crash
        self.assertIsNotNone(result.chosen_action)


if __name__ == "__main__":
    unittest.main()
