"""
Tests for V4 NSCK improvements:
- Construction grammar fixes (gerunds, expanded COMMON_VERBS, new constructions)
- Multi-hop transitive inference in SemanticMemory
- PMILearner
- PredictiveCodingLayer
- ActiveInferencePlanner
- DistributionalCodebook online learning + expanded corpus
"""
import pytest


# ---------------------------------------------------------------------------
# Construction Grammar — word classification
# ---------------------------------------------------------------------------

class TestWordClassification:
    """Verify _classify_word() correctly handles edge cases."""

    def setup_method(self):
        from python.core.language.construction_grammar import _classify_word
        self._cls = _classify_word

    def test_gerund_nouns_not_verb(self):
        """Words ending in -ing that are nouns must not be classified as VERB."""
        for word in ("flooding", "building", "morning", "setting", "training"):
            assert self._cls(word) == "NOUN", f"{word!r} should be NOUN"

    def test_common_verbs_classified_as_verb(self):
        """Verbs in the COMMON_VERBS set should always be VERB."""
        for word in ("loves", "wants", "runs", "causes", "believes", "thinks"):
            assert self._cls(word) == "VERB", f"{word!r} should be VERB"

    def test_nominal_suffixes_are_noun(self):
        """Words with -tion/-ness/-ment/-ity suffixes should be NOUN."""
        for word in ("information", "happiness", "development", "electricity"):
            assert self._cls(word) == "NOUN", f"{word!r} should be NOUN"

    def test_adjectival_suffixes(self):
        """Words with -ful/-less/-ous/-able suffixes should be ADJ."""
        for word in ("beautiful", "careless", "dangerous", "capable", "flexible"):
            assert self._cls(word) == "ADJ", f"{word!r} should be ADJ"

    def test_negators_classified_as_neg(self):
        """Negation particles should be NEG — checked before COMMON_VERBS."""
        # These must be NEG (not VERB), confirming NEG is higher priority
        for word in ("not", "no", "never"):
            assert self._cls(word) == "NEG", f"{word!r} should be NEG"

    def test_ambiguous_nouns_not_adj(self):
        """Nouns ending in -al/-ary/-ic should not be classified as ADJ."""
        for word in ("mammal", "animal", "signal", "canal"):
            assert self._cls(word) == "NOUN", f"{word!r} should be NOUN (not ADJ)"

    def test_chemistry_nouns(self):
        """Chemical element/compound names should be NOUN."""
        for word in ("oxygen", "hydrogen", "carbon", "electron", "nitrogen"):
            assert self._cls(word) == "NOUN", f"{word!r} should be NOUN"

    def test_derivational_verb_suffixes(self):
        """Words with -ize/-ise/-ate/-ify suffixes should be VERB."""
        for word in ("summarize", "calculate", "magnify", "realise"):
            assert self._cls(word) == "VERB", f"{word!r} should be VERB"


# ---------------------------------------------------------------------------
# Construction Grammar — matching
# ---------------------------------------------------------------------------

class TestConstructionMatcher:
    """Verify ConstructionMatcher handles key sentence patterns."""

    def setup_method(self):
        from python.core.language.construction_grammar import ConstructionMatcher
        self.cm = ConstructionMatcher()

    def _match_first(self, words):
        matches = self.cm.match(words)
        return matches[0] if matches else None

    def test_causative_with_gerund_object(self):
        """Rain causes flooding — 'flooding' must be NOUN object, not VERB."""
        m = self._match_first(["Rain", "causes", "flooding"])
        assert m is not None
        assert m.construction.relation == "causes"
        assert m.role_fillers["cause"] == "Rain"
        assert m.role_fillers["effect"] == "flooding"

    def test_svo_with_arbitrary_verb(self):
        """Alice loves Bob — 'loves' is in expanded COMMON_VERBS."""
        m = self._match_first(["Alice", "loves", "Bob"])
        assert m is not None
        assert m.role_fillers["subject"] == "Alice"
        assert m.role_fillers["object"] == "Bob"

    def test_copular_is_a_with_article(self):
        """'The cat is a mammal' → copular_is_a."""
        m = self._match_first(["The", "cat", "is", "a", "mammal"])
        assert m is not None
        assert m.construction.name == "copular_is_a"
        assert m.role_fillers["attribute"] == "mammal"

    def test_copular_is_adj(self):
        """'Water is helpful' (clear -ful ADJ suffix) → has_property."""
        m = self._match_first(["Water", "is", "helpful"])
        assert m is not None
        assert m.construction.relation == "has_property"

    def test_copular_is_adj_essential(self):
        """'Water is essential' (-ial suffix → ADJ) → has_property."""
        m = self._match_first(["Water", "is", "essential"])
        assert m is not None
        assert m.construction.relation == "has_property"

    def test_negation_cannot_vi(self):
        """'Cat cannot fly' → negation."""
        m = self._match_first(["Cat", "cannot", "fly"])
        assert m is not None
        assert "cannot" in m.construction.name.lower() or "neg" in m.construction.relation.lower()

    def test_negation_is_not_a(self):
        """'Cat is not a dog' → is_not."""
        m = self._match_first(["Cat", "is", "not", "a", "dog"])
        assert m is not None
        assert m.construction.relation == "is_not"
        assert m.role_fillers["subject"] == "Cat"
        assert m.role_fillers["attribute"] == "dog"

    def test_becomes(self):
        """'X becomes Y' → becomes relation."""
        m = self._match_first(["Dog", "becomes", "wolf"])
        assert m is not None
        assert m.construction.relation == "becomes"

    def test_wants_to_vi(self):
        """'Alice wants to learn' → wants_to_do."""
        m = self._match_first(["Alice", "wants", "to", "learn"])
        assert m is not None
        assert "wants" in m.construction.relation.lower()

    def test_depends_on(self):
        """'Fire depends on oxygen' → depends_on."""
        m = self._match_first(["Fire", "depends", "on", "oxygen"])
        assert m is not None
        assert m.construction.relation == "depends_on"
        assert m.role_fillers["dependency"] == "oxygen"

    def test_leads_to(self):
        """'Stress leads to illness' → leads_to."""
        m = self._match_first(["Stress", "leads", "to", "illness"])
        assert m is not None
        assert m.construction.relation == "leads_to"

    def test_construction_count(self):
        """At least 60 constructions loaded (V4 expands from 31)."""
        assert len(self.cm.constructions) >= 60


# ---------------------------------------------------------------------------
# SemanticMemory — multi-hop reasoning
# ---------------------------------------------------------------------------

class TestSemanticMemoryTransitiveInference:

    def setup_method(self):
        from python.core.memory.semantic_memory import SemanticMemory
        self.sm = SemanticMemory()
        # Taxonomy
        for c in ["dog","mammal","animal","living_thing",
                  "warm_blooded","breathes","eats"]:
            self.sm.add_concept(c, {})
        self.sm.add_relation("dog", "is_a", "mammal")
        self.sm.add_relation("mammal", "is_a", "animal")
        self.sm.add_relation("animal", "is_a", "living_thing")
        self.sm.add_relation("mammal", "has_property", "warm_blooded")
        self.sm.add_relation("animal", "has_property", "breathes")
        self.sm.add_relation("living_thing", "has_property", "eats")
        # Causal chain
        for c in ["virus","infection","disease","organ_failure","death"]:
            self.sm.add_concept(c, {})
        self.sm.add_relation("virus", "causes", "infection")
        self.sm.add_relation("infection", "leads_to", "disease")
        self.sm.add_relation("disease", "results_in", "organ_failure")
        self.sm.add_relation("organ_failure", "causes", "death")

    def test_infer_transitive_is_a(self):
        result = self.sm.infer_transitive("dog", "is_a")
        names = [r[0] for r in result]
        assert "mammal" in names
        assert "animal" in names
        assert "living_thing" in names

    def test_infer_transitive_respects_max_hops(self):
        result = self.sm.infer_transitive("dog", "is_a", max_hops=1)
        assert len(result) == 1
        assert result[0] == ("mammal", 1)

    def test_infer_transitive_unknown_concept(self):
        result = self.sm.infer_transitive("unicorn", "is_a")
        assert result == []

    def test_infer_inherited_properties(self):
        props = self.sm.infer_inherited_properties("dog")
        assert "warm_blooded" in props
        assert "breathes" in props
        assert "eats" in props

    def test_find_causal_chain_exists(self):
        chain = self.sm.find_causal_chain("virus", "death")
        assert chain is not None
        assert chain[0] == "virus"
        assert chain[-1] == "death"
        assert len(chain) == 5  # virus→infection→disease→organ_failure→death

    def test_find_causal_chain_no_path(self):
        chain = self.sm.find_causal_chain("dog", "death")
        assert chain is None

    def test_find_causal_chain_unknown_concept(self):
        chain = self.sm.find_causal_chain("unicorn", "death")
        assert chain is None

    def test_hop_distances_are_correct(self):
        result = dict(self.sm.infer_transitive("dog", "is_a"))
        assert result["mammal"] == 1
        assert result["animal"] == 2
        assert result["living_thing"] == 3


# ---------------------------------------------------------------------------
# PMILearner
# ---------------------------------------------------------------------------

class TestPMILearner:

    def setup_method(self):
        from python.core.learning.pmi_learner import PMILearner
        self.pmi = PMILearner()
        self.pmi.observe("cat", "chases", "mouse", 10)
        self.pmi.observe("dog", "chases", "cat", 5)
        self.pmi.observe("cat", "eats", "fish", 8)
        self.pmi.observe("dog", "eats", "bone", 7)

    def test_high_weight_for_seen_triplet(self):
        w = self.pmi.relation_weight("cat", "chases", "mouse")
        assert w > 0.7, f"Expected high PMI weight for seen pair, got {w}"

    def test_lower_weight_for_unseen_triplet(self):
        w_seen = self.pmi.relation_weight("cat", "chases", "mouse")
        w_unseen = self.pmi.relation_weight("cat", "chases", "elephant")
        assert w_seen > w_unseen

    def test_top_relations(self):
        top = self.pmi.top_relations("cat", "chases")
        assert len(top) > 0
        assert top[0][0] == "mouse"

    def test_stats(self):
        s = self.pmi.stats()
        assert s["total_observations"] == 30
        assert s["unique_triplets"] == 4

    def test_weight_in_unit_interval(self):
        for args in [("cat","chases","mouse"), ("cat","eats","fish"),
                     ("dog","chases","cat"), ("dog","eats","bone")]:
            w = self.pmi.relation_weight(*args)
            assert 0.0 <= w <= 1.0, f"Weight {w} out of [0,1] for {args}"

    def test_observe_batch(self):
        from python.core.learning.pmi_learner import PMILearner
        pmi2 = PMILearner()
        pmi2.observe_batch([("cat","chases","mouse"), ("dog","chases","cat")])
        assert pmi2.stats()["unique_triplets"] == 2


# ---------------------------------------------------------------------------
# PredictiveCodingLayer
# ---------------------------------------------------------------------------

class TestPredictiveCodingLayer:

    def setup_method(self):
        from python.core.memory.semantic_memory import SemanticMemory
        from python.core.learning.predictive_coding import PredictiveCodingLayer
        import python.core.vsa.hypervec_shim as hvs

        self.sm = SemanticMemory()
        for c in ("cat", "dog", "mouse", "fish"):
            self.sm.add_concept(c, {})

        self.pc = PredictiveCodingLayer(self.sm, error_threshold=0.15)
        self.hvs = hvs

        self.cat_hv = hvs.HyperVector(hash("cat") % (2**32))
        self.surprise_hv = hvs.HyperVector(hash("SURPRISE_SEED_99999") % (2**32))

        # Seed a prediction for 'cat'
        self.pc.update_prediction("cat", self.cat_hv, learning_rate=1.0)

    def test_zero_error_for_known_prediction(self):
        err = self.pc.compute_error("cat", self.cat_hv)
        assert err < 0.01, f"Error should be near 0 for identical HVs, got {err}"

    def test_high_error_for_unexpected_hv(self):
        err = self.pc.compute_error("cat", self.surprise_hv)
        assert err > 0.1, f"Error should be non-trivial for different HVs, got {err}"

    def test_unknown_concept_returns_half(self):
        err = self.pc.compute_error("unicorn", self.cat_hv)
        assert err == 0.5

    def test_is_surprising(self):
        assert not self.pc.is_surprising("cat", self.cat_hv)
        # Surprise HV should exceed threshold
        assert self.pc.is_surprising("cat", self.surprise_hv)

    def test_update_prediction_changes_error(self):
        # Start with surprise HV as prediction
        self.pc.update_prediction("dog", self.surprise_hv, learning_rate=1.0)
        err_before = self.pc.compute_error("dog", self.cat_hv)
        # Update toward cat_hv
        for _ in range(10):
            self.pc.update_prediction("dog", self.cat_hv, learning_rate=0.5)
        err_after = self.pc.compute_error("dog", self.cat_hv)
        assert err_after < err_before

    def test_stats(self):
        s = self.pc.stats()
        assert s["concepts_tracked"] >= 1
        assert "mean_error" in s

    def test_surprise_score_positive(self):
        score = self.pc.surprise_score("cat", self.surprise_hv)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# ActiveInferencePlanner
# ---------------------------------------------------------------------------

class TestActiveInferencePlanner:

    def setup_method(self):
        from python.core.memory.semantic_memory import SemanticMemory
        from python.core.learning.active_inference import ActiveInferencePlanner
        import python.core.vsa.hypervec_shim as hvs

        self.sm = SemanticMemory()
        for c in ("cat", "dog", "mouse", "fish", "bird"):
            self.sm.add_concept(c, {})

        self.aip = ActiveInferencePlanner(self.sm)

    def test_select_returns_candidates(self):
        q = self.aip.select_next_query(["cat","dog","mouse"], k=2)
        assert len(q) == 2
        assert all(c in ("cat","dog","mouse") for c in q)

    def test_select_respects_k(self):
        q = self.aip.select_next_query(["cat","dog","mouse","fish","bird"], k=3)
        assert len(q) == 3

    def test_preference_influences_selection(self):
        """Greedy selection should prefer high-preference concept."""
        self.aip.set_preference("cat", 10.0)
        # With high preference weight (0.5) and exploration_bonus=0.2,
        # cat should be selected in greedy mode.
        results = []
        for _ in range(20):
            q = self.aip.select_next_query(
                ["cat","dog"], k=1, stochastic=False
            )
            results.extend(q)
        assert "cat" in results

    def test_stats(self):
        self.aip.select_next_query(["cat","dog"], k=1)
        s = self.aip.stats()
        assert s["concepts_visited"] >= 1
        assert s["total_queries"] >= 1

    def test_empty_candidates_returns_empty(self):
        q = self.aip.select_next_query([], k=3)
        assert q == []

    def test_set_preferences_batch(self):
        self.aip.set_preferences({"cat": 1.0, "dog": 0.5})
        assert self.aip.extrinsic_value("cat") == 1.0
        assert self.aip.extrinsic_value("dog") == 0.5


# ---------------------------------------------------------------------------
# DistributionalCodebook — online learning + expanded corpus
# ---------------------------------------------------------------------------

class TestDistributionalCodebook:

    def setup_method(self):
        from python.core.language.distributional_semantics import (
            DistributionalCodebook, BUILTIN_CORPUS
        )
        self.cb = DistributionalCodebook.build_default()
        self.BUILTIN_CORPUS = BUILTIN_CORPUS

    def test_expanded_corpus_size(self):
        """Corpus should have at least 150 sentences (V4 target: 200+)."""
        assert len(self.BUILTIN_CORPUS) >= 150

    def test_codebook_covers_common_words(self):
        """Key domain words should have HVs after build."""
        for word in ("gravity", "evolution", "entropy", "neurons", "probability"):
            hv = self.cb.get_hv(word)
            assert hv is not None, f"Word {word!r} missing from codebook"

    def test_similarity_returns_float(self):
        s = self.cb.similarity("cat", "dog")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_unknown_word_similarity_is_zero(self):
        s = self.cb.similarity("xyzzy_nonexistent", "cat")
        assert s == 0.0

    def test_online_observe_adds_word(self):
        """online_observe() should add a previously unknown word."""
        word = "quasar"
        assert self.cb.get_hv(word) is None
        self.cb.online_observe(["quasar", "is", "a", "distant", "galaxy"])
        assert self.cb.get_hv(word) is not None

    def test_online_observe_updates_existing(self):
        """online_observe() should update (bundle) existing entries."""
        import numpy as np
        cat_hv_before = np.array(self.cb.get_hv("cat").bits, dtype=np.float32)
        self.cb.online_observe(["cat", "and", "kitten", "are", "related"])
        cat_hv_after = np.array(self.cb.get_hv("cat").bits, dtype=np.float32)
        # HV bits should change after bundling new context
        assert not np.array_equal(cat_hv_before, cat_hv_after)
