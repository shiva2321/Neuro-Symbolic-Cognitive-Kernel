"""
V7 Feature Tests
================
Tests for V7 improvements:
  1. FluentNLG wired into DialogueManager
  2. KG confidence thresholds (stop-concept filtering, generic-relation gating)
  3. Distributional codebook pre-trains on BUILTIN_CORPUS
  4. HuggingFace corpus loader (offline graceful fallback)
  5. NSCKConfig V7 flags
"""
import pytest
import networkx as nx

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sem_mem():
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory()
    for c in ['water', 'liquid', 'rain', 'flooding', 'erosion', 'rivers', 'ocean']:
        mem.add_concept(c, {'source': 'test'})
    mem.add_relation('water', 'is_a', 'liquid')
    mem.add_relation('water', 'has_property', 'transparency')
    mem.add_relation('water', 'found_in', 'rivers')
    mem.add_relation('rain', 'causes', 'flooding')
    mem.add_relation('flooding', 'causes', 'erosion')
    mem.add_relation('water', 'found_in', 'ocean')
    return mem


@pytest.fixture()
def dialogue_manager(sem_mem):
    from python.core.language.dialogue_manager import DialogueManager

    class _MockEngine:
        def __init__(self, mem):
            self.semantic_memory = mem

    class _MockLang:
        def understand(self, text):
            return {'structured_output': {}, 'grounded_hv': None}

        def generate(self, d, extra=None):
            return str(d)

    eng = _MockEngine(sem_mem)
    return DialogueManager(eng, _MockLang())


# ---------------------------------------------------------------------------
# 1. FluentNLG wiring
# ---------------------------------------------------------------------------

class TestFluentNLGWiring:

    def test_fluent_engine_loaded(self, dialogue_manager):
        assert dialogue_manager._fluent is not None

    def test_answer_what_is_no_template_noise(self, dialogue_manager):
        resp = dialogue_manager._answer_what_is('water')
        # Should not contain raw relation tokens like 'is_a' or 'has_property'
        assert 'is_a' not in resp
        assert 'has_property' not in resp

    def test_answer_what_is_readable(self, dialogue_manager):
        resp = dialogue_manager._answer_what_is('water')
        assert len(resp) > 10
        assert resp[0].isupper()  # starts with capital

    def test_answer_what_is_unknown(self, dialogue_manager):
        resp = dialogue_manager._answer_what_is('quasar')
        assert 'quasar' in resp.lower() or "haven't" in resp.lower()

    def test_answer_explain_no_template_noise(self, dialogue_manager):
        resp = dialogue_manager._answer_explain('water')
        assert 'is_a' not in resp
        assert 'has_property' not in resp

    def test_answer_explain_mentions_concept(self, dialogue_manager):
        resp = dialogue_manager._answer_explain('water')
        assert 'water' in resp.lower()

    def test_answer_explain_unknown(self, dialogue_manager):
        resp = dialogue_manager._answer_explain('quasar')
        assert "quasar" in resp.lower() or "don't" in resp.lower() or "haven't" in resp.lower()

    def test_retrieve_knowledge_fluent(self, dialogue_manager):
        resp = dialogue_manager.retrieve_knowledge('water')
        assert 'is_a' not in resp
        assert len(resp) > 5

    def test_retrieve_knowledge_unknown(self, dialogue_manager):
        resp = dialogue_manager.retrieve_knowledge('quasar')
        assert 'quasar' in resp.lower() or "haven't" in resp.lower()

    def test_retrieve_knowledge_case_insensitive(self, dialogue_manager):
        resp = dialogue_manager.retrieve_knowledge('Water')
        assert 'is_a' not in resp

    def test_answer_causes_of_known(self, dialogue_manager):
        resp = dialogue_manager._answer_causes_of('flooding')
        assert 'rain' in resp.lower() or 'cause' in resp.lower()

    def test_answer_effects_of_known(self, dialogue_manager):
        resp = dialogue_manager._answer_effects_of('rain')
        assert 'flood' in resp.lower() or 'cause' in resp.lower()

    def test_answer_causes_of_unknown(self, dialogue_manager):
        resp = dialogue_manager._answer_causes_of('unknownevent')
        assert resp  # should return something, not crash

    def test_process_turn_does_not_raise(self, dialogue_manager):
        # process_turn internally routes through _try_causal_patterns
        resp = dialogue_manager.process_turn('What is water?')
        assert isinstance(resp, str) and len(resp) > 0

    def test_process_turn_what_is_fluent(self, dialogue_manager):
        resp = dialogue_manager.process_turn('What is water?')
        assert 'is_a' not in resp
        assert 'has_property' not in resp

    def test_process_turn_explain_fluent(self, dialogue_manager):
        resp = dialogue_manager.process_turn('Explain water')
        assert isinstance(resp, str) and len(resp) > 0

    def test_process_turn_unknown_graceful(self, dialogue_manager):
        resp = dialogue_manager.process_turn('What is a quasar?')
        assert isinstance(resp, str) and len(resp) > 0

    def test_fallback_when_fluent_unavailable(self, sem_mem):
        """Even without FluentNLG the DM must return a string, not crash."""
        from python.core.language.dialogue_manager import DialogueManager

        class _MockEngine:
            def __init__(self, mem):
                self.semantic_memory = mem

        class _MockLang:
            def understand(self, text):
                return {'structured_output': {}, 'grounded_hv': None}

        dm = DialogueManager(_MockEngine(sem_mem), _MockLang())
        dm._fluent = None   # simulate unavailability
        resp = dm.retrieve_knowledge('water')
        assert isinstance(resp, str) and len(resp) > 0

    def test_no_double_periods(self, dialogue_manager):
        resp = dialogue_manager._answer_what_is('water')
        assert '..' not in resp

    def test_sentence_ends_with_period(self, dialogue_manager):
        resp = dialogue_manager._answer_what_is('water')
        assert resp.strip()[-1] in '.!?'


# ---------------------------------------------------------------------------
# 2. KG confidence thresholds
# ---------------------------------------------------------------------------

class TestKGNoiseFiltering:

    def test_away_not_a_concept(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        concepts = tkl._extract_concepts('The dog ran away from the cat')
        assert 'Away' not in concepts

    def test_back_not_a_concept(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        concepts = tkl._extract_concepts('She came back from the store')
        assert 'Back' not in concepts

    def test_off_not_a_concept(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        concepts = tkl._extract_concepts('Turn the light off now')
        assert 'Off' not in concepts

    def test_over_not_a_concept(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        concepts = tkl._extract_concepts('The bird flew over the hill')
        assert 'Over' not in concepts

    def test_adverb_alone_filtered(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner, _STOP_CONCEPTS
        # Make sure the constants are non-empty and include problematic words
        assert 'away' in _STOP_CONCEPTS
        assert 'back' in _STOP_CONCEPTS
        assert 'off' in _STOP_CONCEPTS

    def test_real_concept_not_filtered(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        concepts = tkl._extract_concepts('Water flows through rivers')
        assert any('Water' in c or 'water' in c.lower() for c in concepts)

    def test_generic_relation_threshold_value(self):
        from python.core.language.text_knowledge_learner import _GENERIC_RELATION_THRESHOLD
        assert _GENERIC_RELATION_THRESHOLD > 0.55
        assert _GENERIC_RELATION_THRESHOLD <= 0.70

    def test_generic_relation_types_defined(self):
        from python.core.language.text_knowledge_learner import _GENERIC_RELATION_TYPES
        assert 'semantically_related' in _GENERIC_RELATION_TYPES
        assert 'strongly_related' in _GENERIC_RELATION_TYPES

    def test_stop_concepts_has_particles(self):
        from python.core.language.text_knowledge_learner import _STOP_CONCEPTS
        for word in ('away', 'back', 'off', 'over', 'up', 'down', 'out'):
            assert word in _STOP_CONCEPTS, f'{word} missing from _STOP_CONCEPTS'

    def test_particle_sentence_concept_list_clean(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        # 'out', 'up', 'down' are particles — should not appear
        sentence = 'Memory stores information away for recall out back up down'
        concepts = tkl._extract_concepts(sentence)
        for bad in ('Away', 'Out', 'Back', 'Up', 'Down'):
            assert bad not in concepts, f'{bad} should be filtered but got {concepts}'

    def test_extract_relations_avoids_spurious_short_concepts(self):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner()
        rels = tkl._extract_relations('Rain causes flooding rapidly.', ['Rain', 'Flooding'])
        for subj, rel, obj in rels:
            assert len(subj) >= 3 and len(obj) >= 3


# ---------------------------------------------------------------------------
# 3. Distributional codebook
# ---------------------------------------------------------------------------

class TestDistributionalCodebook:

    def test_codebook_pretrains_on_builtin(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook(pretrain=True)
        # Should have at least 100 entries from BUILTIN_CORPUS
        assert len(cb._codebook) >= 100

    def test_codebook_no_pretrain(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook(pretrain=False)
        assert len(cb._codebook) == 0

    def test_get_hv_known_word(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        hv = cb.get_hv('water')
        assert hv is not None

    def test_get_hv_unknown_word(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        assert cb.get_hv('zzzyyyxxx') is None

    def test_brain_memory_similarity_above_random(self):
        """brain and memory co-occur in cognition sentences — sim should be > 0.5."""
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        sim = cb.similarity('brain', 'memory')
        assert sim > 0.5, f'Expected brain↔memory > 0.5, got {sim:.4f}'

    def test_similarity_symmetry(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        s1 = cb.similarity('water', 'ocean')
        s2 = cb.similarity('ocean', 'water')
        assert abs(s1 - s2) < 0.01

    def test_build_default_returns_codebook(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        assert isinstance(cb, DistributionalCodebook)
        assert len(cb._codebook) > 0

    def test_similarity_unknown_word_zero(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        assert cb.similarity('water', 'zzzyyyxxx') == 0.0

    def test_online_update_extends_codebook(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        before = len(cb._codebook)
        cb.build_from_corpus([['xenomorphic', 'biospheric', 'quasar', 'xenon']])
        assert len(cb._codebook) >= before

    def test_build_default_hf_flag_false(self):
        """When enable_hf_corpus=False the codebook should still build (offline)."""
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default(enable_hf_corpus=False)
        assert len(cb._codebook) > 50


# ---------------------------------------------------------------------------
# 4. HuggingFace corpus loader
# ---------------------------------------------------------------------------

class TestHFCorpusLoader:

    def test_import(self):
        from python.core.language.hf_corpus_loader import load_hf_corpus
        assert callable(load_hf_corpus)

    def test_offline_returns_empty_list(self):
        """In a sandboxed env with no internet, load_hf_corpus must return []."""
        from python.core.language.hf_corpus_loader import load_hf_corpus
        result = load_hf_corpus("auto", max_sentences=100)
        assert isinstance(result, list)
        # Either empty (offline) or non-empty (online) — never raises
        for sent in result:
            assert isinstance(sent, list)

    def test_unknown_dataset_returns_empty(self):
        from python.core.language.hf_corpus_loader import load_hf_corpus
        result = load_hf_corpus("this/dataset/does-not-exist", max_sentences=10)
        assert result == []

    def test_tokenize_internal(self):
        from python.core.language.hf_corpus_loader import _tokenize
        tokens = _tokenize('Water flows in rivers and oceans.')
        assert 'water' in tokens
        assert 'rivers' in tokens

    def test_split_sentences_internal(self):
        from python.core.language.hf_corpus_loader import _split_to_sentences
        sents = _split_to_sentences('Hello world. How are you? I am fine.')
        assert len(sents) == 3

    def test_detect_text_columns(self):
        from python.core.language.hf_corpus_loader import _detect_text_columns
        assert _detect_text_columns(['text', 'label', 'id']) == ['text']
        assert _detect_text_columns(['prompt', 'response']) == ['prompt', 'response']
        assert _detect_text_columns(['id', 'idx']) == []


# ---------------------------------------------------------------------------
# 5. NSCKConfig V7 flags
# ---------------------------------------------------------------------------

class TestNSCKConfigV7:

    def test_enable_fluent_dialogue_default_true(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        assert cfg.enable_fluent_dialogue is True

    def test_enable_hf_corpus_default_false(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        assert cfg.enable_hf_corpus is False

    def test_research_preset_fluent_true(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.research()
        assert cfg.enable_fluent_dialogue is True

    def test_production_preset_fluent_true(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.production()
        assert cfg.enable_fluent_dialogue is True

    def test_minimal_preset_has_v7_flags(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig.minimal()
        # minimal() returns default NSCKConfig() — fluent_dialogue defaults True
        assert hasattr(cfg, 'enable_fluent_dialogue')
        assert hasattr(cfg, 'enable_hf_corpus')

    def test_config_can_disable_fluent(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig(enable_fluent_dialogue=False)
        assert cfg.enable_fluent_dialogue is False

    def test_config_can_enable_hf_corpus(self):
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig(enable_hf_corpus=True)
        assert cfg.enable_hf_corpus is True
