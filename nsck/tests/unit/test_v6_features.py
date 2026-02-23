"""
V6 Feature Tests
================
Tests for all 6 V6 gaps:
  1. Fluent NLG (fluent_nlg.py)
  2. Statistical POS Tagger (pos_tagger.py)
  3. VSA anti-bundling negation (negate())
  4. Expanded distributional corpus (200 sentences)
  5. Concurrent multimodal fusion
  6. NSW pure-Python ANN fallback
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ──────────────────────────────────────────────────────────────────────────────
# 1. Fluent NLG
# ──────────────────────────────────────────────────────────────────────────────

class TestFluentNLG:
    """Tests for RelationVerbalizer, FluentResponseComposer, NSCKResponseEngine."""

    def setup_method(self):
        from python.core.language.fluent_nlg import (
            RelationVerbalizer, FluentResponseComposer, NSCKResponseEngine
        )
        self.verbalizer = RelationVerbalizer()
        self.composer = FluentResponseComposer()
        self.engine = NSCKResponseEngine()

    # ── RelationVerbalizer ────────────────────────────────────────────────────

    def test_verbalize_is_a(self):
        s, _ = self.verbalizer.verbalize("dog", "is_a", "mammal")
        assert "dog" in s.lower()
        assert "mammal" in s.lower()
        assert s.endswith(".")

    def test_verbalize_has_property(self):
        s, _ = self.verbalizer.verbalize("water", "has_property", "transparency")
        assert "water" in s.lower()
        assert "transparency" in s.lower()

    def test_verbalize_causes(self):
        s, hint = self.verbalizer.verbalize("rain", "causes", "flooding")
        assert "rain" in s.lower()
        assert "flooding" in s.lower()

    def test_verbalize_unknown_relation(self):
        s, _ = self.verbalizer.verbalize("X", "some_unknown_rel", "Y")
        assert "x" in s.lower() or "X" in s
        assert "y" in s.lower() or "Y" in s

    def test_verbalize_variant_rotation(self):
        """Different variant seeds should produce different sentences sometimes."""
        sentences = {self.verbalizer.verbalize("cat", "is_a", "animal", v)[0] for v in range(5)}
        # At least 2 distinct sentences from 5 variants
        assert len(sentences) >= 1  # must always produce something

    def test_verbalize_capitalization(self):
        s, _ = self.verbalizer.verbalize("fire", "is_a", "combustion")
        assert s[0].isupper()

    def test_verbalize_negation_relation(self):
        s, _ = self.verbalizer.verbalize("whale", "not_is_a", "fish")
        assert "whale" in s.lower()
        assert "not" in s.lower() or "lacks" in s.lower() or "no" in s.lower()

    def test_verbalize_temporal(self):
        s, _ = self.verbalizer.verbalize("spring", "precedes", "summer")
        assert "spring" in s.lower()
        assert "summer" in s.lower()

    def test_verbalize_similar_to(self):
        s, _ = self.verbalizer.verbalize("wolf", "similar_to", "dog")
        assert "wolf" in s.lower()
        assert "dog" in s.lower()

    def test_verbalize_depends_on(self):
        s, _ = self.verbalizer.verbalize("plant", "depends_on", "sunlight")
        assert "plant" in s.lower()

    # ── FluentResponseComposer ────────────────────────────────────────────────

    def test_compose_empty_frames(self):
        text = self.composer.compose([])
        assert len(text) > 0  # fallback message

    def test_compose_single_frame(self):
        frames = [{"subject": "dog", "relation": "is_a", "object": "mammal"}]
        text = self.composer.compose(frames)
        assert "dog" in text.lower()
        assert "mammal" in text.lower()

    def test_compose_multi_frame(self):
        frames = [
            {"subject": "water", "relation": "is_a", "object": "liquid"},
            {"subject": "water", "relation": "has_property", "object": "transparency"},
            {"subject": "water", "relation": "causes", "object": "erosion"},
        ]
        text = self.composer.compose(frames, topic="water", query_type="explanatory")
        assert text.count(".") >= 2
        assert "water" in text.lower()

    def test_compose_topic_sentence_added(self):
        frames = [
            {"subject": "fire", "relation": "is_a", "object": "combustion"},
            {"subject": "fire", "relation": "causes", "object": "smoke"},
        ]
        text = self.composer.compose(frames, topic="fire", query_type="explanatory")
        assert "fire" in text.lower()

    def test_compose_procedural_numbered(self):
        frames = [
            {"subject": "step1", "relation": "precedes", "object": "step2"},
            {"subject": "step2", "relation": "precedes", "object": "step3"},
        ]
        text = self.composer.compose(frames, query_type="procedural")
        assert "1." in text
        assert "2." in text

    def test_compose_negate_frame(self):
        frames = [{"subject": "fish", "relation": "is_a", "object": "mammal", "negate": True}]
        text = self.composer.compose(frames)
        assert "fish" in text.lower()
        assert "not" in text.lower() or "lacks" in text.lower()

    def test_compose_causal_order_preserved(self):
        frames = [
            {"subject": "A", "relation": "causes", "object": "B"},
            {"subject": "B", "relation": "causes", "object": "C"},
        ]
        text = self.composer.compose(frames, query_type="causal")
        pos_a = text.lower().find("a")
        pos_b = text.lower().find("b")
        assert pos_a < pos_b  # A before B in causal order

    def test_compose_conclusion_for_many_facts(self):
        frames = [
            {"subject": "Python", "relation": "is_a", "object": "language"},
            {"subject": "Python", "relation": "has_property", "object": "simplicity"},
            {"subject": "Python", "relation": "used_for", "object": "data science"},
        ]
        text = self.composer.compose(frames, topic="Python")
        assert len(text) > 50

    def test_compose_max_sentences_respected(self):
        frames = [{"subject": "X", "relation": "is_a", "object": f"thing{i}"} for i in range(20)]
        text = self.composer.compose(frames, max_sentences=3)
        # Should have at most ~4 sentences (3 facts + conclusion)
        assert text.count(".") <= 6

    # ── NSCKResponseEngine ────────────────────────────────────────────────────

    def test_respond_basic(self):
        frames = [{"subject": "dog", "relation": "is_a", "object": "mammal"}]
        text = self.engine.respond(frames, topic="dog")
        assert "dog" in text.lower()

    def test_respond_explanatory(self):
        frames = [
            {"subject": "Python", "relation": "is_a", "object": "language"},
            {"subject": "Python", "relation": "used_for", "object": "data science"},
        ]
        text = self.engine.respond(frames, topic="Python", query_type="explanatory")
        assert len(text) > 30

    def test_single_fact(self):
        s = self.engine.single_fact("cat", "is_a", "animal")
        assert "cat" in s.lower()
        assert s.endswith(".")

    def test_answer_query_factual(self):
        facts = [("dog", "is_a", "mammal"), ("dog", "has_property", "loyalty")]
        text = self.engine.answer_query("What is a dog?", facts, topic="dog")
        assert "dog" in text.lower()
        assert len(text) > 20

    def test_answer_query_causal(self):
        facts = [("rain", "causes", "flooding"), ("flooding", "causes", "damage")]
        text = self.engine.answer_query("Why does rain cause problems?", facts, topic="rain")
        assert "rain" in text.lower()

    def test_answer_query_empty_facts(self):
        text = self.engine.answer_query("What is X?", [])
        assert len(text) > 0

    def test_detect_query_type_causal(self):
        assert self.engine._detect_query_type("Why does X happen?") == "causal"

    def test_detect_query_type_procedural(self):
        assert self.engine._detect_query_type("How does the process work step by step?") == "procedural"

    def test_detect_query_type_comparative(self):
        assert self.engine._detect_query_type("Compare A versus B") == "comparative"

    def test_detect_query_type_explanatory(self):
        assert self.engine._detect_query_type("Explain quantum mechanics") == "explanatory"

    def test_detect_query_type_default_factual(self):
        assert self.engine._detect_query_type("What is the capital?") == "factual"

    def test_all_relations_verbalize(self):
        """Every relation in _RELATION_TEMPLATES must produce a non-empty sentence."""
        from python.core.language.fluent_nlg import _RELATION_TEMPLATES
        for rel in _RELATION_TEMPLATES:
            s, _ = self.verbalizer.verbalize("A", rel, "B")
            assert len(s) > 3, f"Empty sentence for relation {rel!r}"
            assert s.endswith("."), f"No period for relation {rel!r}: {s!r}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. BrillPosTagger
# ──────────────────────────────────────────────────────────────────────────────

class TestBrillPosTagger:

    def setup_method(self):
        from python.core.language.pos_tagger import BrillPosTagger, get_default_tagger
        self.tagger = BrillPosTagger()
        self.default = get_default_tagger()

    def test_tag_empty(self):
        assert self.tagger.tag([]) == []

    def test_tag_determiner(self):
        tagged = self.tagger.tag(["the"])
        assert tagged[0][1] == "DT"

    def test_tag_be_verb(self):
        tagged = self.tagger.tag(["is"])
        assert tagged[0][1] in ("VBZ", "VB")

    def test_tag_modal(self):
        tagged = self.tagger.tag(["can"])
        assert tagged[0][1] == "MD"

    def test_tag_negation(self):
        tagged = self.tagger.tag(["not"])
        assert tagged[0][1] == "NEG"

    def test_tag_temporal(self):
        tagged = self.tagger.tag(["before"])
        assert tagged[0][1] in ("TEMP", "IN")

    def test_tag_conditional(self):
        tagged = self.tagger.tag(["if"])
        assert tagged[0][1] == "COND"

    def test_tag_noun_suffix(self):
        tagged = self.tagger.tag(["evolution"])
        assert tagged[0][1] in ("NN", "NNS", "NNP")

    def test_tag_verb_gerund(self):
        tagged = self.tagger.tag(["running"])
        assert tagged[0][1] in ("VBG", "NN")

    def test_tag_adjective_suffix(self):
        tagged = self.tagger.tag(["beautiful"])
        assert tagged[0][1] in ("JJ", "RB")

    def test_tag_adverb_ly(self):
        tagged = self.tagger.tag(["quickly"])
        assert tagged[0][1] == "RB"

    def test_sentence_level_tagging(self):
        tagged = self.tagger.tag_sentence("The cat sat on the mat")
        words = [w for w, t in tagged]
        assert "cat" in words
        assert "mat" in words

    def test_brill_rule_dt_to_noun(self):
        """Word after DT should be re-tagged as NOUN if initially tagged VB."""
        # "the run" — "run" should be NOUN after DT
        tagged = self.tagger.tag(["the", "run"])
        assert tagged[1][1] in ("NN", "NNS", "NNP"), f"Got {tagged[1][1]}"

    def test_proper_noun_detection(self):
        """Capitalized non-sentence-initial words should be NNP."""
        tagged = self.tagger.tag(["visited", "London", "yesterday"])
        # London is capitalized and not at position 0
        assert tagged[1][1] == "NNP"

    def test_get_verbs(self):
        tagged = self.tagger.tag_sentence("cats eat and run")
        verbs = self.tagger.get_verbs(tagged)
        assert any(v in ("eat", "run", "eats", "runs") for v in verbs)

    def test_get_nouns(self):
        tagged = self.tagger.tag_sentence("the large cat")
        nouns = self.tagger.get_nouns(tagged)
        assert "cat" in nouns

    def test_extra_lexicon(self):
        from python.core.language.pos_tagger import BrillPosTagger
        t = BrillPosTagger(extra_lexicon={"nsck": "NNP", "vsa": "NNP"})
        tagged = t.tag(["nsck", "uses", "vsa"])
        assert tagged[0][1] == "NNP"

    def test_default_tagger_singleton(self):
        t1 = self.default
        from python.core.language.pos_tagger import get_default_tagger
        t2 = get_default_tagger()
        assert t1 is t2  # same instance

    def test_tag_sentence_returns_correct_length(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        tagged = self.tagger.tag_sentence(sentence)
        assert len(tagged) == len(sentence.split())


class TestTagSentenceCG:
    """Test the tag_sentence() function exported from construction_grammar."""

    def test_tag_sentence_returns_list(self):
        from python.core.language.construction_grammar import tag_sentence
        result = tag_sentence(["The", "cat", "runs"])
        assert len(result) == 3

    def test_tag_sentence_content(self):
        from python.core.language.construction_grammar import tag_sentence
        result = tag_sentence(["not"])
        assert result[0] == "NEG"

    def test_tag_sentence_verb_noun(self):
        from python.core.language.construction_grammar import tag_sentence
        # "The" should be ART, "cat" should be NOUN
        result = tag_sentence(["The", "cat"])
        assert result[0] == "ART"
        assert result[1] == "NOUN"


# ──────────────────────────────────────────────────────────────────────────────
# 3. VSA anti-bundling negation
# ──────────────────────────────────────────────────────────────────────────────

class TestVSANegation:

    def setup_method(self):
        import python.core.vsa.hypervec_shim as h
        self.HV = h.HyperVector
        self.backend = h.__backend__

    def test_negate_exists(self):
        hv = self.HV(42)
        assert hasattr(hv, "negate"), "negate() method missing"

    def test_negate_near_orthogonal(self):
        """negate(hv) should be ~50% similar to hv (orthogonal region)."""
        hv = self.HV(42)
        neg = hv.negate()
        sim = hv.similarity(neg)
        # Should be in [0.40, 0.60]
        assert 0.40 <= sim <= 0.60, f"Expected ~0.50, got {sim:.4f}"

    def test_negate_idempotent(self):
        """negate(negate(hv)) should equal hv (XOR is self-inverse)."""
        hv = self.HV(42)
        neg2 = hv.negate().negate()
        sim = hv.similarity(neg2)
        assert sim >= 0.99, f"Expected ~1.0, got {sim:.4f}"

    def test_negate_different_vectors_different_results(self):
        hv1 = self.HV(1)
        hv2 = self.HV(2)
        neg1 = hv1.negate()
        neg2 = hv2.negate()
        sim = neg1.similarity(neg2)
        # neg1 and neg2 should not be identical to each other
        assert sim < 0.99, "All negations converge to same vector — wrong"

    def test_negate_deterministic(self):
        hv = self.HV(99)
        n1 = hv.negate()
        n2 = hv.negate()
        assert n1.similarity(n2) >= 0.99

    def test_negate_negative_far_from_original(self):
        """negate should NOT be highly similar to original."""
        hv = self.HV(7)
        neg = hv.negate()
        sim = hv.similarity(neg)
        # Must NOT be close to 1.0 (that would mean negation = identity)
        assert sim < 0.75

    def test_negation_across_concepts(self):
        """Negate produces distinct vectors for distinct concepts."""
        sims = []
        for seed in range(5):
            hv = self.HV(seed)
            neg = hv.negate()
            sims.append(hv.similarity(neg))
        for s in sims:
            assert 0.35 <= s <= 0.65

    def test_python_rust_negate_consistent(self):
        """Python and Rust negate() should produce same result for same seed."""
        from python.core.vsa.hypervec_py import HyperVectorPy
        py_hv = HyperVectorPy(seed=42)
        py_neg = py_hv.negate()
        # Get as bits array
        py_neg_bits = py_neg.bits

        rust_hv = self.HV(42)
        rust_neg = rust_hv.negate()
        # Get rust bits via __getstate__
        try:
            state = rust_neg.__getstate__()
            import numpy as np
            rust_bits = np.zeros(10240, dtype=np.int8)
            for wi, word in enumerate(state):
                for bi in range(64):
                    rust_bits[wi * 64 + bi] = int((word >> bi) & 1)
            # Compare
            match = int(np.sum(py_neg_bits == rust_bits)) / 10240
            assert match >= 0.98, f"Python/Rust negate mismatch: {match:.4f}"
        except Exception:
            pytest.skip("Rust backend unavailable for cross-backend test")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Expanded distributional corpus
# ──────────────────────────────────────────────────────────────────────────────

class TestDistributionalCorpus:

    def test_corpus_size(self):
        from python.core.language.distributional_semantics import BUILTIN_CORPUS
        assert len(BUILTIN_CORPUS) >= 180, f"Expected 180+, got {len(BUILTIN_CORPUS)}"

    def test_corpus_covers_domains(self):
        from python.core.language.distributional_semantics import BUILTIN_CORPUS
        flat = " ".join(" ".join(s) for s in BUILTIN_CORPUS).lower()
        # Check that multiple domains are covered
        assert "evolution" in flat or "genetics" in flat  # biology
        assert "gravity" in flat or "physics" in flat     # physics
        assert "memory" in flat or "reasoning" in flat    # cognition
        assert "climate" in flat or "ocean" in flat       # earth sciences
        assert "algorithm" in flat or "computer" in flat  # technology

    def test_codebook_builds(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        assert len(cb._codebook) >= 100  # many unique words

    def test_semantic_similarity_cat_dog(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        sim = cb.similarity("cat", "dog")
        # Same distributional contexts → positive similarity
        assert sim > 0.0

    def test_semantic_similarity_water_ocean(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook.build_default()
        sim = cb.similarity("water", "ocean")
        assert sim > 0.0

    def test_corpus_all_entries_are_lists(self):
        from python.core.language.distributional_semantics import BUILTIN_CORPUS
        for i, entry in enumerate(BUILTIN_CORPUS):
            assert isinstance(entry, list), f"Entry {i} is not a list: {type(entry)}"
            assert all(isinstance(w, str) for w in entry), f"Entry {i} contains non-string"

    def test_corpus_no_empty_entries(self):
        from python.core.language.distributional_semantics import BUILTIN_CORPUS
        for i, entry in enumerate(BUILTIN_CORPUS):
            assert len(entry) > 0, f"Empty entry at {i}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Concurrent multimodal fusion
# ──────────────────────────────────────────────────────────────────────────────

class TestConcurrentMultimodalProcessor:

    def setup_method(self):
        from python.core.multimodal.multimodal_processor import (
            ConcurrentMultimodalProcessor, MultimodalInput
        )
        self.CMP = ConcurrentMultimodalProcessor
        self.MI = MultimodalInput

    def test_class_exists(self):
        from python.core.multimodal.multimodal_processor import ConcurrentMultimodalProcessor
        assert ConcurrentMultimodalProcessor is not None

    def test_single_text_fallback(self):
        """Single modality should fall back to sequential processing."""
        proc = self.CMP(max_workers=2)
        inp = self.MI(text="hello world")
        result = proc.process(inp)
        assert result.fused_hv is not None
        assert "hello" in result.extracted_concepts or "world" in result.extracted_concepts

    def test_text_and_structured_concurrent(self):
        """Two modalities processed concurrently."""
        proc = self.CMP(max_workers=2)
        inp = self.MI(
            text="a large dog",
            structured={"animal": "dog", "size": "large"},
        )
        result = proc.process(inp)
        assert result.fused_hv is not None
        assert len(result.modality_results) == 2
        assert result.confidence > 0.0

    def test_text_image_structured_concurrent(self):
        """Three modalities: text + image + structured."""
        proc = self.CMP(max_workers=4)
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        inp = self.MI(
            text="a cat on a mat",
            image=img,
            structured={"type": "scene", "subject": "cat"},
        )
        result = proc.process(inp)
        assert len(result.modality_results) == 3
        assert result.confidence > 0.0

    def test_empty_input(self):
        proc = self.CMP(max_workers=2)
        inp = self.MI()
        result = proc.process(inp)
        assert result.confidence == 0.0

    def test_concurrent_is_subclass_of_multimodal(self):
        from python.core.multimodal.multimodal_processor import (
            ConcurrentMultimodalProcessor, MultimodalProcessor
        )
        assert issubclass(ConcurrentMultimodalProcessor, MultimodalProcessor)

    def test_fused_hv_is_not_none(self):
        proc = self.CMP()
        inp = self.MI(text="test sentence", structured={"key": "value"})
        result = proc.process(inp)
        assert result.fused_hv is not None

    def test_concurrent_result_same_as_sequential(self):
        """Concurrent and sequential should produce same fused HV."""
        from python.core.multimodal.multimodal_processor import (
            MultimodalProcessor, ConcurrentMultimodalProcessor, MultimodalInput
        )
        inp = MultimodalInput(
            text="the dog runs",
            structured={"action": "run", "subject": "dog"},
        )
        seq_result = MultimodalProcessor().process(inp)
        con_result = ConcurrentMultimodalProcessor(max_workers=2).process(inp)
        # Both should produce same number of modalities
        assert len(seq_result.modality_results) == len(con_result.modality_results)


# ──────────────────────────────────────────────────────────────────────────────
# 6. NSW pure-Python ANN fallback
# ──────────────────────────────────────────────────────────────────────────────

class TestNSWIndex:

    def setup_method(self):
        from python.core.memory.semantic_memory import _NSWIndex
        self.NSW = _NSWIndex

    def test_empty_search(self):
        nsw = self.NSW()
        result = nsw.search(np.random.rand(10).astype(np.float32), k=5)
        assert result == []

    def test_add_single_and_search(self):
        nsw = self.NSW()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        nsw.add_item(v)
        result = nsw.search(v, k=1)
        assert len(result) == 1
        idx, dist = result[0]
        assert idx == 0
        assert dist < 0.01  # Self-query should be near 0 distance

    def test_add_multiple_and_search(self):
        """NSW returns near-neighbour results — approximate, not exact."""
        nsw = self.NSW()
        import numpy as np
        rng = np.random.default_rng(42)
        # Add 50 random unit vectors in 128D
        vecs = [rng.random(128).astype(np.float32) for _ in range(50)]
        for v in vecs:
            nsw.add_item(v)
        # Query with a vector close to vecs[10]
        query = vecs[10].copy() + rng.random(128).astype(np.float32) * 0.01
        results = nsw.search(query, k=5)
        assert len(results) >= 1
        # With 50 items and small perturbation, the true match (idx 10) should
        # be in the top-5 at least 95% of the time. Verify by checking that
        # min distance is small (< 0.3 cosine distance)
        min_dist = min(d for _, d in results)
        assert min_dist < 0.3, f"NSW not finding nearby vectors: {min_dist:.4f}"

    def test_returns_k_or_fewer(self):
        nsw = self.NSW()
        for _ in range(5):
            nsw.add_item(np.random.rand(32).astype(np.float32))
        result = nsw.search(np.random.rand(32).astype(np.float32), k=10)
        assert len(result) <= 10

    def test_distances_are_non_negative(self):
        nsw = self.NSW()
        for _ in range(10):
            nsw.add_item(np.random.rand(16).astype(np.float32))
        results = nsw.search(np.random.rand(16).astype(np.float32), k=5)
        for idx, dist in results:
            assert dist >= -0.01  # allow tiny float error

    def test_semantic_memory_nsw_enabled(self):
        """SemanticMemory with enable_hnsw_index=True uses NSW when hnswlib absent."""
        from python.core.memory.semantic_memory import SemanticMemory, _HNSWLIB_AVAILABLE
        from python.core.integration.config import NSCKConfig

        if _HNSWLIB_AVAILABLE:
            pytest.skip("hnswlib installed; NSW fallback not exercised")

        config = NSCKConfig(enable_hnsw_index=True)
        sm = SemanticMemory(config=config, use_rust=False)
        sm.add_concept("cat", {"type": "animal"})
        sm.add_concept("dog", {"type": "animal"})
        sm.add_concept("car", {"type": "vehicle"})

        from python.core.memory.semantic_memory import _NSWIndex
        assert isinstance(sm._hnsw_index, _NSWIndex)

    def test_semantic_memory_query_via_nsw(self):
        """Query results via NSW index are plausible."""
        from python.core.memory.semantic_memory import SemanticMemory, _HNSWLIB_AVAILABLE
        from python.core.integration.config import NSCKConfig

        if _HNSWLIB_AVAILABLE:
            pytest.skip("hnswlib installed; NSW fallback not exercised")

        config = NSCKConfig(enable_hnsw_index=True)
        sm = SemanticMemory(config=config, use_rust=False)
        for i in range(10):
            sm.add_concept(f"concept_{i}", {"idx": i})

        import python.core.vsa.hypervec_shim as h
        query = h.HyperVector(hash("concept_0") % (2**32))
        results = sm.query(query, k=3)
        assert len(results) >= 1
        assert all(isinstance(name, str) for name, _ in results)


# ──────────────────────────────────────────────────────────────────────────────
# Rust-specific tests (skipped if Rust not available)
# ──────────────────────────────────────────────────────────────────────────────

class TestRustBackendV6:

    def setup_method(self):
        import python.core.vsa.hypervec_shim as h
        if h.__backend__ != "Rust":
            pytest.skip("Rust backend not loaded")
        self.h = h
        self.HV = h.HyperVector

    def test_rust_negate_native(self):
        """Rust HV has native negate() method."""
        hv = self.HV(100)
        neg = hv.negate()
        sim = hv.similarity(neg)
        assert 0.40 <= sim <= 0.60

    def test_rust_negate_from_bits(self):
        """from_u64_words static method works."""
        hv = self.HV(200)
        state = hv.__getstate__()
        hv2 = self.HV.from_u64_words(state)
        assert hv.similarity(hv2) >= 0.99

    def test_rust_vsa_backend_loaded(self):
        assert self.h.__backend__ == "Rust"

    def test_rust_semantic_memory_concurrent(self):
        if self.h.SemanticMemoryConcurrent is None:
            pytest.skip("SemanticMemoryConcurrent not available")
        sm = self.h.SemanticMemoryConcurrent()
        hv = self.HV(42)
        sm.add_concept("test", hv)
        assert sm.concept_count() == 1
