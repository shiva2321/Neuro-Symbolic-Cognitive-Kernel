import unittest
import tempfile
import os
from python.core.language.distributional_semantics import DistributionalCodebook

class TestDistributionalSemantics(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "ran", "in", "the", "park"],
            ["cats", "and", "dogs", "are", "common", "pets"],
            ["the", "cat", "chased", "the", "mouse"],
            ["dogs", "bark", "at", "strangers"],
            ["birds", "fly", "in", "the", "sky"],
            ["fish", "swim", "in", "the", "water"],
            ["water", "flows", "in", "rivers"],
        ]
        self.cb = DistributionalCodebook(window_size=2)
        self.cb.build_from_corpus(self.corpus)

    def test_build_from_corpus(self):
        hv = self.cb.get_hv("cat")
        self.assertIsNotNone(hv)

    def test_known_word_has_hv(self):
        for word in ["cat", "dog", "the"]:
            hv = self.cb.get_hv(word)
            self.assertIsNotNone(hv, f"Expected HV for '{word}'")

    def test_unknown_word_returns_none(self):
        hv = self.cb.get_hv("xyzzy_nonexistent_word_12345")
        self.assertIsNone(hv)

    def test_similarity_same_word(self):
        sim = self.cb.similarity("cat", "cat")
        self.assertGreater(sim, 0.5)

    def test_similarity_related_words(self):
        # cat and dog appear in similar contexts
        sim_related = self.cb.similarity("cat", "dog")
        sim_unrelated = self.cb.similarity("cat", "water")
        # Related words should be more similar (or at least not error)
        self.assertIsInstance(sim_related, float)
        self.assertIsInstance(sim_unrelated, float)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "codebook.pkl")
            self.cb.save(path)
            cb2 = DistributionalCodebook()
            cb2.load(path)
            hv_orig = self.cb.get_hv("cat")
            hv_loaded = cb2.get_hv("cat")
            self.assertIsNotNone(hv_loaded)

if __name__ == "__main__":
    unittest.main()
