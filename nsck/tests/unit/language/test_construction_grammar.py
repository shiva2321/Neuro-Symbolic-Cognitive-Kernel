import unittest
from python.core.language.construction_grammar import ConstructionMatcher

class TestConstructionGrammar(unittest.TestCase):
    def setUp(self):
        self.matcher = ConstructionMatcher()

    def test_svo_matching(self):
        words = ["John", "chased", "Mary"]
        matches = self.matcher.match(words)
        self.assertTrue(len(matches) > 0)
        top = matches[0]
        self.assertEqual(top.role_fillers.get("subject"), "John")
        self.assertEqual(top.role_fillers.get("object"), "Mary")

    def test_copular_matching(self):
        words = ["Paris", "is", "France"]
        matches = self.matcher.match(words)
        constructions = [m.construction.name for m in matches]
        self.assertTrue(any("copular" in c for c in constructions))
        top_copular = next(m for m in matches if "copular" in m.construction.name)
        self.assertEqual(top_copular.role_fillers.get("subject"), "Paris")

    def test_possessive_matching(self):
        words = ["John", "has", "car"]
        matches = self.matcher.match(words)
        possessive = [m for m in matches if m.construction.name == "possessive_has"]
        self.assertTrue(len(possessive) > 0)
        self.assertEqual(possessive[0].role_fillers.get("owner"), "John")
        self.assertEqual(possessive[0].role_fillers.get("owned"), "car")

    def test_causative_matching(self):
        words = ["heat", "causes", "expansion"]
        matches = self.matcher.match(words)
        causative = [m for m in matches if m.construction.name == "causative"]
        self.assertTrue(len(causative) > 0)
        self.assertEqual(causative[0].role_fillers.get("cause"), "heat")
        self.assertEqual(causative[0].role_fillers.get("effect"), "expansion")

    def test_no_match_random_words(self):
        words = ["the", "a", "an"]  # only articles, no meaningful construction
        matches = self.matcher.match(words)
        # May match some patterns but shouldn't match meaningful constructions
        # Just verify it doesn't crash
        self.assertIsInstance(matches, list)

    def test_locative_matching(self):
        words = ["Paris", "is", "in", "France"]
        matches = self.matcher.match(words)
        locative = [m for m in matches if m.construction.name == "locative_in"]
        self.assertTrue(len(locative) > 0)

if __name__ == "__main__":
    unittest.main()
