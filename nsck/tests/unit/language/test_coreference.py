import unittest
from python.core.language.coreference import EntityRegister
import python.core.vsa.hypervec_shim as hypervec_rs

HV = hypervec_rs.HyperVector

class TestCoreference(unittest.TestCase):
    def setUp(self):
        self.register = EntityRegister()

    def tearDown(self):
        self.register.clear()

    def test_pronoun_resolution_he(self):
        self.register.register("John", HV(hash("John") % (2**32)), {"gender": "male", "animacy": "animate", "number": "singular"})
        result = self.register.resolve("he")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "John")

    def test_pronoun_resolution_she(self):
        self.register.register("Mary", HV(hash("Mary") % (2**32)), {"gender": "female", "animacy": "animate", "number": "singular"})
        result = self.register.resolve("she")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Mary")

    def test_pronoun_resolution_it(self):
        self.register.register("table", HV(hash("table") % (2**32)), {"gender": "neuter", "animacy": "inanimate", "number": "singular"})
        result = self.register.resolve("it")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "table")

    def test_register_overflow_max_10(self):
        for i in range(15):
            self.register.register(f"entity_{i}", HV(i), {"gender": "male", "animacy": "animate", "number": "singular"})
        all_entities = self.register.get_all()
        self.assertLessEqual(len(all_entities), 10)

    def test_ambiguous_pronoun_returns_latest(self):
        import time
        self.register.register("Bob", HV(hash("Bob") % (2**32)), {"gender": "male", "animacy": "animate", "number": "singular"})
        time.sleep(0.01)
        self.register.register("Alice_male_alias", HV(hash("Alice_male") % (2**32)), {"gender": "male", "animacy": "animate", "number": "singular"})
        result = self.register.resolve("he")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Alice_male_alias")

    def test_no_compatible_entity(self):
        self.register.register("table", HV(1), {"gender": "neuter", "animacy": "inanimate", "number": "singular"})
        result = self.register.resolve("she")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
