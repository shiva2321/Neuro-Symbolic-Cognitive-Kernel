"""Quick test runner for Phase 1 invariant tests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
from tests.test_invariants import *

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_invariants.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
