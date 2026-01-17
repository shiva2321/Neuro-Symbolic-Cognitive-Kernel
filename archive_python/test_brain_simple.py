#!/usr/bin/env python
"""Quick test to verify semantic brain works"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Starting import test...")

try:
    print("Importing SemanticBrain...")
    from semantic.context_driver import SemanticBrain
    print("✓ Import successful")

    print("Creating brain instance...")
    brain = SemanticBrain()
    print("✓ Brain instance created")

    print("Resetting brain...")
    brain.factory_reset()
    print("✓ Brain reset complete")

    print("Teaching simple fact...")
    brain.learn_rdf("THE DOG IS ANIMAL.")
    print("✓ Fact learned")

    print("Querying brain...")
    brain.query("What is a dog?")
    print("✓ Query complete")

    print("Closing brain...")
    brain.brain.close()
    print("✓ Brain closed")

    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
