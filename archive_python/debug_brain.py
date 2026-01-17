#!/usr/bin/env python
"""
Minimal debug script to test brain creation
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

output_file = PROJECT_ROOT / "debug_output.txt"

try:
    with open(output_file, 'w') as f:
        f.write("Starting debug...\n")
        f.flush()

        f.write("Importing SemanticBrain...\n")
        f.flush()
        from semantic.context_driver import SemanticBrain
        f.write("✓ Import successful\n")
        f.flush()

        f.write("Creating instance...\n")
        f.flush()
        brain = SemanticBrain()
        f.write("✓ Instance created\n")
        f.flush()

        f.write("Calling factory_reset...\n")
        f.flush()
        brain.factory_reset()
        f.write("✓ Factory reset done\n")
        f.flush()

        f.write("Teaching simple fact...\n")
        f.flush()
        brain.learn_rdf("THE DOG IS ANIMAL.")
        f.write("✓ Fact learned\n")
        f.flush()

        f.write("Brain ready! Checking state...\n")
        f.flush()
        f.write(f"Words learned: {len(brain.word_to_id)}\n")
        f.flush()

        f.write("Closing brain...\n")
        f.flush()
        brain.brain.close()
        f.write("✓ Brain closed\n")
        f.flush()

        f.write("\nSUCCESS\n")

except Exception as e:
    with open(output_file, 'a') as f:
        f.write(f"\nERROR: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())
