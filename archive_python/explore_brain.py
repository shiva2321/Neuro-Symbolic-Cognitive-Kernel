#!/usr/bin/env python
"""
Interactive Brain Explorer
Play with your trained English language brain!
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from semantic.context_driver import SemanticBrain

def main():
    """Interactive brain explorer."""

    print("\n" + "=" * 80)
    print(" ENGLISH LANGUAGE BRAIN - INTERACTIVE EXPLORER")
    print("=" * 80)
    print("\nConnecting to brain...\n")

    try:
        brain = SemanticBrain()

        print("Brain is ready!")
        print("\nYour brain knows:")
        print(f"  - {len(brain.word_to_id)} unique concepts")
        print("  - Relationships: IS, HAS, EATS, NEEDS, LOVES, TEACHES, etc.")
        print("  - A forest story with a deer, bird, and lessons about friendship")

        print("\n" + "-" * 80)
        print("Try these queries:")
        print("-" * 80)
        print("  What is an apple?")
        print("  What is a fruit?")
        print("  What are animals?")
        print("  What is friendship?")
        print("  What is wisdom?")
        print("  What is morning?")
        print("  Who are friends?")
        print("  What do animals need?")
        print("  What is in the forest?")
        print("\nType 'quit' to exit, 'help' for more commands")
        print("-" * 80 + "\n")

        while True:
            try:
                query = input("Query> ").strip()

                if query.lower() == 'quit':
                    print("\nClosing brain connection...")
                    brain.brain.close()
                    print("Brain safely closed. Goodbye!")
                    break

                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  quit      - Exit the explorer")
                    print("  help      - Show this help")
                    print("  stats     - Show brain statistics")
                    print("  learn     - Enter learning mode")
                    print("  (query)   - Ask the brain a question")
                    print()

                elif query.lower() == 'stats':
                    print(f"\nBrain Statistics:")
                    print(f"  Vocabulary: {len(brain.word_to_id)} words")
                    print(f"  Words learned: {', '.join(list(brain.id_to_word.values())[:20])}...")
                    print()

                elif query.lower() == 'learn':
                    print("\nTeach the brain new concepts!")
                    print("Format: SUBJECT VERB OBJECT (e.g., 'THE APPLE IS RED')")
                    print("Type 'done' when finished")
                    print()
                    while True:
                        fact = input("Fact> ").strip()
                        if fact.lower() == 'done':
                            print()
                            break
                        if fact:
                            brain.learn_rdf(fact)
                            print(f"  Learned: {fact}")

                elif query:
                    print()
                    brain.query(query)
                    print()

            except KeyboardInterrupt:
                print("\n\nClosing brain connection...")
                brain.brain.close()
                print("Brain safely closed. Goodbye!")
                break

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
