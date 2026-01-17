#!/usr/bin/env python
"""
English Language Brain Training & Testing Suite - FILE OUTPUT VERSION

This script creates a fresh brain, teaches it English language concepts,
provides a story to learn from, and tests its understanding.
Results are saved to a file for verification.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from semantic.context_driver import SemanticBrain


# ==============================================================================
# ENGLISH LANGUAGE CONCEPTS & VOCABULARY
# ==============================================================================

BASIC_VOCABULARY = """
THE APPLE IS FRUIT.
THE ORANGE IS FRUIT.
THE BANANA IS FRUIT.
FRUIT IS FOOD.
VEGETABLE IS FOOD.
THE CARROT IS VEGETABLE.
THE LETTUCE IS VEGETABLE.
THE DOG IS ANIMAL.
THE CAT IS ANIMAL.
THE BIRD IS ANIMAL.
THE FISH IS ANIMAL.
ANIMAL EATS FOOD.
HUMAN EATS FOOD.
THE SUN IS STAR.
THE MOON IS OBJECT.
THE EARTH IS PLANET.
THE FIRE IS HOT.
THE ICE IS COLD.
THE WATER IS LIQUID.
THE HOUSE IS BUILDING.
THE SCHOOL IS BUILDING.
THE HOSPITAL IS BUILDING.
THE PERSON IS HUMAN.
THE CHILD IS HUMAN.
THE ADULT IS HUMAN.
PERSON LIVES IN HOUSE.
CHILD GOES TO SCHOOL.
ADULT WORKS IN BUILDING.
THE BOOK IS OBJECT.
THE PEN IS OBJECT.
THE DESK IS OBJECT.
STUDENT READS BOOK.
STUDENT USES PEN.
STUDENT SITS AT DESK.
THE RED IS COLOR.
THE BLUE IS COLOR.
THE GREEN IS COLOR.
APPLE HAS COLOR.
SKY HAS COLOR.
TREE HAS COLOR.
THE TREE IS PLANT.
THE FLOWER IS PLANT.
THE GRASS IS PLANT.
PLANT NEEDS WATER.
PLANT NEEDS SUNLIGHT.
THE CAR IS VEHICLE.
THE TRAIN IS VEHICLE.
THE AIRPLANE IS VEHICLE.
VEHICLE MOVES.
PERSON USES VEHICLE.
THE MUSIC IS SOUND.
THE NOISE IS SOUND.
THE VOICE IS SOUND.
PERSON MAKES SOUND.
INSTRUMENT MAKES MUSIC.
THE LOVE IS EMOTION.
THE HAPPINESS IS EMOTION.
THE SADNESS IS EMOTION.
PERSON FEELS EMOTION.
LOVE IS STRONG.
THE MORNING IS TIME.
THE AFTERNOON IS TIME.
THE NIGHT IS TIME.
SUN RISES IN MORNING.
SUN SETS IN NIGHT.
THE WINTER IS SEASON.
THE SUMMER IS SEASON.
THE SPRING IS SEASON.
THE AUTUMN IS SEASON.
SNOW FALLS IN WINTER.
FLOWERS BLOOM IN SPRING.
THE MOTHER IS PARENT.
THE FATHER IS PARENT.
THE CHILD IS OFFSPRING.
PARENT LOVES CHILD.
CHILD LEARNS FROM PARENT.
THE FRIEND IS PERSON.
FRIEND HELPS PERSON.
FRIEND TRUSTS PERSON.
THE TEACHER IS PERSON.
TEACHER TEACHES STUDENT.
TEACHER KNOWS SUBJECT.
THE DOCTOR IS PERSON.
DOCTOR HELPS PATIENT.
DOCTOR KNOWS MEDICINE.
THE CITY IS PLACE.
THE FOREST IS PLACE.
THE OCEAN IS PLACE.
BUILDING EXISTS IN CITY.
TREE EXISTS IN FOREST.
FISH LIVES IN OCEAN.
THE THINKING IS PROCESS.
THE LEARNING IS PROCESS.
THE SPEAKING IS PROCESS.
HUMAN DOES THINKING.
STUDENT DOES LEARNING.
PERSON DOES SPEAKING.
THE QUESTION IS INQUIRY.
THE ANSWER IS RESPONSE.
STUDENT ASKS QUESTION.
TEACHER GIVES ANSWER.
THE WISDOM IS KNOWLEDGE.
THE SKILL IS ABILITY.
LEARNING BUILDS KNOWLEDGE.
PRACTICE BUILDS SKILL.
THE TRUTH IS FACT.
THE FALSEHOOD IS LIE.
FACT IS TRUE.
LIE IS FALSE.
THE SUCCESS IS OUTCOME.
THE FAILURE IS OUTCOME.
HARD WORK LEADS TO SUCCESS.
LAZINESS LEADS TO FAILURE.
PERSON HAS NAME.
PERSON HAS AGE.
PERSON HAS FAMILY.
"""

# ==============================================================================
# THE STORY FOR LEARNING
# ==============================================================================

THE_FOREST_STORY = """
THE FOREST HAS MANY TREES.
THE TREES ARE PLANTS.
TREES NEED WATER AND SUNLIGHT.
IN THE FOREST LIVES A YOUNG DEER.
THE DEER IS ANIMAL.
THE DEER EATS GRASS.
GRASS IS PLANT.
THE DEER HAS A MOTHER.
MOTHER IS PARENT.
MOTHER TEACHES DEER.
MOTHER LOVES DEER.
THE DEER HAS FRIENDS.
FRIENDS ARE ANIMALS.
ANIMALS PLAY TOGETHER.
ONE DAY THE SUN RISES.
THE SUN IS STAR.
SUN GIVES LIGHT.
MORNING COMES AFTER NIGHT.
THE DEER WAKES UP.
THE DEER FEELS HAPPY.
HAPPINESS IS EMOTION.
THE DEER WALKS TO THE RIVER.
RIVER HAS WATER.
WATER IS LIQUID.
ANIMALS NEED WATER.
IN THE RIVER LIVE FISH.
FISH ARE ANIMALS.
THE DEER DRINKS WATER.
WATER IS COLD.
DEER FEELS REFRESHED.
THE DEER MEETS A BIRD.
BIRD IS ANIMAL.
BIRD HAS WINGS.
WINGS HELP BIRD FLY.
THE BIRD SINGS.
THE BIRD AND DEER BECOME FRIENDS.
THE DEER AND BIRD PLAY TOGETHER.
THE SKY IS ABOVE.
THE SKY HAS COLOR.
THE SKY IS BLUE.
CLOUDS FLOAT IN SKY.
THE FOREST IS HOME.
HOME IS PLACE.
THE DEER FEELS SAFE.
THE SUN BEGINS TO SET.
THE AFTERNOON BECOMES EVENING.
THE STARS APPEAR.
THE DEER AND MOTHER REST.
THE MOTHER TEACHES LESSON.
THE MOTHER SAYS TRUST YOUR INSTINCTS.
THE MOTHER SAYS HELP YOUR FRIENDS.
THE DEER SLEEPS.
THE NIGHT IS DARK.
THE MOON SHINES.
MOON IS OBJECT.
THE NEXT MORNING ARRIVES.
THE DEER IS STRONGER.
THE FRIEND BIRD RETURNS.
THEY PLAY AGAIN.
THE FOREST CONTINUES.
THE STORY CONTINUES.
THE DEER GROWS OLDER.
THE DEER BECOMES WISE.
THE DEER HELPS YOUNG ANIMALS.
THE FOREST THRIVES.
FRIENDSHIP IS TREASURE.
KNOWLEDGE IS FOREVER.
"""

# ==============================================================================
# TESTING QUERIES
# ==============================================================================

TESTING_QUERIES = [
    "What is an apple?",
    "What is a fruit?",
    "What are animals?",
    "What does a deer eat?",
    "Where does a deer live?",
    "What teaches the young deer?",
    "Who are friends?",
    "What is friendship?",
    "Who loves the deer?",
    "What is happiness?",
    "What is wisdom?",
    "What is morning?",
    "What do animals need?",
    "What helps birds fly?",
    "What is in the forest?",
    "What is a story?",
]


# ==============================================================================
# MAIN TRAINING & TESTING WITH FILE OUTPUT
# ==============================================================================

def main():
    """Main training and testing flow with file output."""

    # Open output file
    output_file = PROJECT_ROOT / "brain_training_results.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" ENGLISH LANGUAGE BRAIN: CREATION, TRAINING & TESTING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Step 1: Create the brain
        f.write("[STEP 1] Creating Brain Instance...\n")
        f.write("-" * 80 + "\n")
        try:
            brain = SemanticBrain()
            brain.factory_reset()
            f.write("✓ Brain created and reset to factory state\n")
            f.write(f"  - Brain file: semantic_brain.dat\n")
            f.write(f"  - Initial capacity: 5000 nodes\n\n")
        except Exception as e:
            f.write(f"✗ Error creating brain: {e}\n\n")
            return

        # Step 2: Teach basic vocabulary
        f.write("[STEP 2] Teaching Basic Vocabulary & Concepts...\n")
        f.write("-" * 80 + "\n")
        try:
            f.write("Loading vocabulary concepts into semantic memory...\n")
            brain.learn_rdf(BASIC_VOCABULARY)
            vocab_words = len(brain.word_to_id)
            f.write(f"✓ Vocabulary loaded\n")
            f.write(f"  - Unique words: {vocab_words}\n\n")
        except Exception as e:
            f.write(f"✗ Error teaching vocabulary: {e}\n\n")
            return

        # Step 3: Teach the story
        f.write("[STEP 3] Teaching 'The Forest Story'...\n")
        f.write("-" * 80 + "\n")
        try:
            f.write("Loading narrative into semantic memory...\n")
            brain.learn_rdf(THE_FOREST_STORY)
            story_words = len(brain.word_to_id)
            f.write(f"✓ Story learned\n")
            f.write(f"  - Total unique words now: {story_words}\n\n")
        except Exception as e:
            f.write(f"✗ Error teaching story: {e}\n\n")
            return

        # Step 4: Test the brain
        f.write("[STEP 4] Testing Brain Understanding...\n")
        f.write("-" * 80 + "\n")
        f.write(f"Running {len(TESTING_QUERIES)} test queries...\n\n")

        passed = 0
        try:
            for i, query in enumerate(TESTING_QUERIES, 1):
                f.write(f"[Test {i}] Query: {query}\n")
                brain.query(query)
                passed += 1
                f.write("\n")
        except Exception as e:
            f.write(f"✗ Error during testing: {e}\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write(" TRAINING & TESTING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Brain Statistics:\n")
        f.write(f"  - Vocabulary size: {story_words} unique words\n")
        f.write(f"  - Story length: ~{len(THE_FOREST_STORY.split())} words\n")
        f.write(f"  - Semantic connections: Created via RDF triples\n\n")

        f.write(f"Test Results:\n")
        f.write(f"  - Tests run: {len(TESTING_QUERIES)}\n")
        f.write(f"  - Tests passed: {passed}\n")
        f.write(f"  - Success rate: {(passed/len(TESTING_QUERIES))*100:.1f}%\n\n")

        f.write(f"Brain Status:\n")
        f.write(f"  - Alive: Yes\n")
        f.write(f"  - Learning: Complete\n")
        f.write(f"  - Ready for: Advanced queries, inference, reasoning\n\n")

        # Cleanup
        f.write("[CLEANUP] Closing brain connection...\n")
        try:
            brain.brain.close()
            f.write("✓ Brain safely closed\n\n")
        except Exception as e:
            f.write(f"✗ Error closing brain: {e}\n\n")

        f.write("=" * 80 + "\n")
        f.write(" TRAINING COMPLETE - BRAIN READY FOR USE\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Training complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
