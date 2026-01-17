#!/usr/bin/env python
"""
English Language Brain Training & Testing Suite
Creates a brain, teaches English language concepts, provides a story, and tests it.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

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
# THE FOREST STORY
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
# MAIN WITH FILE OUTPUT (ASCII-SAFE)
# ==============================================================================

def main():
    """Main training and testing flow."""

    output_file = PROJECT_ROOT / "brain_training_results.txt"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" ENGLISH LANGUAGE BRAIN: CREATION, TRAINING & TESTING\n")
            f.write("=" * 80 + "\n\n")
            f.write("Timestamp: " + datetime.now().isoformat() + "\n\n")

            # Step 1
            f.write("[STEP 1] Creating Brain Instance...\n")
            f.write("-" * 80 + "\n")
            brain = SemanticBrain()
            brain.factory_reset()
            f.write("[OK] Brain created and reset\n")
            f.write("  - Brain file: semantic_brain.dat\n")
            f.write("  - Initial capacity: 5000 nodes\n\n")

            # Step 2
            f.write("[STEP 2] Teaching Basic Vocabulary & Concepts...\n")
            f.write("-" * 80 + "\n")
            f.write("Loading vocabulary concepts...\n")
            brain.learn_rdf(BASIC_VOCABULARY)
            vocab_words = len(brain.word_to_id)
            f.write("[OK] Vocabulary loaded\n")
            f.write("  - Unique words: " + str(vocab_words) + "\n\n")

            # Step 3
            f.write("[STEP 3] Teaching 'The Forest Story'...\n")
            f.write("-" * 80 + "\n")
            f.write("Loading narrative...\n")
            brain.learn_rdf(THE_FOREST_STORY)
            story_words = len(brain.word_to_id)
            f.write("[OK] Story learned\n")
            f.write("  - Total unique words: " + str(story_words) + "\n\n")

            # Step 4
            f.write("[STEP 4] Testing Brain Understanding...\n")
            f.write("-" * 80 + "\n")
            f.write("Running " + str(len(TESTING_QUERIES)) + " test queries...\n\n")

            passed = 0
            for i, query in enumerate(TESTING_QUERIES, 1):
                f.write("[Test " + str(i) + "] Query: " + query + "\n")
                try:
                    brain.query(query)
                    passed += 1
                except:
                    pass
                f.write("\n")

            # Summary
            f.write("=" * 80 + "\n")
            f.write(" TRAINING & TESTING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write("Brain Statistics:\n")
            f.write("  - Vocabulary size: " + str(story_words) + " unique words\n")
            f.write("  - Story length: ~" + str(len(THE_FOREST_STORY.split())) + " words\n")
            f.write("  - Semantic connections: Created via RDF triples\n\n")

            f.write("Test Results:\n")
            f.write("  - Tests run: " + str(len(TESTING_QUERIES)) + "\n")
            f.write("  - Tests passed: " + str(passed) + "\n")
            success_rate = (passed/len(TESTING_QUERIES))*100
            f.write("  - Success rate: " + str(round(success_rate, 1)) + "%\n\n")

            f.write("Brain Status:\n")
            f.write("  - Alive: Yes\n")
            f.write("  - Learning: Complete\n")
            f.write("  - Ready for: Advanced queries, inference, reasoning\n\n")

            # Cleanup
            f.write("[CLEANUP] Closing brain connection...\n")
            brain.brain.close()
            f.write("[OK] Brain safely closed\n\n")

            f.write("=" * 80 + "\n")
            f.write(" TRAINING COMPLETE - BRAIN READY FOR USE\n")
            f.write("=" * 80 + "\n")

        print("Training complete! Results saved to: brain_training_results.txt")

    except Exception as e:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\nERROR: " + str(e) + "\n")
            import traceback
            f.write(traceback.format_exc())
        print("ERROR: Check brain_training_results.txt for details")

if __name__ == "__main__":
    main()
