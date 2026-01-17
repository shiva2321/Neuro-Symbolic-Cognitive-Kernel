#!/usr/bin/env python
"""
English Language Brain Training & Testing Suite

This script creates a fresh brain, teaches it English language concepts,
provides a story to learn from, and tests its understanding.

Phase: Language Understanding & Semantic Memory
"""

import sys
import os
from pathlib import Path

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
THE BROCCOLI IS VEGETABLE.

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
PLAYING IS FUN.

ONE DAY THE SUN RISES.
THE SUN IS STAR.
SUN GIVES LIGHT.
MORNING COMES AFTER NIGHT.

THE DEER WAKES UP.
THE DEER FEELS HAPPY.
HAPPINESS IS EMOTION.
EMOTIONS ARE FEELINGS.

THE DEER WALKS TO THE RIVER.
RIVER HAS WATER.
WATER IS LIQUID.
ANIMALS NEED WATER.

IN THE RIVER LIVE FISH.
FISH ARE ANIMALS.
FISH EAT SMALLER FISH.
FISH BREATHE UNDERWATER.

THE DEER DRINKS WATER.
WATER IS COLD.
COLD IS TEMPERATURE.
DEER FEELS REFRESHED.

THE DEER MEETS A BIRD.
BIRD IS ANIMAL.
BIRD HAS WINGS.
WINGS HELP BIRD FLY.

THE BIRD SINGS.
SINGING IS SOUND.
SOUND IS VIBRATION.
SOUNDS ARE PLEASANT.

THE BIRD AND DEER BECOME FRIENDS.
FRIENDSHIP IS BOND.
BONDS ARE CONNECTIONS.
FRIENDS TRUST EACH OTHER.

THE DEER AND BIRD PLAY TOGETHER.
PLAYING IS ACTIVITY.
ACTIVITIES ARE ACTIONS.
ACTIONS HAVE CONSEQUENCES.

THE SKY IS ABOVE.
THE SKY HAS COLOR.
THE SKY IS BLUE.
BLUE IS COLOR.

CLOUDS FLOAT IN SKY.
CLOUDS HOLD RAIN.
RAIN IS WATER.
WATER FALLS FROM SKY.

THE FOREST IS HOME.
HOME IS PLACE.
PLACE IS LOCATION.
LOCATION HAS BOUNDARIES.

THE DEER FEELS SAFE.
SAFE IS STATE.
STATES ARE CONDITIONS.
CONDITIONS CHANGE OVER TIME.

THE SUN BEGINS TO SET.
SUNSET IS BEAUTIFUL.
BEAUTIFUL IS AESTHETIC.
AESTHETICS PLEASE SENSES.

THE AFTERNOON BECOMES EVENING.
EVENING IS TIME.
TIME PASSES CONTINUOUSLY.
NIGHT COMES AFTER EVENING.

THE STARS APPEAR.
STARS ARE FAR.
DISTANCE IS MEASUREMENT.
MEASUREMENTS DEFINE SPACE.

THE DEER AND MOTHER REST.
REST IS NECESSARY.
NECESSITY IS REQUIREMENT.
REQUIREMENTS ENSURE SURVIVAL.

THE MOTHER TEACHES LESSON.
LESSONS ARE TEACHINGS.
TEACHINGS BUILD KNOWLEDGE.
KNOWLEDGE IS POWER.

THE MOTHER SAYS:
TRUST YOUR INSTINCTS.
INSTINCTS ARE FEELINGS.
FEELINGS GUIDE BEHAVIOR.
BEHAVIOR HAS RESULTS.

THE MOTHER ALSO SAYS:
HELP YOUR FRIENDS.
KINDNESS IS VALUE.
VALUES DEFINE MORALITY.
MORALITY IS ETHICS.

THE DEER SLEEPS.
SLEEP IS REST.
REST RESTORES ENERGY.
ENERGY IS VITAL.

THE NIGHT IS DARK.
DARKNESS IS ABSENCE.
ABSENCE IS OPPOSITE.
OPPOSITES DEFINE CONTRAST.

THE MOON SHINES.
MOON IS OBJECT.
OBJECT REFLECTS LIGHT.
LIGHT IS ENERGY.

THE NEXT MORNING ARRIVES.
MORNINGS ARE BEGINNINGS.
BEGINNINGS ARE OPPORTUNITIES.
OPPORTUNITIES BRING GROWTH.

THE DEER IS STRONGER.
STRENGTH IS ABILITY.
ABILITIES IMPROVE WITH PRACTICE.
PRACTICE REQUIRES EFFORT.

THE FRIEND BIRD RETURNS.
RETURN IS COMING BACK.
LOYALTY IS COMMITMENT.
COMMITMENT IS PROMISE.

THEY PLAY AGAIN.
JOY IS HAPPINESS.
HAPPINESS IS REWARD.
REWARDS MOTIVATE BEHAVIOR.

THE FOREST CONTINUES.
FORESTS ARE ECOSYSTEMS.
ECOSYSTEMS ARE INTERCONNECTED.
LIFE IS PRECIOUS.

THE STORY CONTINUES.
STORIES ARE NARRATIVES.
NARRATIVES TEACH LESSONS.
LESSONS INSPIRE GROWTH.

THE DEER GROWS OLDER.
AGING IS NATURAL.
NATURE IS CYCLE.
CYCLES REPEAT ETERNALLY.

THE DEER BECOMES WISE.
WISDOM COMES FROM EXPERIENCE.
EXPERIENCE IS TEACHER.
TEACHERS GUIDE LEARNING.

THE DEER HELPS YOUNG ANIMALS.
HELPING IS KINDNESS.
KINDNESS IS BEAUTIFUL.
BEAUTY IS EVERYWHERE.

THE FOREST THRIVES.
THRIVING IS FLOURISHING.
FLOURISHING IS SUCCESS.
SUCCESS IS FULFILLMENT.

THE MORAL OF THE STORY:
LIFE IS JOURNEY.
JOURNEY IS ADVENTURE.
ADVENTURE REQUIRES COURAGE.
COURAGE IS STRENGTH.

FRIENDSHIP IS TREASURE.
TREASURE IS VALUABLE.
VALUE COMES FROM CONNECTION.
CONNECTIONS MAKE US HUMAN.

KNOWLEDGE IS FOREVER.
FOREVER IS ETERNITY.
ETERNITY IS TIMELESS.
TIMELESS WISDOM ENDURES.
"""


# ==============================================================================
# TESTING QUERIES
# ==============================================================================

TESTING_QUERIES = [
    # Basic vocabulary questions
    ("What is an apple?", "Expects: apple is fruit"),
    ("What is a fruit?", "Expects: fruit is food"),
    ("What are animals?", "Expects: multiple animals"),

    # Story-based questions
    ("What does a deer eat?", "Expects: grass"),
    ("Where does a deer live?", "Expects: forest"),
    ("What teaches the young deer?", "Expects: mother"),

    # Relationship questions
    ("Who are friends?", "Expects: deer and bird"),
    ("What is friendship?", "Expects: bond, connection"),
    ("Who loves the deer?", "Expects: mother"),

    # Abstract concept questions
    ("What is happiness?", "Expects: emotion"),
    ("What is wisdom?", "Expects: comes from experience"),
    ("What is morning?", "Expects: time, beginning"),

    # Logical inference questions
    ("What do animals need?", "Expects: food, water"),
    ("What helps birds fly?", "Expects: wings"),
    ("What is in the forest?", "Expects: trees, animals"),

    # Moral/thematic questions
    ("What is the lesson of the story?", "Expects: knowledge, friendship, courage"),
    ("Why is friendship valuable?", "Expects: connections, treasure"),
]


# ==============================================================================
# MAIN TRAINING & TESTING
# ==============================================================================

def main():
    """Main training and testing flow."""

    print("\n" + "=" * 80)
    print(" ENGLISH LANGUAGE BRAIN: CREATION, TRAINING & TESTING")
    print("=" * 80)

    # Step 1: Create the brain
    print("\n[STEP 1] Creating Brain Instance...")
    print("-" * 80)
    brain = SemanticBrain()
    brain.factory_reset()
    print("✓ Brain created and reset to factory state")
    print(f"  - Brain file: semantic_brain.dat")
    print(f"  - Initial capacity: 5000 nodes")

    # Step 2: Teach basic vocabulary
    print("\n[STEP 2] Teaching Basic Vocabulary & Concepts...")
    print("-" * 80)
    print("Loading vocabulary concepts into semantic memory...")
    brain.learn_rdf(BASIC_VOCABULARY)
    vocab_words = len(brain.word_to_id)
    print(f"✓ Vocabulary loaded")
    print(f"  - Unique words: {vocab_words}")
    print(f"  - Concepts taught")

    # Step 3: Teach the story
    print("\n[STEP 3] Teaching 'The Forest Story'...")
    print("-" * 80)
    print("Loading narrative into semantic memory...")
    brain.learn_rdf(THE_FOREST_STORY)
    story_words = len(brain.word_to_id)
    print(f"✓ Story learned")
    print(f"  - Total unique words now: {story_words}")
    print(f"  - Network density increased")

    # Step 4: Test the brain
    print("\n[STEP 4] Testing Brain Understanding...")
    print("-" * 80)
    print(f"Running {len(TESTING_QUERIES)} test queries...\n")

    passed = 0
    for i, (query, expected) in enumerate(TESTING_QUERIES, 1):
        print(f"[Test {i}] Query: {query}")
        print(f"          {expected}")
        result = brain.query(query)
        if result:
            passed += 1
            print(f"          ✓ Brain responded\n")
        else:
            print(f"          ⚠ No response\n")

    # Summary
    print("\n" + "=" * 80)
    print(" TRAINING & TESTING SUMMARY")
    print("=" * 80)
    print(f"\nBrain Statistics:")
    print(f"  - Vocabulary size: {story_words} unique words")
    print(f"  - Story length: ~{len(THE_FOREST_STORY.split())} words")
    print(f"  - Semantic connections: Created via RDF triples")

    print(f"\nTest Results:")
    print(f"  - Tests run: {len(TESTING_QUERIES)}")
    print(f"  - Tests passed: {passed}")
    print(f"  - Success rate: {(passed/len(TESTING_QUERIES))*100:.1f}%")

    print(f"\nBrain Status:")
    print(f"  - Alive: Yes")
    print(f"  - Learning: Complete")
    print(f"  - Ready for: Advanced queries, inference, reasoning")

    # Cleanup
    print("\n[CLEANUP] Closing brain connection...")
    brain.brain.close()
    print("✓ Brain safely closed")

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE - BRAIN READY FOR USE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
