from context_driver import SemanticBrain

TRAINING_DATA = """
    THE SUN IS STAR. 
    THE EARTH IS PLANET.
    FIRE IS HOT.
    ICE IS COLD.
    BIRDS EAT WORMS.
    FISH EAT WORMS.
    PROGRAMMERS WRITE CODE.
    BANK KEEPS MONEY.
    RIVER HAS BANK.
"""

def run_test():
    ai = SemanticBrain()
    ai.factory_reset()
    ai.learn_rdf(TRAINING_DATA)

    # 1. The "Hot" Test (Previously failed)
    # Logic: Find X where X --[IS]--> HOT.
    # At node HOT, we look for reverse link -IS.
    # FIRE connects to HOT with IS. EARTH connects to PLANET with IS.
    # They are physically separated now. No cross-talk.
    ai.query("What is hot?")

    # 2. The "Worm" Test (Polymorphism)
    ai.query("What eat worms?")

    # 3. The "Code" Test
    ai.query("Who write code?")

    # 4. The "Disambiguation" Test
    # Bank (Money) vs Bank (River)
    ai.query("What keeps money?")

    ai.brain.close()

if __name__ == "__main__":
    run_test()