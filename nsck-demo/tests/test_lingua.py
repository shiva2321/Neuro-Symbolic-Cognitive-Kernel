
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lingua_cortex import SemanticMap

def test_semantic_learning():
    print("Initializing Semantic Cortex...")
    cortex = SemanticMap()
    
    # 1. Train on some simple "contexts" (snippets)
    print("Learning snippets...")
    
    # Context 1: Food-related
    cortex.learn_text_snippet("The snake eats the apple is food")
    cortex.learn_text_snippet("food provides energy to the body")
    cortex.learn_text_snippet("apple is a red fruit tasty")
    
    # Context 2: Tech/Game related
    cortex.learn_text_snippet("python is a coding language")
    cortex.learn_text_snippet("the engine runs the game loop")
    cortex.learn_text_snippet("coding requires logic and coffee")
    
    # 2. Check overlap
    # 'snake' and 'apple' should be statistically related due to snippet 1
    snake = cortex.get_fingerprint("snake")
    apple = cortex.get_fingerprint("apple")
    coding = cortex.get_fingerprint("coding")
    
    sim_snake_apple = snake.overlap(apple)
    sim_snake_coding = snake.overlap(coding)
    
    print(f"Similarity(Snake, Apple): {sim_snake_apple:.4f}")
    print(f"Similarity(Snake, Coding): {sim_snake_coding:.4f}")
    
    if sim_snake_apple > sim_snake_coding:
        print("[SUCCESS] Contextual overlap works: Snake is closer to Apple than Coding.")
    else:
        print("[FAILURE] Context differentiation failed.")
        
    # 3. Disambiguation (Logic)
    # "Apple" (Fruit) vs "Apple" (Tech - if we had trained it)
    # Here we test simple boolean operations
    
    print("\nTesting Boolean Algebra...")
    # New concept: "fruit food"
    fruit = cortex.get_fingerprint("fruit")
    food = cortex.get_fingerprint("food")
    
    # Union (Superposition)
    fruit_food = fruit.union(food)
    print(f"Union Density: {fruit_food.density:.4f}")
    
    # Intersection logic
    # Intersection of Snake and Apple should be non-zero (shared context)
    shared_context = snake.intersection(apple)
    print(f"Shared Context Density: {shared_context.density:.4f}")
    
    if shared_context.density > 0:
         print("[SUCCESS] Intersection logic works.")

if __name__ == "__main__":
    test_semantic_learning()
