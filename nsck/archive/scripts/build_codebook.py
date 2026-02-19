import pickle
import sys
import os

# Ensure rust_vsa is available
# In a real scenario, maturin develop would install it to site-packages.
# Here we assume it's installed or we might need to point sys.path if built locally in place.
try:
    import python.core.vsa.hypervec_py
except ImportError:
    print("hypervec_py not found. Ensure you have built the rust crate with 'maturin develop'.")
    sys.exit(1)

def build_codebook():
    print("Generating VSA Codebook...")
    
    # Atomic Vectors
    concepts = [
        "NORTH", "SOUTH", "EAST", "WEST",
        "APPLE", "WALL", "EMPTY", "SNAKE_BODY",
        "INTENT_PLAY_SNAKE", "INTENT_UNKNOWN"
    ]
    
    codebook = {}
    
    for concept in concepts:
        # Create a random hypervector for each atomic concept
        hv = hypervec_py.HyperVector(seed=None) 
        codebook[concept] = hv
        
    # Example Composite Rules (Perception -> Action mapping in memory)
    # If Perception is APPLE and Direction is NORTH -> Move NORTH
    # We can represent rules as Bundles of Bound pairs. 
    # Rule = (Context * Condition) + (Action * Result) ... dependent on VSA algebra used.
    # For this prototype key-value store might be sufficient context.
    
    print(f"Codebook generated with {len(codebook)} concepts.")
    
    with open("codebook.pkl", "wb") as f:
        pickle.dump(codebook, f)
        
    print("Saved to codebook.pkl")

if __name__ == "__main__":
    build_codebook()
