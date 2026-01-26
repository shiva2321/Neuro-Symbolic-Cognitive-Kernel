import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ncgn.brain import Brain
from ncgn.config import DEFAULT_CONFIG

def main():
    print("Initializing NCGN v2.0 (Brain-First Architecture)...")
    
    # Initialize Brain without LLM for now (Phase 1 verification)
    brain = Brain(config=DEFAULT_CONFIG, use_mock_reasoner=True)
    
    print(f"Brain loaded: {brain}")
    print("Ready for input. Type 'q' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        # Basic processing
        result = brain.process_input(user_input)
        
        print("\nBrain:")
        print(f"> Active Concepts: {len(result['active_after'])}")
        print(f"> Surprise: {result['surprise']:.4f}")
        
        top_concepts = brain.get_top_concepts(5)
        if top_concepts:
            print(f"> Top: {', '.join([f'{k}({v:.2f})' for k,v in top_concepts.items()])}")

if __name__ == "__main__":
    main()
