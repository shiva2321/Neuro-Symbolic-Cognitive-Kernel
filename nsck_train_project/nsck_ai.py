#!/usr/bin/env python3
"""
NSCK AI - Production System
==========================

Load and use the production AI model.

Usage:
    python nsck_ai.py              # Interactive mode
    python nsck_ai.py "Your question"  # Direct query
"""

import pickle
import sys
from pathlib import Path

def load_model():
    """Load the production model."""
    model_path = Path(__file__).parent / 'models' / 'production_model.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure models/production_model.pkl exists")
        sys.exit(1)
    
    print("Loading production model...")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def interactive_mode(ai):
    """Run interactive conversation mode."""
    print("\n" + "="*70)
    print("NSCK AI v2.0.0 - Production")
    print("="*70)
    print("Type 'quit' to exit, 'clear' to reset context\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if query.lower() == 'clear':
                ai.clear_context()
                print("Context cleared.")
                continue
            
            response = ai.query(query)
            print(f"\nAI: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    """Main entry point."""
    ai = load_model()
    
    if len(sys.argv) > 1:
        # Direct query mode
        query = ' '.join(sys.argv[1:])
        response = ai.query(query)
        print(response)
    else:
        # Interactive mode
        interactive_mode(ai)

if __name__ == '__main__':
    main()
