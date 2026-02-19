import argparse
import os
import sys
from python.core.language.lingua_cortex import SemanticMap, SemanticFoldingTrainer, get_lingua_cortex

# Small demo corpus for testing without 10GB Wikipedia
DEMO_CORPUS = [
    "The cat chased the mouse via the garden.",
    "A dog barked at the mailman in the street.",
    "Artificial intelligence is the future of computing.",
    "Neural networks learn to recognize patterns in data.",
    "The garden was full of blooming flowers and bees.",
    "Computing power has increased exponentially over the years.",
    "The mouse ate the cheese in the kitchen.",
    "Robots can perform tasks that are dangerous for humans.",
    "Machine learning algorithms require large datasets.",
    "The sun shines brightly in the clear blue sky."
]

def load_corpus(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"Corpus file {path} not found. Using demo corpus.")
        return DEMO_CORPUS
    
    with open(path, 'r', encoding='utf-8') as f:
        # Read lines, filter empty
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Train NSCK Semantic Folding Model")
    parser.add_argument("--corpus", type=str, default="demo", help="Path to text corpus (line separated)")
    parser.add_argument("--grid-size", type=int, default=128, help="Size of the semantic grid (e.g. 128)")
    parser.add_argument("--output", type=str, default="semantic_grid.pkl", help="Output path for the pickle file")
    
    args = parser.parse_args()
    
    # Initialize Map
    semantic_map = SemanticMap(size=args.grid_size)
    trainer = SemanticFoldingTrainer(semantic_map, grid_size=args.grid_size)
    
    # Load Data
    if args.corpus == "demo":
        print("Using built-in demo corpus.")
        corpus = DEMO_CORPUS
    else:
        corpus = load_corpus(args.corpus)
        
    print(f"Loaded {len(corpus)} documents.")
    
    # Train
    try:
        trainer.train_on_corpus(corpus)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
        
    # Save
    semantic_map.save(args.output)
    print(f"Saved Semantic Map to {args.output}")
    
    # Verification: Print some neighbors
    print("\n--- Verification ---")
    test_words = ["cat", "dog", "artificial", "intelligence", "garden"]
    for w in test_words:
        if w in semantic_map.vocab:
            neighbors = semantic_map.similar_words(w, top_k=3)
            print(f"{w}: {neighbors}")

if __name__ == "__main__":
    main()
