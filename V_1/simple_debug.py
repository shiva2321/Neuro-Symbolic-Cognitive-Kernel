from ncgn.brain import Brain
import torch
import numpy as np

def debug():
    b = Brain(use_embeddings=False, use_mock_reasoner=True)
    b.add_concept("Test")
    idx = b.topology.registry.get_index("Test")
    with open("debug_out.txt", "w") as f:
        f.write(f"Index: {idx}\n")
        
        b.inject("Test", 2.0)
        
        # Check buffer
        buf = b.engine._external_buffer[idx]
        f.write(f"Buffer before think: {buf}\n")
        
        b.think(1)
        
        activation = b.get_concept_energy("Test")
        f.write(f"Activation: {activation}\n")
        
        # Check stats
        f.write(f"Stats: {b.get_stats()}\n")
    
    print("Done")

if __name__ == "__main__":
    debug()
