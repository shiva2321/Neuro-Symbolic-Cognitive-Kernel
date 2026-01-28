"""
NCGN v2.0 Core Verification Script
Verifies:
1. Auto-Hebbian Learning (No LLM)
2. Confidence Tracking
3. Conflict Engine instantiation
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ncgn.brain import Brain
from ncgn.config import Config

def verify_core():
    print("Initializing Brain...")
    brain = Brain(use_mock_reasoner=True, use_embeddings=False)
    
    # Check components
    assert brain.confidence is not None, "ConfidenceTracker missing"
    assert brain.conflicts is not None, "ConflictDetector missing"
    assert brain.decision is not None, "DecisionEngine missing"
    print("✅ Components initialized")
    
    # Test Auto-Hebbian Learning
    print("\nTesting Auto-Hebbian Learning...")
    brain.add_concept("node_a")
    brain.add_concept("node_b")
    
    # Verify no edge initially
    assert brain.get_connection_weight("node_a", "node_b") is None
    
    # Co-activate multiple times
    # HebbianLearner.auto_learn trigger threshold is 5 (default)
    # We call learn(reward=1.0) to trigger it.
    
    for i in range(10):
        # Manually set high activation
        brain.set_concept_energy("node_a", 1.0)
        brain.set_concept_energy("node_b", 1.0)
        
        # Apply reward (triggers auto_learn)
        updates = brain.learn(reward=1.0)
        print(f"Update {i+1}: {updates} synapses updated")
        
        # Check if edge formed
        w = brain.get_connection_weight("node_a", "node_b")
        if w is not None:
            print(f"✅ Edge formed at iteration {i+1} with weight {w:.2f}")
            break
            
    assert brain.get_connection_weight("node_a", "node_b") is not None, "Auto-learning failed to create edge"
    
    # Test Confidence
    conf = brain.confidence.get_confidence("node_a", "node_b")
    print(f"✅ New edge confidence: {conf}")
    assert conf > 0.0, "Confidence not tracked"

    print("\n✅ CORE BRAIN UPGRADES VERIFIED")

if __name__ == "__main__":
    verify_core()
