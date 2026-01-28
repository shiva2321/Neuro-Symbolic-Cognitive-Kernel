import pytest
import torch
import torch.nn as nn
from ncgn.brain import Brain

class TestFastLoop:
    
    @pytest.fixture
    def brain(self):
        b = Brain(use_embeddings=False, use_mock_reasoner=True)
        return b
        
    def test_pipeline(self, brain):
        """
        Verify the SNN -> VSA -> Graph injection loop.
        """
        if not brain._has_perception:
            pytest.skip("Perception modules not available.")
            
        # 1. Setup: Create a concept "Digit_5" in the brain
        brain.add_concept("Digit_5")
        
        # Add distractors to form a noise floor for Z-score
        for i in range(10):
            brain.add_concept(f"Noise_{i}")
            brain.associative.add_concept(f"Noise_{i}")
        
        # 2. "One-Shot Learning" Preparation
        # Create a fake input image (Batch, 1, 28, 28)
        # In a real scenario, we'd use a real training loop, but for One-Shot,
        # we can just run the encoder once to get a prototype vector.
        device = brain.vsa.device
        dummy_image = torch.rand(1, 1, 28, 28).to(device)
        
        # Get the hypervector representation of this image
        with torch.no_grad():
            spike_rate = brain.snn(dummy_image)
            hv = brain.bridge(spike_rate)
            hv = hv[0] # Unbatch
            
        # Register this vector as "Digit_5" in the associative memory
        brain.associative.add_concept("Digit_5", hv)
        
        # 3. Test Perception
        # Pass the same image (or slightly noisy)
        noisy_image = dummy_image + torch.randn_like(dummy_image) * 0.1
        
        result = brain.process_perception(noisy_image)
        
        print("\nPerception Result:", result)
        
        # 4. Verify Recognition
        assert result["label"] == "Digit_5"
        assert result["confidence_z"] > 2.0
        
        # Verify concept exists
        assert brain.has_concept("Digit_5"), "Concept Digit_5 missing from Brain topology!"
        idx = brain.topology.registry.get_index("Digit_5")
        print(f"Digit_5 Index: {idx}")
        
        # Test Manual Injection first
        print("Attempting manual injection...")
        brain.inject("Digit_5", 0.5)
        brain.think(1)
        manual_energy = brain.get_concept_energy("Digit_5")
        print(f"Manual Injection Energy: {manual_energy}")
        
        # Reset and try Perception
        # (Though state is accumulated, so result should increase)
        
        # Propagate to convert injected buffer from Perception (which ran before)
        # Wait, Perception ran inject() BEFORE this manual inject. 
        # so previous buffer was processed?
        # NO. We called brain.think(1) above for manual.
        # Let's check energy NOW.
        # Check energy of Digit_5
        # Check energy of Digit_5 or if it fired
        energy = brain.get_concept_energy("Digit_5")
        firing_set = brain.engine.get_firing_set()
        print(f"Graph Energy for Digit_5: {energy}")
        print(f"Firing Set: {firing_set}")
        
        assert (energy > 0.0) or ("Digit_5" in firing_set), "Concept should have energy or have fired"
        
if __name__ == "__main__":
    t = TestFastLoop()
    # Manual setup
    b = Brain(use_embeddings=False, use_mock_reasoner=True)
    t.test_pipeline(b)
    print("Test passed!")
