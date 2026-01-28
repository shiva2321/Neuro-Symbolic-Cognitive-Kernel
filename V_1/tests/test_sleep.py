import pytest
import torch
from ncgn.brain import Brain

class TestSleep:
    
    def test_sleep_consolidation(self):
        # Setup
        brain = Brain(use_embeddings=False, use_mock_reasoner=True)
        if not hasattr(brain, 'sleep_manager'):
            pytest.skip("Sleep module not available.")
            
        device = brain.vsa.device
        
        # 1. Wake Phase: Experience a concept multiple times
        print("\n--- Wake Phase ---")
        img = torch.zeros(1, 1, 28, 28).to(device)
        img[:, :, 10:20, 10:20] = 5.0 # Square
        
        # Register it once to allow recognition
        with torch.no_grad():
            hv = brain.bridge(brain.snn(img))[0]
        brain.associative.add_concept("Square", hv)
        brain.add_concept("Square", initial_energy=0.1) # Weak initial memory
        
        # Add distractors for Z-score
        for i in range(10):
            brain.associative.add_concept(f"Noise_{i}")
        
        # Experience it 5 times
        for _ in range(5):
            brain.step(img)
            
        # Verify STM Buffer
        print(f"STM Buffer Size: {len(brain.stm.buffer)}")
        assert len(brain.stm.buffer) == 5
        
        # Check LTM Energy before sleep (should be low/decayed)
        energy_before = brain.get_concept_energy("Square")
        print(f"LTM Energy Before Sleep: {energy_before:.3f}")
        
        # 2. Sleep Phase
        print("\n--- Sleep Phase ---")
        stats = brain.sleep()
        print(f"Sleep Stats: {stats}")
        
        # 3. Verification
        # STM should be empty
        assert len(brain.stm.buffer) == 0
        
        # LTM should be strengthened
        # (Avg energy of experiences was high because injection happens on confident rec)
        # Default injection is confidence-weighted, likely > 0.5
        energy_after = brain.get_concept_energy("Square")
        firing_set = brain.engine.get_firing_set()
        print(f"LTM Energy After Sleep: {energy_after:.3f}")
        print(f"Firing Set After Sleep: {firing_set}")
        
        assert stats["strengthened"] == 1
        # Energy might be 0 if it just fired (Refractory), so check if it fired
        assert (energy_after > energy_before) or ("Square" in firing_set)
        
if __name__ == "__main__":
    t = TestSleep()
    t.test_sleep_consolidation()
    print("Test passed!")
