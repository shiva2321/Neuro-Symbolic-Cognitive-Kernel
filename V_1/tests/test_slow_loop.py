import pytest
import torch
from ncgn.brain import Brain

class TestSlowLoop:
    
    def test_decision_logic(self):
        # Setup
        brain = Brain(use_embeddings=False, use_mock_reasoner=True)
        if not brain._has_perception:
            pytest.skip("Perception/Control modules not available.")
            
        # 1. Teach "Target" using a specific distinct pattern
        device = brain.vsa.device
        # Pattern A: Vertical Stripe
        clean_image = torch.zeros(1, 1, 28, 28).to(device)
        clean_image[:, :, :, 14] = 10.0 # High intensity stripe
        
        with torch.no_grad():
            hv = brain.bridge(brain.snn(clean_image))[0]
        brain.associative.add_concept("Target", hv)
        brain.add_concept("Target")
        
        # Add distractors for Z-score
        for i in range(10):
            brain.associative.add_concept(f"Noise_{i}")
            
        # 2. Test High Confidence (System 1)
        print("\n--- Testing High Confidence ---")
        # Use same image, should match perfectly
        res1 = brain.step(clean_image)
        decision1 = res1["decision"]
        z1 = res1["confidence_z"]
        print(f"Decision: {decision1} (Z={z1:.2f})")
        print(f"Label Found: {res1['label']}")
        
        # Lower threshold for this test since we adjusted Decider to 80
        # If perfect match Z is around 90-100.
        assert "SYSTEM_1" in decision1
        
        # 3. Test Low Confidence (System 2 - Self Doubt)
        print("\n--- Testing Ambiguity (Self-Doubt) ---")
        # Pattern B: Horizontal Stripe (Should be distinct in SNN Convolution)
        noise_image = torch.zeros(1, 1, 28, 28).to(device)
        noise_image[:, :, 14, :] = 10.0
        
        res2 = brain.step(noise_image)
        decision2 = res2["decision"]
        z2 = res2["confidence_z"]
        print(f"Decision: {decision2} (Z={z2:.2f})")
        print(f"Label Found: {res2['label']}")
        
        # Should trigger System 2
        # Z score should be low (unless we got incredibly lucky)
        assert "SYSTEM_2" in decision2
        assert z2 < 80.0 # Threshold default is 80.0
        
if __name__ == "__main__":
    t = TestSlowLoop()
    t.test_decision_logic()
    print("Test passed!")
