
import sys
import os
import numpy as np

# Ensure we can import from the current directory

from python.core.perception.perception import FusionEngine, CleanupMemory, HV_DIM
from python.interfaces.voice_hd import VoiceHDEngine # To generate audio vectors

def test_fusion():
    print("Initializing Sensor Fusion...")
    fusion = FusionEngine(seed=101)
    cleanup_net = CleanupMemory()
    
    # 1. Create Data
    # Audio: VoiceHD vector for "Dog Bark" (Simulated random for now, or use encode)
    # Visual: Random vector for "Dog Image"
    
    rng = np.random.RandomState(99)
    
    # Concept 1: DOG
    v_dog_audio = rng.randint(0, 2, size=HV_DIM).astype(bool)
    v_dog_visual = rng.randint(0, 2, size=HV_DIM).astype(bool)
    
    # Concept 2: CAT
    v_cat_audio = rng.randint(0, 2, size=HV_DIM).astype(bool)
    v_cat_visual = rng.randint(0, 2, size=HV_DIM).astype(bool)
    
    # Teach Cleanup Memory the "Canonical" vectors
    cleanup_net.learn("Audio_Dog", v_dog_audio)
    cleanup_net.learn("Audio_Cat", v_cat_audio)
    cleanup_net.learn("Visual_Dog", v_dog_visual)
    cleanup_net.learn("Visual_Cat", v_cat_visual)
    
    # 2. Fuse
    print("\nFusing Dog Audio + Dog Visual...")
    v_dog_fused = fusion.fuse(v_dog_audio, v_dog_visual)
    
    # 3. Retrieve Audio from Fusion
    print("\nRetrieving Audio from Dog Fusion (Target: Audio_Dog)...")
    label, sim = fusion.retrieve(v_dog_fused, "audio", cleanup_net)
    
    print(f"Result: {label} (Sim: {sim:.4f})")
    
    # Baseline check
    # Sim should be around 0.75 (0.5 + 0.5 * 0.5)
    # 0.5 matching bits from A.
    # 0.5 matching bits from Noise (chance).
    # Total 0.75.
    
    if label == "Audio_Dog" and sim > 0.7:
        print(">> PASSED: Audio retrieval successful.")
    else:
        print(">> FAILED: Retrieval match too low or wrong label.")
        
    # 4. Cross-Modal Association Test
    # "I see a Cat, what does it sound like?"
    # This requires an associative memory linking Visual -> Audio directly?
    # OR, we assume the system holds the Fused concept "Cat Concept".
    # And we query that.
    
    print("\nFusing Cat Audio + Cat Visual...")
    v_cat_fused = fusion.fuse(v_cat_audio, v_cat_visual)
    
    print("Retrieving Visual from Cat Fusion (Target: Visual_Cat)...")
    label_v, sim_v = fusion.retrieve(v_cat_fused, "visual", cleanup_net)
    print(f"Result: {label_v} (Sim: {sim_v:.4f})")
    
    if label_v == "Visual_Cat":
        print(">> PASSED: Visual retrieval successful.")

if __name__ == "__main__":
    test_fusion()
