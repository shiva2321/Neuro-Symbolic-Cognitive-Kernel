"""
Verify Episodic Memory Upgrade (Phase 3.1)
==========================================
Tests storage of Emotion and TheoryOfMind states in EpisodicMemory.
"""

import time
from episodic_memory import LiveEpisode

# Mock HyperVector for testing
class MockHV:
    def __init__(self):
        self.bits = b'\x00' * 8
    def lsh_hash(self, seed, bits): return 0
    def similarity(self, other): return 0

def main():
    print("--- Testing LiveEpisode Upgrade ---")
    
    # 1. Create Episode with new fields
    ep = LiveEpisode(
        timestamp=time.time(),
        task_tag="snake",
        situation_hv=MockHV(),
        state={"head": (5,5), "food": (2,2), "body": [(5,5), (4,5)]},
        action="UP",
        outcome="move",
        reward=0.1,
        emotion="joy",  # New field
        tom_beliefs={"Sally": "basket"} # New field
    )
    
    print(f"Created Episode with Emotion: {ep.emotion}")
    print(f"Created Episode with ToM: {ep.tom_beliefs}")
    
    # 2. Test Persistence Conversion (Sketch Extraction)
    stored = ep.to_stored()
    sketch = stored.state_sketch
    
    print(f"\nStored Sketch: {sketch}")
    
    # 3. Assertions
    if sketch.get("emotion") == "joy":
        print("[PASS] Emotion preserved in storage sketch.")
    else:
        print("[FAIL] Emotion lost in storage sketch.")
        
    if sketch.get("tom_beliefs") == {"Sally": "basket"}:
        print("[PASS] ToM Beliefs preserved in storage sketch.")
    else:
        print("[FAIL] ToM Beliefs lost or corrupted.")

if __name__ == "__main__":
    main()
