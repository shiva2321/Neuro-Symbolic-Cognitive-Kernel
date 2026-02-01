
import sys
import os
import time

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from homeostasis import HomeostaticMonitor

def test_proto_self():
    print("Initializing Proto-Self...")
    me = HomeostaticMonitor()
    
    print(f"Initial State: {me}")
    
    # 1. Simulate Time Passage (Starvation)
    print("\nSimulating 500 ticks of starvation...")
    for _ in range(500):
        me.update()
        
    print(f"Starved State: {me}")
    print(f"Dominant Drive: {me.get_dominant_drive()}")
    
    hunger = me.drives['hunger']
    if hunger > 0.1:
        print("[SUCCESS] Starvation correctly generated 'Hunger' drive.")
    else:
        print("[FAILURE] Hunger drive did not rise.")
        
    # 2. Simulate Feeding
    print("\nFeeding...")
    me.consume("energy", 0.8)
    me.update()
    
    print(f"Fed State: {me}")
    if me.drives['hunger'] < hunger:
        print("[SUCCESS] Feeding correctly reduced 'Hunger' drive.")
    else:
        print("[FAILURE] Hunger did not decrease.")

if __name__ == "__main__":
    test_proto_self()
