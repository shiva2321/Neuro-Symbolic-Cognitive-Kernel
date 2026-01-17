import sys
import time
from storage.flash_colony import FlashColony

BRAIN_FILE = "main_brain.dat"
VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"
CHAR_TO_ID = {c: i+1 for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i+1: c for i, c in enumerate(VOCAB)}

def inspect_brain():
    print(f"🩺 BRAIN SURGEON: Inspecting {BRAIN_FILE}...")

    try:
        brain = FlashColony(BRAIN_FILE)
    except:
        print("❌ CRITICAL: Could not open brain file.")
        return

    brain.connect_to_file()
    print(f"   Nodes: {brain.max_nodes}")

    # 1. CHECK WEIGHTS (Did training work?)
    print("\n🔍 CHECKING MEMORIES (Synaptic Weights)")

    # Check "N" -> " " (End of HUMAN)
    n_id = CHAR_TO_ID['N']
    space_id = CHAR_TO_ID[' ']

    # We need to find the synapse index
    n_node = brain.get_node(n_id)
    found_link = False

    print(f"   Inspecting Node 'N' (ID {n_id})...")
    print(f"   Edge Count: {n_node.edge_count}")

    for idx, (tid, w, t, p) in enumerate(n_node.iter_synapses()):
        if tid == space_id:
            print(f"   ✅ LINK FOUND: 'N' -> 'SPACE'")
            print(f"      Weight: {w:.4f} (Should be > 0.1)")
            if w < 0.1: print("      ⚠️ WARNING: Weak connection!")
            found_link = True
            break

    if not found_link:
        print("   ❌ CRITICAL: No link between 'N' and 'SPACE'. Training failed.")

    # Check "SPACE" -> "W" (Start of WOKE/WAS/WANTED)
    w_id = CHAR_TO_ID['W']
    space_node = brain.get_node(space_id)
    found_link = False

    print(f"\n   Inspecting Node 'SPACE' (ID {space_id})...")
    for idx, (tid, w, t, p) in enumerate(space_node.iter_synapses()):
        if tid == w_id:
            print(f"   ✅ LINK FOUND: 'SPACE' -> 'W'")
            print(f"      Weight: {w:.4f}")
            found_link = True
            break

    # 2. RUN PHYSICS DIAGNOSTIC
    print("\n⚡ PHYSICS STIMULATION TEST")
    print("   Firing 'SPACE' neuron manually...")

    brain.reset()

    # Force Fire
    space_node = brain.get_node(space_id)
    space_node.refractory_counter = 0 # Force Ready
    brain.set_input(space_id, 2.0)
    brain.step(0,0)

    # Check neighbors
    print("   Checking resulting potentials in neighbors...")
    active_count = 0
    for nid in ID_TO_CHAR:
        if nid == space_id: continue
        pot = brain.get_node(nid).potential
        if pot > 0.0:
            char = ID_TO_CHAR[nid]
            print(f"   -> '{char}' Potential: {pot:.4f}")
            active_count += 1

    if active_count == 0:
        print("   ❌ SILENCE: Source fired, but no energy propagated.")
    else:
        print(f"   ✅ SUCCESS: Energy propagated to {active_count} nodes.")

    brain.close()

if __name__ == "__main__":
    inspect_brain()