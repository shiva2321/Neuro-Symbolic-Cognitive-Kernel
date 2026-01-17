import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.flash import FlashColony

BRAIN_FILE = "smart_brain.dat"


def setup_fresh_brain():
    """Create and wire a small colony representing the hot stove circuit."""
    print(f"\U0001f9e0 Creating new brain: {BRAIN_FILE}...")
    if os.path.exists(BRAIN_FILE):
        os.remove(BRAIN_FILE)

    colony = FlashColony(BRAIN_FILE, max_nodes=100)
    colony.create_new(max_nodes=100)

    # Node 0: Sight of Stove (Input)
    # Node 1: Heat/Pain (Input)
    # Node 2: Pull Hand (Motor Output)
    colony.add_node(0, node_type="input")
    colony.add_node(1, node_type="input")
    colony.add_node(2, node_type="output", threshold=0.8)

    # Instinct: Heat -> Pull Hand (strong)
    colony.connect_binary(1, 2, weight=1.5)

    # Ignorance baseline: Sight -> Pull Hand (weak)
    colony.connect_binary(0, 2, weight=0.1)

    print("\u2713 Circuit wired.")
    print("  [1] Heat  ---(1.5)--> [2] Action (Instinct)")
    print("  [0] Sight ---(0.1)--> [2] Action (Ignorance)")
    return colony


def run_training_session():
    """Phase 1: teach the brain that Sight implies Pain."""
    print("\nSESSION 1: TRAINING PROTOCOL")
    print("-" * 40)

    colony = setup_fresh_brain()

    try:
        # Verify ignorance first
        print("   Testing Baseline: Showing 'Sight' only...")
        colony.set_input(0, 1.0)  # Input: Sight
        colony.step()
        if colony.is_firing(2):
            print("   \u274c FAILURE: Brain reacted before learning!")
            return
        else:
            print("   \u2713 Brain ignored the sight (as expected).")

        print("\n   Conditioning: Pairing Sight + Heat...")
        for _ in range(10):
            colony.set_input(0, 1.0)  # Sight
            colony.set_input(1, 1.0)  # Heat (triggers Pull Hand)
            current = colony.get_synapse_weight(0, 0) or 0.1
            colony.update_synapse_weight(0, 0, current + 0.08)  # reinforce association
            colony.step(global_dopamine=0.8, learning_rate=0.1)
        print("   \u2713 Training Complete. FlashColony auto-saved to disk.")

    finally:
        colony.close()
        print("SIMULATION TERMINATED. MEMORY DUMPED TO DISK.")


def run_recall_session():
    """Phase 2: wake up the brain and test memory."""
    print("\nSESSION 2: RECALL TEST (New Process)")
    print("-" * 40)

    if not os.path.exists(BRAIN_FILE):
        print("\u274c No brain file found! Run training first.")
        return

    colony = FlashColony(BRAIN_FILE)
    colony.connect_to_file()
    for nid in (0, 1, 2):
        colony.get_node(nid)

    try:
        print("   Loading brain state from disk...")
        print("   ACTION: Showing 'Sight' (Node 0)...")

        colony.set_input(0, 1.0)
        colony.step(global_dopamine=0.0)
        colony.step(global_dopamine=0.0)

        motor_node = colony.get_node(2)
        potential = motor_node.potential if motor_node else 0.0
        print(f"   OBSERVATION: Motor Node Potential: {potential:.2f}")

        if colony.is_firing(2):
            print("   SUCCESS: The brain PULLED THE HAND!")
            print("   The association (Sight -> Action) was successfully retrieved from disk.")
        else:
            print("   \u274c FAILURE: The brain stared blankly.")
            print("   Memory was lost or weight update failed.")

    finally:
        colony.close()


if __name__ == "__main__":
    run_training_session()
    print("\n... Simulating time passing (Process Restart) ...\n")
    time.sleep(2)
    run_recall_session()
