"""
Pavlov's Silicon Dog: Classical Conditioning Experiment
Demonstrates 3-Factor STDP learning without backpropagation.

Experiment:
1. Setup: Bell (input 0), Food (input 1), Salivate (output 2)
2. Instinct: Strong connection Food -> Salivate (hardcoded)
3. Phase 1: Bell only -> No salivation (baseline)
4. Phase 2: Bell + Food + Dopamine (training, 15 trials)
5. Phase 3: Bell only -> Salivation! (learned association)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.core import NeuromorphicNetwork
import time


class PavlovExperiment:
    def __init__(self):
        self.network = NeuromorphicNetwork()
        self.setup_network()

        # Tracking for visualization
        self.history = {
            'bell_salivate_weight': [],
            'food_salivate_weight': [],
            'salivate_fired': []
        }

    def setup_network(self):
        """
        Create the 3-node network:
        - Node 0: Bell (input)
        - Node 1: Food (input)
        - Node 2: Salivate (output)
        """
        # Create nodes
        self.network.add_node(0, node_type="input", threshold=0.5)  # Bell
        self.network.add_node(1, node_type="input", threshold=0.5)  # Food
        self.network.add_node(2, node_type="output", threshold=1.0, refractory_period=5)  # Salivate

        # Bell -> Salivate: Weak initial connection (to be learned)
        self.network.connect(source_id=0, target_id=2, weight=0.1)

        # Food -> Salivate: STRONG instinctive connection (hardcoded)
        self.network.connect(source_id=1, target_id=2, weight=1.5)

        print("✓ Network initialized")
        print(f"  Bell -> Salivate: {self.network.get_weight(0, 2):.3f}")
        print(f"  Food -> Salivate: {self.network.get_weight(1, 2):.3f}")

    def ring_bell(self):
        """Activate Bell neuron."""
        self.network.set_input(0, 1.5)  # Strong input

    def present_food(self):
        """Activate Food neuron."""
        self.network.set_input(1, 1.5)  # Strong input

    def clear_inputs(self):
        """Reset all input neurons."""
        self.network.set_input(0, 0.0)
        self.network.set_input(1, 0.0)

    def run_trial(self, bell=False, food=False, dopamine=0.0, ticks=10):
        """
        Run a single trial with specified stimuli.

        Args:
            bell: Present bell stimulus
            food: Present food stimulus
            dopamine: Reward signal strength
            ticks: Duration of trial in timesteps

        Returns:
            True if Salivate node fired during trial
        """
        salivated = False

        for t in range(ticks):
            # Set inputs
            self.clear_inputs()
            if bell:
                self.ring_bell()
            if food:
                self.present_food()

            # Step the network
            self.network.step(global_dopamine=dopamine, learning_rate=0.05)

            # Check if salivation occurred
            if self.network.is_firing(2):
                salivated = True

        return salivated

    def record_state(self):
        """Record current weights and states for visualization."""
        self.history['bell_salivate_weight'].append(
            self.network.get_weight(0, 2)
        )
        self.history['food_salivate_weight'].append(
            self.network.get_weight(1, 2)
        )
        self.history['salivate_fired'].append(
            1 if self.network.is_firing(2) else 0
        )

    def print_dashboard(self, phase, trial, salivated):
        """Real-time text dashboard showing learning progress."""
        bell_weight = self.network.get_weight(0, 2)
        food_weight = self.network.get_weight(1, 2)

        # Create visual weight bars
        bell_bar = '█' * int(bell_weight * 20) + '░' * (40 - int(bell_weight * 20))
        food_bar = '█' * int(food_weight * 20) + '░' * (40 - int(food_weight * 20))

        print(f"\n{'='*60}")
        print(f"  {phase} - Trial {trial}")
        print(f"{'='*60}")
        print(f"  Bell → Salivate: [{bell_bar}] {bell_weight:.4f}")
        print(f"  Food → Salivate: [{food_bar}] {food_weight:.4f}")
        print(f"  Salivate Node:   {'🔔 FIRED!' if salivated else '💤 Silent'}")
        print(f"{'='*60}")

    def run_experiment(self):
        """Execute the full Pavlov experiment."""
        print("\n" + "="*60)
        print("  🧠 PAVLOV'S SILICON DOG EXPERIMENT")
        print("  Neuromorphic 3-Factor STDP Learning")
        print("="*60)

        # ============ PHASE 1: Baseline (Bell Only) ============
        print("\n📋 PHASE 1: BASELINE TEST (Bell Only, No Dopamine)")
        print("-" * 60)

        self.network.reset()
        salivated = self.run_trial(bell=True, food=False, dopamine=0.0, ticks=10)
        self.print_dashboard("PHASE 1: BASELINE", 1, salivated)

        if salivated:
            print("⚠️  WARNING: Salivated before training! (Should not happen)")
        else:
            print("✓ PASS: No salivation to bell alone (expected)")

        time.sleep(1)

        # ============ PHASE 2: Training (Bell + Food + Dopamine) ============
        print("\n📋 PHASE 2: TRAINING (Bell + Food + Dopamine)")
        print("-" * 60)

        training_trials = 15
        for trial in range(1, training_trials + 1):
            self.network.reset()
            salivated = self.run_trial(
                bell=True,
                food=True,
                dopamine=1.0,  # REWARD SIGNAL
                ticks=10
            )

            if trial % 3 == 0 or trial == 1:  # Show every 3rd trial
                self.print_dashboard("PHASE 2: TRAINING", trial, salivated)
                time.sleep(0.3)

        print("\n✓ Training complete!")
        time.sleep(1)

        # ============ PHASE 3: Test (Bell Only, Post-Training) ============
        print("\n📋 PHASE 3: TEST (Bell Only, No Food, No Dopamine)")
        print("-" * 60)

        self.network.reset()
        salivated = self.run_trial(bell=True, food=False, dopamine=0.0, ticks=10)
        self.print_dashboard("PHASE 3: TEST", 1, salivated)

        # ============ RESULTS ============
        print("\n" + "="*60)
        print("  📊 EXPERIMENT RESULTS")
        print("="*60)

        bell_weight_final = self.network.get_weight(0, 2)
        bell_weight_initial = 0.1
        weight_increase = bell_weight_final - bell_weight_initial

        print(f"  Initial Bell→Salivate Weight: {bell_weight_initial:.4f}")
        print(f"  Final Bell→Salivate Weight:   {bell_weight_final:.4f}")
        print(f"  Weight Increase:              +{weight_increase:.4f}")
        print(f"  Learned Association:          {'✓ YES' if salivated else '✗ NO'}")
        print("="*60)

        if salivated and weight_increase > 0.5:
            print("\n🎉 SUCCESS! Classical conditioning achieved!")
            print("   The Bell neuron alone now triggers salivation.")
            print("   The synapse strengthened through 3-Factor STDP.")
        else:
            print("\n⚠️  Learning did not fully complete.")
            print("   Try increasing training trials or learning rate.")

        print("\n" + "="*60)
        print("  🔬 KEY INSIGHTS")
        print("="*60)
        print("  ✓ NO matrix multiplication used")
        print("  ✓ NO backpropagation performed")
        print("  ✓ Learning via 3-Factor STDP (trace × dopamine)")
        print("  ✓ Sparse graph architecture (flash-ready)")
        print("  ✓ Biologically plausible (refractory period, leak, etc.)")
        print("="*60 + "\n")


def main():
    """Run the Pavlov's Dog experiment."""
    experiment = PavlovExperiment()
    experiment.run_experiment()


if __name__ == "__main__":
    main()

