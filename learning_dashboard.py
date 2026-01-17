"""
Interactive Learning Dashboard - Comprehensive Monitoring System
Showcases the successful Pavlov's Dog experiment with full visualization and controls.
"""

from pavlov_experiment import PavlovExperiment
import time
import os


class ComprehensiveDashboard:
    def __init__(self):
        self.experiment = None
        self.running = True
        self.trained = False

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print dashboard header."""
        print("╔" + "═" * 98 + "╗")
        print("║" + " " * 30 + "🧠 NEUROMORPHIC LEARNING DASHBOARD" + " " * 33 + "║")
        print("║" + " " * 32 + "3-Factor STDP Neural Network" + " " * 37 + "║")
        print("╚" + "═" * 98 + "╝")

    def print_main_menu(self):
        """Print main menu."""
        print("\n┌─ MAIN MENU " + "─" * 86 + "┐")
        print("│                                                                                                  │")
        print("│  [1] Initialize New Network                                                                      │")
        print("│  [2] Run Full Training (Pavlov's Dog Experiment)                                                 │")
        print("│  [3] View Current Weights                                                                        │")
        print("│  [4] Manual Test Mode (Test individual inputs)                                                   │")
        print("│  [5] View Learning Statistics                                                                    │")
        print("│  [6] Run Custom Training (specify epochs)                                                        │")
        print("│  [0] Exit Dashboard                                                                              │")
        print("│                                                                                                  │")
        print("└" + "─" * 98 + "┘")

    def show_status(self):
        """Show current status."""
        if self.experiment:
            bell_weight = self.experiment.network.get_weight(0, 2)
            food_weight = self.experiment.network.get_weight(1, 2)
            training_status = "🟢 TRAINED" if self.trained else "🟡 UNTRAINED"

            print(f"\n  Status: {training_status}")
            print(f"  Bell→Salivate: {bell_weight:.4f}  |  Food→Salivate: {food_weight:.4f}")
        else:
            print("\n  Status: 🔴 NO NETWORK LOADED")

    def initialize_network(self):
        """Create new network."""
        self.clear_screen()
        self.print_header()

        print("\n┌─ NETWORK INITIALIZATION " + "─" * 73 + "┐")
        print("│                                                                                                  │")
        print("│  Creating new Pavlov's Dog network...                                                            │")
        print("│                                                                                                  │")
        print("└" + "─" * 98 + "┘\n")

        self.experiment = PavlovExperiment()
        self.trained = False

        print("\n✓ Network initialized successfully!\n")

        # Show architecture
        print("  📐 NETWORK ARCHITECTURE:")
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │                                                 │")
        print("  │    [Bell Input] ─────┐                          │")
        print("  │                      │                          │")
        print("  │                      └──→ [Salivate Output]     │")
        print("  │                      ┌──→                       │")
        print("  │    [Food Input] ─────┘                          │")
        print("  │                                                 │")
        print("  └─────────────────────────────────────────────────┘")

        print("\n  📊 INITIAL WEIGHTS:")
        bell_w = self.experiment.network.get_weight(0, 2)
        food_w = self.experiment.network.get_weight(1, 2)

        bell_bar = '█' * int(bell_w * 30)
        food_bar = '█' * int(food_w * 30)

        print(f"    Bell → Salivate: [{bell_bar:<30}] {bell_w:.4f} (weak, to be learned)")
        print(f"    Food → Salivate: [{food_bar:<30}] {food_w:.4f} (strong instinct)")

        input("\n  [Press ENTER to continue]")

    def run_full_training(self):
        """Run complete Pavlov experiment."""
        if not self.experiment:
            print("\n  ⚠️  No network initialized! Please initialize first.")
            input("\n  [Press ENTER to continue]")
            return

        self.clear_screen()
        self.print_header()

        print("\n┌─ PAVLOV'S DOG EXPERIMENT " + "─" * 71 + "┐")
        print("│                                                                                                  │")
        print("│  This experiment demonstrates classical conditioning through neuromorphic learning.              │")
        print("│                                                                                                  │")
        print("└" + "─" * 98 + "┘\n")

        print("  📋 EXPERIMENT PROTOCOL:\n")
        print("    Phase 1: Baseline Test")
        print("      → Ring bell alone (no food, no reward)")
        print("      → Expected: No salivation\n")

        print("    Phase 2: Training (15 trials)")
        print("      → Ring bell + Present food + Release dopamine")
        print("      → The Bell→Salivate synapse should strengthen\n")

        print("    Phase 3: Test")
        print("      → Ring bell alone (no food, no reward)")
        print("      → Expected: Salivation! (learned response)\n")

        input("  Press ENTER to start experiment...")

        # Run experiment with detailed output
        print("\n" + "═" * 100)
        print("  PHASE 1: BASELINE TEST")
        print("═" * 100)

        self.experiment.network.reset()
        salivated_baseline = self.experiment.run_trial(bell=True, food=False, dopamine=0.0, ticks=10)

        bell_w = self.experiment.network.get_weight(0, 2)
        print(f"\n  Bell→Salivate Weight: {bell_w:.4f}")
        print(f"  Salivation Response: {'🔔 YES (unexpected!)' if salivated_baseline else '💤 NO (expected)'}")

        if not salivated_baseline:
            print("  ✓ BASELINE PASSED: No conditioned response yet")

        time.sleep(2)

        # Training phase
        print("\n" + "═" * 100)
        print("  PHASE 2: TRAINING")
        print("═" * 100)
        print("\n  Training with simultaneous Bell + Food + Dopamine reward signal...")
        print("\n  ┌" + "─" * 96 + "┐")
        print("  │ EPOCH │ BELL→SAL WEIGHT │ FOOD→SAL WEIGHT │ SALIVATION │ STATUS                              │")
        print("  ├" + "─" * 96 + "┤")

        for trial in range(1, 16):
            self.experiment.network.reset()
            salivated = self.experiment.run_trial(bell=True, food=True, dopamine=1.0, ticks=10)

            bell_w = self.experiment.network.get_weight(0, 2)
            food_w = self.experiment.network.get_weight(1, 2)

            bell_bar = '█' * min(int(bell_w * 10), 15)
            food_bar = '█' * min(int(food_w * 10), 15)

            sal_status = "🔔 YES" if salivated else "💤 NO "

            if trial % 3 == 0 or trial == 1:
                increase = bell_w - 0.1
                status_msg = f"Learning... (+{increase:.3f})"
                print(f"  │  {trial:2d}   │ {bell_w:.4f} {bell_bar:<15} │ {food_w:.4f} {food_bar:<15} │   {sal_status}    │ {status_msg:<35} │")

        print("  └" + "─" * 96 + "┘")

        self.trained = True
        time.sleep(2)

        # Test phase
        print("\n" + "═" * 100)
        print("  PHASE 3: FINAL TEST")
        print("═" * 100)

        print("\n  Testing with Bell ONLY (no food, no dopamine)...")
        time.sleep(1)

        self.experiment.network.reset()
        salivated_test = self.experiment.run_trial(bell=True, food=False, dopamine=0.0, ticks=10)

        final_bell_w = self.experiment.network.get_weight(0, 2)
        weight_increase = final_bell_w - 0.1

        print(f"\n  Bell→Salivate Weight: {final_bell_w:.4f} (increased by +{weight_increase:.4f})")
        print(f"  Salivation Response: {'🔔 YES! Learned!' if salivated_test else '💤 NO (learning failed)'}")

        # Results
        print("\n" + "═" * 100)
        print("  📊 EXPERIMENT RESULTS")
        print("═" * 100)

        print(f"\n  ✓ Baseline (before):  {'No salivation' if not salivated_baseline else 'Unexpected salivation'}")
        print(f"  ✓ After Training:     {'Salivation to bell!' if salivated_test else 'No salivation (failed)'}")
        print(f"  ✓ Weight Change:      0.1000 → {final_bell_w:.4f} (+{weight_increase:.4f})")

        if salivated_test and weight_increase > 0.5:
            print("\n  🎉 SUCCESS! Classical conditioning achieved through 3-Factor STDP!")
            print("     The network learned the Bell→Food association without backpropagation.")
        else:
            print("\n  ⚠️  Learning incomplete. Consider adjusting parameters.")

        print("\n" + "═" * 100)

        input("\n  [Press ENTER to continue]")

    def view_weights(self):
        """Display detailed weight information."""
        if not self.experiment:
            print("\n  ⚠️  No network initialized!")
            input("\n  [Press ENTER to continue]")
            return

        self.clear_screen()
        self.print_header()

        print("\n┌─ SYNAPTIC WEIGHTS " + "─" * 79 + "┐")

        bell_w = self.experiment.network.get_weight(0, 2)
        food_w = self.experiment.network.get_weight(1, 2)

        # Visual representation
        bell_bar = '█' * min(int(bell_w * 40), 80)
        food_bar = '█' * min(int(food_w * 40), 80)

        print("│                                                                                                  │")
        print("│  CONNECTION STRENGTHS:                                                                           │")
        print("│                                                                                                  │")
        print(f"│  Bell → Salivate: {bell_w:.4f}                                                                      │")
        print(f"│  [{bell_bar:<80}]     │")
        print("│                                                                                                  │")
        print(f"│  Food → Salivate: {food_w:.4f}                                                                      │")
        print(f"│  [{food_bar:<80}]     │")
        print("│                                                                                                  │")

        # Interpretation
        print("│  INTERPRETATION:                                                                                 │")
        print("│                                                                                                  │")

        if bell_w < 0.5:
            print("│  🟡 Bell synapse is WEAK - little/no learning has occurred                                      │")
        elif bell_w < 1.0:
            print("│  🟢 Bell synapse is MODERATE - partial learning                                                 │")
        else:
            print("│  🟢 Bell synapse is STRONG - significant learning achieved!                                     │")

        if food_w >= 1.0:
            print("│  🟢 Food synapse remains STRONG - instinctive connection intact                                 │")

        print("│                                                                                                  │")
        print("└" + "─" * 98 + "┘")

        # Show network diagram with weights
        print("\n  📐 NETWORK TOPOLOGY WITH WEIGHTS:\n")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │                                                             │")
        print(f"  │   [Bell]                     (w={bell_w:.3f})                 │")
        print(f"  │      ├───────────────────────────┐                          │")
        print(f"  │      │                           ↓                          │")
        print(f"  │      │                     [Salivate]                       │")
        print(f"  │      │                           ↑                          │")
        print(f"  │   [Food]                     (w={food_w:.3f})                 │")
        print(f"  │      └───────────────────────────┘                          │")
        print(f"  │                                                             │")
        print("  └─────────────────────────────────────────────────────────────┘")

        input("\n  [Press ENTER to continue]")

    def manual_test_mode(self):
        """Interactive testing."""
        if not self.experiment:
            print("\n  ⚠️  No network initialized!")
            input("\n  [Press ENTER to continue]")
            return

        while True:
            self.clear_screen()
            self.print_header()

            print("\n┌─ MANUAL TEST MODE " + "─" * 79 + "┐")
            print("│                                                                                                  │")
            print("│  Test the network by presenting stimuli manually.                                               │")
            print("│                                                                                                  │")
            print("└" + "─" * 98 + "┘\n")

            print("  Select stimulus to present:")
            print("    [1] Bell only")
            print("    [2] Food only")
            print("    [3] Bell + Food")
            print("    [0] Return to main menu\n")

            choice = input("  Choice: ").strip()

            if choice == '0':
                break
            elif choice not in ['1', '2', '3']:
                continue

            # Run test
            bell = choice in ['1', '3']
            food = choice in ['2', '3']

            self.experiment.network.reset()
            salivated = self.experiment.run_trial(bell=bell, food=food, dopamine=0.0, ticks=10)

            # Show results
            print("\n" + "─" * 100)
            print("  TEST RESULTS")
            print("─" * 100)
            print(f"\n  Stimuli Presented:")
            print(f"    Bell: {'🔔 YES' if bell else '❌ NO'}")
            print(f"    Food: {'🍖 YES' if food else '❌ NO'}")
            print(f"\n  Network Response:")
            print(f"    Salivation: {'💧 YES' if salivated else '💤 NO'}")

            # Neuron states
            print(f"\n  Neuron Activity:")
            print(f"    Bell Input:  {'🔥 ACTIVE' if bell else '💤 SILENT'}")
            print(f"    Food Input:  {'🔥 ACTIVE' if food else '💤 SILENT'}")
            print(f"    Salivate Out: {'🔥 FIRED!' if salivated else '💤 SILENT'}")

            print("─" * 100)

            input("\n  [Press ENTER to continue]")

    def view_statistics(self):
        """Show learning statistics."""
        if not self.experiment or not self.trained:
            print("\n  ⚠️  No training data available! Run training first.")
            input("\n  [Press ENTER to continue]")
            return

        self.clear_screen()
        self.print_header()

        print("\n┌─ LEARNING STATISTICS " + "─" * 76 + "┐")

        bell_w = self.experiment.network.get_weight(0, 2)
        food_w = self.experiment.network.get_weight(1, 2)
        weight_change = bell_w - 0.1
        percent_increase = (weight_change / 0.1) * 100

        print("│                                                                                                  │")
        print("│  WEIGHT CHANGES:                                                                                 │")
        print(f"│    Initial Bell→Salivate:  0.1000                                                                │")
        print(f"│    Final Bell→Salivate:    {bell_w:.4f}                                                              │")
        print(f"│    Change:                 +{weight_change:.4f} ({percent_increase:.0f}% increase)                              │")
        print("│                                                                                                  │")
        print("│  LEARNING METRICS:                                                                               │")
        print(f"│    Convergence:            {'✓ ACHIEVED' if bell_w > 1.0 else '⚠ PARTIAL'}                                  │")
        print(f"│    Instinct Preserved:     {'✓ YES' if food_w >= 1.0 else '⚠ DEGRADED'}                                      │")
        print("│                                                                                                  │")
        print("│  BIOLOGICAL PLAUSIBILITY:                                                                        │")
        print("│    ✓ No backpropagation used                                                                     │")
        print("│    ✓ Local learning rules (STDP)                                                                 │")
        print("│    ✓ Neuromodulation (dopamine)                                                                  │")
        print("│    ✓ Spike-based computation                                                                     │")
        print("│                                                                                                  │")
        print("└" + "─" * 98 + "┘")

        input("\n  [Press ENTER to continue]")

    def run(self):
        """Main loop."""
        while self.running:
            self.clear_screen()
            self.print_header()
            self.show_status()
            self.print_main_menu()

            choice = input("\n  Select option: ").strip()

            if choice == '1':
                self.initialize_network()
            elif choice == '2':
                self.run_full_training()
            elif choice == '3':
                self.view_weights()
            elif choice == '4':
                self.manual_test_mode()
            elif choice == '5':
                self.view_statistics()
            elif choice == '6':
                print("\n  ⚠️  Custom training coming soon!")
                time.sleep(1)
            elif choice == '0':
                self.running = False
                self.clear_screen()
                print("\n  👋 Thank you for exploring neuromorphic learning!")
                print("     Dashboard closed.\n")
            else:
                print("\n  ⚠️  Invalid option!")
                time.sleep(1)


def main():
    """Launch comprehensive dashboard."""
    dashboard = ComprehensiveDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

