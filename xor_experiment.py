"""
XOR Pattern Learning: The "Fight Club" Edition 🥊
Features Lateral Inhibition to force specialization.
H2 (AND) directly inhibits H1 (OR) to create XOR = (A OR B) AND NOT (A AND B)
"""

from network import NeuromorphicNetwork
import random
import time


class XORExperiment:
    def __init__(self):
        self.network = NeuromorphicNetwork()
        self.setup_network()
        self.patterns = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]
        self.history = {'epoch': [], 'accuracy': []}

    def setup_network(self):
        """
        Fight Club Topology:
        2 inputs -> 2 specialized hidden (H1: OR detector, H2: AND detector) -> 1 output

        Key Innovation: H2 (AND) laterally inhibits H1 (OR)
        This forces: XOR = (A OR B) AND NOT (A AND B)
        """
        # Inputs
        self.network.add_node(0, node_type="input", threshold=0.3)  # A
        self.network.add_node(1, node_type="input", threshold=0.3)  # B

        # Hidden layer with SPECIALIZED ROLES
        # H1: The "OR" Detector
        # (Needs high target rate because OR is true 75% of time: 01, 10, 11)
        # Lower threshold so it fires with single input
        self.network.add_node(2, node_type="hidden", threshold=1.2, refractory_period=3)
        self.network.nodes[2].target_rate = 0.60  # 🥊 Let it be loud!

        # H2: The "AND" Detector
        # (Needs low target rate because AND is true 25% of time: only 11)
        # Higher threshold so it ONLY fires with both inputs
        self.network.add_node(3, node_type="hidden", threshold=2.2, refractory_period=3)
        self.network.nodes[3].target_rate = 0.15  # 🥊 Force it to be picky!

        # Output
        self.network.add_node(5, node_type="output", threshold=0.8, refractory_period=4)

        # Input -> Hidden connections (balanced initialization)
        self.network.connect(0, 2, weight=1.0)  # A -> H1 (OR)
        self.network.connect(1, 2, weight=1.0)  # B -> H1 (OR)
        self.network.connect(0, 3, weight=1.0)  # A -> H2 (AND)
        self.network.connect(1, 3, weight=1.0)  # B -> H2 (AND)

        # THE FIGHT CLUB CONNECTIONS 🥊
        # H1 drives the output (Excitatory)
        self.network.connect(2, 5, weight=1.5)  # H1 (OR) -> Output

        # H2 KILLS H1 (Lateral Inhibition)
        # If H2 (AND) fires, it silences H1 (OR).
        # This prevents both from firing simultaneously for (1,1)
        self.network.connect(3, 2, weight=2.0, is_inhibitory=True)  # H2 -> H1 (inhibit!)

        # H2 also inhibits output (redundant safety)
        self.network.connect(3, 5, weight=1.5, is_inhibitory=True)  # H2 -> Output (inhibit)

        print("✓ XOR Fight Club Network Initialized")
        print("  Strategy: H2 (AND) laterally inhibits H1 (OR)")
        print("  Homeostasis: H1 target=0.60, H2 target=0.15")
        print("  Logic: XOR = (A OR B) AND NOT (A AND B)")

    def run_pattern(self, pattern, target, dopamine=0.0):
        """Run one training pattern with teacher forcing."""
        self.network.reset()
        output_fired = False

        # Set inputs
        self.network.set_input(0, pattern[0] * 1.5)
        self.network.set_input(1, pattern[1] * 1.5)

        # Teacher Forcing (Only for Output)
        for t in range(15):
            # Proactive Dopamine + Forcing
            dopa = 0.0
            if dopamine > 0:
                if target == 1:
                    if 2 < t < 10:
                        self.network.set_input(5, 1.0)  # Force Output
                        dopa = 1.0
                elif target == 0:
                    # Punish false positives
                    if self.network.is_firing(5):
                        dopa = -1.0

            self.network.step(global_dopamine=dopa, learning_rate=0.05)
            if self.network.is_firing(5):
                output_fired = True

        return (output_fired and target == 1) or (not output_fired and target == 0)

    def run_training(self, max_epochs=400, target_accuracy=100.0):
        """Train the network."""
        print("\n" + "=" * 80)
        print("  🥊 XOR FIGHT CLUB TRAINING")
        print("  H2 (AND) vs H1 (OR) - May the best detector win!")
        print("=" * 80)
        print()

        for epoch in range(1, max_epochs + 1):
            shuffled = self.patterns.copy()
            random.shuffle(shuffled)

            for p, t in shuffled:
                self.run_pattern(p, t, dopamine=1.0)

            if epoch % 50 == 0 or epoch == 1 or epoch <= 5:
                score = self.test_all()
                bar = "█" * int(score / 5)
                print(f"  Epoch {epoch:3d}: Accuracy [{bar:<20}] {score:.1f}%")
                self.history['epoch'].append(epoch)
                self.history['accuracy'].append(score)

                if score >= target_accuracy:
                    print(f"\n  ✓ Converged at epoch {epoch}!")
                    break

        final_acc = self.test_all()
        print(f"\n  ✓ Final Accuracy: {final_acc:.1f}%")
        return final_acc >= target_accuracy

    def test_all(self):
        """Test on all patterns."""
        correct = 0
        for p, t in self.patterns:
            if self.run_pattern(p, t, dopamine=0.0):
                correct += 1
        return (correct / 4) * 100

    def test_all_patterns(self, show_details=True):
        """Test all patterns with optional details."""
        if show_details:
            print("\n" + "─" * 80)
            print("  Testing All Patterns")
            print("─" * 80)

        correct = 0
        for pattern, target in self.patterns:
            result = self.run_pattern(pattern, target, dopamine=0.0)
            if result:
                correct += 1

            if show_details:
                status = "✓" if result else "✗"
                predicted = 1 if self.run_pattern(pattern, target, dopamine=0.0) else 0
                print(f"  {status} {pattern} → Expected: {target}, Got: {predicted}")

        accuracy = (correct / len(self.patterns)) * 100
        if show_details:
            print("─" * 80)

        return accuracy, correct

    def print_weight_matrix(self):
        """Display network weights."""
        print("\n" + "=" * 80)
        print("  NETWORK WEIGHT MATRIX")
        print("=" * 80)

        print("\n  Input → Hidden Layer:")
        print(f"    A → H1 (OR):  {self.network.get_weight(0, 2):.4f}")
        print(f"    B → H1 (OR):  {self.network.get_weight(1, 2):.4f}")
        print(f"    A → H2 (AND): {self.network.get_weight(0, 3):.4f}")
        print(f"    B → H2 (AND): {self.network.get_weight(1, 3):.4f}")

        print("\n  Hidden → Output Layer:")
        print(f"    H1 (OR) → XOR:  {self.network.get_weight(2, 5):.4f} [excitatory]")
        print(f"    H2 (AND) → XOR: {self.network.get_weight(3, 5):.4f} [INHIBITORY]")

        print("\n  🥊 FIGHT CLUB CONNECTION (Lateral Inhibition):")
        print(f"    H2 → H1: {self.network.get_weight(3, 2):.4f} [INHIBITORY]")

        print("\n  Firing Rates vs Target:")
        print(f"    H1 (OR):  {self.network.nodes[2].avg_firing_rate:.3f} (target: 0.60)")
        print(f"    H2 (AND): {self.network.nodes[3].avg_firing_rate:.3f} (target: 0.15)")

        print("=" * 80)


def main():
    """Run the XOR Fight Club experiment."""
    exp = XORExperiment()
    success = exp.run_training(max_epochs=400, target_accuracy=100.0)

    print("\n" + "=" * 80)
    if success:
        print("  🌟 XOR LEARNED SUCCESSFULLY!")
        print("  The Fight Club strategy worked!")
        print("  H2 (AND) successfully neutralized H1 (OR) when needed.")
    else:
        print("  ⚠️ Still training... XOR is tough!")
    print("=" * 80)

    exp.test_all_patterns(show_details=True)
    exp.print_weight_matrix()


if __name__ == "__main__":
    main()

