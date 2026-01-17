"""
Temporal Sequence Learning: Learn to predict next stimulus in a pattern
This demonstrates the temporal dynamics of spiking neurons.

Task: Learn the sequence A → B → C → A → B → C...
The network must predict what comes next based on current input.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.core import NeuromorphicNetwork


class SequenceLearningExperiment:
    def __init__(self):
        self.network = NeuromorphicNetwork()
        self.setup_network()

        # Sequence to learn
        self.sequence = ['A', 'B', 'C']
        self.node_map = {'A': 0, 'B': 1, 'C': 2}
        self.output_map = {'A': 3, 'B': 4, 'C': 5}

        # Tracking
        self.training_history = {
            'epoch': [],
            'accuracy': [],
            'weights_snapshot': []
        }

    def setup_network(self):
        """
        Network: 3 inputs (A, B, C) → 3 outputs (predict next A, B, C)
        """
        # Input nodes
        self.network.add_node(0, node_type="input", threshold=0.2)  # A
        self.network.add_node(1, node_type="input", threshold=0.2)  # B
        self.network.add_node(2, node_type="input", threshold=0.2)  # C

        # Output nodes (predictions)
        self.network.add_node(3, node_type="output", threshold=0.9, refractory_period=5)  # Predict A
        self.network.add_node(4, node_type="output", threshold=0.9, refractory_period=5)  # Predict B
        self.network.add_node(5, node_type="output", threshold=0.9, refractory_period=5)  # Predict C

        # Create connections: A→B, B→C, C→A pattern
        # A should predict B
        self.network.connect(0, 4, weight=0.3)  # A input → Predict B
        self.network.connect(0, 3, weight=0.1)  # A → Predict A (weak)
        self.network.connect(0, 5, weight=0.1)  # A → Predict C (weak)

        # B should predict C
        self.network.connect(1, 5, weight=0.3)  # B input → Predict C
        self.network.connect(1, 3, weight=0.1)  # B → Predict A (weak)
        self.network.connect(1, 4, weight=0.1)  # B → Predict B (weak)

        # C should predict A
        self.network.connect(2, 3, weight=0.3)  # C input → Predict A
        self.network.connect(2, 4, weight=0.1)  # C → Predict B (weak)
        self.network.connect(2, 5, weight=0.1)  # C → Predict C (weak)

        print("✓ Sequence Network Initialized")
        print("  Sequence to learn: A → B → C → A ...")
        print("  Topology: 3 inputs → 3 outputs (9 connections)")

    def present_stimulus(self, stimulus):
        """Present one of the stimuli."""
        # Clear all inputs
        for node_id in [0, 1, 2]:
            self.network.set_input(node_id, 0.0)

        # Set the stimulus
        if stimulus in self.node_map:
            self.network.set_input(self.node_map[stimulus], 1.5)

    def get_next_in_sequence(self, current):
        """Get what should come next."""
        idx = self.sequence.index(current)
        next_idx = (idx + 1) % len(self.sequence)
        return self.sequence[next_idx]

    def train_step(self, current_stimulus, next_stimulus, learning_rate=0.06):
        """Train on one stimulus transition with TEACHER FORCING.

        Fix: Apply dopamine proactively during the forcing window so the
        post-synaptic spike and dopamine coincide (avoids refractory timing trap).
        """
        self.network.reset()

        # 1. Present current stimulus (The Context)
        self.present_stimulus(current_stimulus)

        # 2. Identify the Target (The 'Correct' Answer)
        expected_output_node = self.output_map[next_stimulus]

        fired_outputs = []

        # Track firing for competitive inhibition (punishment)
        last_fired = {3: False, 4: False, 5: False}

        for t in range(15):
            # Present stimulus each tick
            self.present_stimulus(current_stimulus)

            # --- DOPAMINE LOGIC FIX ---
            dopa = 0.0

            # A) TEACHER FORCING & REWARD
            # If we are in the forcing window, we apply force AND reward simultaneously.
            # This ensures (Post_Fire + Dopa) happen in the same tick.
            if 2 < t < 10:
                self.network.set_input(expected_output_node, 1.2)
                dopa = 1.0 # <--- PROACTIVE REWARD (The Fix)

            # B) PUNISHMENT (Competitive Inhibition)
            # If a WRONG node fired last tick, punish it now.
            for out_node in [3, 4, 5]:
                if out_node != expected_output_node and last_fired[out_node]:
                    dopa = -0.5

            # Step with Dopamine
            self.network.step(global_dopamine=dopa, learning_rate=learning_rate)

            # Record firing for next tick's punishment logic
            for out_node in [3, 4, 5]:
                last_fired[out_node] = self.network.is_firing(out_node)
                if out_node == expected_output_node and last_fired[out_node]:
                    if expected_output_node not in fired_outputs:
                        fired_outputs.append(expected_output_node)

        # Check correctness
        correct = expected_output_node in fired_outputs
        return correct

    def test_sequence(self, length=9):
        """Test the network on a sequence."""
        correct_predictions = 0
        total_predictions = length

        results = []

        for i in range(length):
            current_idx = i % len(self.sequence)
            current = self.sequence[current_idx]
            expected_next = self.get_next_in_sequence(current)

            self.network.reset()
            self.present_stimulus(current)

            # Run for several ticks
            fired_outputs = []
            for t in range(12):
                self.network.step(global_dopamine=0.0, learning_rate=0.0)

                for node_id, label in [(3, 'A'), (4, 'B'), (5, 'C')]:
                    if self.network.is_firing(node_id):
                        if label not in fired_outputs:
                            fired_outputs.append(label)

            # Check prediction
            predicted = fired_outputs[0] if fired_outputs else None
            correct = (predicted == expected_next)

            if correct:
                correct_predictions += 1

            results.append({
                'current': current,
                'expected': expected_next,
                'predicted': predicted,
                'correct': correct
            })

        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy, results

    def train_epoch(self, epoch_num):
        """Train on complete sequence."""
        # Train on several repetitions
        for _ in range(3):
            for i in range(len(self.sequence)):
                current = self.sequence[i]
                next_stim = self.get_next_in_sequence(current)
                self.train_step(current, next_stim)

        # Test
        accuracy, _ = self.test_sequence(length=6)

        # Record
        self.training_history['epoch'].append(epoch_num)
        self.training_history['accuracy'].append(accuracy)
        self.record_weights()

        return accuracy

    def record_weights(self):
        """Record key weights."""
        snapshot = {
            'A→PredictB': self.network.get_weight(0, 4),
            'B→PredictC': self.network.get_weight(1, 5),
            'C→PredictA': self.network.get_weight(2, 3),
        }
        self.training_history['weights_snapshot'].append(snapshot)

    def print_weights(self):
        """Display weight matrix."""
        print("\n" + "═" * 70)
        print("  SEQUENCE PREDICTION WEIGHTS")
        print("═" * 70)
        print("\n  Correct Associations (should be strong):")

        w_a_b = self.network.get_weight(0, 4)
        w_b_c = self.network.get_weight(1, 5)
        w_c_a = self.network.get_weight(2, 3)

        bar_a_b = '█' * min(int(w_a_b * 20), 40)
        bar_b_c = '█' * min(int(w_b_c * 20), 40)
        bar_c_a = '█' * min(int(w_c_a * 20), 40)

        print(f"    A → Predict B: [{bar_a_b:<40}] {w_a_b:.3f}")
        print(f"    B → Predict C: [{bar_b_c:<40}] {w_b_c:.3f}")
        print(f"    C → Predict A: [{bar_c_a:<40}] {w_c_a:.3f}")

        print("\n  Incorrect Associations (should be weak):")
        w_a_c = self.network.get_weight(0, 5)
        w_b_a = self.network.get_weight(1, 3)
        w_c_b = self.network.get_weight(2, 4)

        print(f"    A → Predict C: {w_a_c:.3f}")
        print(f"    B → Predict A: {w_b_a:.3f}")
        print(f"    C → Predict B: {w_c_b:.3f}")
        print("═" * 70)

    def run_training(self, max_epochs=100):
        """Run full training."""
        print("\n" + "═" * 70)
        print("  🧠 TEMPORAL SEQUENCE LEARNING")
        print("  Predict Next Element in Pattern")
        print("═" * 70)
        print("\n  Task: Learn A → B → C → A → B → C...")
        print("  Success: Predict what comes next given current input")
        print("\n" + "─" * 70)
        print("  TRAINING PHASE")
        print("─" * 70)

        for epoch in range(1, max_epochs + 1):
            accuracy = self.train_epoch(epoch)

            if epoch % 10 == 0:
                acc_bar = "█" * int(accuracy / 5) + "░" * (20 - int(accuracy / 5))
                print(f"  Epoch {epoch:3d} | Accuracy: [{acc_bar}] {accuracy:5.1f}%")

            if accuracy >= 100.0:
                print(f"\n  ✓ Perfect learning achieved at epoch {epoch}!")
                break

        # Final test
        print("\n" + "─" * 70)
        print("  FINAL EVALUATION")
        print("─" * 70)

        accuracy, results = self.test_sequence(length=12)

        print("\n  Prediction Results:")
        for r in results:
            status = "✓" if r['correct'] else "✗"
            pred_str = r['predicted'] if r['predicted'] else "?"
            print(f"    {status} {r['current']} → {r['expected']} (predicted: {pred_str})")

        print("\n" + "═" * 70)
        print("  📊 FINAL RESULTS")
        print("═" * 70)
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Status: {'✓ SUCCESS' if accuracy >= 80 else '⚠ PARTIAL'}")
        print("═" * 70)

        self.print_weights()

        return accuracy >= 80


def main():
    """Run sequence learning experiment."""
    experiment = SequenceLearningExperiment()
    success = experiment.run_training(max_epochs=100)

    if success:
        print("\n🎉 Sequence learning successful!")
        print("   The network learned temporal associations.")
    else:
        print("\n⚠️  Learning incomplete.")


if __name__ == "__main__":
    main()

