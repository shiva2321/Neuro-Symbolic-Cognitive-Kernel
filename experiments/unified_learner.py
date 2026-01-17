"""
Unified Neuromorphic Learner: ONE Network for All Experiments
This demonstrates continuous learning - the network retains knowledge
from Pavlov, then learns Sequence, then learns XOR, all in one brain.
"""

from core.network import NeuromorphicNetwork
import random
import time


class UnifiedLearner:
    """
    A single neuromorphic network that learns all three tasks:
    1. Pavlov (Classical Conditioning)
    2. Sequence (Temporal Prediction)
    3. XOR (Non-linear Classification)

    Node Architecture:
    - Nodes 0-2: Pavlov (Bell, Food, Salivate)
    - Nodes 10-15: Sequence (3 inputs A,B,C + 3 outputs predict A,B,C)
    - Nodes 20-25: XOR (2 inputs + 3 hidden + 1 output)
    """

    def __init__(self):
        self.network = NeuromorphicNetwork()
        self.setup_unified_network()

        # Tracking
        self.learning_history = {
            'pavlov': {'epochs': [], 'success': []},
            'sequence': {'epochs': [], 'accuracy': []},
            'xor': {'epochs': [], 'accuracy': []}
        }

    def setup_unified_network(self):
        """Build ONE network with all three experiment topologies."""
        print("\n" + "="*70)
        print("🧠 BUILDING UNIFIED NEUROMORPHIC NETWORK")
        print("="*70)

        # ============ PAVLOV SECTION (Nodes 0-2) ============
        self.network.add_node(0, node_type="input", threshold=0.5)   # Bell
        self.network.add_node(1, node_type="input", threshold=0.5)   # Food
        self.network.add_node(2, node_type="output", threshold=1.0, refractory_period=5)  # Salivate

        self.network.connect(0, 2, weight=0.1)   # Bell -> Salivate (to be learned)
        self.network.connect(1, 2, weight=1.5)   # Food -> Salivate (instinct)

        print("  ✓ Pavlov Module: Nodes 0-2 (Bell, Food, Salivate)")

        # ============ SEQUENCE SECTION (Nodes 10-15) ============
        # Inputs: A, B, C
        self.network.add_node(10, node_type="input", threshold=0.2)
        self.network.add_node(11, node_type="input", threshold=0.2)
        self.network.add_node(12, node_type="input", threshold=0.2)

        # Outputs: Predict A, B, C
        self.network.add_node(13, node_type="output", threshold=0.9, refractory_period=5)
        self.network.add_node(14, node_type="output", threshold=0.9, refractory_period=5)
        self.network.add_node(15, node_type="output", threshold=0.9, refractory_period=5)

        # Connections: A→B, B→C, C→A pattern
        self.network.connect(10, 14, weight=0.3)  # A -> Predict B
        self.network.connect(10, 13, weight=0.1)  # A -> Predict A (weak)
        self.network.connect(10, 15, weight=0.1)  # A -> Predict C (weak)

        self.network.connect(11, 15, weight=0.3)  # B -> Predict C
        self.network.connect(11, 13, weight=0.1)  # B -> Predict A (weak)
        self.network.connect(11, 14, weight=0.1)  # B -> Predict B (weak)

        self.network.connect(12, 13, weight=0.3)  # C -> Predict A
        self.network.connect(12, 14, weight=0.1)  # C -> Predict B (weak)
        self.network.connect(12, 15, weight=0.1)  # C -> Predict C (weak)

        print("  ✓ Sequence Module: Nodes 10-15 (A,B,C -> Predict A,B,C)")

        # ============ XOR SECTION (Nodes 20-25) ============
        # Inputs
        self.network.add_node(20, node_type="input", threshold=0.3)   # A
        self.network.add_node(21, node_type="input", threshold=0.3)   # B

        # Hidden layer - FIGHT CLUB EDITION 🥊
        # Node 22: OR Detector (fires when A OR B is true)
        self.network.add_node(22, node_type="hidden", threshold=1.2, refractory_period=3)
        self.network.nodes[22].target_rate = 0.60  # 🥊 Let it be loud!

        # Node 23: AND Detector (fires when A AND B is true)
        self.network.add_node(23, node_type="hidden", threshold=2.2, refractory_period=3)
        self.network.nodes[23].target_rate = 0.15  # 🥊 Force it to be picky!

        # Output
        self.network.add_node(25, node_type="output", threshold=0.8, refractory_period=4)   # XOR

        # Input -> Hidden connections (balanced)
        self.network.connect(20, 22, weight=1.0)  # A -> OR
        self.network.connect(21, 22, weight=1.0)  # B -> OR
        self.network.connect(20, 23, weight=1.0)  # A -> AND
        self.network.connect(21, 23, weight=1.0)  # B -> AND

        # Hidden -> Output connections
        self.network.connect(22, 25, weight=1.5)  # OR -> Output (excitatory)
        self.network.connect(23, 25, weight=1.5, is_inhibitory=True)  # AND -> Output (inhibitory)

        # 🥊 FIGHT CLUB CONNECTION: AND inhibits OR (Lateral Inhibition)
        self.network.connect(23, 22, weight=2.0, is_inhibitory=True)  # AND -> OR (inhibitory!)

        print("  ✓ XOR Module: Nodes 20-25 (Fight Club Edition - 2 hidden + lateral inhibition)")

        print("="*70)
        print(f"  Total Nodes: {len(self.network.nodes)}")
        print("  Total Synapses: 21")
        print("="*70 + "\n")

    # ============================================================
    # PAVLOV EXPERIMENT METHODS
    # ============================================================

    def pavlov_trial(self, bell=False, food=False, dopamine=0.0, ticks=10):
        """Run a Pavlov trial."""
        self.network.reset()
        salivated = False

        for t in range(ticks):
            # Set inputs
            self.network.set_input(0, 1.5 if bell else 0.0)
            self.network.set_input(1, 1.5 if food else 0.0)

            # Step network
            self.network.step(global_dopamine=dopamine, learning_rate=0.05)

            if self.network.is_firing(2):
                salivated = True

        return salivated

    def train_pavlov(self, trials=15, show_progress=True):
        """Train Pavlov conditioning."""
        print("\n" + "🔔"*35)
        print("📋 PAVLOV TRAINING: Classical Conditioning")
        print("🔔"*35 + "\n")

        initial_weight = self.network.get_weight(0, 2)

        for trial in range(1, trials + 1):
            salivated = self.pavlov_trial(bell=True, food=True, dopamine=1.0)

            if show_progress and (trial % 5 == 0 or trial == 1):
                weight = self.network.get_weight(0, 2)
                bar = '█' * int(weight * 20)
                print(f"  Trial {trial:2d}: Bell→Salivate = [{bar:<40}] {weight:.4f}")

        # Test
        salivated = self.pavlov_trial(bell=True, food=False, dopamine=0.0)
        final_weight = self.network.get_weight(0, 2)

        print(f"\n  ✓ Training Complete!")
        print(f"    Initial Weight: {initial_weight:.4f}")
        print(f"    Final Weight:   {final_weight:.4f}")
        print(f"    Bell Alone → {'🔔 SALIVATION!' if salivated else '💤 No response'}")
        print(f"    Learning: {'✓ SUCCESS' if salivated else '✗ FAILED'}\n")

        self.learning_history['pavlov']['epochs'].append(trials)
        self.learning_history['pavlov']['success'].append(salivated)

        return salivated

    # ============================================================
    # SEQUENCE EXPERIMENT METHODS
    # ============================================================

    def sequence_train_step(self, current, next_stim, learning_rate=0.06):
        """Train one sequence transition with TEACHER FORCING.

        Fix: Apply dopamine proactively during the forcing window so the
        post-synaptic spike and dopamine coincide (avoids refractory timing trap).
        """
        self.network.reset()

        # Node mappings
        node_map = {'A': 10, 'B': 11, 'C': 12}
        output_map = {'A': 13, 'B': 14, 'C': 15}

        # Clear all sequence nodes first
        for node_id in range(10, 16):
            self.network.set_input(node_id, 0.0)

        # Expected output
        expected_output = output_map[next_stim]
        fired_outputs = []
        last_fired = {13: False, 14: False, 15: False}

        for t in range(15):
            # 1. Present current stimulus
            self.network.set_input(node_map[current], 1.5)

            # --- DOPAMINE FIX START ---
            dopa = 0.0

            # 2. TEACHER FORCING + SIMULTANEOUS REWARD
            if 2 < t < 10:
                self.network.set_input(expected_output, 1.2)
                dopa = 1.0 # Reward NOW, while it's firing!

            # 3. Competitive Punishment (for wrong guesses)
            for out_node in [13, 14, 15]:
                if out_node != expected_output and last_fired[out_node]:
                    dopa = -0.8
            # --- DOPAMINE FIX END ---

            # 4. Step with dopamine
            self.network.step(global_dopamine=dopa, learning_rate=learning_rate)

            # 5. Record current firing
            for out_node in [13, 14, 15]:
                last_fired[out_node] = self.network.is_firing(out_node)
                if out_node == expected_output and last_fired[out_node]:
                    if expected_output not in fired_outputs:
                        fired_outputs.append(expected_output)

        return expected_output in fired_outputs

    def test_sequence(self):
        """Test sequence prediction."""
        sequence = ['A', 'B', 'C']
        node_map = {'A': 10, 'B': 11, 'C': 12}
        output_map = {'A': 13, 'B': 14, 'C': 15}

        correct = 0
        total = 3

        for i, current in enumerate(sequence):
            expected_next = sequence[(i + 1) % len(sequence)]

            self.network.reset()

            # Clear all sequence inputs first
            for node_id in range(10, 16):
                self.network.set_input(node_id, 0.0)

            # Set only the current input
            self.network.set_input(node_map[current], 1.5)

            fired = []
            for t in range(12):
                self.network.step(global_dopamine=0.0, learning_rate=0.0)
                for label, node_id in [('A', 13), ('B', 14), ('C', 15)]:
                    if self.network.is_firing(node_id) and label not in fired:
                        fired.append(label)

            predicted = fired[0] if fired else None
            if predicted == expected_next:
                correct += 1

        return (correct / total) * 100

    def train_sequence(self, epochs=100, show_progress=True):
        """Train sequence learning."""
        print("\n" + "🔄"*35)
        print("📋 SEQUENCE TRAINING: Temporal Pattern Learning")
        print("🔄"*35 + "\n")

        sequence = ['A', 'B', 'C']

        for epoch in range(1, epochs + 1):
            # Train on transitions
            for _ in range(3):
                for i, current in enumerate(sequence):
                    next_stim = sequence[(i + 1) % len(sequence)]
                    self.sequence_train_step(current, next_stim)

            # Test periodically
            if epoch % 20 == 0 or epoch == 1:
                accuracy = self.test_sequence()
                self.learning_history['sequence']['epochs'].append(epoch)
                self.learning_history['sequence']['accuracy'].append(accuracy)

                if show_progress:
                    bar = '█' * int(accuracy / 5)
                    print(f"  Epoch {epoch:3d}: Accuracy = [{bar:<20}] {accuracy:.1f}%")

        final_accuracy = self.test_sequence()
        print(f"\n  ✓ Training Complete! Final Accuracy: {final_accuracy:.1f}%")
        print(f"    Learning: {'✓ SUCCESS' if final_accuracy >= 80 else '✗ NEEDS MORE TRAINING'}\n")

        return final_accuracy

    # ============================================================
    # XOR EXPERIMENT METHODS
    # ============================================================

    def xor_run_pattern(self, pattern, target, dopamine=0.0, ticks=15):
        """Run XOR pattern with TEACHER FORCING."""
        self.network.reset()

        # Clear XOR inputs first
        for node_id in range(20, 26):
            self.network.set_input(node_id, 0.0)

        output_fired = False

        for t in range(ticks):
            # Set inputs
            if t < 10:
                self.network.set_input(20, pattern[0] * 1.5)
                self.network.set_input(21, pattern[1] * 1.5)
            else:
                self.network.set_input(20, 0.0)
                self.network.set_input(21, 0.0)

            # TEACHER FORCING for positive examples
            if dopamine > 0 and target == 1 and 2 < t < 8:
                self.network.set_input(25, 1.0)

            # Apply dopamine continuously during training window
            if dopamine > 0:
                if target == 1 and 2 < t < 10:
                    # Positive example - apply reward during forcing window
                    dopa = 1.0
                elif target == 0:
                    # Negative example - punish if it fires
                    current_output = self.network.is_firing(25)
                    dopa = -1.0 if current_output else 0.0
                else:
                    dopa = 0.0
            else:
                dopa = 0.0

            self.network.step(global_dopamine=dopa, learning_rate=0.08)

            if self.network.is_firing(25):
                output_fired = True

        correct = (output_fired and target == 1) or (not output_fired and target == 0)
        return correct, output_fired

    def test_xor(self):
        """Test XOR on all patterns."""
        patterns = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]

        correct = 0
        for pattern, target in patterns:
            # Clear XOR nodes before test
            for node_id in range(20, 26):
                self.network.set_input(node_id, 0.0)

            is_correct, _ = self.xor_run_pattern(pattern, target, dopamine=0.0)
            if is_correct:
                correct += 1

        return (correct / len(patterns)) * 100

    def train_xor(self, epochs=200, show_progress=True):
        """Train XOR problem."""
        print("\n" + "⊕"*35)
        print("📋 XOR TRAINING: Non-Linear Classification")
        print("⊕"*35 + "\n")

        patterns = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]

        for epoch in range(1, epochs + 1):
            shuffled = patterns.copy()
            random.shuffle(shuffled)

            for pattern, target in shuffled:
                self.xor_run_pattern(pattern, target, dopamine=1.0)

            # Test periodically
            if epoch % 40 == 0 or epoch == 1:
                accuracy = self.test_xor()
                self.learning_history['xor']['epochs'].append(epoch)
                self.learning_history['xor']['accuracy'].append(accuracy)

                if show_progress:
                    bar = '█' * int(accuracy / 5)
                    print(f"  Epoch {epoch:3d}: Accuracy = [{bar:<20}] {accuracy:.1f}%")

        final_accuracy = self.test_xor()
        print(f"\n  ✓ Training Complete! Final Accuracy: {final_accuracy:.1f}%")
        print(f"    Learning: {'✓ SUCCESS' if final_accuracy >= 80 else '✗ NEEDS MORE TRAINING'}\n")

        return final_accuracy

    # ============================================================
    # UNIFIED TRAINING PIPELINE
    # ============================================================

    def run_full_curriculum(self):
        """Train all three experiments sequentially on ONE network."""
        print("\n" + "="*70)
        print("🎓 UNIFIED NEUROMORPHIC CURRICULUM")
        print("   One Brain, Three Skills, Continuous Learning")
        print("="*70)

        start_time = time.time()

        # Stage 1: Pavlov
        print("\n📚 STAGE 1/3: Learning Classical Conditioning...")
        pavlov_success = self.train_pavlov(trials=15, show_progress=True)

        time.sleep(1)

        # Stage 2: Sequence
        print("\n📚 STAGE 2/3: Learning Temporal Sequences...")
        sequence_accuracy = self.train_sequence(epochs=100, show_progress=True)

        time.sleep(1)

        # Stage 3: XOR
        print("\n📚 STAGE 3/3: Learning Non-Linear Logic...")
        xor_accuracy = self.train_xor(epochs=200, show_progress=True)

        elapsed = time.time() - start_time

        # Final Summary
        print("\n" + "="*70)
        print("🎉 CURRICULUM COMPLETE!")
        print("="*70)
        print(f"  Training Time: {elapsed:.2f} seconds")
        print(f"\n  📊 RESULTS:")
        print(f"    Pavlov (Conditioning):  {'✓ PASSED' if pavlov_success else '✗ FAILED'}")
        print(f"    Sequence (Prediction):  {sequence_accuracy:.1f}% {'✓' if sequence_accuracy >= 80 else '✗'}")
        print(f"    XOR (Classification):   {xor_accuracy:.1f}% {'✓' if xor_accuracy >= 80 else '✗'}")

        # Check if Pavlov is still intact
        print(f"\n  🧪 RETENTION TEST: Does the network still remember Pavlov?")
        still_remembers = self.pavlov_trial(bell=True, food=False, dopamine=0.0)
        print(f"    Bell Alone → {'🔔 SALIVATION!' if still_remembers else '💤 Forgot!'}")
        print(f"    Memory Retention: {'✓ INTACT' if still_remembers else '✗ CATASTROPHIC FORGETTING'}")

        print("="*70 + "\n")

        return {
            'pavlov': pavlov_success,
            'sequence': sequence_accuracy,
            'xor': xor_accuracy,
            'retention': still_remembers
        }


def main():
    """Run the unified learning experiment."""
    learner = UnifiedLearner()
    results = learner.run_full_curriculum()

    # Final message
    if all([results['pavlov'], results['sequence'] >= 80, results['xor'] >= 80, results['retention']]):
        print("🌟 PERFECT SCORE! The network learned all tasks and retained memory!")
    else:
        print("💡 The network is learning! Try more epochs for better results.")


if __name__ == "__main__":
    main()

