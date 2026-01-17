"""
Test with correct timing - dopamine applied SAME tick as firing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.core import NeuromorphicNetwork

net = NeuromorphicNetwork()

# Setup
net.add_node(10, node_type="input", threshold=0.2)
net.add_node(14, node_type="output", threshold=0.9, refractory_period=5)
net.connect(10, 14, weight=0.3)

print("TESTING CORRECT DOPAMINE TIMING")
print("="*70)
print(f"Initial weight: {net.get_weight(10, 14):.4f}\n")

net.reset()
net.set_input(10, 0.0)
net.set_input(14, 0.0)

for t in range(15):
    # Set inputs
    net.set_input(10, 1.5)

    if 2 < t < 10:
        net.set_input(14, 1.2)

    #  Get node
    node_14 = net.nodes[14]
    synapse = node_14.inputs[10]

    trace_before = synapse.trace
    weight_before = synapse.weight

    # Step with dopamine
    net.step(global_dopamine=1.2, learning_rate=0.06)  # ALWAYS apply dopamine

    firing_after = net.is_firing(14)
    trace_after = synapse.trace
    weight_after = synapse.weight

    delta_w = weight_after - weight_before

    print(f"t={t:2d}: Fire={firing_after}, Trace={trace_after:.3f}, "
          f"Weight={weight_after:.4f}, ΔW={delta_w:+.4f}")

print(f"\nFinal weight: {net.get_weight(10, 14):.4f}")
print(f"Change: {net.get_weight(10, 14) - 0.3:.4f}")

