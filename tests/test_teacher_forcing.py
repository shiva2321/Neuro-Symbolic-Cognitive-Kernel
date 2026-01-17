"""
Debug test for teacher forcing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.core import NeuromorphicNetwork

# Create a simple 2-node network
net = NeuromorphicNetwork()

# Input node
net.add_node(0, node_type="input", threshold=0.2)
# Output node
net.add_node(1, node_type="output", threshold=0.9, refractory_period=5)

# Weak connection
net.connect(0, 1, weight=0.3)

print("Testing Teacher Forcing")
print("=" * 50)
print(f"Initial weight 0→1: {net.get_weight(0, 1):.3f}")
print()

# Test 1: Without teacher forcing
print("Test 1: WITHOUT Teacher Forcing")
net.reset()
for t in range(10):
    net.set_input(0, 1.5)  # Strong input
    net.step(global_dopamine=1.0, learning_rate=0.05)
    if net.is_firing(1):
        print(f"  Tick {t}: Output FIRED! Potential={net.get_node_potential(1):.3f}")
    else:
        print(f"  Tick {t}: Output silent. Potential={net.get_node_potential(1):.3f}")

print(f"  Final weight: {net.get_weight(0, 1):.3f}")
print()

# Reset for test 2
net = NeuromorphicNetwork()
net.add_node(0, node_type="input", threshold=0.2)
net.add_node(1, node_type="output", threshold=0.9, refractory_period=5)
net.connect(0, 1, weight=0.3)

# Test 2: With teacher forcing
print("Test 2: WITH Teacher Forcing")
net.reset()
for t in range(10):
    net.set_input(0, 1.5)  # Input
    if 2 < t < 8:
        net.set_input(1, 1.0)  # FORCE output to fire
    net.step(global_dopamine=1.0, learning_rate=0.05)
    if net.is_firing(1):
        print(f"  Tick {t}: Output FIRED! Potential={net.get_node_potential(1):.3f}")
    else:
        print(f"  Tick {t}: Output silent. Potential={net.get_node_potential(1):.3f}")

print(f"  Final weight: {net.get_weight(0, 1):.3f}")
print()

print("=" * 50)
if net.get_weight(0, 1) > 0.3:
    print("✓ Teacher forcing WORKS! Weight increased.")
else:
    print("✗ Teacher forcing FAILED. Weight unchanged.")

