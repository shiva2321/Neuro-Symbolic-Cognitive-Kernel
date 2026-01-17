# Quick Reference Guide

**Last Updated:** January 16, 2026

---

## 🚀 Quick Commands

### Run Experiments

```bash
# Interactive menu
python launcher.py

# Individual experiments
python pavlov_experiment.py       # 2 seconds
python sequence_experiment.py     # 30 seconds
python xor_experiment.py          # 60 seconds
python unified_learner.py         # 5 minutes

# Semantic demo
python simple_demo.py
```

### Run Tests

```bash
cd tests
pytest                           # All tests
pytest test_teacher_forcing.py  # Specific test
pytest -v                        # Verbose output
```

---

## 🧠 Core API Reference

### Creating a Network

```python
from network import NeuromorphicNetwork

net = NeuromorphicNetwork()

# Add neurons
net.add_node(
    node_id=0,
    node_type="input",      # "input", "hidden", or "output"
    threshold=1.0,          # Firing threshold
    refractory_period=3     # Cooldown ticks
)

# Connect neurons
net.connect(
    source_id=0,
    target_id=1,
    weight=0.5,             # Connection strength
    is_inhibitory=False     # Excitatory or inhibitory
)
```

### Running Simulation

```python
# Set inputs
net.set_input(node_id=0, value=1.0)

# Step simulation
net.step(
    global_dopamine=0.5,    # Reward signal (0-1)
    learning_rate=0.01      # Learning speed
)

# Check outputs
if net.is_firing(node_id=1):
    print("Neuron 1 fired!")

# Get membrane potential
potential = net.nodes[1].potential

# Reset eligibility traces (between training phases)
net.reset()
```

### Saving/Loading (BioNode Network)

```python
# Manual save (not implemented - use Flash Colony for persistence)
# BioNode networks are ephemeral by default

# For persistence, use Flash Colony instead
```

---

## 💾 Flash Colony API

### Creating a Brain File

```python
from flash_colony import FlashColony

colony = FlashColony(
    brain_file="my_brain.dat",
    max_nodes=1_000_000,
    max_edges_per_node=10
)

# Add neurons
node_id = colony.add_node(
    node_type=0,      # 0=input, 1=hidden, 2=output
    threshold=1.0
)

# Connect neurons
colony.connect_binary(
    source_id=0,
    target_id=1,
    weight=0.5
)
```

### Simulation Loop

```python
# Set inputs
colony.inject_input(node_id=0, current=1.0)

# Step (automatically persists to disk)
colony.step(
    global_dopamine=0.5,
    learning_rate=0.01
)

# Check if neuron fired
if colony.did_fire(node_id=1):
    print("Neuron 1 fired!")

# Close (automatically saves)
colony.close()
```

### Reloading Existing Brain

```python
# Brain automatically loads from existing file
colony = FlashColony(brain_file="my_brain.dat")

# Continue simulation...
colony.step(0.0, 0.01)
```

---

## 🎯 Common Patterns

### Pattern 1: Simple Association Learning

```python
from network import NeuromorphicNetwork

net = NeuromorphicNetwork()
net.add_node(0, "input")
net.add_node(1, "output")
net.connect(0, 1, weight=0.1)

# Training
for trial in range(15):
    net.set_input(0, 1.0)
    net.step(global_dopamine=1.0, learning_rate=0.05)
    net.reset()

# Test
net.set_input(0, 1.0)
net.step(0.0, 0.0)  # No dopamine, no learning
print(f"Fired: {net.is_firing(1)}")
print(f"Weight: {net.nodes[1].inputs[0].weight}")
```

### Pattern 2: Sequence Learning with Teacher Forcing

```python
from network import NeuromorphicNetwork

sequence = ["A", "B", "C"]
net = NeuromorphicNetwork()

# Build network
for i in range(3):
    net.add_node(i, "input")      # Input nodes
    net.add_node(i+3, "output")   # Output nodes
    for j in range(3):
        net.connect(i, j+3, weight=0.1)

# Training
for epoch in range(100):
    for i, current in enumerate(sequence):
        next_idx = (i + 1) % len(sequence)
        
        # Set current input
        net.set_input(i, 1.0)
        
        # Teacher forcing: activate correct output
        net.nodes[next_idx + 3].external_current = 5.0
        
        # Step with reward
        net.step(global_dopamine=1.0, learning_rate=0.01)
        
        # Clear for next step
        net.set_input(i, 0.0)
        net.reset()
```

### Pattern 3: XOR with Inhibition

```python
from network import NeuromorphicNetwork

net = NeuromorphicNetwork()

# Inputs
net.add_node(0, "input")
net.add_node(1, "input")

# Hidden (OR and AND gates)
net.add_node(2, "hidden", threshold=0.5)  # OR
net.add_node(3, "hidden", threshold=1.5)  # AND

# Output
net.add_node(4, "output", threshold=0.8)

# Connections
net.connect(0, 2, weight=1.0)  # Input → OR
net.connect(1, 2, weight=1.0)
net.connect(0, 3, weight=1.0)  # Input → AND
net.connect(1, 3, weight=1.0)
net.connect(2, 4, weight=1.0)  # OR → Output
net.connect(3, 4, weight=1.0, is_inhibitory=True)  # AND -| Output

# Training
xor_cases = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]
for epoch in range(400):
    for x0, x1, target in xor_cases:
        net.set_input(0, x0)
        net.set_input(1, x1)
        
        # Teacher forcing on output
        if target:
            net.nodes[4].external_current = 5.0
        
        dopamine = 1.0 if (net.is_firing(4) == target) else 0.0
        net.step(dopamine, learning_rate=0.01)
        net.reset()
```

---

## 🔧 Hyperparameter Tuning

### Learning Rate
```python
# Too low: Slow convergence (400+ epochs)
learning_rate = 0.005

# Good balance: Most experiments
learning_rate = 0.01

# Too high: Unstable, oscillating weights
learning_rate = 0.05
```

### Dopamine Amount
```python
# Weak reward: Slow learning
global_dopamine = 0.3

# Standard reward: Good for most tasks
global_dopamine = 1.0

# Strong reward: Fast but may overshoot
global_dopamine = 2.0
```

### Neuron Threshold
```python
# Low threshold: Fires easily, high baseline activity
threshold = 0.5

# Medium threshold: Selective firing
threshold = 1.0

# High threshold: Hard to activate, sparse coding
threshold = 2.0
```

### Refractory Period
```python
# No cooldown: May fire every tick
refractory_period = 0

# Standard: Prevents rapid re-firing
refractory_period = 3

# Long cooldown: Very sparse firing
refractory_period = 10
```

### Homeostasis Parameters
```python
# In bionode.py
target_rate = 0.25          # Target: fire 25% of time
homeostasis_rate = 0.005    # Slow adjustment (stable)
```

---

## 🐛 Debugging Tips

### Network Not Learning

```python
# 1. Print weights before/after
print(f"Initial: {net.nodes[1].inputs[0].weight}")
# ... training ...
print(f"Final: {net.nodes[1].inputs[0].weight}")

# 2. Check if neurons are firing
net.step(1.0, 0.01)
for node_id, node in net.nodes.items():
    print(f"Node {node_id}: firing={node.is_firing}, potential={node.potential:.2f}")

# 3. Verify dopamine timing
print(f"Dopamine: {global_dopamine}, Trace: {synapse.eligibility_trace}")
```

### Weights Saturating

```python
# Check if weights hit ceiling
for node in net.nodes.values():
    for synapse in node.inputs.values():
        if synapse.weight >= 1.9:  # Near max of 2.0
            print(f"⚠ Weight saturated!")
            
# Solutions:
# - Enable homeostatic plasticity (already on in BioNode)
# - Lower learning rate
# - Reduce dopamine amount
```

### Neurons Not Firing

```python
# Check potentials
for node_id, node in net.nodes.items():
    print(f"Node {node_id}:")
    print(f"  Potential: {node.potential:.3f}")
    print(f"  Threshold: {node.threshold:.3f}")
    print(f"  Inputs: {len(node.inputs)}")
    
# Solutions:
# - Lower threshold
# - Increase input weights
# - Reduce decay rate
# - Check for inhibition
```

### Catastrophic Forgetting

```python
# Test if old tasks still work after new training
def test_retention(net):
    # Test task 1
    net.set_input(0, 1.0)
    net.step(0.0, 0.0)
    task1_works = net.is_firing(1)
    
    # Test task 2
    net.set_input(2, 1.0)
    net.step(0.0, 0.0)
    task2_works = net.is_firing(3)
    
    return task1_works and task2_works

# Solution: Use separate node regions for different tasks
```

---

## 📊 Monitoring Training

### Print Progress Every N Epochs

```python
for epoch in range(1000):
    # ... training ...
    
    if epoch % 50 == 0:
        accuracy = calculate_accuracy()
        print(f"Epoch {epoch}: {accuracy:.1f}% accuracy")
```

### Weight Visualization

```python
def print_weights(net, connection_name, source, target):
    weight = net.nodes[target].inputs[source].weight
    bar = "█" * int(weight * 10)
    print(f"{connection_name}: [{bar:<20}] {weight:.4f}")

# Usage
print_weights(net, "Bell → Salivate", 0, 2)
```

### Firing Rate Monitor

```python
def monitor_firing_rates(net, num_ticks=100):
    fire_counts = {nid: 0 for nid in net.nodes}
    
    for _ in range(num_ticks):
        net.step(0.0, 0.0)
        for nid, node in net.nodes.items():
            if node.is_firing:
                fire_counts[nid] += 1
    
    for nid, count in fire_counts.items():
        rate = count / num_ticks
        print(f"Node {nid}: {rate:.1%} firing rate")
```

---

## 📝 Common Errors

### "Target node does not exist"
```python
# Make sure target is created before connecting
net.add_node(1, "output")  # Create first
net.connect(0, 1, 0.5)     # Then connect
```

### "Brain file is locked"
```python
# On Windows, close colony before deleting file
colony.close()
import time
time.sleep(0.1)  # Allow OS to release handle
os.remove("brain.dat")
```

### "Division by zero in homeostasis"
```python
# Check for zero weights before scaling
if synapse.weight > 0:
    synapse.weight *= scaling_factor
```

---

## 🎓 Learning Path

**Beginner:**
1. Run `pavlov_experiment.py`
2. Modify weights and see effects
3. Change learning rate, observe convergence
4. Read `bionode.py` to understand LIF dynamics

**Intermediate:**
5. Implement your own simple network
6. Try `sequence_experiment.py`
7. Add teacher forcing to guide learning
8. Experiment with homeostasis parameters

**Advanced:**
9. Use Flash Colony for large networks
10. Build multi-region brain with `ncgn_anatomy.py`
11. Create custom projection patterns
12. Optimize performance with profiling

---

## 📚 Further Reading

- **Book**: "Neuronal Dynamics" by Gerstner et al. (online, free)
- **Paper**: "Spike-Timing-Dependent Plasticity" by Bi & Poo (1998)
- **Paper**: "A quantitative description of membrane current..." by Hodgkin & Huxley (1952)
- **Project**: SpiNNaker, BrainScaleS (hardware neuromorphic systems)

---

**Last Updated:** January 16, 2026

