# NSCK API Reference

> **Complete API documentation for the Neuro-Symbolic Cognitive Kernel**

This document provides detailed API specifications for all core modules in the NSCK system. Use this as a reference when developing custom tasks or extending the system.

---

## Table of Contents

1. [System 1: Spiking Neural Network](#system-1-spiking-neural-network)
2. [System 2: Vector Symbolic Architecture](#system-2-vector-symbolic-architecture)
3. [Symbol Grounding](#symbol-grounding)
4. [Learning Mechanisms](#learning-mechanisms)
5. [Memory Systems](#memory-systems)
6. [Decision Fusion](#decision-fusion)
7. [Game Environments](#game-environments)
8. [Utilities](#utilities)

---

## System 1: Spiking Neural Network

### Module: `python/snn_qat.py`

#### Class: `TaskAwareSNN`

**Description**: Multi-task spiking neural network with late fusion architecture and ternary quantization.

**Constructor**:
```python
TaskAwareSNN(beta=0.5)
```

**Parameters**:
- `beta` (float, default=0.5): Leak rate for LIF neurons
  - Range: [0.0, 1.0]
  - Lower values → faster decay (shorter memory)
  - Higher values → slower decay (longer memory)
  - Recommended: 0.5 for visual tasks

**Attributes**:
- `conv1` (nn.Conv2d): First convolutional layer (4→16 channels)
- `conv2` (nn.Conv2d): Second convolutional layer (16→32 channels)
- `fc_shared` (nn.Linear): Shared latent space (289→64 dimensions)
- `head_snake` (nn.Linear): Snake task head (64→4 outputs)
- `head_pong` (nn.Linear): Pong task head (64→2 outputs)
- `head_maze` (nn.Linear): Maze task head (64→4 outputs)
- `head_chars` (nn.Linear): Character recognition head (64→62 outputs)

**Methods**:

##### `forward(x, task_id)`

Perform forward pass through the network.

**Parameters**:
- `x` (torch.Tensor): Input tensor, shape `[batch, 4, 10, 10]`
  - 4 frames of 10×10 pixel grids
  - Values: typically [0, 1, 2, 3] for different entity types
- `task_id` (int): Task identifier
  - 0 = Snake
  - 1 = Pong
  - 2 = Maze
  - 3 = Characters

**Returns**:
- `output` (torch.Tensor): Action logits, shape depends on task:
  - Snake/Maze: `[batch, 4]` (UP, DOWN, LEFT, RIGHT)
  - Pong: `[batch, 2]` (UP, DOWN)
  - Characters: `[batch, 62]` (0-9, A-Z, a-z)

**Example**:
```python
import torch
from snn_qat import TaskAwareSNN

# Initialize network
snn = TaskAwareSNN(beta=0.5)

# Create sample input (1 sample, 4 frames, 10×10 grid)
state = torch.randn(1, 4, 10, 10)

# Forward pass for Snake game
logits = snn(state, task_id=0)
action = logits.argmax(dim=1)  # Choose best action

print(f"Logits: {logits}")
print(f"Action: {action.item()}")  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
```

**Computational Cost**:
- Time: ~1.2ms (CPU), ~0.3ms (GPU)
- Memory: 4.7 MB (with ternary weights)
- FLOPs: ~2.1M per inference

---

#### Function: `ternarize_weight(w)`

**Description**: Quantizes weights to {-1, 0, 1} using straight-through estimator.

**Parameters**:
- `w` (torch.Tensor): Full-precision weights

**Returns**:
- `quantized` (torch.Tensor): Ternarized weights in {-1, 0, 1}

**Algorithm**:
```python
scale = w.abs().max() + 1e-6
normalized = w / scale
out = torch.zeros_like(normalized)
out[normalized > 0.1] = 1.0   # Strong positive
out[normalized < -0.1] = -1.0  # Strong negative
# Middle values (|w| < 0.1) → 0 (sparse)
return out * scale
```

**Example**:
```python
weights = torch.randn(16, 4, 3, 3)  # Conv layer weights
ternary = ternarize_weight(weights)

print(f"Unique values: {ternary.unique()}")  # [-1.0, 0.0, 1.0]
print(f"Sparsity: {(ternary == 0).float().mean():.2%}")  # ~20-30%
```

---

## System 2: Vector Symbolic Architecture

### Module: `rust_vsa/src/lib.rs` (Rust) + `python/hypervec_py.py` (Python wrapper)

#### Class: `HyperVector`

**Description**: 10,240-bit hypervector for symbolic reasoning.

**Constructor** (Python interface):
```python
from hypervec_rs import HyperVector

# Create random hypervector
hv = HyperVector.random()

# Create from binary string
hv = HyperVector.from_binary("1010...")  # 10,240 bits

# Create from list of 0s and 1s
hv = HyperVector.from_list([0, 1, 0, 1, ...])
```

**Methods**:

##### `xor(other)`

**Description**: Bind two hypervectors using XOR operation (creates association).

**Parameters**:
- `other` (HyperVector): Second hypervector to bind

**Returns**:
- `result` (HyperVector): Bound hypervector

**Mathematical operation**: `C = A ⊕ B` (bitwise XOR)

**Properties**:
- Reversible: `C ⊕ B = A` (unbind to retrieve original)
- Distributed: No single bit contains full information
- Noise-tolerant: Works with up to 30% bit flips

**Example**:
```python
# Create concept vectors
food = HyperVector.random()
location_above = HyperVector.random()

# Bind them: "food is above"
food_above = food.xor(location_above)

# Retrieve: "what is above?" → food
query = food_above.xor(location_above)
similarity = query.similarity(food)
print(f"Similarity: {similarity:.3f}")  # ~0.98 (high match)
```

---

##### `bundle(other)`

**Description**: Superpose two hypervectors using majority vote (creates set).

**Parameters**:
- `other` (HyperVector): Second hypervector to bundle

**Returns**:
- `result` (HyperVector): Bundled hypervector

**Mathematical operation**: `C = A ⊕ B` (majority vote per bit)

**Example**:
```python
# Bundle multiple concepts into a set
apple = HyperVector.random()
banana = HyperVector.random()
orange = HyperVector.random()

# Create "fruit" concept
fruit = apple.bundle(banana).bundle(orange)

# Check membership
print(f"Apple in fruit: {fruit.similarity(apple):.3f}")    # ~0.67
print(f"Banana in fruit: {fruit.similarity(banana):.3f}")  # ~0.67
print(f"Car in fruit: {fruit.similarity(car):.3f}")        # ~0.50 (random)
```

---

##### `similarity(other)`

**Description**: Compute similarity between two hypervectors using normalized Hamming distance.

**Parameters**:
- `other` (HyperVector): Hypervector to compare against

**Returns**:
- `sim` (float): Similarity score in [0.0, 1.0]
  - 1.0 = identical
  - 0.5 = random/unrelated
  - 0.0 = opposite (rare in practice)

**Formula**: `sim = 1 - (hamming_distance / total_bits)`

**Example**:
```python
hv1 = HyperVector.random()
hv2 = HyperVector.random()
hv3 = hv1.copy()  # Identical copy

print(f"Random vs Random: {hv1.similarity(hv2):.3f}")     # ~0.50
print(f"Identical: {hv1.similarity(hv3):.3f}")             # 1.00

# Test noise tolerance
noisy = hv1.add_noise(0.1)  # Flip 10% of bits
print(f"With 10% noise: {hv1.similarity(noisy):.3f}")     # ~0.90
```

---

## Symbol Grounding

### Module: `python/symbol_grounding.py`

#### Function: `extract_predicates(state, task='snake')`

**Description**: Convert raw game state into symbolic predicates (true/false statements).

**Parameters**:
- `state` (np.ndarray): Game state, shape `[10, 10]` or similar
  - Values represent entity types (empty, wall, agent, goal, etc.)
- `task` (str): Task name ('snake', 'pong', 'maze')

**Returns**:
- `predicates` (dict): Dictionary of predicate names → boolean values

**Example (Snake)**:
```python
import numpy as np
from symbol_grounding import extract_predicates

# Create sample Snake state (10×10 grid)
state = np.zeros((10, 10))
state[3, 5] = 1  # Snake head
state[2, 5] = 2  # Food
state[0, :] = 3  # Top wall

predicates = extract_predicates(state, task='snake')

print(predicates)
# Output:
# {
#     'food_above': True,      # Food is at (2,5), head at (3,5)
#     'food_below': False,
#     'food_left': False,
#     'food_right': False,
#     'wall_up': False,        # No immediate wall above
#     'wall_down': False,
#     'wall_left': False,
#     'wall_right': False,
#     'body_up': False,
#     'body_down': False,
#     'food_close': True,      # Manhattan distance < 3
#     'food_far': False
# }
```

**Predicate Definitions**:

**Spatial (Snake/Maze)**:
- `food_above`: Goal is in direction of -y
- `food_below`: Goal is in direction of +y
- `food_left`: Goal is in direction of -x
- `food_right`: Goal is in direction of +x
- `wall_<dir>`: Obstacle in direction within 1 cell
- `body_<dir>`: Self-collision risk in direction

**Temporal (Pong)**:
- `ball_approaching_left`: Ball moving toward left paddle
- `ball_approaching_right`: Ball moving toward right paddle
- `ball_going_up`: Vertical velocity < 0
- `ball_going_down`: Vertical velocity > 0
- `paddle_aligned`: Paddle y-position matches ball y-position

**Distance**:
- `food_close`: Manhattan distance < 3
- `food_far`: Manhattan distance ≥ 5
- `goal_visible`: Direct line-of-sight exists

---

#### Function: `ground_to_vsa(predicates, codebook)`

**Description**: Convert boolean predicates into a single VSA hypervector.

**Parameters**:
- `predicates` (dict): Predicate dictionary from `extract_predicates()`
- `codebook` (dict): Mapping from predicate names to HyperVectors

**Returns**:
- `state_vector` (HyperVector): Bundled hypervector representing state

**Algorithm**:
```python
def ground_to_vsa(predicates, codebook):
    active = [codebook[name] for name, value in predicates.items() if value]
    if len(active) == 0:
        return HyperVector.random()  # Empty state
    
    state_vec = active[0]
    for hv in active[1:]:
        state_vec = state_vec.bundle(hv)
    
    return state_vec
```

**Example**:
```python
# Load codebook
import pickle
with open('codebook.pkl', 'rb') as f:
    codebook = pickle.load(f)

# Ground predicates
state = np.array([...])  # Game state
predicates = extract_predicates(state)
state_vec = ground_to_vsa(predicates, codebook)

# Query: "What action should I take if food is above?"
food_above_vec = codebook['food_above']
action_query = state_vec.xor(food_above_vec)

# Find closest action
best_action = None
best_sim = 0.0
for action_name in ['move_up', 'move_down', 'move_left', 'move_right']:
    action_vec = codebook[action_name]
    sim = action_query.similarity(action_vec)
    if sim > best_sim:
        best_sim = sim
        best_action = action_name

print(f"Suggested action: {best_action} (similarity: {best_sim:.3f})")
```

---

## Learning Mechanisms

### Module: `python/learning.py`

#### Class: `HybridLearner`

**Description**: Manages three parallel learning mechanisms: Hebbian, RL, and Imitation.

**Constructor**:
```python
HybridLearner(
    snn,                    # TaskAwareSNN instance
    hebbian_lr=0.01,       # Hebbian learning rate
    rl_lr=0.001,           # RL learning rate  
    gamma=0.99,            # Discount factor
    replay_size=1000       # Replay buffer capacity
)
```

**Methods**:

##### `update_hebbian(pre_spikes, post_spikes, reward)`

**Description**: Apply 3-factor Hebbian learning rule.

**Parameters**:
- `pre_spikes` (torch.Tensor): Pre-synaptic spike trains, shape `[batch, in_features]`
- `post_spikes` (torch.Tensor): Post-synaptic spike trains, shape `[batch, out_features]`
- `reward` (float): Modulatory signal (reward value)

**Returns**: None (updates SNN weights in-place)

**Formula**: `Δw_ij = η × reward × pre_i × post_j`

**Example**:
```python
learner = HybridLearner(snn)

# Collect spike data during episode
pre_spikes = []  # Input layer spikes
post_spikes = []  # Output layer spikes

for timestep in range(episode_length):
    pre, post = snn.get_spike_traces()
    pre_spikes.append(pre)
    post_spikes.append(post)

# Apply Hebbian update at episode end
reward = total_episode_reward
learner.update_hebbian(
    torch.stack(pre_spikes),
    torch.stack(post_spikes),
    reward
)
```

---

##### `update_rl(trajectory)`

**Description**: Apply REINFORCE policy gradient update.

**Parameters**:
- `trajectory` (list): List of (state, action, reward) tuples

**Returns**:
- `loss` (float): Policy gradient loss value

**Example**:
```python
trajectory = []

for step in range(episode_length):
    state = env.get_state()
    action = agent.act(state)
    reward = env.step(action)
    trajectory.append((state, action, reward))

# Update policy
loss = learner.update_rl(trajectory)
print(f"Policy loss: {loss:.4f}")
```

---

##### `update_imitation(states, teacher_actions)`

**Description**: Learn from teacher demonstrations using cross-entropy loss.

**Parameters**:
- `states` (torch.Tensor): Batch of states, shape `[batch, 4, 10, 10]`
- `teacher_actions` (torch.Tensor): Ground truth actions, shape `[batch]`

**Returns**:
- `loss` (float): Cross-entropy loss
- `accuracy` (float): Agreement rate with teacher

**Example**:
```python
# Collect teacher demonstrations
teacher_data = []

for episode in range(10):
    state = env.reset()
    while not done:
        # Human player provides action via keyboard
        teacher_action = get_keyboard_input()
        teacher_data.append((state, teacher_action))
        state, done = env.step(teacher_action)

# Train on demonstrations
states, actions = zip(*teacher_data)
states = torch.stack(states)
actions = torch.tensor(actions)

loss, acc = learner.update_imitation(states, actions)
print(f"Imitation loss: {loss:.4f}, Accuracy: {acc:.2%}")
```

---

## Memory Systems

### Module: `python/episodic_memory.py`

#### Class: `EpisodicMemory`

**Description**: VSA-based experience storage with locality-sensitive hashing for fast retrieval.

**Constructor**:
```python
EpisodicMemory(
    capacity=10000,          # Maximum episodes to store
    vsa_dim=10240,          # Hypervector dimension
    similarity_threshold=0.7 # Match threshold for retrieval
)
```

**Methods**:

##### `store(state, action, reward, predicates)`

**Description**: Store an experience tuple.

**Parameters**:
- `state` (np.ndarray): Raw game state
- `action` (int): Action taken
- `reward` (float): Reward received
- `predicates` (dict): Symbolic predicates

**Returns**: None

**Example**:
```python
memory = EpisodicMemory()

# Store experience
state = env.get_state()
action = agent.act(state)
reward = env.step(action)
predicates = extract_predicates(state)

memory.store(state, action, reward, predicates)
```

---

##### `retrieve(query_predicates, k=5)`

**Description**: Retrieve k most similar experiences using VSA similarity.

**Parameters**:
- `query_predicates` (dict): Query predicates
- `k` (int): Number of experiences to retrieve

**Returns**:
- `experiences` (list): List of (state, action, reward, predicates) tuples

**Example**:
```python
# Query: "What happened when food was above and wall was left?"
query = {
    'food_above': True,
    'wall_left': True
}

similar_experiences = memory.retrieve(query, k=5)

for state, action, reward, preds in similar_experiences:
    print(f"Action: {action}, Reward: {reward}")
    print(f"Predicates: {preds}")
```

---

## Decision Fusion

### Module: `python/brain_fusion.py`

#### Function: `fuse_decisions(snn_output, vsa_output, predicates, confidence_threshold=0.7)`

**Description**: Integrate System 1 (SNN) and System 2 (VSA) decisions using free energy gating.

**Parameters**:
- `snn_output` (torch.Tensor): System 1 action logits
- `vsa_output` (int or None): System 2 suggested action (if invoked)
- `predicates` (dict): Current symbolic state
- `confidence_threshold` (float): Surprise threshold for System 2 invocation

**Returns**:
- `action` (int): Final action to execute
- `decision_type` (str): 'system1_trusted', 'system2_veto', or 'system2_improve'
- `explanation` (dict): Reasoning trace

**Example**:
```python
from brain_fusion import fuse_decisions

# Get System 1 proposal
snn_logits = snn(state, task_id=0)

# Extract predicates
predicates = extract_predicates(state)

# Fuse decisions
action, decision_type, explanation = fuse_decisions(
    snn_logits, 
    vsa_output=None,  # System 2 invoked internally if needed
    predicates=predicates,
    confidence_threshold=0.7
)

print(f"Action: {action}")
print(f"Decision type: {decision_type}")
print(f"Explanation: {explanation}")

# Example output:
# Action: 0 (UP)
# Decision type: system2_veto
# Explanation: {
#   'system1_proposal': 2 (LEFT),
#   'system1_confidence': 0.65,
#   'surprise': 0.82,
#   'system2_invoked': True,
#   'veto_reason': 'Simulated collision with wall',
#   'alternative_action': 0 (UP),
#   'alternative_reward': +10
# }
```

---

## Game Environments

### Module: `python/snake_ui.py` (and similar for Pong, Maze)

#### Class: `SnakeEnv`

**Description**: Snake game environment following standard RL interface.

**Constructor**:
```python
SnakeEnv(grid_size=10, food_reward=10, collision_penalty=-100)
```

**Methods**:

##### `reset()`

**Description**: Reset environment to initial state.

**Returns**:
- `state` (np.ndarray): Initial state, shape `[10, 10]`

---

##### `step(action)`

**Description**: Execute action and return result.

**Parameters**:
- `action` (int): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

**Returns**:
- `state` (np.ndarray): New state
- `reward` (float): Immediate reward
- `done` (bool): Episode termination flag
- `info` (dict): Additional information

**Example**:
```python
from snake_ui import SnakeEnv

env = SnakeEnv()
state = env.reset()

for step in range(1000):
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    
    if done:
        print(f"Episode finished. Score: {info['score']}")
        state = env.reset()
```

---

## Utilities

### Module: `python/persistence.py`

#### Function: `save_brain(snn, codebook, metadata, filepath)`

**Description**: Save complete brain state to disk.

**Parameters**:
- `snn` (TaskAwareSNN): Neural network to save
- `codebook` (dict): VSA concept dictionary
- `metadata` (dict): Training statistics
- `filepath` (str): Save path

**Example**:
```python
from persistence import save_brain, load_brain

# Save
save_brain(
    snn=my_snn,
    codebook=my_codebook,
    metadata={
        'episodes': 247,
        'accuracy': 0.87,
        'tasks': ['snake', 'pong']
    },
    filepath='brain_checkpoint.pth'
)

# Load
snn, codebook, metadata = load_brain('brain_checkpoint.pth')
print(f"Loaded brain trained for {metadata['episodes']} episodes")
```

---

### Module: `python/visualization.py`

#### Function: `plot_learning_curve(metrics, save_path=None)`

**Description**: Plot training metrics over time.

**Parameters**:
- `metrics` (dict): Dictionary with keys 'accuracy', 'loss', 'reward'
- `save_path` (str, optional): Path to save figure

**Example**:
```python
from visualization import plot_learning_curve

metrics = {
    'accuracy': [0.45, 0.56, 0.68, 0.79, 0.85],
    'loss': [1.23, 0.98, 0.76, 0.54, 0.42],
    'reward': [-20, -5, 15, 38, 52]
}

plot_learning_curve(metrics, save_path='learning_curve.png')
```

---

## Complete Example: Custom Task

Here's a complete example integrating all components:

```python
import torch
import numpy as np
from snn_qat import TaskAwareSNN
from symbol_grounding import extract_predicates, ground_to_vsa
from learning import HybridLearner
from brain_fusion import fuse_decisions
from episodic_memory import EpisodicMemory

# 1. Initialize components
snn = TaskAwareSNN(beta=0.5)
learner = HybridLearner(snn)
memory = EpisodicMemory()

# 2. Training loop
for episode in range(100):
    state = env.reset()
    trajectory = []
    
    for step in range(200):
        # Get predicates
        predicates = extract_predicates(state, task='snake')
        
        # System 1 forward pass
        state_tensor = torch.tensor(state).unsqueeze(0)
        snn_output = snn(state_tensor, task_id=0)
        
        # Fuse with System 2
        action, decision_type, explanation = fuse_decisions(
            snn_output, None, predicates
        )
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        memory.store(state, action, reward, predicates)
        trajectory.append((state, action, reward))
        
        state = next_state
        if done:
            break
    
    # 3. Learning updates
    # Hebbian
    pre_spikes, post_spikes = snn.get_spike_traces()
    learner.update_hebbian(pre_spikes, post_spikes, sum([r for _, _, r in trajectory]))
    
    # RL
    rl_loss = learner.update_rl(trajectory)
    
    print(f"Episode {episode}: Score={info['score']}, Loss={rl_loss:.4f}")

# 4. Save trained brain
from persistence import save_brain
save_brain(snn, codebook, {'episodes': 100}, 'trained_brain.pth')
```

---

## Performance Tips

1. **Batch processing**: Process multiple states simultaneously for 3-5× speedup
2. **GPU acceleration**: Move SNN to GPU with `snn.cuda()`
3. **Reduce VSA dimension**: Use 2,560 bits instead of 10,240 for 4× faster similarity (slight accuracy loss)
4. **Cache predicates**: Don't recompute predicates if state hasn't changed
5. **Limit replay buffer**: Keep only recent 500 experiences for faster sampling

---

## See Also

- [Main README](./README.md) - Overview and quick start
- [Architecture Documentation](./docs/ARCHITECTURE.md) - System design
- [Technical Report](./NSCK_Technical_Report.md) - Mathematical foundations

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Contributors**: NSCK Development Team
