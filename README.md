# NCGN - Neural Cognitive Graph Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A neuro-symbolic AI framework implementing dual-process cognition through vectorized graph dynamics and LLM reasoning.**

## 🧠 What is NCGN?

NCGN (Neural Cognitive Graph Network) is a cutting-edge artificial intelligence system that bridges the gap between **neural networks** (fast, intuitive) and **symbolic reasoning** (slow, deliberative). Inspired by human cognition, it implements:

- **System 1 (Fast Thinking)**: Massively parallel associative processing using sparse matrix operations
- **System 2 (Slow Thinking)**: Deliberative reasoning using constraint-based Large Language Models
- **Learning**: Biologically-plausible 3-factor Hebbian plasticity with reward modulation
- **Semantics**: Transformer-based embeddings for natural language understanding

### Why NCGN?

Traditional AI systems are either:
- **Pure Neural Networks**: Fast but opaque, struggle with logical reasoning
- **Pure Symbolic Systems**: Interpretable but brittle, struggle with ambiguity

**NCGN combines the best of both worlds**, achieving:
- ⚡ **Speed**: 10-100x faster than pure Python graph traversal
- 🧩 **Flexibility**: Learns from experience through reward signals
- 🔍 **Interpretability**: Explicit knowledge graph with confidence tracking
- 🌐 **Natural Language**: Seamlessly understands and generates human language

---

## 📊 Key Features

### 🚀 High-Performance Associative Engine
- **Rustworkx** backend: Compiled Rust graph library (10-100x faster than NetworkX)
- **Sparse matrix operations**: O(E) propagation using SciPy CSR matrices
- **Vectorized computation**: SIMD-optimized NumPy operations
- **Data-Oriented Design**: Structure-of-Arrays layout for cache efficiency

### 🧪 Biologically-Inspired Learning
- **3-Factor Hebbian Learning**: "Neurons that fire together, wire together" + dopamine modulation
- **Reward Prediction Error**: Learns from surprises (actual reward - expected reward)
- **Auto-Genesis**: Automatically creates connections based on co-occurrence patterns
- **Synaptic Scaling**: Maintains stable weight distributions

### 🛡️ Robust Logic Core (v2.0)
- **Confidence Tracking**: Multi-factor reliability scoring (repetition, source, temporal decay)
- **Conflict Detection**: Identifies contradictions before accepting new knowledge
- **Decision Engine**: Policy-based acceptance/rejection of new facts
- **Chain-of-Thought**: Enforced reasoning in LLM outputs

### 🌍 Natural Language Integration
- **Semantic Embeddings**: all-MiniLM-L6-v2 (384-dim) for concept grounding
- **Entry Point Detection**: Maps text queries to graph concepts using similarity
- **Structured Extraction**: LLM-powered triplet extraction (subject-relation-object)
- **Natural Responses**: Converts mathematical brain state to human-readable text

### 🎮 Interactive Environments
- **Corridor Game**: Simple 1D navigation for learning validation
- **Snake Game**: Complex 2D environment for spatial reasoning
- **Real-time Dashboard**: Web-based visualization of brain activity
- **Persistent State**: Save/load brain snapshots for experiments

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Install dependencies
pip install -r requirements.txt

# Optional: Install LLM for System 2 reasoning
# Download TinyLlama model (or use your own GGUF model)
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Basic Usage

```python
from ncgn.brain import Brain
from ncgn.config import DEFAULT_CONFIG

# Initialize brain
brain = Brain(config=DEFAULT_CONFIG, use_mock_reasoner=True)

# Add concepts and connections
brain.add_concept("fire", initial_energy=0.5)
brain.add_concept("smoke")
brain.add_concept("heat")

brain.connect("fire", "smoke", weight=0.8)
brain.connect("fire", "heat", weight=0.7)

# Inject energy and propagate
brain.inject("fire", 0.9)
active_concepts = brain.think(steps=10)

print("Active concepts:", active_concepts)
# Output: {'fire': 0.85, 'smoke': 0.68, 'heat': 0.59, ...}

# Query the brain
response = brain.ask("What causes smoke?")
print(response)
# Output: "Based on active concepts (fire, smoke, heat), fire typically causes smoke."

# Learn from feedback
brain.learn(reward=1.0)  # Positive reward strengthens active connections
```

### Advanced Example: Natural Language Knowledge Building

```python
from ncgn.brain import Brain

# Initialize with LLM for System 2 reasoning
brain = Brain(
    llm_model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    use_embeddings=True
)

# Process natural language input
result = brain.process_input("Dogs are mammals. Dogs eat meat.")

print(f"Active concepts: {len(result['active_after'])}")
print(f"New nodes: {result['graph_updates']['accepted_nodes']}")
print(f"New edges: {result['graph_updates']['accepted_edges']}")
print(f"Formatted response: {result['formatted_response']}")

# The brain has now learned:
# - Concepts: "dog", "mammal", "meat"
# - Relations: dog→mammal (is_a), dog→meat (eats)

# Query learned knowledge
response = brain.ask("What do dogs eat?")
print(response)
# Output: "Based on the knowledge graph, dogs eat meat."
```

### Game Environment Example

```python
from ncgn.brain import Brain
from ncgn.games.corridor import CorridorGame
from ncgn.games.bridge import GameBrainBridge

# Initialize game and brain
game = CorridorGame()
brain = Brain()
bridge = GameBrainBridge(game, brain)

# Train the brain to play
for episode in range(100):
    bridge.reset()
    total_reward = 0
    
    while not bridge.is_done():
        # Brain selects action based on sensory input
        action = bridge.select_action()
        
        # Execute action in environment
        observation, reward, done = bridge.step(action)
        
        # Brain learns from reward
        brain.learn(reward)
        
        total_reward += reward
    
    print(f"Episode {episode}: Reward = {total_reward}")

# After training, the brain has learned to navigate to the goal!
```

---

## 📂 Project Structure

```
Node_network/
├── ncgn/                      # Core NCGN library
│   ├── brain.py               # Main Brain orchestrator
│   ├── topology.py            # Rustworkx graph structure
│   ├── state.py               # Cognitive state (SoA arrays)
│   ├── engine.py              # Propagation engine (System 1)
│   ├── learner.py             # Hebbian learning
│   ├── embeddings.py          # Semantic layer
│   ├── confidence.py          # Confidence tracking (v2.0)
│   ├── conflicts.py           # Conflict detection (v2.0)
│   ├── decision.py            # Decision engine (v2.0)
│   ├── linguistic_processor.py # LLM triplet extraction
│   ├── formatter.py           # Natural language formatting
│   ├── reasoner.py            # LLM reasoner (System 2)
│   ├── persistence.py         # Save/load brain state
│   ├── config.py              # Configuration
│   ├── logger.py              # Logging utilities
│   ├── games/                 # Game environments
│   │   ├── corridor.py        # 1D navigation
│   │   ├── snake.py           # 2D snake game
│   │   ├── bridge.py          # Game-brain interface
│   │   └── interface.py       # Base game interface
│   └── schemas/               # Pydantic schemas
│       └── cognitive.py       # LLM output schemas
├── ui/                        # Web dashboard
│   ├── dashboard.py           # Flask server
│   └── templates/             # HTML templates
├── nsck-demo/                 # Demo applications
│   └── python/                # Python demos
├── tests/                     # Test suite
│   ├── test_brain/            # Unit tests
│   ├── test_v7/               # v7 tests
│   ├── test_integration.py    # Integration tests
│   └── test_v7_rigorous.py    # Rigorous validation
├── docs/                      # Documentation
├── saved_brains/              # Persistent brain states
├── run.py                     # Simple CLI interface
├── requirements.txt           # Python dependencies
├── ARCHITECTURE.md            # Architecture documentation
├── README.md                  # This file
└── DOCUMENTATION.md           # Technical documentation
```

---

## 🔧 Configuration

The brain can be configured through the `Config` class:

```python
from ncgn.config import Config

config = Config(
    # Capacity and sizing
    initial_capacity=10000,
    embedding_dim=384,
    
    # Physics parameters
    decay_delta=0.1,          # Energy decay per tick
    flow_alpha=0.8,           # Propagation conductivity
    norm_beta=0.01,           # Divisive normalization
    default_threshold=0.75,   # Firing threshold
    refractory_period=10,     # Cooldown after firing
    seizure_threshold=50.0,   # Max total energy
    seizure_damping=0.5,      # Damping when exceeded
    
    # Learning parameters
    learning_rate=0.01,
    weight_min=0.0,
    weight_max=1.0,
    
    # LLM settings
    llm_model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    llm_context_window=4096,
    llm_gpu_layers=-1,        # -1 = all layers on GPU
    
    # Embedding model
    embedding_model="all-MiniLM-L6-v2",
)

brain = Brain(config=config)
```

---

## 📚 Core Concepts

### System 1: Fast Associative Processing

**What it does**: Spreads activation through the knowledge graph like neural firing.

**Mathematical Model**:
```
A_{t+1} = σ((A_t × (1-δ) + W·A_t × α + I_ext) / (1 + β·ΣA)) ⊙ (1-R)
```

- **A_t**: Activation vector at time t
- **δ**: Decay rate (thoughts fade without reinforcement)
- **W**: Sparse adjacency matrix (connections between concepts)
- **α**: Flow rate (synaptic conductivity)
- **I_ext**: External input (sensory stimuli)
- **β**: Normalization constant (prevents runaway activation)
- **σ**: Sigmoid activation function
- **R**: Refractory mask (cooldown period after firing)

**Key Properties**:
- **Parallel**: All nodes update simultaneously
- **Fast**: O(E) complexity where E = number of edges
- **Continuous**: Runs every tick (typically 10-100 ticks per second)
- **Emergent**: Complex behavior from simple local rules

### System 2: Slow Deliberative Reasoning

**What it does**: Uses LLM to extract structured knowledge and make logical decisions.

**Pipeline**:
1. **Input**: Natural language text
2. **LLM Processing**: Extract concepts and relations
3. **Schema Validation**: Ensure structured output (Pydantic)
4. **Conflict Detection**: Check for contradictions
5. **Decision**: Accept, reject, or flag for curiosity
6. **Application**: Update graph if accepted

**Example Schema**:
```python
{
  "reasoning": "Dogs are carnivores, so they eat meat.",
  "new_nodes": [
    {"label": "dog", "properties": {"is_alive": true}},
    {"label": "meat", "properties": {"is_edible": true}}
  ],
  "new_edges": [
    {"source": "dog", "target": "meat", "relation_type": "eats", "weight": 0.8}
  ]
}
```

### Hebbian Learning: "Neurons that Fire Together, Wire Together"

**3-Factor Learning Rule**:
```
ΔW_{ij} = η × M × A_i × A_j
```

- **η**: Learning rate (0.01)
- **M**: Reward modulator (dopamine-like signal)
  - M > 0 → LTP (Long-Term Potentiation) - strengthen connection
  - M < 0 → LTD (Long-Term Depression) - weaken connection
- **A_i**: Pre-synaptic activation
- **A_j**: Post-synaptic activation

**Reward Prediction Error**:
```
M = r - r̂
where r̂ = 0.1 × r + 0.9 × r̂_{previous}
```

**Example**:
```python
# Brain activates: "fire" → "smoke" → "danger"
brain.inject("fire", 0.9)
brain.think(steps=10)

# Positive reward strengthens the pathway
brain.learn(reward=1.0)  # LTP: fire→smoke and smoke→danger get stronger

# Negative reward weakens the pathway
brain.learn(reward=-0.5)  # LTD: connections weaken
```

### Semantic Embeddings: Bridging Language and Concepts

**Purpose**: Map natural language to graph concepts using vector similarity.

**How it works**:
1. **Encode text**: "What causes fire?" → 384-dim vector
2. **Compare to concepts**: Compute cosine similarity to all graph concepts
3. **Find entry points**: Top-5 most similar concepts
4. **Inject energy**: Activate those concepts proportional to similarity

**Example**:
```
Query: "What causes fire?"
↓ (encode)
Vector: [0.23, -0.15, 0.67, ...]
↓ (compare to graph)
Similarities:
  "fire": 0.85
  "combustion": 0.72
  "heat": 0.68
  "oxygen": 0.61
  "smoke": 0.55
↓ (inject energy)
brain.inject("fire", 0.85 * 0.5)
brain.inject("combustion", 0.72 * 0.5)
...
```

### Confidence & Conflict System (v2.0)

**Confidence Tracking**: 3-factor reliability scoring

1. **Repetition**: More reinforcements → higher confidence
   ```
   C_reinforce = min(0.8, 0.1 × count)
   ```

2. **Source Credibility**:
   - Training data: base = 0.5
   - User input: base = 0.2
   - Guesses: base = 0.0

3. **Temporal Decay**: Unused knowledge fades
   ```
   C_{t+1} = λ × C_t
   where λ = 0.901 (training) or 0.99 (user)
   ```

**Conflict Detection**: Logical contradiction checking

| Confidence | Conflict Type | Action |
|------------|---------------|--------|
| > 90% | HARD_REJECT | Reject new contradictory fact |
| 50-90% | CURIOSITY | Flag for investigation |
| < 50% | ACCEPT | Overwrite low-confidence fact |
| N/A | NEW_KNOWLEDGE | Accept new fact |

**Example**:
```python
# High-confidence fact
brain.connect("bird", "fly", weight=0.9)
brain.confidence.reinforce("bird", "fly", source_type="training", count=10)
# Confidence: ~0.9

# Conflicting fact
brain.process_input("Birds cannot fly.")
# → HARD_REJECT: Confidence too high to override

# But this works:
brain.process_input("Penguins cannot fly.")
# → ACCEPT: "penguin" is a new concept, no conflict
```

---

## 🎯 Use Cases

### 1. Knowledge Base Construction
Build interpretable knowledge graphs from unstructured text.

```python
brain = Brain(llm_model_path="models/tinyllama.gguf")

texts = [
    "Water boils at 100 degrees Celsius.",
    "Ice is frozen water.",
    "Steam is water vapor.",
]

for text in texts:
    brain.process_input(text)

# Query the knowledge
print(brain.ask("What are the states of water?"))
# Output: "Water has three states: ice (solid), water (liquid), and steam (gas)."
```

### 2. Conversational AI
Natural language understanding and generation.

```python
brain = Brain(llm_model_path="models/tinyllama.gguf", use_embeddings=True)

# Build knowledge through conversation
conversation = [
    "My name is Alice.",
    "I like pizza.",
    "Pizza has cheese and tomato sauce.",
]

for utterance in conversation:
    response = brain.process_input(utterance)
    print(f"User: {utterance}")
    print(f"Bot: {response['formatted_response']}")

# Query personal knowledge
print(brain.ask("What does Alice like?"))
# Output: "Alice likes pizza."
```

### 3. Game AI
Learn to play games through trial and error.

```python
from ncgn.games.snake import SnakeGame
from ncgn.games.bridge import GameBrainBridge

game = SnakeGame(grid_size=8)
brain = Brain()
bridge = GameBrainBridge(game, brain)

# Training loop
for episode in range(1000):
    bridge.reset()
    
    while not bridge.is_done():
        action = bridge.select_action()
        obs, reward, done = bridge.step(action)
        brain.learn(reward)
    
    if episode % 100 == 0:
        stats = bridge.get_stats()
        print(f"Episode {episode}: Score = {stats['score']}")

# The brain learns to:
# - Avoid walls
# - Avoid self-collision
# - Navigate to food
# - Maximize score
```

### 4. Semantic Search
Find relevant concepts using natural language queries.

```python
brain = Brain(use_embeddings=True)

# Build knowledge base
concepts = ["dog", "cat", "car", "bicycle", "apple", "orange"]
for concept in concepts:
    brain.add_concept(concept)

# Semantic search
query = "What pets are there?"
entry_points = brain.semantics.find_entry_points(query, brain.topology, top_k=3)

print("Relevant concepts:")
for label, similarity in entry_points:
    print(f"  {label}: {similarity:.2f}")

# Output:
#   dog: 0.72
#   cat: 0.68
#   bicycle: 0.15  (less relevant)
```

### 5. Causal Reasoning
Infer causes and effects.

```python
brain = Brain()

# Build causal knowledge
brain.connect("rain", "wet_ground", weight=0.9)
brain.connect("rain", "umbrella_use", weight=0.7)
brain.connect("wet_ground", "slippery", weight=0.8)

# Forward inference: "What happens when it rains?"
brain.inject("rain", 1.0)
effects = brain.think(steps=10)
print("Effects of rain:", list(effects.keys()))
# Output: ['rain', 'wet_ground', 'slippery', 'umbrella_use']

# Backward inference: "What causes slippery ground?"
brain.clear()  # Reset activations
# (Implement backward propagation or use graph traversal)
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/test_brain/
pytest tests/test_v7/

# Integration tests
pytest tests/test_integration.py

# Rigorous validation (with visual feedback)
python tests/test_v7_rigorous.py
```

**Test Coverage**:
- ✅ Topology management (add/remove concepts, edges)
- ✅ State synchronization (arrays, matrices)
- ✅ Propagation physics (decay, firing, refractory)
- ✅ Hebbian learning (LTP, LTD, auto-genesis)
- ✅ Confidence tracking
- ✅ Conflict detection
- ✅ Decision engine
- ✅ Semantic embeddings
- ✅ End-to-end cognitive cycles

---

## 🎨 Dashboard

Launch the interactive web dashboard:

```bash
python ui/dashboard.py
```

Then open http://localhost:5000 in your browser.

**Features**:
- 📊 Real-time graph visualization (D3.js)
- 🔥 Live activation display
- 📁 File upload for knowledge extraction
- 💾 Save/load brain states
- 🎮 Integrated game environments (Corridor, Snake)
- 📈 Performance metrics

---

## 📖 Documentation

Comprehensive documentation is available:

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into system architecture, mathematical foundations, and design patterns
- **[DOCUMENTATION.md](DOCUMENTATION.md)**: Module-by-module technical documentation, API reference, and implementation details
- **[docs/V7_COMPLETE_DOCUMENTATION.md](docs/V7_COMPLETE_DOCUMENTATION.md)**: Original v7 documentation
- **[docs/V7_SETUP_GUIDE.md](docs/V7_SETUP_GUIDE.md)**: Setup and configuration guide

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Write tests**: Ensure your code is tested
4. **Follow PEP 8**: Use consistent code style
5. **Document your changes**: Update relevant documentation
6. **Submit a pull request**: Describe your changes clearly

**Areas for Contribution**:
- 🚀 Performance optimization (GPU acceleration, parallelization)
- 🧪 New learning algorithms (meta-learning, curriculum learning)
- 🎮 Additional game environments
- 🌐 Multi-modal support (vision, audio)
- 📊 Enhanced visualization
- 🐛 Bug fixes and code quality improvements

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Theoretical Foundations**:
- Dual-process theory (Kahneman, 2011)
- Hebbian learning (Hebb, 1949)
- Spreading activation (Anderson, 1983)
- Reward prediction error (Schultz, 1997)

**Technical Inspiration**:
- Rustworkx graph library
- SentenceTransformers for embeddings
- Llama.cpp for efficient LLM inference
- SciPy sparse matrices

**Research**:
- Neural-symbolic integration literature
- Cognitive architecture frameworks (ACT-R, Soar, CLARION)
- Graph neural networks
- Knowledge graph embeddings

---

## 📞 Contact

- **Repository**: [https://github.com/shiva2321/Node_network](https://github.com/shiva2321/Node_network)
- **Issues**: [https://github.com/shiva2321/Node_network/issues](https://github.com/shiva2321/Node_network/issues)
- **Discussions**: [https://github.com/shiva2321/Node_network/discussions](https://github.com/shiva2321/Node_network/discussions)

---

## 🌟 Star History

If you find NCGN useful, please consider giving it a star! ⭐

---

## 📈 Roadmap

### Version 7.x (Current)
- ✅ Data-Oriented Design (Structure of Arrays)
- ✅ Rustworkx integration
- ✅ Confidence tracking
- ✅ Conflict detection
- ✅ Semantic embeddings
- ✅ Game environments

### Version 8.0 (Planned)
- 🚀 GPU acceleration (CuPy, PyTorch)
- 🚀 Parallelized propagation
- 🚀 Distributed graph sharding
- 🚀 Enhanced observability dashboard

### Version 9.0 (Future)
- 🌐 Multi-modal embeddings (vision, audio)
- 🧠 Meta-learning capabilities
- 🔮 Planning algorithms (MCTS, A*)
- 🌍 Multi-agent coordination

---

**Built with ❤️ for advancing neuro-symbolic AI**
