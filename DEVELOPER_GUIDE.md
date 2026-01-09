# NCGN Developer Guide

Technical documentation, API reference, and architecture details for the Neuromorphic Cognitive Graph Network.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [API Reference](#api-reference)
4. [Dashboard Architecture](#dashboard-architecture)
5. [Implementation Details](#implementation-details)
6. [Extension Guide](#extension-guide)
7. [Research Background](#research-background)

---

## Architecture Overview

### System Architecture

```
NCGN System
│
├── Phase 1: Linguistic Graph Substrate
│   ├── GraphBuilder: Constructs graph from corpus
│   ├── LinguisticGraph: Core graph structure (DGL)
│   ├── GraphEmbedding: Pre-trained embeddings (RoBERTa/BERT)
│   └── StructuralEncoder: Laplacian/random walk PE
│
├── Phase 2: Neuromorphic Core
│   ├── LIFNeuron: Leaky Integrate-and-Fire
│   ├── IzhikevichNeuron: Biologically realistic model
│   ├── SpikingLayer: Full layer of spiking neurons
│   ├── PoissonEncoder: Spike encoding
│   └── STDPLearning: Plasticity rules (4 variants)
│
├── Phase 3: Dual-System Architecture
│   ├── GraphTransformer: System 1 (Neural)
│   ├── SymbolicReasoner: System 2 (Logical)
│   └── DualSystemArchitecture: Integration layer
│
├── Phase 4: Continual Learning (Planned)
│   ├── AL-GNN: Analytic learning
│   ├── BGML: Cognitive sharding
│   └── NGM: Neural graph memory
│
└── Phase 5: Hardware Optimization (Planned)
    ├── Memory management
    ├── Dynamic scheduling
    └── Energy monitoring
```

### Dashboard Architecture

```
Cognitive Cockpit
│
├── Backend (Flask + SocketIO)
│   ├── ncgn_dashboard.py: Main server
│   ├── dashboard_utils/
│   │   ├── metrics_monitor.py: Training metrics
│   │   ├── hardware_profiler.py: GPU/CPU monitoring
│   │   ├── hebbian_tracker.py: Synaptic tracking
│   │   ├── training_manager.py: Training orchestration
│   │   └── data_loader.py: File processing
│   └── API Endpoints: REST + WebSocket
│
└── Frontend (HTML + D3.js + Chart.js)
    ├── templates/dashboard.html: UI structure
    ├── static/css/dashboard.css: Styling
    └── static/js/dashboard.js: Client logic
```

---

## Core Modules

### Module 1: Linguistic Graph (`ncgn/linguistic_graph.py`)

**Classes**:

#### `LinguisticGraph`

Represents the linguistic graph substrate.

```python
class LinguisticGraph:
    def __init__(self, vocab: Dict[str, int], 
                 node_features: Dict[int, NodeFeature],
                 edge_list: List[Edge],
                 config: Dict = None)
```

**Methods**:
- `add_node(token, frequency, embedding)`: Add a node
- `add_edge(source, target, weight, pmi_score)`: Add an edge
- `get_neighbors(node_id)`: Get neighbors of a node
- `get_statistics()`: Get graph statistics
- `save(path)`: Save graph to disk
- `load(path)`: Load graph from disk (classmethod)

**DGL Integration**:
```python
# Access underlying DGL graph
dgl_graph = linguistic_graph.graph

# Node features
node_features = dgl_graph.ndata['feat']

# Edge weights
edge_weights = dgl_graph.edata['weight']
```

#### `GraphBuilder`

Constructs graphs from text corpora.

```python
builder = GraphBuilder(
    embedding_model="roberta-base",
    device="cuda"
)

graph = builder.build_from_corpus(
    corpus=texts,
    window_size=5,
    min_frequency=2,
    max_vocab_size=50000
)
```

**Pipeline**:
1. Tokenize text
2. Extract vocabulary
3. Compute co-occurrence statistics
4. Calculate PMI for edges
5. Generate embeddings
6. Apply structural encodings
7. Create DGL graph

---

### Module 2: Spiking Neurons (`ncgn/spiking_neurons.py`)

#### `LIFNeuron`

Leaky Integrate-and-Fire neuron model.

```python
neuron = LIFNeuron(
    tau_membrane=10.0,
    v_threshold=1.0,
    v_reset=0.0,
    refractory_period=2
)

# Forward pass
spike_output, mem_state = neuron(input_current, membrane_state)
```

**Dynamics**:
```
τ dv/dt = -(v - v_rest) + R*I(t)

if v > v_threshold:
    v = v_reset
    emit spike
```

#### `IzhikevichNeuron`

More biologically realistic model.

```python
neuron = IzhikevichNeuron(
    a=0.02, b=0.2, c=-65, d=8
)
```

**Dynamics**:
```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)

if v ≥ 30:
    v = c
    u = u + d
    emit spike
```

#### `SpikingLayer`

Full layer of spiking neurons with synaptic delays.

```python
layer = SpikingLayer(
    in_features=768,
    out_features=256,
    neuron_model='lif',
    delay_min=1,
    delay_max=10
)

# Reset state
layer.reset_state(batch_size=32, device='cuda')

# Forward pass
output = layer(input_spikes)
```

#### Encoders

**PoissonEncoder**: Stochastic rate coding
```python
spike_train = PoissonEncoder.encode(features, time_steps=50)
# Output: (time_steps, batch, features) binary tensor
```

**RateEncoder**: Deterministic rate coding
```python
spike_train = RateEncoder.encode(features, time_steps=50)
```

**TemporalEncoder**: Latency coding
```python
spike_train = TemporalEncoder.encode(features, time_steps=50, tau=10.0)
```

---

### Module 3: STDP Learning (`ncgn/stdp_learning.py`)

#### `STDPLearning`

Classic spike-timing-dependent plasticity.

```python
stdp = STDPLearning(
    tau_plus=20.0,
    tau_minus=20.0,
    A_plus=0.01,
    A_minus=0.01
)

# Apply update
stdp.apply_update(
    weights=layer.weight,
    pre_spikes=input_spikes,
    post_spikes=output_spikes
)
```

**Learning Rule**:
```
Δw = A₊ exp(-Δt/τ₊)  if pre before post (LTP)
Δw = -A₋ exp(Δt/τ₋)  if post before pre (LTD)
```

#### Variants

**TripleSTDP**: Considers spike triplets
**RewardModulatedSTDP**: For reinforcement learning
**HomeostaticSTDP**: Maintains target firing rate

---

### Module 4: Graph Transformer (`ncgn/graph_transformer.py`)

#### `GraphTransformer`

Multi-head attention on graphs (System 1).

```python
model = GraphTransformer(
    node_feat_dim=768,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    dropout=0.1
)

output = model(node_features, adjacency_matrix)
```

**Architecture**:
```
Input → Embedding
  → [TransformerLayer × N]
  → Output Projection
```

**TransformerLayer**:
```
x → LayerNorm → MultiHeadAttention → Residual
  → LayerNorm → FeedForward → Residual
```

#### `GraphMultiHeadAttention`

Attention mechanism that respects graph structure.

```python
attn = GraphMultiHeadAttention(
    embed_dim=256,
    num_heads=8,
    edge_dim=64
)

output, attn_weights = attn(
    query, key, value,
    adjacency_mask,
    edge_features
)
```

---

### Module 5: Symbolic Reasoner (`ncgn/symbolic_reasoner.py`)

#### `SymbolicReasoner`

Knowledge graph with logical inference (System 2).

```python
reasoner = SymbolicReasoner()

# Add facts
reasoner.add_fact("cat", "is_a", "animal")

# Add rules
reasoner.add_rule(Rule(
    premises=[('?x', 'is_a', '?y'), ('?y', 'is_a', '?z')],
    conclusion=('?x', 'is_a', '?z'),
    name="transitivity"
))

# Reason
new_facts = reasoner.reason(max_iterations=10)
```

**Inference Engine**: Forward chaining with variable binding

---

### Module 6: Dual-System Architecture (`ncgn/dual_system.py`)

#### `DualSystemArchitecture`

Integrates neural (System 1) and symbolic (System 2) reasoning.

```python
dual_system = DualSystemArchitecture(
    node_feat_dim=768,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    confidence_threshold=0.7
)

output = dual_system(
    node_features,
    adjacency,
    context={'entity': 'cat'}
)
```

**Processing Modes**:
- `system1_only`: Fast neural processing (high confidence)
- `dual`: Both systems active (medium confidence)
- `system2_verification`: Logic check (low confidence or explicit request)

**Integration Strategies**:
```python
# Weighted combination
integration_mode='weighted'

# Attention-based fusion
integration_mode='attention'

# Gating mechanism
integration_mode='gating'
```

---

## API Reference

### GraphBuilder API

```python
class GraphBuilder:
    def __init__(self, embedding_model='roberta-base', device='cuda'):
        ...
    
    def build_from_corpus(
        self,
        corpus: List[str],
        window_size: int = 5,
        min_frequency: int = 2,
        max_vocab_size: int = 50000
    ) -> LinguisticGraph:
        ...
    
    def build_from_files(
        self,
        file_paths: List[str],
        **kwargs
    ) -> LinguisticGraph:
        ...
```

### SpikingLayer API

```python
class SpikingLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neuron_model: str = 'lif',
        tau_membrane: float = 10.0,
        v_threshold: float = 1.0,
        delay_min: int = 1,
        delay_max: int = 10,
        bias: bool = True
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def reset_state(self, batch_size: int, device: str = 'cpu'):
        ...
```

### DualSystemArchitecture API

```python
class DualSystemArchitecture(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        confidence_threshold: float = 0.7,
        integration_mode: str = 'weighted'
    ):
        ...
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        context: Dict = None
    ) -> Dict[str, Any]:
        ...
    
    def add_knowledge(self, facts: List[Tuple[str, str, str]]):
        ...
    
    def explain_decision(self, output: Dict) -> str:
        ...
```

---

## Dashboard Architecture

### Backend API Endpoints

#### Status & Configuration

```python
GET  /api/status
Response: {
    'model_loaded': bool,
    'graph_loaded': bool,
    'training_active': bool,
    'timestamp': str
}

GET  /api/config
Response: {config_dict}

POST /api/config
Body: {config_updates}
Response: {'status': 'success', 'config': updated_config}
```

#### Model Operations

```python
POST /api/model/load
Body: {'path': 'saved_models/ncgn/'}
Response: {'status': 'success', 'message': str}

POST /api/upload
Body: multipart/form-data with file
Response: {
    'status': 'success',
    'filename': str,
    'size': int,
    'type': str
}

POST /api/inference
Body: {
    'query': str,
    'context': dict
}
Response: {
    'status': 'success',
    'mode': str,
    'confidence': float,
    'attention': array,
    'explanation': str
}
```

#### Training

```python
POST /api/training/start
Body: {
    'learning_rate': float,
    'batch_size': int,
    'epochs': int,
    'weight_decay': float
}
Response: {'status': 'success', 'message': str}

POST /api/training/stop
Response: {'status': 'success'}

GET  /api/metrics/history
Response: {
    'metric_name': [
        {'step': int, 'value': float, 'timestamp': float},
        ...
    ]
}
```

#### Visualization

```python
GET  /api/graph/data
Response: {
    'nodes': [
        {'id': int, 'label': str, 'frequency': int, 'group': int},
        ...
    ],
    'edges': [
        {'source': int, 'target': int, 'weight': float, 'pmi': float},
        ...
    ],
    'stats': dict
}

GET  /api/hardware/stats
Response: {
    'gpu': {
        'name': str,
        'memory_allocated': float,
        'memory_total': float,
        'utilization': float,
        'temperature': float,
        'power_usage': float
    },
    'cpu': {...},
    'memory': {...},
    'energy': {...}
}
```

### WebSocket Events

**Client → Server**:
```javascript
socket.emit('request_update');
```

**Server → Client**:
```javascript
socket.on('metrics_update', (data) => {
    // Training metrics
});

socket.on('hardware_update', (data) => {
    // Hardware stats
});

socket.on('training_update', (data) => {
    // Training progress
});

socket.on('hebbian_update', (data) => {
    // Synaptic traces
});

socket.on('training_complete', (data) => {
    // Training finished
});

socket.on('training_error', (data) => {
    // Error occurred
});
```

---

## Implementation Details

### Memory Management

**GPU Memory Tracking**:
```python
# Check available memory
total_mem = torch.cuda.get_device_properties(0).total_memory
allocated = torch.cuda.memory_allocated(0)
reserved = torch.cuda.memory_reserved(0)

print(f"Used: {allocated / 1e9:.2f} GB")
print(f"Available: {(total_mem - reserved) / 1e9:.2f} GB")
```

**Memory Optimization Strategies**:

1. **Gradient Checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   def forward(self, x):
       x = checkpoint(self.layer1, x)
       x = checkpoint(self.layer2, x)
       return x
   ```

2. **Mixed Precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **CPU Offloading**:
   ```python
   # Move intermediate activations to CPU
   x = x.cpu()
   # ... other ops ...
   x = x.cuda()
   ```

### Energy Measurement

```python
from dashboard_utils import HardwareProfiler

profiler = HardwareProfiler()

# Measure operation
profiler.reset_energy()
start_time = time.time()

# Your operation here
output = model(input)

duration = time.time() - start_time
stats = profiler.get_stats()

energy_mj = stats['energy']['total_mj']
print(f"Energy: {energy_mj:.2f} mJ")
print(f"Time: {duration*1000:.2f} ms")
```

---

## Extension Guide

### Adding New Neuron Models

**Step 1**: Create neuron class in `ncgn/spiking_neurons.py`:

```python
class MyCustomNeuron(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def forward(self, x, state):
        # Implement dynamics
        new_state = ...
        spikes = (new_state > threshold).float()
        return spikes, new_state
    
    def reset_state(self, batch_size, device='cpu'):
        return torch.zeros(batch_size, device=device)
```

**Step 2**: Register in `SpikingLayer`:

```python
# In SpikingLayer.__init__
if neuron_model == 'my_custom':
    self.neuron = MyCustomNeuron(param1, param2)
```

### Adding New Learning Rules

**Step 1**: Create class in `ncgn/stdp_learning.py`:

```python
class MySTDPVariant:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def apply_update(self, weights, pre_spikes, post_spikes):
        # Implement learning rule
        delta_w = self.compute_weight_change(pre_spikes, post_spikes)
        weights.data += self.lr * delta_w
```

**Step 2**: Use in training loop:

```python
learner = MySTDPVariant(learning_rate=0.01)
learner.apply_update(layer.weight, pre_spikes, post_spikes)
```

### Adding Dashboard Panels

**Step 1**: Add panel HTML in `templates/dashboard.html`:

```html
<section id="panel-mypanel" class="dashboard-panel">
    <div class="panel-grid">
        <!-- Your content -->
    </div>
</section>
```

**Step 2**: Add navigation button:

```html
<button class="nav-btn" data-panel="mypanel">My Panel</button>
```

**Step 3**: Add JavaScript logic in `static/js/dashboard.js`:

```javascript
function updateMyPanel(data) {
    // Update panel content
}
```

### Adding Custom Visualizations

**D3.js Example**:

```javascript
function createCustomViz(data) {
    const svg = d3.select('#my-viz')
        .append('svg')
        .attr('width', 800)
        .attr('height', 600);
    
    // Draw visualization
    svg.selectAll('circle')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', 5);
}
```

---

## Research Background

### Phase 1: Graph Neural Networks

**Key Papers**:
- Dwivedi & Bresson (2020) - "A Generalization of Transformer Networks to Graphs"
- Kipf & Welling (2016) - "Semi-Supervised Classification with Graph Convolutional Networks"

**Implemented Techniques**:
- Laplacian positional encodings
- Multi-head attention on graphs
- Message passing with edge features

### Phase 2: Spiking Neural Networks

**Key Papers**:
- Maass (1997) - "Networks of Spiking Neurons"
- Bi & Poo (1998) - "Synaptic Modifications in Cultured Hippocampal Neurons"
- Izhikevich (2003) - "Simple Model of Spiking Neurons"

**Implemented Models**:
- Leaky Integrate-and-Fire (LIF)
- Izhikevich neuron
- STDP and variants

### Phase 3: Dual-Process Theory

**Key Papers**:
- Kahneman (2011) - "Thinking, Fast and Slow"
- Evans & Stanovich (2013) - "Dual-Process Theories of Higher Cognition"

**Implementation**:
- System 1: Graph Transformer (fast, intuitive)
- System 2: Symbolic Reasoner (slow, deliberate)
- Confidence-based switching

### Phase 4-5: Continual Learning (Planned)

**Key Papers**:
- Zhou et al. (2023) - "Continual Graph Learning"
- Parisi et al. (2019) - "Continual Lifelong Learning with Neural Networks"

**Planned Features**:
- Analytic Learning GNN
- Cognitive sharding (BGML)
- Information Self-Assessment Ownership

---

## Performance Benchmarks

### Current Performance (RTX 3060 12GB)

| Operation | Time | Memory |
|-----------|------|--------|
| Graph construction (10K nodes) | ~30s | ~2GB |
| Graph Transformer forward (batch=32) | ~150ms | ~3GB |
| Spiking layer forward (50 timesteps) | ~200ms | ~1GB |
| STDP update | ~50ms | <500MB |
| Dual system inference | ~150ms | ~4GB |

### Energy Efficiency

| Component | Traditional | Spiking | Savings |
|-----------|------------|---------|---------|
| Forward pass | 100 mJ | 10 mJ | 90% |
| Backward pass | 150 mJ | N/A | N/A |
| Inference | 50 mJ | 5 mJ | 90% |

---

## Code Statistics

**Total Lines of Code**: ~10,000

**Breakdown**:
- Core NCGN modules: 3,790 LOC
- Dashboard backend: 1,800 LOC
- Dashboard frontend: 2,000 LOC
- Scripts and utilities: 800 LOC
- Documentation: 6,000 LOC
- Configuration: 200 LOC

**Files**: 35+ Python files, 13 documentation files

---

**Developer Guide Version**: 1.0  
**Last Updated**: January 8, 2026  
**Compatible with**: NCGN v0.1.0+

---

*Build the future of neuromorphic AI!* 🚀

