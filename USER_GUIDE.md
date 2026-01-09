# NCGN Complete User Guide

Comprehensive guide for using the Neuromorphic Cognitive Graph Network and Cognitive Cockpit Dashboard.

## 📋 Table of Contents

**Part I: NCGN Core**
1. [Getting Started](#getting-started)
2. [Core Usage](#core-usage)
3. [Code Examples](#code-examples)

**Part II: Cognitive Cockpit Dashboard**
4. [Dashboard Overview](#dashboard-overview)
5. [Training with Dashboard](#training-with-dashboard)
6. [Visualization](#visualization)
7. [Interactive Playground](#interactive-playground)
8. [Hardware Monitoring](#hardware-monitoring)

**Part III: Advanced Topics**
9. [Interpreting Metrics](#interpreting-metrics)
10. [Optimization Tips](#optimization-tips)
11. [Troubleshooting](#troubleshooting)
12. [Quick Reference](#quick-reference)

---

# Part I: NCGN Core

## Getting Started

### First Run

```bash
# Activate virtual environment
.\ncgn_env\Scripts\Activate.ps1  # Windows
source ncgn_env/bin/activate      # Linux/Mac

# Run complete demo
python scripts/ncgn_demo.py

# Or run specific phases
python scripts/ncgn_demo.py --phase 1  # Linguistic Graph
python scripts/ncgn_demo.py --phase 2  # Spiking Networks
python scripts/ncgn_demo.py --phase 3  # Dual System
```

### Verify Installation

```bash
python verify_installation.py
```

---

## Core Usage

### 1. Build Linguistic Graph

**From Text Corpus:**

```python
from ncgn import GraphBuilder

# Prepare your corpus
corpus = [
    "Neural networks learn patterns from data",
    "Deep learning uses multiple layers",
    "Graph neural networks process graph-structured data"
]

# Build graph
builder = GraphBuilder(embedding_model="roberta-base")
graph = builder.build_from_corpus(
    corpus=corpus,
    window_size=5,
    min_frequency=2,
    max_vocab_size=50000
)

# Save graph
graph.save("my_linguistic_graph/")

# Get statistics
stats = graph.get_statistics()
print(f"Nodes: {stats['unique_tokens']}")
print(f"Edges: {stats['num_edges']}")
```

**Load Existing Graph:**

```python
from ncgn import LinguisticGraph

graph = LinguisticGraph.load("my_linguistic_graph/")
print(graph.get_statistics())
```

### 2. Use Spiking Neural Networks

**Create Spiking Layer:**

```python
from ncgn import SpikingLayer, PoissonEncoder, STDPLearning
import torch

# Create layer with LIF neurons
layer = SpikingLayer(
    in_features=768,
    out_features=256,
    neuron_model='lif',
    bias=True
)

# Reset neuron state
layer.reset_state(batch_size=1, device='cuda')

# Encode continuous features to spikes
features = torch.randn(1, 768)
spike_train = PoissonEncoder.encode(features, time_steps=50)

# Forward pass through spiking network
output_spikes = []
for t in range(50):
    out = layer(spike_train[t])
    output_spikes.append(out)

# Apply STDP learning
stdp = STDPLearning(tau_plus=20.0, tau_minus=20.0)
stdp.apply_update(layer.weight, spike_train[0], output_spikes[0])
```

### 3. Dual-System Architecture

**Initialize and Use:**

```python
from ncgn import DualSystemArchitecture

# Create dual system
dual_system = DualSystemArchitecture(
    node_feat_dim=768,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    confidence_threshold=0.7
)

# Add knowledge to System 2 (Symbolic)
dual_system.add_knowledge([
    ("cat", "is_a", "animal"),
    ("dog", "is_a", "animal"),
    ("animal", "is_a", "living_thing")
])

# Prepare input
node_features = torch.randn(1, 50, 768)
adjacency = torch.eye(50).unsqueeze(0)

# Run inference
context = {'entity': 'cat'}
output = dual_system(node_features, adjacency, context=context)

# Check results
print(f"Processing Mode: {output['mode']}")
print(f"Confidence: {output['confidence'].item():.3f}")

# Get explanation
explanation = dual_system.explain_decision(output)
print(f"\nExplanation:\n{explanation}")
```

---

## Code Examples

### Example 1: Complete Pipeline

```python
from ncgn import GraphBuilder, DualSystemArchitecture
import torch

# 1. Build graph from your data
corpus = load_your_text_data()  # Your function
builder = GraphBuilder("roberta-base")
graph = builder.build_from_corpus(corpus)

# 2. Create model
model = DualSystemArchitecture(
    node_feat_dim=768,
    embed_dim=256
)

# 3. Add domain knowledge
model.add_knowledge([
    ("transformer", "uses", "attention"),
    ("attention", "is_a", "mechanism")
])

# 4. Run inference
node_features = graph.graph.ndata['feat'][:50].unsqueeze(0)
adjacency = torch.eye(50).unsqueeze(0)
output = model(node_features, adjacency)

print(f"Result: {output['confidence']}")
```

### Example 2: Training Loop

```python
import torch.optim as optim

# Setup
model = DualSystemArchitecture(node_feat_dim=768, embed_dim=256)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Training
for epoch in range(10):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(node_features, adjacency)
    
    # Compute loss (example)
    loss = compute_your_loss(output)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

# Part II: Cognitive Cockpit Dashboard

## Dashboard Overview

### Starting the Dashboard

```bash
# Start server
python ncgn_dashboard.py
# or
launch_cockpit.bat

# Access dashboard
# Open browser: http://localhost:5000
```

### Interface Layout

The dashboard has **5 main panels**:

1. **Overview** - Status, metrics summary, hardware snapshot
2. **Training** - Configure training, upload datasets, monitor progress
3. **Visualization** - Graph structure, attention maps, Hebbian traces
4. **Playground** - Interactive queries, ablation studies
5. **Hardware** - Detailed GPU/CPU monitoring, energy tracking

**Navigation**: Click panel buttons in header to switch views

---

## Training with Dashboard

### Step 1: Upload Dataset

1. Go to **Training Panel**
2. Drag-and-drop files into upload area OR click to browse
3. **Supported formats**: TXT, PDF, DOCX, PY, JAVA, CPP, JS
4. Wait for "Upload successful" message

### Step 2: Configure Training

**Key Parameters:**

- **Learning Rate** (default: 0.0001)
  - Lower (1e-5): Stable but slow
  - Higher (1e-3): Fast but may diverge
  - **Tip**: Start with 0.0001

- **Batch Size** (default: 32)
  - For RTX 3060 12GB: Use 32
  - For 6GB VRAM: Use 16
  - For 24GB VRAM: Use 64

- **Epochs** (default: 10)
  - More epochs = better convergence
  - Stop early if loss plateaus

- **Weight Decay** (default: 0.01)
  - Regularization strength
  - Higher = more regularization

### Step 3: Start Training

1. Click **"▶️ Start Training"**
2. Monitor **Loss Curves** in real-time
3. Watch **Hardware Panel** for memory usage
4. Check **Activity Log** for progress messages

**Stop Training**: Click **"⏹️ Stop Training"** anytime

### Reading Loss Curves

**Good Training** ✅:
```
Loss: 1.5 → 0.8 → 0.5 → 0.3 → 0.25 (smooth decrease)
```

**Overfitting** ⚠️:
```
Train Loss: Decreasing
Validation Loss: Increasing
→ Action: Stop training or add regularization
```

**Divergence** 🔴:
```
Loss: 0.5 → 2.0 → NaN
→ Action: STOP! Reduce learning rate
```

---

## Visualization

### Linguistic Graph Structure

**What You See**:
- **Nodes**: Words/concepts (size = frequency)
- **Edges**: Semantic relationships (thickness = strength)
- **Clusters**: Related concepts group together

**Controls**:
- **Hover**: Show node details
- **Drag**: Reposition nodes
- **Zoom**: Mouse wheel
- **Layout**: Choose from dropdown
  - Force: Physics-based
  - Circular: Ring layout
  - Hierarchical: Tree structure

**What to Look For**:
- Do similar words cluster? ✅
- Are hub nodes meaningful? ✅
- Any unexpected connections? 🔍

### Attention Heatmap

**Purpose**: See which words the model focuses on

**Reading the Heatmap**:
- **Bright colors**: Strong attention
- **Dark colors**: Weak attention
- **Diagonal**: Self-attention (always bright)
- **Vertical stripes**: Important words

**Example**:
```
Query: "Neural networks learn patterns"
→ "learn" attends strongly to "patterns" (semantic link)
→ "neural" attends to "networks" (phrase dependency)
```

### Hebbian Trace Monitor

**Purpose**: Watch synaptic weights evolve over time

**Patterns**:

1. **Learning (Potentiation)**:
   ```
   Weight: 0.1 → 0.3 → 0.5 ↑ (strengthening)
   ```

2. **Forgetting (Depression)**:
   ```
   Weight: 0.8 → 0.6 → 0.4 ↓ (weakening)
   ```

3. **Stable Memory (Engram)**:
   ```
   Weight: 0.7 → 0.7 → 0.7 → (persistent)
   ```

**Interpretation**: Stable high weights = core knowledge retained!

---

## Interactive Playground

### Query Interface

**How to Use**:

1. Enter natural language query
2. (Optional) Add context: `entity: neural_network`
3. Click **"🧠 Run Inference"**
4. Review response

**Example Queries**:
```
"What is a neural network?"
"Explain transformers"
"Is a cat an animal?"  # Tests System 2 logic
```

**Response Fields**:

- **Mode**: `system1_only`, `dual`, `system2_verification`
- **Confidence**: 0-100% certainty
- **Explanation**: Reasoning trace

### Ablation Controls

**Purpose**: Measure component contributions

**Switches**:
- ☑️ System 1 (Neural) - Pattern recognition
- ☑️ System 2 (Symbolic) - Logical reasoning
- ☑️ Spiking Network - Energy efficiency
- ☑️ Graph Attention - Long-range dependencies

**Experiment**:
```
Query: "Is a dog a mammal?"

All ON:
  Confidence: 95%, Mode: dual, Energy: 10mJ

System 2 OFF:
  Confidence: 70%, Mode: system1_only
  → Less certain without logic!

SNN OFF:
  Confidence: 95%, Mode: dual, Energy: 45mJ
  → 4.5x more energy!
```

**Insight**: Quantify each component's impact

---

## Hardware Monitoring

### GPU Status

**Key Metrics**:

- **Memory Used**: 
  - Safe: <9GB (75%)
  - Caution: 9-11GB
  - Critical: >11GB

- **Utilization**:
  - Training: 80-100% ✅
  - Inference: 50-80%
  - Idle: <10%

- **Temperature**:
  - Normal: <75°C
  - Warm: 75-85°C
  - Hot: >85°C (check cooling)

- **Power**: ~170W max for RTX 3060

### Memory Wall Management

**Status Indicators**:

| Status | Pressure | Action |
|--------|----------|--------|
| 🟢 Healthy | <70% | Continue |
| 🟡 Caution | 70-85% | Monitor |
| 🟠 Warning | 85-95% | Enable checkpointing |
| 🔴 Critical | >95% | Reduce batch size NOW |

**Recommendations** (shown in dashboard):
- Reduce batch size
- Enable gradient checkpointing
- Enable CPU offloading

### Energy Consumption

**Metrics**:
- **Total (mJ)**: Cumulative since start
- **Per Inference (mJ)**: Energy per query
  - Spiking: 5-10 mJ ✅
  - Traditional: 50-100 mJ
- **Savings (%)**: Usually 90-95%

**Real-World Context**:
```
10 mJ = Lift 1 gram by 1 meter
50,000 inferences at 10mJ = 0.14 Wh
≈ Running LED bulb for 1 second
```

---

# Part III: Advanced Topics

## Interpreting Metrics

### Continual Learning Metrics

**Average Performance (AP)**:
```
AP = (1/T) * Σ accuracy_on_task_i
```
- **Good**: >85%
- **Excellent**: >90%
- **Poor**: <70%

**Average Forgetting (AF)**:
```
AF = (1/(T-1)) * Σ (max_accuracy - final_accuracy)
```
- **Good**: <10%
- **Excellent**: <5%
- **Poor**: >20%

**Example**:
```
3 tasks learned:
Task 1: 90% → AP includes this
Task 2: 85% → AP includes this
Task 3: 88% → AP includes this
AP = (90+85+88)/3 = 87.7% ✅

Task 1 started at 95%, now 90%
Forgetting = 5% ✅ Minimal!
```

### Edge of Chaos

**What**: Balance metric for spiking network

**Scale**: 0 to 1
- **0.4-0.6**: Optimal (healthy dynamics)
- **<0.3**: Too ordered (rigid)
- **>0.7**: Too chaotic (unstable)

**Dashboard Display**: Shown in metrics panel

---

## Optimization Tips

### For RTX 3060 12GB

```yaml
# configs/ncgn_config.yaml
hardware:
  device: "cuda"
  gpu_memory_gb: 12
  use_mixed_precision: true

memory_optimization:
  batch_size: 32
  gradient_checkpointing: true
  offload_to_cpu: false
```

### Speed Up Training

1. **Mixed Precision**:
   ```yaml
   use_mixed_precision: true  # 2x faster
   ```

2. **Larger Batch Size** (if memory allows):
   ```yaml
   batch_size: 48  # Up from 32
   ```

3. **Reduce Logging**:
   ```yaml
   validation_interval: 1000  # Up from 500
   ```

4. **Enable cuDNN Benchmark**:
   ```python
   import torch
   torch.backends.cudnn.benchmark = True
   ```

### Reduce Memory Usage

1. **Gradient Checkpointing**:
   ```yaml
   gradient_checkpointing: true
   ```

2. **Smaller Batch**:
   ```yaml
   batch_size: 16
   ```

3. **CPU Offloading**:
   ```yaml
   offload_to_cpu: true
   ```

---

## Troubleshooting

### Training Issues

**Problem**: Training won't start

**Solutions**:
1. Check model loaded: Overview panel
2. Verify dataset uploaded
3. Check Activity Log for errors
4. Restart dashboard

---

**Problem**: Loss is NaN

**Solutions**:
```yaml
training:
  learning_rate: 0.00001  # Reduce 10x
  gradient_clip: 1.0      # Add clipping
```

---

**Problem**: Training very slow

**Solutions**:
1. Check GPU utilization (should be >80%)
2. Verify using CUDA: `device: "cuda"`
3. Enable mixed precision
4. Increase batch size

---

### Dashboard Issues

**Problem**: Dashboard won't start

**Solutions**:
```bash
# Check dependencies
pip install flask flask-socketio python-socketio eventlet

# Verify installation
python verify_installation.py

# Check port availability
netstat -ano | findstr :5000
```

---

**Problem**: Connection lost / No updates

**Solutions**:
1. Refresh browser (Ctrl+R)
2. Check terminal for errors
3. Restart dashboard
4. Clear browser cache

---

**Problem**: Visualization not loading

**Solutions**:
1. Click "🔄 Reload" button
2. Check if graph exists
3. Reduce graph size in config
4. Try different browser (Chrome recommended)

---

### Memory Issues

**Problem**: CUDA out of memory

**Immediate Actions**:
```yaml
memory_optimization:
  batch_size: 16          # Reduce
  gradient_checkpointing: true
  offload_to_cpu: true
```

**Restart Training** after config change

---

## Quick Reference

### Command Cheat Sheet

```bash
# Start dashboard
python ncgn_dashboard.py
launch_cockpit.bat

# Run demos
python scripts/ncgn_demo.py
python scripts/ncgn_demo.py --phase 1

# Verify installation
python verify_installation.py

# Check GPU
nvidia-smi
```

### Configuration Files

**Main Config**: `configs/ncgn_config.yaml`

**Quick edits**:
```yaml
# Memory
memory_optimization:
  batch_size: 32

# Speed
hardware:
  use_mixed_precision: true

# Device
hardware:
  device: "cuda"  # or "cpu"
```

### Python Quick Reference

```python
# Import core modules
from ncgn import (
    LinguisticGraph, GraphBuilder,
    SpikingLayer, PoissonEncoder,
    DualSystemArchitecture
)

# Build graph
builder = GraphBuilder("roberta-base")
graph = builder.build_from_corpus(corpus)

# Create model
model = DualSystemArchitecture(node_feat_dim=768, embed_dim=256)

# Add knowledge
model.add_knowledge([("cat", "is_a", "animal")])

# Inference
output = model(node_features, adjacency)
```

### Dashboard Panels Quick Guide

| Panel | Use For |
|-------|---------|
| **Overview** | Status check, metrics summary |
| **Training** | Upload data, configure, monitor |
| **Visualization** | Explore graph, attention, memory |
| **Playground** | Test queries, ablation studies |
| **Hardware** | Monitor GPU/CPU, optimize memory |

### Metric Interpretation

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| **AP** | >85% | 70-85% | <70% |
| **AF** | <10% | 10-20% | >20% |
| **GPU Memory** | <75% | 75-90% | >90% |
| **Temperature** | <75°C | 75-85°C | >85°C |
| **Edge of Chaos** | 0.4-0.6 | 0.3-0.7 | <0.3 or >0.7 |

### File Locations

```
configs/ncgn_config.yaml     # Main configuration
saved_models/ncgn/           # Model checkpoints
uploads/                     # Uploaded datasets
logs/                        # Training logs
```

---

## 📚 Additional Resources

- **Technical Details**: See `DEVELOPER_GUIDE.md`
- **Installation Help**: See `INSTALLATION_GUIDE.md`
- **Main README**: See `README.md`

---

**User Guide Version**: 1.0  
**Last Updated**: January 8, 2026  
**Compatible with**: NCGN v0.1.0+

---

*Master your neuromorphic AI system!* 🧠✨

