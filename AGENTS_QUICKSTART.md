# NCGN Specialized Agents - Quick Start Guide

## 🚀 Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**New dependencies added**:
- `ogb>=1.3.6` - Open Graph Benchmark
- `requests>=2.31.0` - Dataset downloads
- `seaborn>=0.13.0` - Enhanced visualizations

### Step 2: Verify Installation
```bash
python setup_agents.py
```

This will:
- ✅ Check Python version (3.8+ required)
- ✅ Verify all dependencies
- ✅ Test agent imports
- ✅ Create necessary directories
- ✅ Run basic functionality tests

### Step 3: Run Demo
```bash
python agents_demo.py
```

Select:
- **Option 1**: Quick demo (data harvesting only, ~1 minute)
- **Option 2**: Full pipeline (all 5 agents, ~5 minutes)

---

## 📚 Agent Overview

### Agent 1: Data Harvester 🤖
**What it does**: Automatically finds and downloads graph datasets based on natural language commands

**Example**:
```python
from agents.data_harvester import DataHarvester

harvester = DataHarvester()
result = harvester.harvest("Train for English language")
graph = result['graph']
```

**Commands you can use**:
- "Train for English language" → Linguistic datasets
- "Train for protein networks" → Biological datasets  
- "Train for citation networks" → Academic datasets
- "Train for social networks" → Social media graphs

---

### Agent 2: Topological Converter 🔄
**What it does**: Transforms graphs into enhanced representations with positional encodings

**Example**:
```python
from agents.topological_converter import TopologicalConverter

converter = TopologicalConverter()
result = converter.convert(graph)
mcteg = result['mcteg']  # Enhanced graph
graph_words = result['graph_word_embeddings']  # Learned tokens
```

**What you get**:
- MC-TEG graph with context features
- Laplacian positional encodings (spectral coordinates)
- Random Walk positional encodings
- Graph Word embeddings (like Word2Vec for graphs)

---

### Agent 3: Bottleneck Optimizer ⚡
**What it does**: Fixes structural problems in graphs (oversquashing, oversmoothing)

**Example**:
```python
from agents.bottleneck_optimizer import BottleneckOptimizer

optimizer = BottleneckOptimizer()
result = optimizer.optimize(graph)
optimized_graph = result['optimized_graph']
```

**What it fixes**:
- **Oversquashing**: Information getting squeezed in narrow pathways
- **Oversmoothing**: Nodes becoming too similar in dense regions
- **Long-range connections**: Adds virtual node for global communication

---

### Agent 4: Analytic Learner 🧠
**What it does**: Trains models WITHOUT backpropagation, NO catastrophic forgetting

**Example**:
```python
from agents.analytic_learner import AnalyticLearner

learner = AnalyticLearner()

# Train on Task 1
learner.train_step(graph1, features1, labels1, "task1")

# Train on Task 2 - still remembers Task 1!
learner.continual_learn(graph2, features2, labels2, "task1")
```

**Why it's special**:
- One-pass learning (see data once)
- No forgetting (keeps old knowledge)
- Memory efficient (12GB VRAM friendly)
- Biologically plausible (like real brains)

---

### Agent 5: Analytics Suite 📊
**What it does**: Evaluates performance and creates visualizations

**Example**:
```python
from agents.analytics_suite import AnalyticsSuite

analytics = AnalyticsSuite()

# Evaluate multiple tasks
for task_id, (graph, features, labels) in enumerate(tasks):
    analytics.evaluate_task(model, graph, features, labels, task_id, f"task_{task_id}")

# Generate comprehensive report
report = analytics.generate_cglb_report()
print(f"Average Performance: {report.average_performance:.2%}")
print(f"Average Forgetting: {report.average_forgetting:.2%}")
```

**What you get**:
- Performance metrics (AP, AF, FT, BT)
- Learning curves
- Hebbian trace visualizations (which connections strengthened)
- Privacy analysis (Membership Inference Attacks)

---

## 🎯 Common Use Cases

### Use Case 1: Train on New Language Dataset
```python
from agents import DataHarvester, TopologicalConverter, AnalyticLearner

# 1. Get data
harvester = DataHarvester()
data = harvester.harvest("Train for English language")

# 2. Enhance graph
converter = TopologicalConverter()
enhanced = converter.convert(data['graph'])

# 3. Train
learner = AnalyticLearner()
learner.train_step(enhanced['mcteg'], features, labels)
```

### Use Case 2: Continual Learning (No Forgetting)
```python
learner = AnalyticLearner()

# Learn Task 1
learner.train_step(graph1, features1, labels1, "english")

# Learn Task 2 - no forgetting!
learner.continual_learn(graph2, features2, labels2, "english")

# Learn Task 3 - still remembers Task 1 and 2!
learner.continual_learn(graph3, features3, labels3, "english")
```

### Use Case 3: Fix Graph Bottlenecks
```python
from agents import BottleneckOptimizer

optimizer = BottleneckOptimizer()

# Analyze problems
stats = optimizer.get_bottleneck_statistics(graph)
print(f"Found {stats['num_bottleneck_edges']} bottlenecks")

# Fix them
result = optimizer.optimize(graph)
print(f"Added {result['edges_added_total']} shortcuts")
```

### Use Case 4: Complete Pipeline
```python
from agents import *

# Full pipeline
harvester = DataHarvester()
converter = TopologicalConverter()
optimizer = BottleneckOptimizer()
learner = AnalyticLearner()
analytics = AnalyticsSuite()

# 1. Get data
data = harvester.harvest("Train for English language")

# 2. Convert
enhanced = converter.convert(data['graph'])

# 3. Optimize
optimized = optimizer.optimize(enhanced['mcteg'])

# 4. Train
learner.train_step(optimized['optimized_graph'], features, labels)

# 5. Evaluate
report = analytics.generate_cglb_report()
```

---

## 🎓 Understanding the Research

### What is MC-TEG?
**Multilevel Context Textual-Edge Graph** - A graph where:
- **Nodes** have descriptions (e.g., "This is a sentence node")
- **Edges** have types (e.g., "contains", "cites", "related to")
- **Positions** are encoded (nodes have coordinates in semantic space)

Like a map where each location has a description and roads have labels.

### What is Oversquashing?
Imagine a highway with a narrow bottleneck:
- Many cars (information) → Bottleneck → Information loss
- **Solution**: Add more lanes (edges) around the bottleneck

### What is Ricci Curvature?
- **Negative curvature**: Bottleneck (information gets squeezed)
- **Positive curvature**: Dense region (redundant connections)
- **Zero curvature**: Balanced

Like measuring if a pipe is too narrow or too wide.

### What is RLS (Recursive Least Squares)?
Traditional learning: See all data many times (epochs)  
RLS learning: See each data point ONCE, update immediately

Like learning by doing (practice) vs. studying (memorization).

---

## 📊 Expected Results

### Data Harvester
- **Time**: 30 seconds - 5 minutes
- **Output**: DGL graph with 100K - 10M nodes
- **Memory**: 2-8 GB RAM

### Topological Converter
- **Time**: 2-5 minutes for 100K nodes
- **Output**: Enhanced graph with positional encodings
- **Memory**: 4-8 GB RAM

### Bottleneck Optimizer
- **Time**: 1-3 minutes for 100K nodes
- **Output**: Optimized graph with +/- edges
- **Memory**: 3-6 GB RAM

### Analytic Learner
- **Time**: 10-60 seconds for 100K samples
- **Accuracy**: 70-90% (depends on task)
- **Memory**: 1-3 GB RAM

### Analytics Suite
- **Time**: 1-2 minutes
- **Output**: JSON reports, PNG visualizations
- **Memory**: 2-4 GB RAM

---

## 🐛 Common Issues

### Issue 1: Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
config.batch_size = 1000  # instead of 10000

# Enable memory efficient mode
learner = AnalyticLearner(AnalyticLearningConfig(
    memory_efficient=True
))

# Use CPU if necessary
learner = AnalyticLearner(device='cpu')
```

### Issue 2: Slow Performance
**Problem**: Takes too long to process

**Solutions**:
```python
# Reduce graph size
harvester_config.max_nodes = 50000  # instead of 1000000

# Reduce iterations
rewiring_config.rewiring_iterations = 2  # instead of 5

# Sample nodes
graph_sample = dgl.node_subgraph(graph, torch.randperm(graph.num_nodes())[:10000])
```

### Issue 3: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'ogb'`

**Solution**:
```bash
pip install ogb requests seaborn
```

### Issue 4: No Datasets Found
**Problem**: Harvester can't find datasets

**Solution**:
```python
# Check catalog
harvester = DataHarvester()
datasets = harvester.list_available_datasets()
for d in datasets:
    print(d.name, d.domain)

# Try exact dataset name
result = harvester.harvest("ogbn-arxiv")
```

---

## 📁 File Structure

```
Node_network/
├── agents/                          # 🆕 New specialized agents
│   ├── __init__.py
│   ├── data_harvester.py           # Agent 1
│   ├── topological_converter.py    # Agent 2
│   ├── bottleneck_optimizer.py     # Agent 3
│   ├── analytic_learner.py         # Agent 4
│   └── analytics_suite.py          # Agent 5
│
├── agents_demo.py                   # 🆕 Demo script
├── setup_agents.py                  # 🆕 Setup/verification
├── AGENTS_README.md                 # 🆕 Quick start
├── AGENTS_DOCUMENTATION.md          # 🆕 Full docs
├── AGENTS_IMPLEMENTATION_SUMMARY.md # 🆕 Summary
│
├── ncgn/                            # Existing NCGN components
│   ├── linguistic_graph.py
│   ├── graph_transformer.py
│   └── ...
│
└── requirements.txt                 # 🔄 Updated with new deps
```

---

## 🎯 Next Steps

### 1. Try the Quick Demo (1 minute)
```bash
python agents_demo.py  # Select option 1
```

### 2. Read the Full Documentation
```bash
# Open in your editor
AGENTS_DOCUMENTATION.md
```

### 3. Experiment with Individual Agents
```python
# Try each agent separately
from agents import DataHarvester
harvester = DataHarvester()
print(harvester.list_available_datasets())
```

### 4. Integrate with Your Workflow
```python
# Use with existing NCGN components
from ncgn.dual_system import DualSystem
from agents import DataHarvester, AnalyticLearner

# Your custom pipeline here
```

---

## 💡 Tips & Tricks

### Tip 1: Start Small
```python
# Test with small graphs first
config = DataHarvesterConfig(max_nodes=10000)
harvester = DataHarvester(config)
```

### Tip 2: Save Intermediate Results
```python
# Save processed graphs
import pickle
with open('processed_graph.pkl', 'wb') as f:
    pickle.dump(optimized_graph, f)
```

### Tip 3: Monitor Memory
```python
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```

### Tip 4: Use Configuration Files
```python
import yaml

# Save config
config_dict = {
    'max_nodes': 100000,
    'use_semantic_filter': True,
}
with open('harvester_config.yaml', 'w') as f:
    yaml.dump(config_dict, f)

# Load config
with open('harvester_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
config = DataHarvesterConfig(**config_dict)
```

---

## 📞 Getting Help

1. **Check Documentation**:
   - `AGENTS_README.md` - This guide
   - `AGENTS_DOCUMENTATION.md` - Detailed reference

2. **Run Examples**:
   - `agents_demo.py` - Working examples

3. **Check Source Code**:
   - Each agent has detailed docstrings
   - Look at the method implementations

4. **Verify Installation**:
   ```bash
   python setup_agents.py
   ```

---

## 🎉 You're Ready!

All agents are installed and ready to use. Start with:

```bash
python agents_demo.py
```

**Happy training! 🚀**

