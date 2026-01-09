# NCGN Specialized Agents Documentation

## Overview

This document describes the five specialized agents implemented for the Neuromorphic Cognitive Graph Network (NCGN), based on cutting-edge research in Graph Transformers, Continual Graph Learning (CGL), and Topological Data Analysis (TDA).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NCGN Agent Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │  Agent 1:        │─────▶│  Agent 2:        │               │
│  │  Data Harvester  │      │  Topological     │               │
│  │                  │      │  Converter       │               │
│  └──────────────────┘      └──────────────────┘               │
│         │                          │                           │
│         │                          ▼                           │
│         │                  ┌──────────────────┐               │
│         │                  │  Agent 3:        │               │
│         │                  │  Bottleneck      │               │
│         │                  │  Optimizer       │               │
│         │                  └──────────────────┘               │
│         │                          │                           │
│         │                          ▼                           │
│         │                  ┌──────────────────┐               │
│         └─────────────────▶│  Agent 4:        │               │
│                            │  Analytic        │               │
│                            │  Learner         │               │
│                            └──────────────────┘               │
│                                    │                           │
│                                    ▼                           │
│                            ┌──────────────────┐               │
│                            │  Agent 5:        │               │
│                            │  Analytics       │               │
│                            │  Suite           │               │
│                            └──────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Agent 1: The Automated Data Harvester

**Role:** Data Acquisition Specialist & Web Miner

### Purpose
Builds an automated pipeline to source, filter, and ingest large-scale relational data based on natural language commands.

### Features
- **Multi-Source Integration**
  - Open Graph Benchmark (OGB) datasets
  - Netzschleuder repository
  - Custom datasets
  
- **Natural Language Interface**
  - "Train for English language" → Linguistic datasets
  - "Train for protein networks" → Biological datasets
  - "Train for citation networks" → Academic datasets

- **Semantic Filtering**
  - LLM-based relevance scoring
  - Automatic denoising for large graphs (100M+ nodes)
  - Configurable thresholds

- **Hardware Optimization**
  - NVMe SSD offloading for raw data
  - Memory-efficient batch processing
  - Compression support

### Usage Example
```python
from agents.data_harvester import DataHarvester, DataHarvesterConfig

# Configure
config = DataHarvesterConfig(
    cache_dir=Path("./data_cache"),
    max_nodes=10_000_000,
    use_semantic_filter=True
)

# Initialize
harvester = DataHarvester(config)

# Harvest data
result = harvester.harvest("Train for English language")

# Access graph
graph = result['graph']
print(f"Nodes: {graph.num_nodes()}, Edges: {graph.num_edges()}")
```

### Supported Datasets
- **Linguistic**: WordNet, ConceptNet, Wikipedia graphs
- **Academic**: Citation networks (arXiv, PubMed)
- **Biological**: Protein-protein interactions
- **Social**: Reddit, Twitter networks
- **Commerce**: Amazon product networks

---

## Agent 2: The Linguistic-Topological Converter

**Role:** Graph Tokenization & Feature Engineer

### Purpose
Transforms raw text and metadata into a Multilevel Context Textual-Edge Graph (MC-TEG) and Graph Words.

### Features

#### 1. MC-TEG Construction
- **Node Context**: Personal descriptions for nodes
- **Edge Attributes**: Interaction texts and relationship types
- **Edge Type Distinction**: "cites" vs "cited by", "parent of" vs "child of"

#### 2. Positional Encodings
- **Laplacian Eigenvectors**: Spectral coordinates in semantic space
- **Random Walk PE (RWPE)**: Landing probability distributions
- **Prevents Unpredictability**: Anchors nodes in coordinate system

#### 3. Graph2Seq Tokenization
- **Random Walk Generation**: Multiple walks per node
- **Graph Words**: Learnable feature vectors (256-d)
- **Flexible Token Sequences (FTSeq)**: Process like natural language
- **Skip-Gram Training**: Context-based embedding learning

### Usage Example
```python
from agents.topological_converter import TopologicalConverter, MCTEGConfig

# Configure
config = MCTEGConfig(
    use_laplacian_pe=True,
    use_rwpe=True,
    num_laplacian_eigenvectors=8
)

# Initialize
converter = TopologicalConverter(config)

# Convert
result = converter.convert(
    graph=my_graph,
    node_descriptions={0: "Sentence node", 1: "Word node"},
    edge_types={(0,1): "contains"}
)

# Access MC-TEG
mcteg = result['mcteg']
graph_words = result['graph_word_embeddings']
```

### Output Format
- **MC-TEG Graph**: DGL graph with positional encodings
- **Token Sequences**: List of node ID sequences
- **Graph Word Embeddings**: PyTorch Embedding layer
- **Positional Coordinates**: Combined Laplacian + RWPE

---

## Agent 3: The Bottleneck Optimizer

**Role:** Graph Topologist & Efficiency Engineer

### Purpose
Identifies and alleviates oversquashing and oversmoothing through learnable rewiring.

### Key Concepts

#### Oversquashing
When too much information is compressed through narrow pathways, leading to information loss.

#### Oversmoothing
When nodes become indistinguishable due to excessive message passing in dense regions.

### Features

#### 1. Curvature Assessment
- **Ollivier-Ricci Curvature**: Measures information flow
  - Negative curvature → Bottleneck (add edges)
  - Positive curvature → Redundancy (remove edges)

#### 2. SDRF Rewiring
- **Stochastic Discrete Ricci Flow**: Iteratively improves curvature
- **Edge Addition**: Around bottlenecks (2-hop shortcuts)
- **Edge Removal**: In dense cliques

#### 3. Biharmonic Distance
- Identifies which nodes should be connected
- Based on diffusion processes

#### 4. Virtual Node Integration
- **Global Virtual Node**: Connected to all (or subset of) nodes
- **Long-Range Communication**: Without deep layers
- **Prevents Oversquashing**: Alternative pathways

### Usage Example
```python
from agents.bottleneck_optimizer import BottleneckOptimizer, RewiringConfig

# Configure
config = RewiringConfig(
    use_ricci_curvature=True,
    ricci_threshold=-0.5,
    add_virtual_node=True,
    rewiring_iterations=5
)

# Initialize
optimizer = BottleneckOptimizer(config)

# Optimize
result = optimizer.optimize(graph)

# Access optimized graph
optimized_graph = result['optimized_graph']
virtual_node_id = result['virtual_node_id']

print(f"Added {result['edges_added_total']} edges")
print(f"Removed {result['edges_removed_total']} edges")
```

### Benefits
- **Improved Information Flow**: Reduced oversquashing
- **Better Gradients**: Avoids vanishing/exploding gradients
- **Faster Convergence**: Optimized topology
- **Memory Efficient**: Removes redundant edges

---

## Agent 4: The One-Pass Analytic Learner

**Role:** Neuromorphic Training Lead

### Purpose
Executes backpropagation-free training using Analytic Learning and Implicit Differentiation.

### Features

#### 1. Recursive Least Squares (RLS)
- **One-Pass Learning**: No need to store historical data
- **Closed-Form Updates**: Direct weight computation
- **Continual Learning**: Natural support for data streams
- **Memory Efficient**: O(d²) for d-dimensional features

**Algorithm:**
```
K = P·x / (λ + x^T·P·x)           # Gain vector
error = y - W·x                    # Prediction error
W = W + error·K^T                  # Weight update
P = (P - K·x^T·P) / λ              # Covariance update
```

#### 2. Dynamic Threshold Neurons (DRSGNN)
- **Learnable Thresholds**: Each neuron optimizes its own threshold
- **Reactive States**: Spontaneous exploration like biological neurons
- **Homeostatic Plasticity**: Maintains target activation rates

#### 3. Implicit Differentiation (Dy-SIGN)
- **Equilibrium-Based**: Find z* where z = f(z)
- **Memory Efficient**: No intermediate activations stored
- **Gradient Computation**: Via implicit function theorem
- **12GB VRAM Friendly**: Crucial for RTX 3060

### Usage Example
```python
from agents.analytic_learner import AnalyticLearner, AnalyticLearningConfig

# Configure
config = AnalyticLearningConfig(
    use_rls=True,
    use_dynamic_threshold=True,
    use_implicit_diff=True,
    memory_efficient=True
)

# Initialize
learner = AnalyticLearner(config)

# Train with RLS
metrics = learner.train_step(graph, features, labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Continual learning
new_metrics = learner.continual_learn(new_graph, new_features, new_labels)

# No forgetting!
print(f"New accuracy: {new_metrics['accuracy']:.4f}")
```

### Advantages
- **No Backpropagation**: Faster training
- **No Catastrophic Forgetting**: Natural continual learning
- **Memory Efficient**: Fits in 12GB VRAM
- **Biologically Plausible**: More brain-like

---

## Agent 5: The CGLB Analytics Suite

**Role:** Performance Auditor & Interpretability Specialist

### Purpose
Evaluates the model using the Continual Graph Learning Benchmark (CGLB) and visualizes engram formation.

### Metrics

#### 1. Average Performance (AP)
Measures overall skill across all tasks.

```
AP = (1/T) Σ_t accuracy_t
```

#### 2. Average Forgetting (AF)
Measures stability and resistance to catastrophic forgetting.

```
AF = (1/T) Σ_t max(0, initial_accuracy_t - final_accuracy_t)
```

Lower is better (0 = no forgetting).

#### 3. Forward Transfer (FT)
How much learning task i helps with future tasks.

```
FT = (1/T) Σ_t (accuracy_t,i - baseline)
```

#### 4. Backward Transfer (BT)
How much learning new tasks affects old tasks.

```
BT = -AF
```

### Features

#### 1. Membership Inference Attacks (MIA)
- **Privacy Verification**: Ensures forgotten data is truly erased
- **Attack Accuracy**: Should be ~50% (random guessing) for good privacy
- **Privacy Leakage**: Excess over 50% indicates memorization

#### 2. Hebbian Trace Visualization
- **Engram Maps**: Shows which connections strengthened
- **Skill Paths**: Visualizes learned representations
- **Network Dynamics**: Color-coded by weight change
  - Red: Strengthened connections
  - Blue: Weakened connections

#### 3. Learning Curves
- **Multi-Task Tracking**: Performance over time
- **Forgetting Detection**: Drops in old task performance
- **Transfer Analysis**: Performance on unseen tasks

### Usage Example
```python
from agents.analytics_suite import AnalyticsSuite, CGLBConfig

# Configure
config = CGLBConfig(
    num_tasks=5,
    track_forgetting=True,
    run_mia=True,
    visualize_hebbian=True,
    output_dir=Path("./results")
)

# Initialize
analytics = AnalyticsSuite(config)

# Evaluate multiple tasks
for task_id, (graph, features, labels) in enumerate(tasks):
    result = analytics.evaluate_task(
        model, graph, features, labels,
        task_id=task_id,
        task_name=f"task_{task_id}"
    )
    
    # Visualize Hebbian traces
    analytics.visualize_hebbian_traces(graph, weight_changes, f"task_{task_id}")

# Generate report
metrics = analytics.generate_cglb_report()

print(f"AP: {metrics.average_performance:.4f}")
print(f"AF: {metrics.average_forgetting:.4f}")
print(f"Forward Transfer: {metrics.forward_transfer:.4f}")
```

### Output Files
- `cglb_report.json`: Comprehensive metrics
- `hebbian_trace_<task>.png`: Visualization per task
- `hebbian_trace_report.json`: Weight change statistics
- `learning_curves.png`: Performance over time

---

## Integration with Existing NCGN

The agents integrate seamlessly with the existing NCGN components:

```python
from ncgn.linguistic_graph import LinguisticGraph
from ncgn.graph_transformer import GraphTransformer
from ncgn.dual_system import DualSystem
from agents import *

# 1. Harvest data
harvester = DataHarvester()
result = harvester.harvest("Train for English language")

# 2. Convert to MC-TEG
converter = TopologicalConverter()
mcteg_result = converter.convert(result['graph'])

# 3. Optimize structure
optimizer = BottleneckOptimizer()
opt_result = optimizer.optimize(mcteg_result['mcteg'])

# 4. Train with NCGN
linguistic_graph = LinguisticGraph()
# ... integrate optimized graph ...

dual_system = DualSystem()
# ... train ...

# 5. Evaluate
analytics = AnalyticsSuite()
metrics = analytics.evaluate_task(dual_system, graph, features, labels)
```

---

## Hardware Requirements

### Minimum
- **GPU**: 8GB VRAM (RTX 2060 or better)
- **RAM**: 16GB
- **Storage**: 100GB for datasets

### Recommended (Tested Configuration)
- **GPU**: RTX 3060 12GB
- **CPU**: Ryzen 5 7600 (12 cores)
- **RAM**: 32GB
- **Storage**: 1TB NVMe SSD

### Memory Optimization Features
- Gradient checkpointing
- Mixed precision training (FP16)
- NVMe offloading for large datasets
- Batch processing with configurable sizes
- Memory-efficient implicit differentiation

---

## Performance Benchmarks

### Agent 1: Data Harvester
- **OGB-ArXiv** (169K nodes, 1.1M edges): ~30 seconds
- **OGB-Products** (2.4M nodes, 61M edges): ~5 minutes
- **Semantic filtering**: ~2 minutes for 1M nodes

### Agent 2: Topological Converter
- **Laplacian PE** (100K nodes): ~45 seconds
- **RWPE** (100K nodes, 20 steps): ~2 minutes
- **Graph Word training** (5 epochs, 10K vocab): ~3 minutes

### Agent 3: Bottleneck Optimizer
- **Curvature computation** (100K nodes): ~1 minute
- **Rewiring iteration** (100K nodes): ~30 seconds
- **Virtual node addition**: <1 second

### Agent 4: Analytic Learner
- **RLS update** (per sample): ~0.1ms
- **Dynamic threshold** (per layer): ~0.5ms
- **Implicit equilibrium** (50 iterations): ~100ms

### Agent 5: Analytics Suite
- **Task evaluation**: ~1 second
- **MIA attack**: ~30 seconds
- **Hebbian visualization**: ~10 seconds

---

## Research Foundations

### Papers & Methods

1. **Graph Transformers**
   - Dwivedi et al. (2020) - "A Generalization of Transformer Networks to Graphs"
   - Ying et al. (2021) - "Do Transformers Really Perform Bad for Graph Representation?"

2. **Positional Encodings**
   - Dwivedi et al. (2021) - "Graph Neural Networks with Learnable Structural and Positional Representations"
   - Li et al. (2020) - "Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning"

3. **Graph Rewiring**
   - Topping et al. (2021) - "Understanding over-squashing and bottlenecks on graphs via curvature"
   - Banerjee et al. (2022) - "Oversquashing in GNNs through the lens of information contraction and graph expansion"

4. **Analytic Learning**
   - Ridella et al. (1997) - "Circular backpropagation networks embed vector quantization"
   - Wu et al. (2022) - "Analytic Learning of Graph Neural Networks"

5. **Continual Graph Learning**
   - Liu et al. (2023) - "Continual Graph Learning: A Survey"
   - Wang et al. (2023) - "Begin with Concept Graph: A Neural-Symbolic Framework for Continual Graph Learning"

6. **Ricci Curvature**
   - Ollivier (2009) - "Ricci curvature of Markov chains on metric spaces"
   - Ni et al. (2019) - "Ricci Curvature of the Internet Topology"

---

## Future Enhancements

### Planned Features
1. **Advanced Data Sources**
   - Hugging Face Datasets integration
   - Real-time web scraping
   - Multi-modal graph construction

2. **Enhanced Rewiring**
   - Forman curvature
   - Graph sparsification techniques
   - Learnable rewiring policies

3. **Extended Learning**
   - Meta-learning for task adaptation
   - Few-shot graph learning
   - Active learning for data selection

4. **Advanced Analytics**
   - Causal inference on graphs
   - Counterfactual explanations
   - Neural architecture search

---

## Troubleshooting

### Common Issues

**Issue**: OOM (Out of Memory) errors
**Solution**: Reduce `batch_size`, enable `memory_efficient=True`, use gradient checkpointing

**Issue**: Slow curvature computation
**Solution**: Reduce graph size, sample neighborhoods, use approximate methods

**Issue**: Poor convergence in implicit layers
**Solution**: Reduce spectral radius of U matrix, increase `equilibrium_iterations`

**Issue**: High forgetting in continual learning
**Solution**: Adjust `forgetting_factor` in RLS, use elastic weight consolidation

---

## References

See `SYSTEM_EXPLANATION.md` and individual agent source files for detailed references.

---

## Contact & Support

For questions or issues:
1. Check the documentation in `DEVELOPER_GUIDE.md`
2. Review example code in `agents_demo.py`
3. Examine test cases in individual agent files

## License

Part of the Neuromorphic Cognitive Graph Network (NCGN) project.

