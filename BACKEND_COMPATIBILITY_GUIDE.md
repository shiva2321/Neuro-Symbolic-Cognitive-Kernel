# Graph Backend Compatibility Guide

## Overview

The Node_network system now supports **multiple graph backends** for maximum flexibility and compatibility:

- **PyTorch Geometric (PyG)** - Recommended for new installations
- **DGL (Deep Graph Library)** - Legacy support, fully backward compatible

## Why Multiple Backends?

1. **Flexibility**: Choose the backend that best fits your needs
2. **Future-proofing**: Not locked into a single library
3. **Compatibility**: Works with existing DGL code seamlessly
4. **Performance**: PyG offers native PyTorch integration

## Quick Start

### Check Available Backends

```bash
python utils/check_backend.py
```

This will show:
- Which backends are installed
- Which backend is the default
- Test results for each backend

### Using PyTorch Geometric (Recommended)

**Install PyG:**
```bash
# For CUDA 11.8
pip install torch-geometric

# Or specify CUDA version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**Use in code:**
```python
from ncgn.linguistic_graph import LinguisticGraph

# Automatically uses PyG if available
graph = LinguisticGraph(device='cuda', backend='pyg')

# Or let it auto-detect (prefers PyG)
graph = LinguisticGraph(device='cuda')
```

### Using DGL (Legacy)

**Keep using DGL:**
```python
from ncgn.linguistic_graph import LinguisticGraph

# Explicitly use DGL
graph = LinguisticGraph(device='cuda', backend='dgl')
```

**Existing code works unchanged:**
All existing DGL code continues to work without modifications!

## Backend Comparison

| Feature | PyTorch Geometric | DGL |
|---------|------------------|-----|
| PyTorch Integration | ✓ Native | ✓ Good |
| Performance | ✓ Excellent | ✓ Excellent |
| Memory Efficiency | ✓ Better | ✓ Good |
| Layer Library | ✓ Extensive | ✓ Good |
| Community | ✓ Very Active | ✓ Active |
| Documentation | ✓ Excellent | ✓ Excellent |
| CUDA Support | ✓ Yes | ✓ Yes |
| Message Passing | ✓ Native | ✓ Yes |
| **Recommendation** | **Preferred** | Legacy |

## Migration Guide

### For New Projects

Simply use the default backend (PyG if available):

```python
from ncgn.linguistic_graph import LinguisticGraph, GraphBuilder

# Build a graph
builder = GraphBuilder(embedding_model="roberta-base")
graph = builder.build_from_corpus(corpus)

# The backend is automatically selected
print(graph.get_statistics())  # Shows which backend is used
```

### For Existing Projects

**No changes needed!** Your existing code continues to work:

```python
# This still works exactly as before
from ncgn.linguistic_graph import LinguisticGraph

graph = LinguisticGraph(device='cuda')
# Will use DGL if that's what you have installed
```

### Switching Backends

To switch from DGL to PyG:

1. Install PyG: `pip install torch-geometric`
2. Run tests: `python utils/check_backend.py`
3. Update code (optional):
   ```python
   # Old (still works)
   graph = LinguisticGraph(device='cuda')
   
   # New (explicit)
   graph = LinguisticGraph(device='cuda', backend='pyg')
   ```
4. Re-save models for optimal performance:
   ```python
   graph.save('path/to/model')  # Saves in new format
   ```

## API Reference

### GraphBackend Class

The compatibility layer that wraps both DGL and PyG:

```python
from utils.graph_backend import GraphBackend

# Create backend
backend = GraphBackend(backend='pyg')  # or 'dgl' or None for auto

# Create graph
graph = backend.create_graph(src_nodes, dst_nodes, num_nodes)

# Node features
backend.set_node_features('feat', features)
features = backend.get_node_features('feat')

# Edge features
backend.set_edge_features('weight', weights)
weights = backend.get_edge_features('weight')

# Graph operations
num_nodes = backend.num_nodes()
num_edges = backend.num_edges()
src, dst = backend.edges()

# Device movement
backend.to('cuda')

# NetworkX conversion
nx_graph = backend.to_networkx()

# Save/Load
backend.save('graph.bin')
backend = GraphBackend.load('graph.bin')
```

### LinguisticGraph Updates

New optional `backend` parameter:

```python
from ncgn.linguistic_graph import LinguisticGraph

# Auto-detect backend
graph = LinguisticGraph(device='cuda')

# Force specific backend
graph = LinguisticGraph(device='cuda', backend='pyg')
graph = LinguisticGraph(device='cuda', backend='dgl')

# Load with specific backend
graph = LinguisticGraph.load('path/to/graph', backend='pyg')
```

## Testing

### Run Full Backend Tests

```bash
python utils/check_backend.py
```

### Test Specific Backend

```python
from utils.check_backend import test_backend

test_backend('pyg')  # Test PyG
test_backend('dgl')  # Test DGL
```

### Test LinguisticGraph

```python
from utils.check_backend import test_linguistic_graph

test_linguistic_graph()  # Tests with all available backends
```

## Troubleshooting

### "No graph backend available"

Install at least one backend:
```bash
# Recommended
pip install torch-geometric

# Or legacy
pip install dgl
```

### "Backend requested but not available"

If you explicitly request a backend that isn't installed:
```python
graph = LinguisticGraph(backend='pyg')  # Error if PyG not installed
```

Solution: Either install the backend or use `backend=None` for auto-detection.

### CUDA Version Mismatch

PyG requires matching CUDA versions:
```bash
# Check your PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching PyG
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Legacy Graph Files

Old DGL graph files are automatically converted when loaded:
```python
# Old file with DGL format
graph = LinguisticGraph.load('old_model', backend='pyg')
# Automatically converts to PyG

# Re-save for better performance
graph.save('old_model')  # Now in new format
```

## Performance Considerations

### Memory Usage

PyG generally uses less memory due to:
- Sparse edge storage (COO format)
- No separate edge list structure
- Efficient tensor operations

### Training Speed

Both backends offer similar training speeds:
- DGL: Optimized for graph operations
- PyG: Optimized for PyTorch integration

### Recommendation

- **New projects**: Use PyG
- **Existing projects**: Keep DGL or migrate gradually
- **Research**: Try both and benchmark

## Advanced Usage

### Custom Backend Selection

Set environment variable for global default:
```bash
export GRAPH_BACKEND=pyg
```

Or in code:
```python
import os
os.environ['GRAPH_BACKEND'] = 'pyg'
```

### Mixed Backend Usage

You can use both backends in the same project:
```python
# Use PyG for training
train_graph = LinguisticGraph(backend='pyg')

# Use DGL for inference (if you have legacy models)
inference_graph = LinguisticGraph.load('legacy_model.bin', backend='dgl')
```

### Backend-Specific Optimizations

Access the underlying graph for backend-specific features:
```python
from ncgn.linguistic_graph import LinguisticGraph

graph = LinguisticGraph(backend='pyg')
graph.build_dgl_graph()

# Access underlying PyG graph
if graph._backend is not None:
    pyg_graph = graph._backend.graph
    # Now use PyG-specific features
```

## FAQ

**Q: Should I switch from DGL to PyG?**
A: For new projects, yes. For existing projects, only if you need PyG-specific features.

**Q: Will DGL support be removed?**
A: No, DGL support is maintained for backward compatibility.

**Q: Does this affect performance?**
A: No, the compatibility layer has minimal overhead. Both backends perform similarly.

**Q: Can I contribute to the backend layer?**
A: Yes! See `utils/graph_backend.py` and submit PRs.

**Q: What about other backends (igraph, NetworkX)?**
A: NetworkX is used for utility functions. Other backends can be added to the compatibility layer.

## Support

- Check backend status: `python utils/check_backend.py`
- Report issues: GitHub Issues
- See also: `INSTALLATION_GUIDE.md`, `DEVELOPER_GUIDE.md`

---

**Version**: 0.2.0  
**Last Updated**: January 8, 2026  
**Status**: Stable

