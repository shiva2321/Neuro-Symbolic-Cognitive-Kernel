# Quick Reference: Graph Backends

## TL;DR

**PyTorch Geometric (PyG)** is now the recommended alternative to DGL. Your existing code works unchanged.

## Install

```bash
# Recommended: PyTorch Geometric
pip install torch-geometric

# OR Legacy: DGL
pip install dgl
```

## Check Status

```bash
python utils/check_backend.py
```

## Usage

### Auto (Recommended)
```python
from ncgn.linguistic_graph import LinguisticGraph

# Uses PyG if available, DGL otherwise
graph = LinguisticGraph(device='cuda')
```

### Explicit Backend
```python
# Force PyG
graph = LinguisticGraph(device='cuda', backend='pyg')

# Force DGL
graph = LinguisticGraph(device='cuda', backend='dgl')
```

## Test

```bash
python test_backend_compatibility.py
```

## Choose Backend

| Feature | PyG | DGL |
|---------|-----|-----|
| Native PyTorch | ✓ | |
| Active Development | ✓ | |
| Memory Efficient | ✓ | |
| **Recommended** | **Yes** | Legacy |

## Migration

### New Projects
✓ Use PyG (default)

### Existing Projects
✓ Keep DGL (works as-is)  
✓ Or switch to PyG (optional)

## Help

- Full guide: `BACKEND_COMPATIBILITY_GUIDE.md`
- Installation: `INSTALLATION_GUIDE.md`
- Summary: `BACKEND_MIGRATION_SUMMARY.md`

---

**No code changes required - backward compatible!**

