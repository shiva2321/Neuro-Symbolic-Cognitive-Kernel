"""
Graph Backend Compatibility Layer
Provides unified interface for both DGL and PyTorch Geometric (PyG).

This allows the system to work with either backend without breaking existing functionality.
Automatically detects and uses the available backend, with PyG as the preferred alternative.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Detect available backends
BACKEND_DGL = False
BACKEND_PYG = False

try:
    import dgl
    BACKEND_DGL = True
    logger.info("DGL backend available")
except (ImportError, FileNotFoundError, OSError) as e:
    # Catch ImportError (not installed) and FileNotFoundError/OSError (corrupted install)
    logger.warning(f"DGL not available: {e}")
    dgl = None  # Define for type checking

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx, from_networkx
    BACKEND_PYG = True
    logger.info("PyTorch Geometric backend available")
except (ImportError, FileNotFoundError, OSError) as e:
    logger.warning(f"PyTorch Geometric not available: {e}")
    torch_geometric = None
    Data = None
    to_networkx = None
    from_networkx = None

if not BACKEND_DGL and not BACKEND_PYG:
    error_msg = (
        "Neither DGL nor PyTorch Geometric is available.\n"
        "Please install at least one:\n"
        "  pip install torch-geometric  (Recommended)\n"
        "  OR\n"
        "  pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html\n"
        "\n"
        "If DGL is installed but showing errors, reinstall it:\n"
        "  pip uninstall dgl -y\n"
        "  pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html"
    )
    raise ImportError(error_msg)


class GraphBackend:
    """
    Unified graph interface that works with both DGL and PyG.

    Attributes:
        backend: Current backend ('dgl' or 'pyg')
        graph: Underlying graph object
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize graph backend.

        Args:
            backend: Force specific backend ('dgl' or 'pyg'). If None, auto-detect.
        """
        if backend is None:
            # Prefer PyG if available, fallback to DGL
            if BACKEND_PYG:
                self.backend = 'pyg'
            elif BACKEND_DGL:
                self.backend = 'dgl'
            else:
                raise RuntimeError("No graph backend available")
        else:
            backend = backend.lower()
            if backend == 'dgl' and not BACKEND_DGL:
                raise RuntimeError("DGL backend requested but not available")
            if backend == 'pyg' and not BACKEND_PYG:
                raise RuntimeError("PyG backend requested but not available")
            self.backend = backend

        self.graph = None
        logger.info(f"Using {self.backend.upper()} backend")

    def create_graph(self, src_nodes: List[int], dst_nodes: List[int],
                    num_nodes: int) -> Any:
        """
        Create a graph from edge list.

        Args:
            src_nodes: Source node indices
            dst_nodes: Destination node indices
            num_nodes: Total number of nodes

        Returns:
            Graph object (DGL or PyG)
        """
        if self.backend == 'dgl':
            self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        else:  # pyg
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            self.graph = Data(edge_index=edge_index, num_nodes=num_nodes)

        return self.graph

    def set_node_features(self, name: str, features: torch.Tensor):
        """Set node features."""
        if self.backend == 'dgl':
            self.graph.ndata[name] = features
        else:  # pyg
            setattr(self.graph, name, features)

    def get_node_features(self, name: str) -> torch.Tensor:
        """Get node features."""
        if self.backend == 'dgl':
            return self.graph.ndata[name]
        else:  # pyg
            return getattr(self.graph, name)

    def set_edge_features(self, name: str, features: torch.Tensor):
        """Set edge features."""
        if self.backend == 'dgl':
            self.graph.edata[name] = features
        else:  # pyg
            edge_attr_name = f'edge_{name}'
            setattr(self.graph, edge_attr_name, features)

    def get_edge_features(self, name: str) -> torch.Tensor:
        """Get edge features."""
        if self.backend == 'dgl':
            return self.graph.edata[name]
        else:  # pyg
            edge_attr_name = f'edge_{name}'
            return getattr(self.graph, edge_attr_name)

    def num_nodes(self) -> int:
        """Get number of nodes."""
        if self.backend == 'dgl':
            return self.graph.num_nodes()
        else:  # pyg
            return self.graph.num_nodes

    def num_edges(self) -> int:
        """Get number of edges."""
        if self.backend == 'dgl':
            return self.graph.num_edges()
        else:  # pyg
            return self.graph.edge_index.size(1)

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get edge list as (src, dst) tensors."""
        if self.backend == 'dgl':
            return self.graph.edges()
        else:  # pyg
            return self.graph.edge_index[0], self.graph.edge_index[1]

    def successors(self, node_id: int) -> torch.Tensor:
        """Get successor nodes for a given node."""
        if self.backend == 'dgl':
            return self.graph.successors(node_id)
        else:  # pyg
            # Find all edges where source is node_id
            mask = self.graph.edge_index[0] == node_id
            return self.graph.edge_index[1][mask]

    def predecessors(self, node_id: int) -> torch.Tensor:
        """Get predecessor nodes for a given node."""
        if self.backend == 'dgl':
            return self.graph.predecessors(node_id)
        else:  # pyg
            # Find all edges where destination is node_id
            mask = self.graph.edge_index[1] == node_id
            return self.graph.edge_index[0][mask]

    def to_networkx(self):
        """Convert to NetworkX graph."""
        if self.backend == 'dgl':
            return self.graph.to_networkx()
        else:  # pyg
            return to_networkx(self.graph, to_undirected=False)

    def to(self, device: Union[str, torch.device]):
        """Move graph to device."""
        if self.backend == 'dgl':
            self.graph = self.graph.to(device)
        else:  # pyg
            self.graph = self.graph.to(device)
        return self

    def save(self, path: str):
        """Save graph to disk."""
        if self.backend == 'dgl':
            dgl.save_graphs(path, [self.graph])
        else:  # pyg
            torch.save(self.graph, path)

    @classmethod
    def load(cls, path: str, backend: Optional[str] = None):
        """Load graph from disk."""
        instance = cls(backend=backend)

        if instance.backend == 'dgl':
            graphs, _ = dgl.load_graphs(path)
            instance.graph = graphs[0]
        else:  # pyg
            instance.graph = torch.load(path)

        return instance

    def get_adj_matrix(self) -> torch.Tensor:
        """Get adjacency matrix as dense tensor."""
        if self.backend == 'dgl':
            return self.graph.adj().to_dense()
        else:  # pyg
            from torch_geometric.utils import to_dense_adj
            # Ensure we get the full matrix including isolated nodes
            return to_dense_adj(self.graph.edge_index, max_num_nodes=self.graph.num_nodes)[0]

    def subgraph(self, nodes: List[int]):
        """Extract subgraph containing specified nodes."""
        if self.backend == 'dgl':
            return self.graph.subgraph(nodes)
        else:  # pyg
            from torch_geometric.utils import subgraph
            subset = torch.tensor(nodes, dtype=torch.long)
            edge_index, edge_attr = subgraph(
                subset,
                self.graph.edge_index,
                relabel_nodes=True,
                num_nodes=self.graph.num_nodes
            )
            data = Data(edge_index=edge_index, num_nodes=len(nodes))
            # Copy node features if they exist
            for key in ['feat', 'pos_enc', 'freq']:
                if hasattr(self.graph, key):
                    setattr(data, key, getattr(self.graph, key)[subset])
            return data


def create_graph_backend(backend: Optional[str] = None) -> GraphBackend:
    """
    Factory function to create a graph backend.

    Args:
        backend: 'dgl', 'pyg', or None for auto-detection

    Returns:
        GraphBackend instance
    """
    return GraphBackend(backend=backend)


def get_available_backends() -> List[str]:
    """Get list of available graph backends."""
    backends = []
    if BACKEND_DGL:
        backends.append('dgl')
    if BACKEND_PYG:
        backends.append('pyg')
    return backends


def get_default_backend() -> str:
    """Get the default backend (prefers PyG)."""
    if BACKEND_PYG:
        return 'pyg'
    elif BACKEND_DGL:
        return 'dgl'
    else:
        raise RuntimeError("No graph backend available")


# Convenience wrappers for backward compatibility
if BACKEND_DGL:
    DGLGraph = dgl.DGLGraph
else:
    # Create a dummy class for type hints
    class DGLGraph:
        pass


if BACKEND_PYG:
    PyGData = Data
else:
    # Create a dummy class for type hints
    class PyGData:
        pass
