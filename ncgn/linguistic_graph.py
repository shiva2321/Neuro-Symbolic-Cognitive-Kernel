"""
Linguistic Graph Substrate
Implements the foundational graph structure G = (V, E) with semantic connectivity.

Agent Prompt 1 Implementation:
- Node initialization with pre-trained embeddings (RoBERTa/BERT)
- Edge creation via Pointwise Mutual Information (PMI)
- Structural positional encodings
- GPU-accelerated operations via DGL
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import graph backend compatibility layer
try:
    from utils.graph_backend import GraphBackend, BACKEND_DGL, BACKEND_PYG
except ImportError as e:
    # Fallback to direct DGL import for backward compatibility
    try:
        import dgl
        BACKEND_DGL = True
        BACKEND_PYG = False
        GraphBackend = None
    except (ImportError, FileNotFoundError, OSError) as dgl_error:
        # DGL also failed - provide helpful error
        error_msg = (
            f"Cannot import graph backend: {e}\n"
            f"DGL fallback also failed: {dgl_error}\n\n"
            "Please install PyTorch Geometric (recommended) or fix DGL:\n"
            "  pip install torch-geometric\n"
            "OR reinstall DGL:\n"
            "  pip uninstall dgl -y\n"
            "  pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html"
        )
        raise ImportError(error_msg) from dgl_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeFeatures:
    """Features associated with each node in the linguistic graph"""
    node_id: int
    token: str
    embedding: np.ndarray  # Pre-trained embedding (768-d for RoBERTa)
    frequency: int = 1
    positional_encoding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class EdgeFeatures:
    """Features associated with each edge"""
    source: int
    target: int
    pmi_score: float  # Pointwise Mutual Information
    co_occurrence: int
    weight: float  # Normalized edge weight
    edge_type: str = "semantic"  # semantic, syntactic, sequential


class LinguisticGraph:
    """
    Core linguistic graph structure that represents vocabulary as a network.

    Attributes:
        graph: Graph object (DGL or PyG) for GPU-accelerated operations
        vocab: Dictionary mapping tokens to node IDs
        node_features: Dictionary of node features
        device: Computation device (CPU/CUDA)
    """

    def __init__(self, device: Optional[str] = None, backend: Optional[str] = None):
        """
        Initialize the linguistic graph.

        Args:
            device: Device for computation ('cpu', 'cuda', or None for auto)
            backend: Graph backend ('dgl', 'pyg', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backend_type = backend

        # Initialize graph backend
        if GraphBackend is not None:
            self._backend = GraphBackend(backend=backend)
            logger.info(f"Initializing LinguisticGraph on device: {self.device} with backend: {self._backend.backend}")
        else:
            self._backend = None
            logger.info(f"Initializing LinguisticGraph on device: {self.device} with DGL backend (fallback)")

        self.graph = None
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.node_features: Dict[int, NodeFeatures] = {}
        self.edge_list: List[EdgeFeatures] = []

        # Statistics
        self.total_tokens: int = 0
        self.unique_tokens: int = 0

    def add_node(self, token: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Add a node to the graph or update existing node.

        Args:
            token: Word or symbol
            embedding: Pre-trained embedding vector
            metadata: Additional node information

        Returns:
            Node ID
        """
        if token in self.vocab:
            # Update frequency
            node_id = self.vocab[token]
            self.node_features[node_id].frequency += 1
            return node_id

        # Create new node
        node_id = len(self.vocab)
        self.vocab[token] = node_id
        self.reverse_vocab[node_id] = token

        self.node_features[node_id] = NodeFeatures(
            node_id=node_id,
            token=token,
            embedding=embedding,
            frequency=1,
            metadata=metadata or {}
        )

        self.unique_tokens += 1
        self.total_tokens += 1

        return node_id

    def add_edge(self, source_token: str, target_token: str,
                 co_occurrence: int = 1, edge_type: str = "semantic"):
        """
        Add an edge between two tokens.

        Args:
            source_token: Source token
            target_token: Target token
            co_occurrence: Number of co-occurrences
            edge_type: Type of relationship
        """
        if source_token not in self.vocab or target_token not in self.vocab:
            logger.warning(f"Cannot add edge: tokens not in vocab")
            return

        source_id = self.vocab[source_token]
        target_id = self.vocab[target_token]

        # Check if edge already exists
        for edge in self.edge_list:
            if edge.source == source_id and edge.target == target_id:
                edge.co_occurrence += co_occurrence
                return

        # Create new edge (PMI will be calculated later)
        self.edge_list.append(EdgeFeatures(
            source=source_id,
            target=target_id,
            pmi_score=0.0,
            co_occurrence=co_occurrence,
            weight=0.0,
            edge_type=edge_type
        ))

    def compute_pmi_scores(self, window_size: int = 5):
        """
        Compute Pointwise Mutual Information (PMI) for all edges.

        PMI(x, y) = log(P(x, y) / (P(x) * P(y)))

        Args:
            window_size: Context window size for co-occurrence
        """
        logger.info("Computing PMI scores for edges...")

        # Calculate token probabilities
        total = sum(nf.frequency for nf in self.node_features.values())
        token_probs = {
            nid: nf.frequency / total
            for nid, nf in self.node_features.items()
        }

        # Calculate joint probabilities and PMI
        total_co_occurrences = sum(e.co_occurrence for e in self.edge_list)

        for edge in tqdm(self.edge_list, desc="Computing PMI"):
            # Joint probability
            p_xy = edge.co_occurrence / total_co_occurrences

            # Marginal probabilities
            p_x = token_probs[edge.source]
            p_y = token_probs[edge.target]

            # PMI with smoothing to avoid log(0)
            if p_x * p_y > 0:
                edge.pmi_score = np.log((p_xy + 1e-10) / (p_x * p_y + 1e-10))
            else:
                edge.pmi_score = 0.0

            # Normalize weight using sigmoid
            edge.weight = 1.0 / (1.0 + np.exp(-edge.pmi_score))

    def build_dgl_graph(self):
        """
        Build a graph from the linguistic graph structure.
        Works with both DGL and PyTorch Geometric backends.

        Returns:
            Graph object ready for GPU operations
        """
        logger.info("Building graph...")

        num_nodes = len(self.vocab)

        # Extract edge indices
        src_nodes = [e.source for e in self.edge_list]
        dst_nodes = [e.target for e in self.edge_list]

        # Create graph using backend
        if self._backend is not None:
            self.graph = self._backend.create_graph(src_nodes, dst_nodes, num_nodes)
        else:
            # Fallback to DGL
            import dgl
            self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

        # Add node features
        embeddings = np.stack([
            self.node_features[i].embedding
            for i in range(num_nodes)
        ])
        frequencies = [self.node_features[i].frequency for i in range(num_nodes)]

        if self._backend is not None:
            self._backend.graph = self.graph
            self._backend.set_node_features('feat', torch.tensor(embeddings, dtype=torch.float32))
            self._backend.set_node_features('freq', torch.tensor(frequencies, dtype=torch.long))
        else:
            # Fallback to DGL
            self.graph.ndata['feat'] = torch.tensor(embeddings, dtype=torch.float32)
            self.graph.ndata['freq'] = torch.tensor(frequencies, dtype=torch.long)

        # Add edge features
        edge_weights = torch.tensor([e.weight for e in self.edge_list], dtype=torch.float32)
        pmi_scores = torch.tensor([e.pmi_score for e in self.edge_list], dtype=torch.float32)

        if self._backend is not None:
            self._backend.set_edge_features('weight', edge_weights)
            self._backend.set_edge_features('pmi', pmi_scores)
        else:
            # Fallback to DGL
            self.graph.edata['weight'] = edge_weights
            self.graph.edata['pmi'] = pmi_scores

        # Move to device
        if self._backend is not None:
            self._backend.to(self.device)
            self.graph = self._backend.graph
        else:
            # Fallback to DGL
            self.graph = self.graph.to(self.device)

        logger.info(f"Graph created: {num_nodes} nodes, {len(self.edge_list)} edges")
        return self.graph

    def compute_structural_encodings(self, k: int = 8):
        """
        Compute structural positional encodings using Laplacian eigenvectors.

        This provides a global coordinate system for nodes based on graph topology.

        Args:
            k: Number of smallest eigenvectors to use
        """
        logger.info(f"Computing structural positional encodings (k={k})...")

        if self.graph is None:
            logger.error("Must build graph first!")
            return

        # Get adjacency matrix
        if self._backend is not None:
            adj_matrix = self._backend.get_adj_matrix().cpu().numpy()
        else:
            # Fallback to DGL
            adj_matrix = self.graph.adj().to_dense().cpu().numpy()

        # Compute degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        D = np.diag(degrees)

        # Compute Laplacian: L = D - A
        L = D - adj_matrix

        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Eigenvalue computation failed: {e}. Using random encodings.")
            num_nodes = len(self.vocab)
            eigenvectors = np.random.randn(num_nodes, k + 1)

        # Take k smallest (excluding the trivial one)
        # Ensure we don't take more than available eigenvectors
        num_vecs = eigenvectors.shape[1]
        if num_vecs <= 1:
             pos_enc = np.zeros((eigenvectors.shape[0], k))
        else:
             end_idx = min(k + 1, num_vecs)
             pos_enc = eigenvectors[:, 1:end_idx]
             
             # Pad with zeros if we have fewer than k eigenvectors
             if pos_enc.shape[1] < k:
                 padding = np.zeros((pos_enc.shape[0], k - pos_enc.shape[1]))
                 pos_enc = np.hstack([pos_enc, padding])

        # Store in node features
        for i in range(len(self.vocab)):
            self.node_features[i].positional_encoding = pos_enc[i]

        # Add to graph
        pos_enc_tensor = torch.tensor(pos_enc, dtype=torch.float32).to(self.device)
        if self._backend is not None:
            self._backend.set_node_features('pos_enc', pos_enc_tensor)
        else:
            # Fallback to DGL
            self.graph.ndata['pos_enc'] = pos_enc_tensor

        logger.info(f"Structural encodings computed: {pos_enc.shape}")

    def save(self, path: Union[str, Path]):
        """
        Save the linguistic graph to disk.

        Args:
            path: Directory path to save graph
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving linguistic graph to {path}")

        # Save vocabulary
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)

        # Save node features
        with open(path / "node_features.pkl", "wb") as f:
            pickle.dump(self.node_features, f)

        # Save edge list
        with open(path / "edge_list.pkl", "wb") as f:
            pickle.dump(self.edge_list, f)

        # Save graph
        if self.graph is not None:
            if self._backend is not None:
                self._backend.save(str(path / "graph.bin"))
            else:
                # Fallback to DGL
                import dgl
                dgl.save_graphs(str(path / "dgl_graph.bin"), [self.graph])

        # Save metadata
        metadata = {
            'total_tokens': self.total_tokens,
            'unique_tokens': self.unique_tokens,
            'num_edges': len(self.edge_list),
            'device': self.device,
            'backend': self._backend.backend if self._backend is not None else 'dgl'
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Graph saved successfully")

    @classmethod
    def load(cls, path: Union[str, Path], backend: Optional[str] = None) -> 'LinguisticGraph':
        """
        Load a linguistic graph from disk.

        Args:
            path: Directory path containing saved graph
            backend: Force specific backend ('dgl' or 'pyg'), or None for auto

        Returns:
            Loaded LinguisticGraph instance
        """
        path = Path(path)
        logger.info(f"Loading linguistic graph from {path}")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Use saved backend if not specified
        if backend is None and 'backend' in metadata:
            backend = metadata['backend']

        # Create instance
        graph = cls(device=metadata['device'], backend=backend)

        # Load vocabulary
        with open(path / "vocab.json", "r") as f:
            graph.vocab = json.load(f)
        graph.reverse_vocab = {v: k for k, v in graph.vocab.items()}

        # Load node features
        with open(path / "node_features.pkl", "rb") as f:
            graph.node_features = pickle.load(f)

        # Load edge list
        with open(path / "edge_list.pkl", "rb") as f:
            graph.edge_list = pickle.load(f)

        # Load graph
        if (path / "graph.bin").exists():
            # New format with backend compatibility
            if graph._backend is not None:
                loaded_backend = GraphBackend.load(str(path / "graph.bin"), backend=backend)
                graph.graph = loaded_backend.graph
                graph._backend = loaded_backend
            else:
                # Fallback
                import dgl
                graphs, _ = dgl.load_graphs(str(path / "graph.bin"))
                graph.graph = graphs[0].to(graph.device)
        elif (path / "dgl_graph.bin").exists():
            # Old DGL format
            if graph._backend is None:
                import dgl
                graphs, _ = dgl.load_graphs(str(path / "dgl_graph.bin"))
                graph.graph = graphs[0].to(graph.device)
            else:
                # Need to convert DGL to new format
                logger.warning("Loading legacy DGL format. Consider re-saving in new format.")
                import dgl
                graphs, _ = dgl.load_graphs(str(path / "dgl_graph.bin"))
                old_graph = graphs[0]

                # Extract edges and recreate
                src, dst = old_graph.edges()
                graph.graph = graph._backend.create_graph(
                    src.tolist(), dst.tolist(), old_graph.num_nodes()
                )
                graph._backend.graph = graph.graph

                # Copy features
                for key in old_graph.ndata.keys():
                    graph._backend.set_node_features(key, old_graph.ndata[key])
                for key in old_graph.edata.keys():
                    graph._backend.set_edge_features(key, old_graph.edata[key])

                graph._backend.to(graph.device)
                graph.graph = graph._backend.graph

        graph.total_tokens = metadata['total_tokens']
        graph.unique_tokens = metadata['unique_tokens']

        logger.info("Graph loaded successfully")
        return graph

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        stats = {
            'total_tokens': self.total_tokens,
            'unique_tokens': self.unique_tokens,
            'num_edges': len(self.edge_list),
            'avg_degree': len(self.edge_list) / max(self.unique_tokens, 1),
            'device': self.device,
            'backend': self._backend.backend if self._backend is not None else 'dgl'
        }

        if self.graph is not None:
            if self._backend is not None:
                stats['graph_nodes'] = self._backend.num_nodes()
                stats['graph_edges'] = self._backend.num_edges()
            else:
                stats['graph_nodes'] = self.graph.num_nodes()
                stats['graph_edges'] = self.graph.num_edges()

        return stats


class GraphBuilder:
    """
    Builder class for constructing linguistic graphs from text corpora.
    """

    def __init__(self, embedding_model: str = "roberta-base", device: Optional[str] = None):
        """
        Initialize the graph builder.

        Args:
            embedding_model: Pre-trained model for embeddings
            device: Computation device
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model_name = embedding_model
        self.graph = LinguisticGraph(device=self.device)

        logger.info(f"Initializing GraphBuilder with {embedding_model}")

    def build_from_corpus(self, corpus: List[str], window_size: int = 5,
                         min_frequency: int = 2, max_vocab_size: Optional[int] = None) -> LinguisticGraph:
        """
        Build a linguistic graph from a text corpus.

        Args:
            corpus: List of text documents
            window_size: Context window for co-occurrence
            min_frequency: Minimum token frequency to include
            max_vocab_size: Maximum vocabulary size (None for unlimited)

        Returns:
            Constructed LinguisticGraph
        """
        logger.info(f"Building linguistic graph from corpus ({len(corpus)} documents)")

        # This is a placeholder - actual implementation would use transformers
        # For now, we'll use a simplified version
        from sklearn.feature_extraction.text import CountVectorizer

        # Tokenize and count
        vectorizer = CountVectorizer(
            max_features=max_vocab_size,
            min_df=min_frequency,
            token_pattern=r'\b\w+\b'
        )

        # Fit on corpus
        X = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out()

        logger.info(f"Vocabulary size: {len(vocab)}")

        # Add nodes with dummy embeddings (should use RoBERTa)
        for token in tqdm(vocab, desc="Adding nodes"):
            # Placeholder: random embedding (should use transformers)
            embedding = np.random.randn(768).astype(np.float32)
            self.graph.add_node(token, embedding)

        # Build co-occurrence matrix
        logger.info("Computing co-occurrences...")
        co_occurrence = defaultdict(int)

        for doc in tqdm(corpus, desc="Processing documents"):
            tokens = doc.lower().split()

            for i, token1 in enumerate(tokens):
                if token1 not in self.graph.vocab:
                    continue

                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                    if i == j:
                        continue
                    token2 = tokens[j]
                    if token2 not in self.graph.vocab:
                        continue

                    co_occurrence[(token1, token2)] += 1

        # Add edges
        logger.info(f"Adding {len(co_occurrence)} edges...")
        for (token1, token2), count in tqdm(co_occurrence.items(), desc="Adding edges"):
            self.graph.add_edge(token1, token2, co_occurrence=count)

        # Compute PMI scores
        self.graph.compute_pmi_scores(window_size)

        # Build DGL graph
        self.graph.build_dgl_graph()

        # Compute structural encodings
        self.graph.compute_structural_encodings(k=8)

        logger.info("Linguistic graph construction complete!")
        return self.graph

