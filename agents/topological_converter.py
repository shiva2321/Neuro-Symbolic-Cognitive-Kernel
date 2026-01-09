"""
Agent 2: The Linguistic-Topological Converter (Data Converter)
Role: Graph Tokenization & Feature Engineer

Transforms raw text and metadata into a Multilevel Context Textual-Edge Graph (MC-TEG)
and Graph Words using Graph2Seq tokenization.

Features:
- Context construction with node descriptions and edge attributes
- Random walk-based tokenization (Graph2Seq)
- Flexible Token Sequence (FTSeq) generation
- Laplacian Eigenvectors and Random Walk Positional Encodings (RWPE)
- Graph Words as learnable feature vectors
"""

import torch
import torch.nn as nn
import numpy as np
import dgl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from scipy.sparse.linalg import eigsh
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCTEGConfig:
    """Configuration for Multilevel Context Textual-Edge Graph"""
    max_context_length: int = 512
    edge_type_vocab_size: int = 100  # Number of distinct edge types
    use_laplacian_pe: bool = True
    use_rwpe: bool = True
    num_laplacian_eigenvectors: int = 8
    rwpe_walk_length: int = 20
    rwpe_num_walks: int = 10


@dataclass
class Graph2SeqConfig:
    """Configuration for Graph2Seq tokenization"""
    walk_length: int = 50
    num_walks_per_node: int = 10
    window_size: int = 5
    graph_word_dim: int = 256
    vocab_size: int = 50000
    negative_samples: int = 5


class PositionalEncoder:
    """
    Generates positional encodings for graph nodes.
    Combines Laplacian Eigenvectors and Random Walk Positional Encodings.
    """

    def __init__(self, config: MCTEGConfig):
        """
        Initialize positional encoder.

        Args:
            config: Configuration object
        """
        self.config = config

    def compute_laplacian_pe(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute Laplacian positional encodings.

        Args:
            graph: DGL graph

        Returns:
            Tensor of shape (num_nodes, num_eigenvectors)
        """
        logger.info("Computing Laplacian positional encodings...")

        # Convert to NetworkX for eigenvalue computation
        nx_graph = graph.to_networkx().to_undirected()

        # Get Laplacian matrix
        import networkx as nx
        L = nx.normalized_laplacian_matrix(nx_graph).astype(float)

        # Compute eigenvectors
        try:
            # Get k smallest eigenvalues and eigenvectors
            k = min(self.config.num_laplacian_eigenvectors, graph.num_nodes() - 2)
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

            # Sort by eigenvalue
            idx = eigenvalues.argsort()
            eigenvectors = eigenvectors[:, idx]

            # Convert to tensor
            pe = torch.from_numpy(eigenvectors).float()

            logger.info(f"Computed {k} Laplacian eigenvectors")
            return pe

        except Exception as e:
            logger.warning(f"Failed to compute Laplacian PE: {e}. Using random PE.")
            return torch.randn(graph.num_nodes(), self.config.num_laplacian_eigenvectors)

    def compute_rwpe(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute Random Walk Positional Encodings.

        Args:
            graph: DGL graph

        Returns:
            Tensor of shape (num_nodes, walk_length)
        """
        logger.info("Computing Random Walk positional encodings...")

        num_nodes = graph.num_nodes()
        walk_length = self.config.rwpe_walk_length

        # Initialize landing probability matrix
        landing_probs = torch.zeros(num_nodes, walk_length)

        # For each node, perform random walks
        for start_node in range(num_nodes):
            current = start_node

            for step in range(walk_length):
                # Record landing at current node
                landing_probs[current, step] += 1

                # Get neighbors
                successors = graph.successors(current).numpy()

                if len(successors) == 0:
                    break

                # Random walk step
                current = np.random.choice(successors)

        # Normalize
        landing_probs = landing_probs / (landing_probs.sum(dim=0, keepdim=True) + 1e-8)

        logger.info(f"Computed RWPE with walk length {walk_length}")
        return landing_probs


class GraphWordTokenizer:
    """
    Converts graph structures into sequences of 'Graph Words'.
    Based on random walks and Skip-Gram-like training.
    """

    def __init__(self, config: Graph2SeqConfig):
        """
        Initialize Graph Word tokenizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.graph_word_embeddings = None
        self.node_to_token_id: Dict[int, int] = {}
        self.token_id_to_node: Dict[int, int] = {}

    def random_walk(self, graph: dgl.DGLGraph, start_node: int, walk_length: int) -> List[int]:
        """
        Perform a random walk starting from a node.

        Args:
            graph: DGL graph
            start_node: Starting node ID
            walk_length: Length of walk

        Returns:
            List of visited node IDs
        """
        walk = [start_node]
        current = start_node

        for _ in range(walk_length - 1):
            successors = graph.successors(current).numpy()

            if len(successors) == 0:
                break

            # Sample next node (can be weighted by edge weights)
            if 'weight' in graph.edata:
                # Get edge IDs
                edge_ids = graph.edge_ids(current, successors)
                weights = graph.edata['weight'][edge_ids].numpy()
                weights = weights / weights.sum()
                next_node = np.random.choice(successors, p=weights)
            else:
                next_node = np.random.choice(successors)

            walk.append(int(next_node))
            current = int(next_node)

        return walk

    def generate_walks(self, graph: dgl.DGLGraph) -> List[List[int]]:
        """
        Generate random walks for all nodes.

        Args:
            graph: DGL graph

        Returns:
            List of walks (each walk is a list of node IDs)
        """
        logger.info(f"Generating {self.config.num_walks_per_node} walks per node...")

        walks = []
        num_nodes = graph.num_nodes()

        for node in range(num_nodes):
            for _ in range(self.config.num_walks_per_node):
                walk = self.random_walk(graph, node, self.config.walk_length)
                walks.append(walk)

        logger.info(f"Generated {len(walks)} walks")
        return walks

    def walks_to_sequences(self, walks: List[List[int]]) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        Convert walks to token sequences.

        Args:
            walks: List of random walks

        Returns:
            Tuple of (token_sequences, vocab_mapping)
        """
        # Build vocabulary from walks
        node_freq = defaultdict(int)
        for walk in walks:
            for node in walk:
                node_freq[node] += 1

        # Sort by frequency and assign token IDs
        sorted_nodes = sorted(node_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_nodes = sorted_nodes[:self.config.vocab_size]

        self.node_to_token_id = {node: idx for idx, (node, _) in enumerate(sorted_nodes)}
        self.token_id_to_node = {idx: node for node, idx in self.node_to_token_id.items()}

        # Convert walks to token sequences
        token_sequences = []
        for walk in walks:
            sequence = [self.node_to_token_id[node] for node in walk if node in self.node_to_token_id]
            if len(sequence) > 0:
                token_sequences.append(sequence)

        return token_sequences, self.node_to_token_id

    def train_graph_words(self, token_sequences: List[List[int]], num_epochs: int = 5) -> nn.Embedding:
        """
        Train Graph Word embeddings using Skip-Gram approach.

        Args:
            token_sequences: List of token sequences
            num_epochs: Number of training epochs

        Returns:
            Trained embedding layer
        """
        logger.info(f"Training Graph Word embeddings (dim={self.config.graph_word_dim})...")

        vocab_size = len(self.node_to_token_id)
        embed_dim = self.config.graph_word_dim

        # Initialize embeddings
        embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(embeddings.weight)

        # Simple Skip-Gram training
        optimizer = torch.optim.Adam(embeddings.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for sequence in token_sequences:
                # Generate context pairs
                for i, center in enumerate(sequence):
                    # Get context window
                    start = max(0, i - self.config.window_size)
                    end = min(len(sequence), i + self.config.window_size + 1)

                    context = [sequence[j] for j in range(start, end) if j != i]

                    if len(context) == 0:
                        continue

                    # Positive samples
                    center_tensor = torch.tensor([center])
                    context_tensor = torch.tensor(context)

                    # Get embeddings
                    center_embed = embeddings(center_tensor)
                    context_embeds = embeddings(context_tensor)

                    # Positive loss
                    pos_scores = torch.matmul(context_embeds, center_embed.T).squeeze()
                    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()

                    # Negative sampling
                    neg_samples = torch.randint(0, vocab_size, (self.config.negative_samples,))
                    neg_embeds = embeddings(neg_samples)
                    neg_scores = torch.matmul(neg_embeds, center_embed.T).squeeze()
                    neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).mean()

                    # Total loss
                    loss = pos_loss + neg_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        self.graph_word_embeddings = embeddings
        return embeddings


class TopologicalConverter:
    """
    Main converter class that transforms raw data into MC-TEG and Graph Words.
    """

    def __init__(self, mcteg_config: Optional[MCTEGConfig] = None,
                 graph2seq_config: Optional[Graph2SeqConfig] = None):
        """
        Initialize the Topological Converter.

        Args:
            mcteg_config: Configuration for MC-TEG
            graph2seq_config: Configuration for Graph2Seq
        """
        self.mcteg_config = mcteg_config or MCTEGConfig()
        self.graph2seq_config = graph2seq_config or Graph2SeqConfig()

        self.pe_encoder = PositionalEncoder(self.mcteg_config)
        self.tokenizer = GraphWordTokenizer(self.graph2seq_config)

        logger.info("TopologicalConverter initialized")

    def construct_mcteg(self, graph: dgl.DGLGraph,
                       node_descriptions: Optional[Dict[int, str]] = None,
                       edge_texts: Optional[Dict[Tuple[int, int], str]] = None,
                       edge_types: Optional[Dict[Tuple[int, int], str]] = None) -> dgl.DGLGraph:
        """
        Construct Multilevel Context Textual-Edge Graph.

        Args:
            graph: Input DGL graph
            node_descriptions: Personal descriptions for nodes
            edge_texts: Interaction texts for edges
            edge_types: Edge type labels (e.g., 'cites', 'cited_by')

        Returns:
            Enhanced DGL graph with context features
        """
        logger.info("Constructing MC-TEG...")

        # Add positional encodings
        if self.mcteg_config.use_laplacian_pe:
            laplacian_pe = self.pe_encoder.compute_laplacian_pe(graph)
            graph.ndata['laplacian_pe'] = laplacian_pe

        if self.mcteg_config.use_rwpe:
            rwpe = self.pe_encoder.compute_rwpe(graph)
            graph.ndata['rwpe'] = rwpe

        # Add node descriptions as features (using simple encoding)
        if node_descriptions:
            # In production, use a proper text encoder (e.g., BERT)
            desc_features = []
            for node_id in range(graph.num_nodes()):
                if node_id in node_descriptions:
                    # Simple hash-based encoding (placeholder)
                    desc_hash = hash(node_descriptions[node_id]) % 1000
                    desc_features.append(desc_hash)
                else:
                    desc_features.append(0)

            graph.ndata['description_feat'] = torch.tensor(desc_features).unsqueeze(1).float()

        # Add edge type features
        if edge_types:
            edge_type_ids = []
            src, dst = graph.edges()

            for s, d in zip(src.numpy(), dst.numpy()):
                edge_key = (int(s), int(d))
                if edge_key in edge_types:
                    # Map edge type to ID
                    edge_type_id = hash(edge_types[edge_key]) % self.mcteg_config.edge_type_vocab_size
                    edge_type_ids.append(edge_type_id)
                else:
                    edge_type_ids.append(0)

            graph.edata['edge_type'] = torch.tensor(edge_type_ids)

        # Add edge text features
        if edge_texts:
            edge_text_features = []
            src, dst = graph.edges()

            for s, d in zip(src.numpy(), dst.numpy()):
                edge_key = (int(s), int(d))
                if edge_key in edge_texts:
                    # Simple encoding
                    text_hash = hash(edge_texts[edge_key]) % 1000
                    edge_text_features.append(text_hash)
                else:
                    edge_text_features.append(0)

            graph.edata['text_feat'] = torch.tensor(edge_text_features).unsqueeze(1).float()

        logger.info(f"MC-TEG constructed with {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        return graph

    def graph_to_sequence(self, graph: dgl.DGLGraph,
                         train_embeddings: bool = True) -> Dict[str, Any]:
        """
        Convert graph to Flexible Token Sequence (FTSeq) of Graph Words.

        Args:
            graph: Input DGL graph
            train_embeddings: Whether to train Graph Word embeddings

        Returns:
            Dictionary with sequences and embeddings
        """
        logger.info("Converting graph to token sequences...")

        # Generate random walks
        walks = self.tokenizer.generate_walks(graph)

        # Convert to token sequences
        token_sequences, vocab_mapping = self.tokenizer.walks_to_sequences(walks)

        # Train Graph Word embeddings
        if train_embeddings:
            embeddings = self.tokenizer.train_graph_words(token_sequences)
        else:
            embeddings = None

        return {
            'token_sequences': token_sequences,
            'vocab_mapping': vocab_mapping,
            'graph_word_embeddings': embeddings,
            'num_unique_tokens': len(vocab_mapping),
            'num_sequences': len(token_sequences)
        }

    def convert(self, graph: dgl.DGLGraph,
               node_descriptions: Optional[Dict[int, str]] = None,
               edge_texts: Optional[Dict[Tuple[int, int], str]] = None,
               edge_types: Optional[Dict[Tuple[int, int], str]] = None,
               generate_sequences: bool = True) -> Dict[str, Any]:
        """
        Main conversion function.

        Args:
            graph: Input DGL graph
            node_descriptions: Node context descriptions
            edge_texts: Edge interaction texts
            edge_types: Edge type labels
            generate_sequences: Whether to generate token sequences

        Returns:
            Dictionary with MC-TEG and Graph Words
        """
        logger.info("Starting topological conversion...")

        # Construct MC-TEG
        mcteg = self.construct_mcteg(graph, node_descriptions, edge_texts, edge_types)

        result = {
            'mcteg': mcteg,
            'num_nodes': mcteg.num_nodes(),
            'num_edges': mcteg.num_edges(),
        }

        # Generate sequences if requested
        if generate_sequences:
            seq_data = self.graph_to_sequence(mcteg)
            result.update(seq_data)

        logger.info("Topological conversion complete")
        return result

    def get_node_positional_coords(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Get positional coordinates for nodes in semantic space.

        Args:
            graph: DGL graph with positional encodings

        Returns:
            Tensor of positional coordinates
        """
        coords = []

        if 'laplacian_pe' in graph.ndata:
            coords.append(graph.ndata['laplacian_pe'])

        if 'rwpe' in graph.ndata:
            coords.append(graph.ndata['rwpe'])

        if not coords:
            # Return node IDs as fallback
            return torch.arange(graph.num_nodes()).unsqueeze(1).float()

        return torch.cat(coords, dim=1)

