"""
Graph Transformer for System 1 (Neural/Intuitive Processing)
Implements attention-based graph neural network for global relationship learning.

Based on:
- "A Generalization of Transformer Networks to Graphs" (Dwivedi et al., 2020)
- Graph attention with structural encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for graphs.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, bias: bool = True):
        """
        Initialize graph multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            edge_attr: Edge attributes (batch_size, seq_len, seq_len, edge_dim)
            attention_mask: Attention mask (batch_size, seq_len, seq_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add edge attributes if provided
        if edge_attr is not None:
            # Project edge attributes to match attention heads
            # edge_attr: (batch, seq_len, seq_len, edge_dim)
            # We need: (batch, heads, seq_len, seq_len)
            edge_scores = edge_attr.mean(dim=-1, keepdim=True).expand(-1, -1, -1, self.num_heads)
            edge_scores = edge_scores.permute(0, 3, 1, 2)
            scores = scores + edge_scores

        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attention_weights.mean(dim=1)  # Average over heads


class GraphTransformerLayer(nn.Module):
    """
    Single layer of graph transformer.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1):
        """
        Initialize graph transformer layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head attention
        self.self_attn = GraphMultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, num_nodes, embed_dim)
            edge_attr: Edge attributes
            attention_mask: Attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x, x, x, edge_attr, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff_network(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights


class GraphTransformer(nn.Module):
    """
    Graph Transformer for System 1 (Neural) processing.
    Learns global relationships across the entire graph.
    """

    def __init__(self, node_feat_dim: int, embed_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8,
                 ff_dim: int = 1024, dropout: float = 0.1,
                 use_edge_features: bool = True,
                 max_nodes: int = 10000):
        """
        Initialize graph transformer.

        Args:
            node_feat_dim: Input node feature dimension
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            use_edge_features: Whether to use edge features
            max_nodes: Maximum number of nodes (for positional encoding)
        """
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, embed_dim)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_nodes, embed_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Edge feature projection (if used)
        if use_edge_features:
            self.edge_proj = nn.Linear(1, embed_dim)  # Simple edge weight projection

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                edge_weights: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node features (batch_size, num_nodes, node_feat_dim)
            adjacency: Adjacency matrix (batch_size, num_nodes, num_nodes)
            edge_weights: Edge weights (batch_size, num_nodes, num_nodes)
            return_attention: Whether to return attention weights

        Returns:
            Output features (batch_size, num_nodes, embed_dim)
            Or tuple of (output, attention_weights) if return_attention=True
        """
        batch_size, num_nodes, _ = node_features.shape

        # Project input features
        x = self.input_proj(node_features)

        # Add positional encoding
        x = x + self.pos_encoding[:, :num_nodes, :]
        x = self.dropout(x)

        # Prepare edge attributes
        edge_attr = None
        if self.use_edge_features and edge_weights is not None:
            edge_attr = self.edge_proj(edge_weights.unsqueeze(-1))

        # Prepare attention mask from adjacency
        attention_mask = adjacency if adjacency is not None else None

        # Apply transformer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, edge_attr, attention_mask)
            if return_attention:
                attention_weights_list.append(attn_weights)

        # Output projection
        x = self.output_proj(x)

        if return_attention:
            return x, attention_weights_list
        return x

    def get_node_embeddings(self, node_features: torch.Tensor,
                           adjacency: Optional[torch.Tensor] = None,
                           edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get node embeddings from the transformer.

        Args:
            node_features: Node features
            adjacency: Adjacency matrix
            edge_weights: Edge weights

        Returns:
            Node embeddings
        """
        return self.forward(node_features, adjacency, edge_weights, return_attention=False)


class SparseGraphTransformer(nn.Module):
    """
    Memory-efficient graph transformer for large sparse graphs.
    Uses sparse attention patterns.
    """

    def __init__(self, node_feat_dim: int, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8,
                 k_neighbors: int = 20, dropout: float = 0.1):
        """
        Initialize sparse graph transformer.

        Args:
            node_feat_dim: Input node feature dimension
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            k_neighbors: Number of neighbors to attend to (for sparsity)
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, embed_dim)

        # Sparse transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def compute_sparse_attention_mask(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse attention mask based on k-nearest neighbors.

        Args:
            adjacency: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
            Sparse attention mask
        """
        batch_size, num_nodes, _ = adjacency.shape

        # Get top-k neighbors for each node
        if self.k_neighbors >= num_nodes:
            return adjacency

        # Keep top-k edges per node
        topk_values, topk_indices = torch.topk(adjacency, self.k_neighbors, dim=-1)

        # Create sparse mask
        mask = torch.zeros_like(adjacency)
        mask.scatter_(2, topk_indices, 1.0)

        # Make symmetric
        mask = torch.maximum(mask, mask.transpose(-2, -1))

        return mask

    def forward(self, node_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse attention.

        Args:
            node_features: Node features
            adjacency: Adjacency matrix

        Returns:
            Output features
        """
        # Project input
        x = self.input_proj(node_features)
        x = self.dropout(x)

        # Compute sparse attention mask
        sparse_mask = self.compute_sparse_attention_mask(adjacency)

        # Apply transformer layers with sparse attention
        for layer in self.layers:
            x, _ = layer(x, attention_mask=sparse_mask)

        return x


class GraphPooling(nn.Module):
    """
    Graph pooling layer for graph-level representations.
    """

    def __init__(self, embed_dim: int, pooling_type: str = 'attention'):
        """
        Initialize graph pooling.

        Args:
            embed_dim: Embedding dimension
            pooling_type: Type of pooling ('mean', 'max', 'sum', 'attention')
        """
        super().__init__()

        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, node_features: torch.Tensor,
                batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            node_features: Node features (batch_size, num_nodes, embed_dim)
            batch_indices: Batch assignment for each node

        Returns:
            Graph-level features (batch_size, embed_dim)
        """
        if self.pooling_type == 'mean':
            return node_features.mean(dim=1)

        elif self.pooling_type == 'max':
            return node_features.max(dim=1)[0]

        elif self.pooling_type == 'sum':
            return node_features.sum(dim=1)

        elif self.pooling_type == 'attention':
            # Compute attention scores
            scores = self.attention_weights(node_features)
            attention = F.softmax(scores, dim=1)
            return (node_features * attention).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


if __name__ == "__main__":
    # Test Graph Transformer
    logger.info("Testing Graph Transformer...")

    batch_size = 2
    num_nodes = 100
    node_feat_dim = 768
    embed_dim = 256

    # Create dummy data
    node_features = torch.randn(batch_size, num_nodes, node_feat_dim)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    adjacency = (adjacency > 0.9).float()  # Sparse adjacency
    edge_weights = torch.rand(batch_size, num_nodes, num_nodes)

    # Initialize model
    model = GraphTransformer(
        node_feat_dim=node_feat_dim,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=8
    )

    # Forward pass
    output = model(node_features, adjacency, edge_weights)

    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test sparse version
    logger.info("\nTesting Sparse Graph Transformer...")

    sparse_model = SparseGraphTransformer(
        node_feat_dim=node_feat_dim,
        embed_dim=embed_dim,
        num_layers=3,
        k_neighbors=20
    )

    sparse_output = sparse_model(node_features, adjacency)

    print(f"Sparse output shape: {sparse_output.shape}")
    print(f"Sparse model parameters: {sum(p.numel() for p in sparse_model.parameters()):,}")

    logger.info("\nGraph Transformer tests complete!")

