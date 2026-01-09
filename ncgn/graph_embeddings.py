"""
Graph Embeddings & Structural Encoders
Provides pre-trained embeddings and structural encodings for graph nodes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, AutoModel
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphEmbedding:
    """
    Wrapper for pre-trained language models to generate node embeddings.
    Supports RoBERTa, BERT, and other HuggingFace models.
    """

    def __init__(self, model_name: str = "roberta-base", device: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Computation device
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        logger.info(f"Loading embedding model: {model_name} on {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dim: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to random embeddings")
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 768

    def embed_tokens(self, tokens: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of tokens.

        Args:
            tokens: List of tokens to embed
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (num_tokens, embedding_dim)
        """
        if self.model is None:
            # Fallback: random embeddings
            logger.warning("Using random embeddings (model not loaded)")
            return np.random.randn(len(tokens), self.embedding_dim).astype(np.float32)

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(tokens), batch_size), desc="Embedding tokens"):
                batch_tokens = tokens[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=32
                ).to(self.device)

                # Get embeddings
                outputs = self.model(**inputs)

                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def embed_single_token(self, token: str) -> np.ndarray:
        """
        Embed a single token.

        Args:
            token: Token to embed

        Returns:
            Embedding vector
        """
        return self.embed_tokens([token])[0]


class StructuralEncoder:
    """
    Computes structural positional encodings for graph nodes.
    Uses spectral methods based on graph Laplacian.
    """

    @staticmethod
    def compute_laplacian_pe(adjacency: np.ndarray, k: int = 8,
                            normalized: bool = True) -> np.ndarray:
        """
        Compute Laplacian positional encodings.

        Args:
            adjacency: Adjacency matrix (N x N)
            k: Number of eigenvectors to use
            normalized: Whether to use normalized Laplacian

        Returns:
            Positional encodings (N x k)
        """
        logger.info(f"Computing Laplacian PE (k={k}, normalized={normalized})")

        n = adjacency.shape[0]

        # Compute degree matrix
        degrees = np.sum(adjacency, axis=1)
        D = np.diag(degrees)

        # Compute Laplacian
        L = D - adjacency

        if normalized:
            # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Sort by eigenvalue (ascending)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take k smallest non-trivial eigenvectors
        # Skip the first one (all constants for connected graphs)
        pos_enc = eigenvectors[:, 1:k+1]

        logger.info(f"Laplacian PE computed: {pos_enc.shape}")
        return pos_enc.astype(np.float32)

    @staticmethod
    def compute_random_walk_pe(adjacency: np.ndarray, k: int = 16) -> np.ndarray:
        """
        Compute random walk-based positional encodings.

        Args:
            adjacency: Adjacency matrix (N x N)
            k: Number of steps

        Returns:
            Landing probabilities for k steps (N x k)
        """
        logger.info(f"Computing Random Walk PE (k={k})")

        n = adjacency.shape[0]

        # Normalize adjacency to get transition matrix
        degrees = np.sum(adjacency, axis=1, keepdims=True)
        P = adjacency / (degrees + 1e-10)

        # Compute powers of transition matrix
        pe = []
        P_k = np.eye(n)

        for i in range(k):
            P_k = P_k @ P
            # Diagonal contains self-return probabilities
            pe.append(np.diag(P_k))

        pe = np.column_stack(pe)

        logger.info(f"Random Walk PE computed: {pe.shape}")
        return pe.astype(np.float32)

    @staticmethod
    def compute_degree_encoding(adjacency: np.ndarray) -> np.ndarray:
        """
        Compute simple degree-based encodings.

        Args:
            adjacency: Adjacency matrix (N x N)

        Returns:
            Degree encodings (N x 3) - [in_degree, out_degree, total_degree]
        """
        in_degree = np.sum(adjacency, axis=0)
        out_degree = np.sum(adjacency, axis=1)
        total_degree = in_degree + out_degree

        # Normalize
        max_degree = max(total_degree.max(), 1)

        encodings = np.column_stack([
            in_degree / max_degree,
            out_degree / max_degree,
            total_degree / max_degree
        ])

        return encodings.astype(np.float32)


class HybridEmbedding(nn.Module):
    """
    Combines pre-trained embeddings with structural encodings.
    """

    def __init__(self, semantic_dim: int = 768, structural_dim: int = 8,
                 hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initialize hybrid embedding layer.

        Args:
            semantic_dim: Dimension of semantic embeddings
            structural_dim: Dimension of structural encodings
            hidden_dim: Hidden dimension for projection
            dropout: Dropout rate
        """
        super().__init__()

        self.semantic_dim = semantic_dim
        self.structural_dim = structural_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.structural_proj = nn.Linear(structural_dim, hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, semantic_emb: torch.Tensor,
                structural_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining semantic and structural embeddings.

        Args:
            semantic_emb: Semantic embeddings (B, semantic_dim)
            structural_emb: Structural encodings (B, structural_dim)

        Returns:
            Hybrid embeddings (B, hidden_dim)
        """
        # Project both embeddings
        sem_proj = self.semantic_proj(semantic_emb)
        struct_proj = self.structural_proj(structural_emb)

        # Concatenate and fuse
        combined = torch.cat([sem_proj, struct_proj], dim=-1)
        fused = self.fusion(combined)

        # Final projection
        output = self.output_proj(fused)

        return output


class AdaptiveEmbedding(nn.Module):
    """
    Learnable embeddings that adapt during training.
    """

    def __init__(self, num_nodes: int, embedding_dim: int,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_pretrained: bool = False):
        """
        Initialize adaptive embeddings.

        Args:
            num_nodes: Number of nodes in graph
            embedding_dim: Embedding dimension
            pretrained_embeddings: Optional pre-trained embeddings to initialize with
            freeze_pretrained: Whether to freeze pre-trained weights
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Initialize embedding layer
        if pretrained_embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_pretrained
            )
        else:
            self.embeddings = nn.Embedding(num_nodes, embedding_dim)
            nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for node IDs.

        Args:
            node_ids: Node indices (B,)

        Returns:
            Embeddings (B, embedding_dim)
        """
        return self.embeddings(node_ids)

    def update_embedding(self, node_id: int, new_embedding: torch.Tensor):
        """
        Update a specific node's embedding.

        Args:
            node_id: Node index
            new_embedding: New embedding vector
        """
        with torch.no_grad():
            self.embeddings.weight[node_id] = new_embedding


if __name__ == "__main__":
    # Test embedding generation
    logger.info("Testing GraphEmbedding...")

    embedder = GraphEmbedding(model_name="roberta-base")

    test_tokens = ["hello", "world", "neural", "network", "graph"]
    embeddings = embedder.embed_tokens(test_tokens)

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")

    # Test structural encoder
    logger.info("\nTesting StructuralEncoder...")

    # Create a simple test graph
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.float32)

    encoder = StructuralEncoder()
    lap_pe = encoder.compute_laplacian_pe(adj, k=2)
    rw_pe = encoder.compute_random_walk_pe(adj, k=4)
    deg_enc = encoder.compute_degree_encoding(adj)

    print(f"\nLaplacian PE shape: {lap_pe.shape}")
    print(f"Random Walk PE shape: {rw_pe.shape}")
    print(f"Degree encoding shape: {deg_enc.shape}")

