"""
Dual-System Cognitive Architecture
Integrates System 1 (Neural/Intuitive) and System 2 (Symbolic/Logical)

Based on dual-process theory from cognitive science.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .graph_transformer import GraphTransformer, GraphPooling
from .symbolic_reasoner import SymbolicReasoner, Fact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class System1Module(nn.Module):
    """
    System 1: Fast, intuitive, pattern-based neural processing.
    Uses Graph Transformer for learning global relationships.
    """

    def __init__(self, node_feat_dim: int, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8):
        """
        Initialize System 1 module.

        Args:
            node_feat_dim: Input node feature dimension
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()

        self.transformer = GraphTransformer(
            node_feat_dim=node_feat_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1
        )

        # Pooling for graph-level representation
        self.pooling = GraphPooling(embed_dim, pooling_type='attention')

        # Output projections
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                edge_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for System 1.

        Args:
            node_features: Node features (batch_size, num_nodes, node_feat_dim)
            adjacency: Adjacency matrix
            edge_weights: Edge weights

        Returns:
            Dictionary with node_embeddings, graph_embedding, confidence
        """
        # Transform through graph transformer
        node_embeddings = self.transformer(node_features, adjacency, edge_weights)

        # Pool to graph-level representation
        graph_embedding = self.pooling(node_embeddings)

        # Project output
        output = self.output_proj(graph_embedding)

        # Estimate confidence
        confidence = self.confidence_head(graph_embedding)

        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': output,
            'confidence': confidence
        }


class System2Module(nn.Module):
    """
    System 2: Slow, deliberate, rule-based symbolic processing.
    Uses symbolic reasoner with knowledge graphs.
    """

    def __init__(self, reasoning_depth: int = 5):
        """
        Initialize System 2 module.

        Args:
            reasoning_depth: Maximum reasoning depth
        """
        super().__init__()

        self.reasoner = SymbolicReasoner(reasoning_depth=reasoning_depth)

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for System 2.

        Args:
            context: Context dictionary

        Returns:
            Reasoning results
        """
        return self.reasoner(context)

    def add_knowledge(self, facts: List[Tuple[str, str, str]]):
        """Add facts to knowledge base"""
        for subject, predicate, obj in facts:
            self.reasoner.add_fact(subject, predicate, obj)


class IntegrationLayer(nn.Module):
    """
    Integration layer that combines System 1 and System 2 outputs.
    """

    def __init__(self, embed_dim: int, integration_method: str = 'weighted'):
        """
        Initialize integration layer.

        Args:
            embed_dim: Embedding dimension
            integration_method: How to combine systems ('weighted', 'attention', 'gating')
        """
        super().__init__()

        self.integration_method = integration_method

        if integration_method == 'attention':
            # Attention-based integration
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)

        elif integration_method == 'gating':
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, system1_output: torch.Tensor,
                system2_features: torch.Tensor,
                system1_confidence: torch.Tensor,
                system2_confidence: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate outputs from both systems.

        Args:
            system1_output: Output from System 1
            system2_features: Features from System 2
            system1_confidence: Confidence from System 1
            system2_confidence: Confidence from System 2

        Returns:
            Tuple of (integrated_output, final_confidence)
        """
        if self.integration_method == 'weighted':
            # Weighted combination based on confidence
            total_conf = system1_confidence + system2_confidence + 1e-8
            w1 = system1_confidence / total_conf
            w2 = system2_confidence / total_conf

            integrated = w1 * system1_output + w2 * system2_features
            final_confidence = (system1_confidence + system2_confidence) / 2

        elif self.integration_method == 'attention':
            # Attention-based integration
            combined = torch.stack([system1_output, system2_features], dim=0)
            integrated, _ = self.attention(combined, combined, combined)
            integrated = integrated.mean(dim=0)
            final_confidence = (system1_confidence + system2_confidence) / 2

        elif self.integration_method == 'gating':
            # Gating mechanism
            concatenated = torch.cat([system1_output, system2_features], dim=-1)
            gate = self.gate(concatenated)

            integrated = gate * system1_output + (1 - gate) * system2_features
            final_confidence = (system1_confidence + system2_confidence) / 2

        else:
            # Default: simple average
            integrated = (system1_output + system2_features) / 2
            final_confidence = (system1_confidence + system2_confidence) / 2

        # Project to output
        output = self.output_proj(integrated)

        return output, final_confidence


class DualSystemArchitecture(nn.Module):
    """
    Complete Dual-System Cognitive Architecture.
    Combines fast neural processing with deliberate symbolic reasoning.
    """

    def __init__(self, node_feat_dim: int, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8,
                 reasoning_depth: int = 5,
                 integration_method: str = 'weighted',
                 confidence_threshold: float = 0.5,
                 use_system2_verification: bool = True):
        """
        Initialize dual-system architecture.

        Args:
            node_feat_dim: Input node feature dimension
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            reasoning_depth: Symbolic reasoning depth
            integration_method: How to integrate systems
            confidence_threshold: Threshold for invoking System 2
            use_system2_verification: Whether to verify System 1 with System 2
        """
        super().__init__()

        self.confidence_threshold = confidence_threshold
        self.use_system2_verification = use_system2_verification

        # System 1: Neural processing
        self.system1 = System1Module(
            node_feat_dim=node_feat_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )

        # System 2: Symbolic reasoning
        self.system2 = System2Module(reasoning_depth=reasoning_depth)

        # Integration layer
        self.integration = IntegrationLayer(embed_dim, integration_method)

        # Symbolic feature encoder (converts symbolic facts to neural features)
        self.symbolic_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

    def forward(self, node_features: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                edge_weights: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forward pass through dual-system architecture.

        Args:
            node_features: Node features
            adjacency: Adjacency matrix
            edge_weights: Edge weights
            context: Additional context for reasoning

        Returns:
            Dictionary with integrated output and metadata
        """
        # System 1: Fast neural processing
        system1_output = self.system1(node_features, adjacency, edge_weights)

        s1_embedding = system1_output['graph_embedding']
        s1_confidence = system1_output['confidence']

        # Decide whether to invoke System 2
        invoke_system2 = (s1_confidence < self.confidence_threshold).any()

        if invoke_system2 or self.use_system2_verification:
            # System 2: Symbolic reasoning
            if context is None:
                context = {}

            system2_output = self.system2(context)

            # Convert symbolic output to neural features
            # (simplified: use System 1 embedding as placeholder)
            s2_features = self.symbolic_encoder(s1_embedding)
            s2_confidence = system2_output.get('confidence', 0.5)

            # Integrate both systems
            integrated_output, final_confidence = self.integration(
                s1_embedding,
                s2_features,
                s1_confidence,
                torch.tensor(s2_confidence).to(s1_embedding.device)
            )

            mode = 'dual'
        else:
            # Use System 1 only
            integrated_output = s1_embedding
            final_confidence = s1_confidence
            system2_output = None
            mode = 'system1_only'

        return {
            'output': integrated_output,
            'confidence': final_confidence,
            'system1_output': system1_output,
            'system2_output': system2_output,
            'mode': mode
        }

    def train_system1(self, enable: bool = True):
        """Enable/disable training for System 1"""
        for param in self.system1.parameters():
            param.requires_grad = enable

    def add_knowledge(self, facts: List[Tuple[str, str, str]]):
        """Add facts to System 2 knowledge base"""
        self.system2.add_knowledge(facts)

    def explain_decision(self, output: Dict[str, Any]) -> str:
        """
        Generate explanation for a decision.

        Args:
            output: Output from forward pass

        Returns:
            Explanation string
        """
        mode = output['mode']
        confidence = output['confidence'].item() if torch.is_tensor(output['confidence']) else output['confidence']

        explanation = f"Decision made in {mode} mode with confidence {confidence:.3f}.\n"

        if mode == 'dual':
            explanation += "Both neural (System 1) and symbolic (System 2) reasoning were used.\n"

            if output['system2_output']:
                facts = output['system2_output'].get('facts', [])
                if facts:
                    explanation += f"System 2 retrieved {len(facts)} relevant facts.\n"
        else:
            explanation += "Only neural (System 1) processing was used.\n"

        return explanation


if __name__ == "__main__":
    # Test Dual-System Architecture
    logger.info("Testing Dual-System Architecture...")

    # Setup
    batch_size = 2
    num_nodes = 50
    node_feat_dim = 128
    embed_dim = 256

    # Create dummy data
    node_features = torch.randn(batch_size, num_nodes, node_feat_dim)
    adjacency = (torch.rand(batch_size, num_nodes, num_nodes) > 0.9).float()

    # Initialize architecture
    dual_system = DualSystemArchitecture(
        node_feat_dim=node_feat_dim,
        embed_dim=embed_dim,
        num_layers=3,
        num_heads=4,
        confidence_threshold=0.7
    )

    # Add some knowledge to System 2
    dual_system.add_knowledge([
        ("neural_network", "is_a", "machine_learning"),
        ("machine_learning", "is_a", "artificial_intelligence"),
        ("graph", "is_a", "data_structure")
    ])

    # Forward pass
    context = {'entity': 'neural_network'}
    output = dual_system(node_features, adjacency, context=context)

    print(f"\nOutput shape: {output['output'].shape}")
    print(f"Confidence: {output['confidence']}")
    print(f"Processing mode: {output['mode']}")

    # Get explanation
    explanation = dual_system.explain_decision(output)
    print(f"\nExplanation:\n{explanation}")

    # Test with high confidence (should use System 1 only)
    print("\n--- Testing with modified confidence threshold ---")
    dual_system.confidence_threshold = 0.3
    output2 = dual_system(node_features, adjacency)

    print(f"Processing mode: {output2['mode']}")

    logger.info("\nDual-system architecture tests complete!")

