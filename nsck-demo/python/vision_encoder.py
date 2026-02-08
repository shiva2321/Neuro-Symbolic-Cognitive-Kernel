"""
NSCK Vision Encoder (Phase 2.1)
================================
Lightweight vision encoder for converting images to VSA concept representations.

Architecture:
  - Lightweight ConvNet encoder (≤128 dims for efficiency)
  - Feature extraction from images (28x28 MNIST, 32x32 CIFAR)
  - VSA concept binding for visual features
  - Integration with existing SNN and multimodal systems

Design Constraints (per AGENT_INSTRUCTIONS.md):
  - Efficiency first: O(n) operations, ≤128 dims
  - CPU-only compatible (4GB+ RAM)
  - Integrate with existing VSA infrastructure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


@dataclass
class VisualFeatures:
    """Extracted visual features from an image."""
    encoded_features: torch.Tensor  # [batch, feature_dim]
    concept_hv: Optional[Any] = None  # VSA HyperVector representation
    attention_map: Optional[torch.Tensor] = None  # [batch, H, W]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LightweightVisionEncoder(nn.Module):
    """
    Lightweight convolutional encoder for visual perception.
    
    Architecture:
      Input (28x28 or 32x32) → Conv layers → Pool → Linear → Features (64-128 dims)
    
    Designed for efficiency:
      - Small number of parameters (~50K)
      - CPU-friendly operations
      - Fast inference
    """
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 64,
                 input_size: int = 28):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            feature_dim: Output feature dimension (≤128 for efficiency)
            input_size: Input image size (28 for MNIST, 32 for CIFAR)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.input_size = input_size
        
        # Lightweight convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        # After 3 pooling operations: size / (2^3) = size / 8
        conv_output_size = (input_size // 8) ** 2 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the vision encoder.
        
        Args:
            x: Input images [batch, channels, height, width]
            return_attention: Whether to return attention map
            
        Returns:
            Tuple of (features, attention_map)
            - features: [batch, feature_dim]
            - attention_map: [batch, H, W] or None
        """
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # size / 2
        x = self.pool(F.relu(self.conv2(x)))  # size / 4
        x = self.pool(F.relu(self.conv3(x)))  # size / 8
        
        # Generate attention map if requested (before flattening)
        attention_map = None
        if return_attention:
            # Use last conv layer activations as attention
            attention_map = torch.mean(x, dim=1)  # [batch, H, W]
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)  # [batch, feature_dim]
        
        return features, attention_map


class VisualConceptMapper:
    """
    Maps visual features to VSA concept representations.
    
    Implements the visual perception → concept grounding pipeline:
      1. Extract neural features from image
      2. Bind features to spatial/semantic roles
      3. Create unified concept HyperVector
    """
    
    def __init__(self, feature_dim: int = 64, seed: int = 42):
        """
        Args:
            feature_dim: Dimension of visual features
            seed: Random seed for reproducibility
        """
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        # Role vectors for binding
        self.role_visual = self._create_role_hv("ROLE_VISUAL")
        self.role_spatial = self._create_role_hv("ROLE_SPATIAL")
        self.role_semantic = self._create_role_hv("ROLE_SEMANTIC")
        
        # Concept codebook: maps feature patterns to concept names
        self.concept_codebook: Dict[str, Any] = {}
        
    def _create_role_hv(self, name: str) -> Any:
        """Create a deterministic role HyperVector."""
        seed_value = int(name.encode().hex(), 16) % (2**32)
        return hypervec_rs.HyperVector(seed_value)
    
    def _features_to_hv(self, features: torch.Tensor) -> Any:
        """
        Convert continuous features to binary HyperVector.
        
        Uses random projection with thresholding.
        """
        # Convert to numpy
        feat_np = features.detach().cpu().numpy()
        
        # For each feature dimension, create a projection
        # This is a simplified approach - could be enhanced
        feat_flat = feat_np.flatten()
        
        # Create HV based on feature hash
        feat_hash = hash(tuple(np.round(feat_flat, decimals=2)))
        seed_val = abs(feat_hash) % (2**32)
        
        return hypervec_rs.HyperVector(seed_val)
    
    def map_to_concept(self, features: torch.Tensor, 
                      spatial_info: Optional[Dict[str, float]] = None,
                      class_label: Optional[str] = None) -> Any:
        """
        Map visual features to a concept HyperVector.
        
        Args:
            features: Visual features [batch, feature_dim] or [feature_dim]
            spatial_info: Optional spatial information (position, scale, etc.)
            class_label: Optional class label for supervised binding
            
        Returns:
            Concept HyperVector
        """
        # Ensure features are 1D
        if features.dim() > 1:
            features = features[0]  # Take first in batch
        
        # Convert features to HV
        feature_hv = self._features_to_hv(features)
        
        # Bind with visual role
        concept_hv = feature_hv .xor(self.role_visual
        
        # Add spatial information if provided
        if spatial_info:
            spatial_hv = self._create_spatial_hv(spatial_info)
            concept_hv = concept_hv .xor((spatial_hv .xor(self.role_spatial)
        
        # Add semantic label if provided
        if class_label:
            label_hv = self._create_role_hv(f"CLASS_{class_label}")
            concept_hv = concept_hv .xor((label_hv .xor(self.role_semantic)
        
        return concept_hv
    
    def _create_spatial_hv(self, spatial_info: Dict[str, float]) -> Any:
        """Create HV from spatial information."""
        # Simple approach: hash the spatial dict
        spatial_str = str(sorted(spatial_info.items()))
        seed_val = abs(hash(spatial_str)) % (2**32)
        return hypervec_rs.HyperVector(seed_val)
    
    def learn_concept(self, concept_name: str, features: torch.Tensor):
        """
        Store a concept in the codebook.
        
        Args:
            concept_name: Name of the concept (e.g., "digit_3", "cat")
            features: Representative features for this concept
        """
        concept_hv = self.map_to_concept(features)
        self.concept_codebook[concept_name] = concept_hv
    
    def recognize_concept(self, features: torch.Tensor, 
                         threshold: float = 0.5) -> Tuple[str, float]:
        """
        Recognize a concept by comparing to codebook.
        
        Args:
            features: Visual features to recognize
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (concept_name, similarity)
        """
        if not self.concept_codebook:
            return "UNKNOWN", 0.0
        
        query_hv = self.map_to_concept(features)
        
        best_match = "UNKNOWN"
        best_sim = 0.0
        
        for concept_name, stored_hv in self.concept_codebook.items():
            similarity = query_hv.similarity(stored_hv)
            if similarity > best_sim:
                best_sim = similarity
                best_match = concept_name
        
        if best_sim < threshold:
            return "UNKNOWN", best_sim
        
        return best_match, best_sim


class VisionPerceptionSystem:
    """
    Complete vision perception system integrating encoder and concept mapping.
    
    Pipeline:
      Image → Encoder → Features → ConceptMapper → VSA HyperVector
    """
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 64,
                 input_size: int = 28, device: str = "cpu"):
        """
        Args:
            input_channels: Number of input channels
            feature_dim: Feature dimension
            input_size: Input image size
            device: Device for computation
        """
        self.device = device
        
        # Create encoder
        self.encoder = LightweightVisionEncoder(
            input_channels=input_channels,
            feature_dim=feature_dim,
            input_size=input_size
        ).to(device)
        
        # Create concept mapper
        self.concept_mapper = VisualConceptMapper(feature_dim=feature_dim)
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "concepts_recognized": 0,
            "avg_confidence": 0.0
        }
    
    def perceive(self, image: torch.Tensor, 
                class_label: Optional[str] = None,
                return_attention: bool = False) -> VisualFeatures:
        """
        Perceive an image and convert to VSA concept representation.
        
        Args:
            image: Input image [batch, channels, H, W] or [channels, H, W]
            class_label: Optional class label for supervised binding
            return_attention: Whether to return attention map
            
        Returns:
            VisualFeatures with encoded features and concept HV
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Encode
        with torch.no_grad():
            features, attention_map = self.encoder(image, return_attention=return_attention)
        
        # Map to concept
        concept_hv = self.concept_mapper.map_to_concept(
            features, 
            class_label=class_label
        )
        
        # Try to recognize if we have a codebook
        recognized_concept, confidence = self.concept_mapper.recognize_concept(features)
        
        # Update statistics
        self.stats["images_processed"] += 1
        if recognized_concept != "UNKNOWN":
            self.stats["concepts_recognized"] += 1
        
        # Create result
        result = VisualFeatures(
            encoded_features=features,
            concept_hv=concept_hv,
            attention_map=attention_map,
            confidence=confidence,
            metadata={
                "recognized_concept": recognized_concept,
                "class_label": class_label
            }
        )
        
        return result
    
    def learn_from_examples(self, images: List[torch.Tensor], 
                          labels: List[str]) -> Dict[str, Any]:
        """
        Learn concepts from labeled examples.
        
        Args:
            images: List of images
            labels: Corresponding labels
            
        Returns:
            Learning statistics
        """
        learned_concepts = set()
        
        for image, label in zip(images, labels):
            # Perceive the image
            visual_features = self.perceive(image, class_label=label)
            
            # Store in codebook
            self.concept_mapper.learn_concept(label, visual_features.encoded_features)
            learned_concepts.add(label)
        
        return {
            "num_examples": len(images),
            "num_concepts": len(learned_concepts),
            "concepts": list(learned_concepts)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get perception statistics."""
        return self.stats.copy()


# Helper functions for integration
def create_vision_system(input_type: str = "mnist", device: str = "cpu") -> VisionPerceptionSystem:
    """
    Create a vision system configured for a specific input type.
    
    Args:
        input_type: "mnist" (28x28 grayscale) or "cifar" (32x32 RGB)
        device: Computation device
        
    Returns:
        Configured VisionPerceptionSystem
    """
    if input_type == "mnist":
        return VisionPerceptionSystem(
            input_channels=1,
            feature_dim=64,
            input_size=28,
            device=device
        )
    elif input_type == "cifar":
        return VisionPerceptionSystem(
            input_channels=3,
            feature_dim=64,
            input_size=32,
            device=device
        )
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
