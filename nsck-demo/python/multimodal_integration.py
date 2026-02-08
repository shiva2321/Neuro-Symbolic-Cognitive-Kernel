"""
NSCK Multimodal Integration (Phase 2.4)
========================================
Unified multimodal perception system integrating vision, audio, and language.

Architecture:
      Vision        Audio        Language
         ↓             ↓             ↓
    [Encoder]     [Encoder]     [Grounding]
         ↓             ↓             ↓
         └─────────────┴─────────────┘
                       ↓
              [Shared Latent Space]
                       ↓
                  [VSA Binding]
                       ↓
             [Unified Concept Space]

Design Constraints (per AGENT_INSTRUCTIONS.md):
  - Efficiency first: O(n) operations
  - VSA binding for grounding
  - CPU-only compatible
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

# Import perception modules
from vision_encoder import VisionPerceptionSystem, VisualFeatures, create_vision_system
from audio_encoder import AudioPerceptionSystem, AudioFeatures, create_audio_system
from language_grounding import EmbodiedLanguageLearner, create_language_grounding_system

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


@dataclass
class MultimodalPerception:
    """Complete multimodal perception result."""
    unified_hv: Any  # Unified VSA HyperVector
    visual_features: Optional[VisualFeatures] = None
    audio_features: Optional[AudioFeatures] = None
    language_grounding: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    modalities_present: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossModalAttention:
    """
    Cross-modal attention mechanism for multimodal fusion.
    
    Implements attention between different modalities to strengthen
    relevant features and suppress irrelevant ones.
    """
    
    def __init__(self, feature_dim: int = 64):
        """
        Args:
            feature_dim: Feature dimension for each modality
        """
        self.feature_dim = feature_dim
        
        # Role vectors for each modality
        self.role_vision = self._create_role_hv("ROLE_VISION_FUSION")
        self.role_audio = self._create_role_hv("ROLE_AUDIO_FUSION")
        self.role_language = self._create_role_hv("ROLE_LANGUAGE_FUSION")
        
    def _create_role_hv(self, name: str) -> Any:
        """Create a deterministic role HyperVector."""
        seed_value = int(name.encode().hex(), 16) % (2**32)
        return hypervec_rs.HyperVector(seed_value)
    
    def attend(self, query_hv: Any, key_hvs: List[Any], 
              value_hvs: List[Any]) -> Any:
        """
        Simplified attention mechanism using VSA similarity.
        
        Args:
            query_hv: Query HyperVector
            key_hvs: List of key HyperVectors
            value_hvs: List of value HyperVectors
            
        Returns:
            Attended HyperVector
        """
        if not key_hvs or not value_hvs:
            return query_hv
        
        # Compute similarities (attention weights)
        similarities = [query_hv.similarity(key) for key in key_hvs]
        
        # Normalize to get weights
        total_sim = sum(similarities) + 1e-10
        weights = [s / total_sim for s in similarities]
        
        # Weighted combination (simplified - in true VSA, use proper bundling)
        # For now, select the highest weight value
        max_idx = np.argmax(weights)
        attended_hv = value_hvs[max_idx]
        
        return attended_hv
    
    def cross_modal_fusion(self, vision_hv: Optional[Any] = None,
                          audio_hv: Optional[Any] = None,
                          language_hv: Optional[Any] = None) -> Any:
        """
        Fuse multiple modalities using cross-modal attention.
        
        Args:
            vision_hv: Visual HyperVector
            audio_hv: Audio HyperVector
            language_hv: Language HyperVector
            
        Returns:
            Fused multimodal HyperVector
        """
        # Collect present modalities
        modalities = []
        if vision_hv is not None:
            modalities.append(("vision", vision_hv, self.role_vision))
        if audio_hv is not None:
            modalities.append(("audio", audio_hv, self.role_audio))
        if language_hv is not None:
            modalities.append(("language", language_hv, self.role_language))
        
        if not modalities:
            # Return empty HV
            return hypervec_rs.HyperVector(0)
        
        if len(modalities) == 1:
            # Single modality - just bind with role
            _, hv, role = modalities[0]
            return hv ^ role
        
        # Multiple modalities - bind each with role and bundle
        bound_hvs = []
        for name, hv, role in modalities:
            bound_hv = hv ^ role
            bound_hvs.append(bound_hv)
        
        # Bundle all modalities (simplified - use first and XOR with others)
        fused_hv = bound_hvs[0]
        for hv in bound_hvs[1:]:
            fused_hv = fused_hv ^ hv
        
        return fused_hv


class MultimodalIntegrationSystem:
    """
    Complete multimodal integration system.
    
    Integrates vision, audio, and language into a unified concept space
    using VSA binding and cross-modal attention.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Computation device
        """
        self.device = device
        
        # Create perception systems
        self.vision_system = create_vision_system(input_type="mnist", device=device)
        self.audio_system = create_audio_system(sample_rate=16000, device=device)
        self.language_system = create_language_grounding_system()
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(feature_dim=64)
        
        # Statistics
        self.stats = {
            "total_perceptions": 0,
            "vision_only": 0,
            "audio_only": 0,
            "language_only": 0,
            "multimodal": 0
        }
    
    def perceive(self, 
                image: Optional[torch.Tensor] = None,
                waveform: Optional[torch.Tensor] = None,
                text: Optional[str] = None,
                use_cross_attention: bool = True) -> MultimodalPerception:
        """
        Perceive multimodal input and create unified concept representation.
        
        Args:
            image: Optional image tensor
            waveform: Optional audio waveform
            text: Optional text description
            use_cross_attention: Whether to use cross-modal attention
            
        Returns:
            MultimodalPerception with unified concept HV
        """
        # Track modalities
        modalities_present = []
        
        # Process each modality
        visual_features = None
        vision_hv = None
        if image is not None:
            visual_features = self.vision_system.perceive(image)
            vision_hv = visual_features.concept_hv
            modalities_present.append("vision")
        
        audio_features = None
        audio_hv = None
        if waveform is not None:
            audio_features = self.audio_system.perceive(waveform)
            audio_hv = audio_features.concept_hv
            modalities_present.append("audio")
        
        language_grounding = None
        language_hv = None
        if text is not None:
            language_hv = self.language_system.grounding_engine.ground_phrase(text)
            language_grounding = {
                "text": text,
                "grounded_hv": language_hv,
                "confidence": 1.0  # Simplified
            }
            modalities_present.append("language")
        
        # Fuse modalities
        if use_cross_attention:
            unified_hv = self.cross_modal_attention.cross_modal_fusion(
                vision_hv=vision_hv,
                audio_hv=audio_hv,
                language_hv=language_hv
            )
        else:
            # Simple binding without attention
            unified_hv = self._simple_fusion(vision_hv, audio_hv, language_hv)
        
        # Calculate overall confidence
        confidences = []
        if visual_features:
            confidences.append(visual_features.confidence)
        if audio_features:
            confidences.append(audio_features.confidence)
        if language_grounding:
            confidences.append(language_grounding["confidence"])
        
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Update statistics
        self.stats["total_perceptions"] += 1
        num_modalities = len(modalities_present)
        if num_modalities == 1:
            if "vision" in modalities_present:
                self.stats["vision_only"] += 1
            elif "audio" in modalities_present:
                self.stats["audio_only"] += 1
            elif "language" in modalities_present:
                self.stats["language_only"] += 1
        else:
            self.stats["multimodal"] += 1
        
        # Create result
        result = MultimodalPerception(
            unified_hv=unified_hv,
            visual_features=visual_features,
            audio_features=audio_features,
            language_grounding=language_grounding,
            confidence=float(overall_confidence),
            modalities_present=modalities_present,
            metadata={
                "num_modalities": num_modalities,
                "used_cross_attention": use_cross_attention
            }
        )
        
        return result
    
    def _simple_fusion(self, vision_hv: Optional[Any],
                      audio_hv: Optional[Any],
                      language_hv: Optional[Any]) -> Any:
        """Simple fusion by XOR binding."""
        hvs = [hv for hv in [vision_hv, audio_hv, language_hv] if hv is not None]
        
        if not hvs:
            return hypervec_rs.HyperVector(0)
        
        fused = hvs[0]
        for hv in hvs[1:]:
            fused = fused ^ hv
        
        return fused
    
    def learn_multimodal_concept(self, concept_name: str,
                                examples: List[Dict[str, Any]]):
        """
        Learn a multimodal concept from examples.
        
        Args:
            concept_name: Name of the concept
            examples: List of example dicts with 'image', 'waveform', 'text' keys
        """
        for example in examples:
            # Perceive the example
            perception = self.perceive(
                image=example.get("image"),
                waveform=example.get("waveform"),
                text=example.get("text")
            )
            
            # If we have vision + language, bind them
            if perception.visual_features and perception.language_grounding:
                self.language_system.observe_with_language(
                    perception_hv=perception.visual_features.concept_hv,
                    description=perception.language_grounding["text"]
                )
            
            # Learn in individual systems
            if example.get("image") is not None:
                self.vision_system.concept_mapper.learn_concept(
                    concept_name,
                    perception.visual_features.encoded_features
                )
            
            if example.get("waveform") is not None:
                self.audio_system.concept_mapper.learn_concept(
                    concept_name,
                    perception.audio_features.encoded_features
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.stats.copy()
        
        # Add individual system stats
        stats["vision_stats"] = self.vision_system.get_stats()
        stats["audio_stats"] = self.audio_system.get_stats()
        stats["language_stats"] = self.language_system.get_stats()
        
        return stats


# Helper function
def create_multimodal_system(device: str = "cpu") -> MultimodalIntegrationSystem:
    """
    Create a complete multimodal integration system.
    
    Args:
        device: Computation device
        
    Returns:
        Configured MultimodalIntegrationSystem
    """
    return MultimodalIntegrationSystem(device=device)
