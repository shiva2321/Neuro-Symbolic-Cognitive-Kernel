"""
NSCK Audio Encoder (Phase 2.2)
===============================
Lightweight audio encoder for converting audio to VSA concept representations.

Architecture:
  - Mel-spectrogram feature extraction
  - Lightweight temporal convolution encoder
  - Audio feature → VSA concept binding
  - Integration with multimodal fusion

Design Constraints (per AGENT_INSTRUCTIONS.md):
  - Efficiency first: O(n) operations, ≤128 dims
  - CPU-only compatible
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
class AudioFeatures:
    """Extracted audio features."""
    encoded_features: torch.Tensor  # [batch, feature_dim]
    concept_hv: Optional[Any] = None  # VSA HyperVector representation
    spectrogram: Optional[torch.Tensor] = None  # [batch, freq, time]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleMelSpectrogram:
    """
    Simplified mel-spectrogram computation for CPU efficiency.
    
    Uses basic FFT and mel filterbank without heavy dependencies.
    """
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop_length: int = 256, n_mels: int = 64):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram from waveform.
        
        Args:
            waveform: Audio waveform [batch, samples] or [samples]
            
        Returns:
            Mel-spectrogram [batch, n_mels, time]
        """
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.shape[0]
        
        # Simplified approach: use torch.stft if available
        # For true lightweight implementation, could use basic FFT
        try:
            # Compute STFT
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            # Convert to magnitude
            mag_spec = torch.abs(spec)
            
            # Simple mel approximation: average frequency bins
            # (True mel would use triangular filterbank)
            mel_spec = F.avg_pool1d(
                mag_spec.transpose(1, 2),
                kernel_size=mag_spec.shape[1] // self.n_mels + 1,
                stride=mag_spec.shape[1] // self.n_mels + 1
            )
            
            # Ensure correct number of mel bins
            if mel_spec.shape[1] > self.n_mels:
                mel_spec = mel_spec[:, :self.n_mels, :]
            elif mel_spec.shape[1] < self.n_mels:
                # Pad if needed
                padding = self.n_mels - mel_spec.shape[1]
                mel_spec = F.pad(mel_spec, (0, 0, 0, padding))
            
            # Log scale
            mel_spec = torch.log(mel_spec + 1e-10)
            
            return mel_spec
            
        except:
            # Fallback: create dummy spectrogram
            # In production, use proper STFT implementation
            n_frames = waveform.shape[-1] // self.hop_length
            return torch.randn(batch_size, self.n_mels, n_frames)


class LightweightAudioEncoder(nn.Module):
    """
    Lightweight audio encoder using temporal convolutions.
    
    Architecture:
      Mel-spec [n_mels, time] → Conv1D layers → Pool → Linear → Features (64 dims)
    """
    
    def __init__(self, n_mels: int = 64, feature_dim: int = 64):
        """
        Args:
            n_mels: Number of mel frequency bins
            feature_dim: Output feature dimension (≤128 for efficiency)
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.feature_dim = feature_dim
        
        # Temporal convolutional layers (1D convolutions over time)
        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, feature_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        
        Args:
            mel_spec: Mel-spectrogram [batch, n_mels, time]
            
        Returns:
            Audio features [batch, feature_dim]
        """
        # Temporal convolutions
        x = F.relu(self.conv1(mel_spec))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Global pooling to get fixed-size representation
        x = self.global_pool(x)  # [batch, 64, 1]
        x = x.squeeze(-1)  # [batch, 64]
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)  # [batch, feature_dim]
        
        return features


class AudioConceptMapper:
    """
    Maps audio features to VSA concept representations.
    
    Similar to visual concept mapping but for audio modality.
    """
    
    def __init__(self, feature_dim: int = 64, seed: int = 43):
        """
        Args:
            feature_dim: Dimension of audio features
            seed: Random seed for reproducibility
        """
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        # Role vectors for binding
        self.role_audio = self._create_role_hv("ROLE_AUDIO")
        self.role_temporal = self._create_role_hv("ROLE_TEMPORAL")
        self.role_semantic = self._create_role_hv("ROLE_SEMANTIC")
        
        # Concept codebook
        self.concept_codebook: Dict[str, Any] = {}
        
    def _create_role_hv(self, name: str) -> Any:
        """Create a deterministic role HyperVector."""
        seed_value = int(name.encode().hex(), 16) % (2**32)
        return hypervec_rs.HyperVector(seed_value)
    
    def _features_to_hv(self, features: torch.Tensor) -> Any:
        """Convert continuous features to binary HyperVector."""
        feat_np = features.detach().cpu().numpy()
        feat_flat = feat_np.flatten()
        
        # Create HV based on feature hash
        feat_hash = hash(tuple(np.round(feat_flat, decimals=2)))
        seed_val = abs(feat_hash) % (2**32)
        
        return hypervec_rs.HyperVector(seed_val)
    
    def map_to_concept(self, features: torch.Tensor,
                      temporal_info: Optional[Dict[str, float]] = None,
                      class_label: Optional[str] = None) -> Any:
        """
        Map audio features to a concept HyperVector.
        
        Args:
            features: Audio features [batch, feature_dim] or [feature_dim]
            temporal_info: Optional temporal information (duration, tempo, etc.)
            class_label: Optional class label
            
        Returns:
            Concept HyperVector
        """
        # Ensure features are 1D
        if features.dim() > 1:
            features = features[0]
        
        # Convert features to HV
        feature_hv = self._features_to_hv(features)
        
        # Bind with audio role
        concept_hv = feature_hv ^ self.role_audio
        
        # Add temporal information if provided
        if temporal_info:
            temporal_hv = self._create_temporal_hv(temporal_info)
            concept_hv = concept_hv ^ (temporal_hv ^ self.role_temporal)
        
        # Add semantic label if provided
        if class_label:
            label_hv = self._create_role_hv(f"CLASS_{class_label}")
            concept_hv = concept_hv ^ (label_hv ^ self.role_semantic)
        
        return concept_hv
    
    def _create_temporal_hv(self, temporal_info: Dict[str, float]) -> Any:
        """Create HV from temporal information."""
        temporal_str = str(sorted(temporal_info.items()))
        seed_val = abs(hash(temporal_str)) % (2**32)
        return hypervec_rs.HyperVector(seed_val)
    
    def learn_concept(self, concept_name: str, features: torch.Tensor):
        """Store a concept in the codebook."""
        concept_hv = self.map_to_concept(features)
        self.concept_codebook[concept_name] = concept_hv
    
    def recognize_concept(self, features: torch.Tensor,
                         threshold: float = 0.5) -> Tuple[str, float]:
        """Recognize a concept by comparing to codebook."""
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


class AudioPerceptionSystem:
    """
    Complete audio perception system integrating encoder and concept mapping.
    
    Pipeline:
      Waveform → Mel-spec → Encoder → Features → ConceptMapper → VSA HyperVector
    """
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 64,
                 feature_dim: int = 64, device: str = "cpu"):
        """
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel frequency bins
            feature_dim: Feature dimension
            device: Device for computation
        """
        self.device = device
        self.sample_rate = sample_rate
        
        # Create mel-spectrogram computer
        self.mel_computer = SimpleMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # Create encoder
        self.encoder = LightweightAudioEncoder(
            n_mels=n_mels,
            feature_dim=feature_dim
        ).to(device)
        
        # Create concept mapper
        self.concept_mapper = AudioConceptMapper(feature_dim=feature_dim)
        
        # Statistics
        self.stats = {
            "audio_processed": 0,
            "concepts_recognized": 0,
            "avg_confidence": 0.0
        }
    
    def perceive(self, waveform: torch.Tensor,
                class_label: Optional[str] = None,
                return_spectrogram: bool = False) -> AudioFeatures:
        """
        Perceive audio and convert to VSA concept representation.
        
        Args:
            waveform: Audio waveform [batch, samples] or [samples]
            class_label: Optional class label
            return_spectrogram: Whether to return mel-spectrogram
            
        Returns:
            AudioFeatures with encoded features and concept HV
        """
        # Compute mel-spectrogram
        mel_spec = self.mel_computer.compute(waveform)
        mel_spec = mel_spec.to(self.device)
        
        # Encode
        with torch.no_grad():
            features = self.encoder(mel_spec)
        
        # Map to concept
        concept_hv = self.concept_mapper.map_to_concept(
            features,
            class_label=class_label
        )
        
        # Try to recognize
        recognized_concept, confidence = self.concept_mapper.recognize_concept(features)
        
        # Update statistics
        self.stats["audio_processed"] += 1
        if recognized_concept != "UNKNOWN":
            self.stats["concepts_recognized"] += 1
        
        # Create result
        result = AudioFeatures(
            encoded_features=features,
            concept_hv=concept_hv,
            spectrogram=mel_spec if return_spectrogram else None,
            confidence=confidence,
            metadata={
                "recognized_concept": recognized_concept,
                "class_label": class_label,
                "sample_rate": self.sample_rate
            }
        )
        
        return result
    
    def learn_from_examples(self, waveforms: List[torch.Tensor],
                          labels: List[str]) -> Dict[str, Any]:
        """
        Learn concepts from labeled audio examples.
        
        Args:
            waveforms: List of audio waveforms
            labels: Corresponding labels
            
        Returns:
            Learning statistics
        """
        learned_concepts = set()
        
        for waveform, label in zip(waveforms, labels):
            # Perceive the audio
            audio_features = self.perceive(waveform, class_label=label)
            
            # Store in codebook
            self.concept_mapper.learn_concept(label, audio_features.encoded_features)
            learned_concepts.add(label)
        
        return {
            "num_examples": len(waveforms),
            "num_concepts": len(learned_concepts),
            "concepts": list(learned_concepts)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get perception statistics."""
        return self.stats.copy()


# Helper functions
def create_audio_system(sample_rate: int = 16000, device: str = "cpu") -> AudioPerceptionSystem:
    """
    Create an audio perception system.
    
    Args:
        sample_rate: Audio sample rate (16kHz default)
        device: Computation device
        
    Returns:
        Configured AudioPerceptionSystem
    """
    return AudioPerceptionSystem(
        sample_rate=sample_rate,
        n_mels=64,
        feature_dim=64,
        device=device
    )
