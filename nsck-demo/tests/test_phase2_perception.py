"""
Tests for Phase 2 Perception Systems.

Tests:
  - Vision encoder with VSA integration
  - Audio encoder with mel-spectrogram
  - Language grounding and symbol binding
  - Multimodal integration
"""
import unittest
import importlib
import torch
import numpy as np

_missing = []
for _mod in [
    "vision_encoder",
    "audio_encoder",
    "language_grounding",
    "multimodal_integration",
]:
    if importlib.util.find_spec(_mod) is None:
        _missing.append(_mod)

if _missing:
    raise unittest.SkipTest(
        f"Missing optional perception modules: {', '.join(_missing)}"
    )

from vision_encoder import (
    LightweightVisionEncoder, VisualConceptMapper, VisionPerceptionSystem,
    create_vision_system
)
from audio_encoder import (
    SimpleMelSpectrogram, LightweightAudioEncoder, AudioConceptMapper,
    AudioPerceptionSystem, create_audio_system
)
from language_grounding import (
    SymbolGroundingEngine, VisionLanguageBinding, EmbodiedLanguageLearner,
    create_language_grounding_system
)
from multimodal_integration import (
    CrossModalAttention, MultimodalIntegrationSystem, create_multimodal_system
)

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector
    class _Shim:
        HyperVector = HyperVector
    hypervec_rs = _Shim()


class TestVisionEncoder(unittest.TestCase):
    """Test vision encoding and VSA integration."""
    
    def test_lightweight_encoder(self):
        """Test lightweight vision encoder forward pass."""
        encoder = LightweightVisionEncoder(
            input_channels=1,
            feature_dim=64,
            input_size=28
        )
        
        # Test forward pass
        x = torch.randn(4, 1, 28, 28)
        features, attention_map = encoder(x, return_attention=True)
        
        self.assertEqual(features.shape, (4, 64), "Feature shape should be [4, 64]")
        self.assertIsNotNone(attention_map, "Should return attention map")
        self.assertTrue(torch.isfinite(features).all(), "Features should be finite")
    
    def test_visual_concept_mapper(self):
        """Test mapping visual features to VSA concepts."""
        mapper = VisualConceptMapper(feature_dim=64)
        
        # Create dummy features
        features = torch.randn(64)
        
        # Map to concept
        concept_hv = mapper.map_to_concept(features, class_label="digit_5")
        
        # Check that we got a HyperVector
        self.assertIsNotNone(concept_hv, "Should return concept HV")
        
        # Learn and recognize
        mapper.learn_concept("digit_5", features)
        recognized, sim = mapper.recognize_concept(features)
        
        self.assertEqual(recognized, "digit_5", "Should recognize learned concept")
        self.assertGreater(sim, 0.9, "Similarity should be high for same features")
    
    def test_vision_perception_system(self):
        """Test complete vision perception pipeline."""
        system = create_vision_system(input_type="mnist")
        
        # Create dummy MNIST-like image
        image = torch.randn(1, 28, 28)
        
        # Perceive
        result = system.perceive(image, class_label="digit_3")
        
        self.assertIsNotNone(result.encoded_features, "Should have features")
        self.assertIsNotNone(result.concept_hv, "Should have concept HV")
        self.assertEqual(result.metadata["class_label"], "digit_3")
    
    def test_vision_learning(self):
        """Test learning concepts from examples."""
        system = create_vision_system(input_type="mnist")
        
        # Create synthetic examples
        images = [torch.randn(1, 28, 28) for _ in range(5)]
        labels = ["digit_0", "digit_1", "digit_2", "digit_0", "digit_1"]
        
        # Learn
        stats = system.learn_from_examples(images, labels)
        
        self.assertEqual(stats["num_examples"], 5)
        self.assertEqual(stats["num_concepts"], 3)  # 3 unique labels


class TestAudioEncoder(unittest.TestCase):
    """Test audio encoding and VSA integration."""
    
    def test_mel_spectrogram(self):
        """Test mel-spectrogram computation."""
        mel_computer = SimpleMelSpectrogram(
            sample_rate=16000,
            n_mels=64
        )
        
        # Create dummy audio
        waveform = torch.randn(16000)  # 1 second at 16kHz
        
        # Compute spectrogram
        mel_spec = mel_computer.compute(waveform)
        
        self.assertEqual(mel_spec.shape[1], 64, "Should have 64 mel bins")
        self.assertTrue(torch.isfinite(mel_spec).all(), "Mel-spec should be finite")
    
    def test_audio_encoder(self):
        """Test audio encoder forward pass."""
        encoder = LightweightAudioEncoder(n_mels=64, feature_dim=64)
        
        # Create dummy mel-spectrogram
        mel_spec = torch.randn(4, 64, 100)  # [batch, mels, time]
        
        # Encode
        features = encoder(mel_spec)
        
        self.assertEqual(features.shape, (4, 64), "Feature shape should be [4, 64]")
        self.assertTrue(torch.isfinite(features).all(), "Features should be finite")
    
    def test_audio_perception_system(self):
        """Test complete audio perception pipeline."""
        system = create_audio_system()
        
        # Create dummy audio
        waveform = torch.randn(16000)
        
        # Perceive
        result = system.perceive(waveform, class_label="speech")
        
        self.assertIsNotNone(result.encoded_features)
        self.assertIsNotNone(result.concept_hv)
        self.assertEqual(result.metadata["class_label"], "speech")


class TestLanguageGrounding(unittest.TestCase):
    """Test language grounding and symbol binding."""
    
    def test_symbol_grounding_engine(self):
        """Test basic symbol grounding."""
        engine = SymbolGroundingEngine()
        
        # Create dummy perception HV
        perception_hv = hypervec_rs.HyperVector(42)
        
        # Add experiences
        engine.add_experience(
            words=["apple", "red", "fruit"],
            perception_hv=perception_hv
        )
        
        # Check grounding
        apple_hv, confidence = engine.ground_word("apple")
        
        self.assertIsNotNone(apple_hv, "Should return grounded HV")
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be non-negative")
        
        # Check stats
        stats = engine.get_stats()
        self.assertGreater(stats["words_grounded"], 0)
        self.assertGreater(stats["experiences_collected"], 0)
    
    def test_vision_language_binding(self):
        """Test CLIP-style vision-language binding."""
        binder = VisionLanguageBinding()
        
        # Create dummy vision HV
        image_hv = hypervec_rs.HyperVector(123)
        
        # Bind with text
        binding = binder.bind_vision_language(
            image_hv=image_hv,
            text="a red apple on a table"
        )
        
        self.assertEqual(binding.text, "a red apple on a table")
        self.assertIsNotNone(binding.visual_hv)
        self.assertIsNotNone(binding.unified_hv)
        self.assertGreater(binding.confidence, 0.0)
    
    def test_embodied_language_learning(self):
        """Test embodied language learning."""
        learner = create_language_grounding_system()
        
        # Simulate embodied experiences
        perception_hv = hypervec_rs.HyperVector(456)
        
        learner.observe_with_language(
            perception_hv=perception_hv,
            description="move forward to reach the goal",
            action="move_forward"
        )
        
        # Try to understand instruction
        understanding = learner.understand_instruction("move forward")
        
        self.assertIn("instruction", understanding)
        self.assertIn("grounded_hv", understanding)
        self.assertIn("word_groundings", understanding)
        self.assertGreaterEqual(understanding["confidence"], 0.0)


class TestMultimodalIntegration(unittest.TestCase):
    """Test multimodal integration."""
    
    def test_cross_modal_attention(self):
        """Test cross-modal attention mechanism."""
        attention = CrossModalAttention(feature_dim=64)
        
        # Create dummy HVs
        vision_hv = hypervec_rs.HyperVector(100)
        audio_hv = hypervec_rs.HyperVector(101)
        language_hv = hypervec_rs.HyperVector(102)
        
        # Fuse modalities
        fused_hv = attention.cross_modal_fusion(
            vision_hv=vision_hv,
            audio_hv=audio_hv,
            language_hv=language_hv
        )
        
        self.assertIsNotNone(fused_hv, "Should return fused HV")
    
    def test_multimodal_perception_vision_only(self):
        """Test multimodal perception with vision only."""
        system = create_multimodal_system()
        
        # Create image
        image = torch.randn(1, 28, 28)
        
        # Perceive
        result = system.perceive(image=image)
        
        self.assertIsNotNone(result.unified_hv)
        self.assertIn("vision", result.modalities_present)
        self.assertIsNotNone(result.visual_features)
        self.assertEqual(len(result.modalities_present), 1)
    
    def test_multimodal_perception_vision_language(self):
        """Test multimodal perception with vision + language."""
        system = create_multimodal_system()
        
        # Create inputs
        image = torch.randn(1, 28, 28)
        text = "a handwritten digit"
        
        # Perceive
        result = system.perceive(image=image, text=text)
        
        self.assertIsNotNone(result.unified_hv)
        self.assertIn("vision", result.modalities_present)
        self.assertIn("language", result.modalities_present)
        self.assertIsNotNone(result.visual_features)
        self.assertIsNotNone(result.language_grounding)
        self.assertEqual(len(result.modalities_present), 2)
    
    def test_multimodal_perception_all_modalities(self):
        """Test multimodal perception with all modalities."""
        system = create_multimodal_system()
        
        # Create inputs
        image = torch.randn(1, 28, 28)
        waveform = torch.randn(16000)
        text = "spoken digit with visual display"
        
        # Perceive
        result = system.perceive(image=image, waveform=waveform, text=text)
        
        self.assertIsNotNone(result.unified_hv)
        self.assertEqual(len(result.modalities_present), 3)
        self.assertIn("vision", result.modalities_present)
        self.assertIn("audio", result.modalities_present)
        self.assertIn("language", result.modalities_present)
        self.assertGreater(result.confidence, 0.0)
    
    def test_multimodal_learning(self):
        """Test learning multimodal concepts."""
        system = create_multimodal_system()
        
        # Create example
        examples = [
            {
                "image": torch.randn(1, 28, 28),
                "text": "digit zero"
            },
            {
                "image": torch.randn(1, 28, 28),
                "text": "number zero"
            }
        ]
        
        # Learn concept
        system.learn_multimodal_concept("zero", examples)
        
        # Check stats
        stats = system.get_stats()
        self.assertGreater(stats["total_perceptions"], 0)


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 components."""
    
    def test_vision_to_language_pipeline(self):
        """Test vision → concept → language pipeline."""
        # Create systems
        vision_system = create_vision_system()
        language_system = create_language_grounding_system()
        
        # Perceive image
        image = torch.randn(1, 28, 28)
        visual_result = vision_system.perceive(image, class_label="digit_5")
        
        # Ground with language
        language_system.observe_with_language(
            perception_hv=visual_result.concept_hv,
            description="the digit five"
        )
        
        # Try to understand
        understanding = language_system.understand_instruction("digit five")
        
        self.assertIsNotNone(understanding["grounded_hv"])
        self.assertIn("digit", understanding["word_groundings"])
    
    def test_complete_multimodal_pipeline(self):
        """Test complete multimodal perception pipeline."""
        system = create_multimodal_system()
        
        # Process multiple inputs
        results = []
        for i in range(3):
            result = system.perceive(
                image=torch.randn(1, 28, 28),
                text=f"example {i}"
            )
            results.append(result)
        
        # Check all succeeded
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.unified_hv)
            self.assertGreater(result.confidence, 0.0)
        
        # Check stats
        stats = system.get_stats()
        self.assertEqual(stats["total_perceptions"], 3)
        self.assertEqual(stats["multimodal"], 3)


if __name__ == "__main__":
    unittest.main()
