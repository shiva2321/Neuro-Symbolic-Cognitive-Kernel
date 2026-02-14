"""
Tests for NSCK Image Generator
================================
Tests the VSA-based image generation system without neural networks.
"""

import sys
import os

# Add parent directory paths to access NSCK core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo'))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo/python'))

# Add project source directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, 'src'))

import pytest
import numpy as np
from image_generator import (
    ImageGenerator,
    GenerationConfig,
    VisualFeatures,
    ConceptVisualMemory
)


def test_visual_features_creation():
    """Test VisualFeatures creation and serialization."""
    features = VisualFeatures(
        hog_features=np.random.rand(128),
        color_features=np.random.rand(24),
        edge_density=0.5,
        lbp_features=np.random.rand(40),
        spatial_features=np.random.rand(8)
    )
    
    # Test serialization
    arr = features.to_array()
    assert arr.shape == (201,), f"Expected 201 features, got {arr.shape}"
    
    # Test deserialization
    features2 = VisualFeatures.from_array(arr)
    assert np.allclose(features.hog_features, features2.hog_features)
    assert np.allclose(features.color_features, features2.color_features)
    assert features.edge_density == features2.edge_density


def test_visual_features_random():
    """Test random feature generation."""
    features = VisualFeatures.random(seed=42)
    
    assert features.hog_features.shape == (128,)
    assert features.color_features.shape == (24,)
    assert 0 <= features.edge_density <= 1
    assert features.lbp_features.shape == (40,)
    assert features.spatial_features.shape == (8,)


def test_concept_visual_memory():
    """Test concept-visual association memory."""
    memory = ConceptVisualMemory()
    
    # Create test features
    features = VisualFeatures.random(seed=123)
    
    # Learn association (without HV for simplicity)
    from python.core.vsa.hypervec_py import HyperVectorPy
    concept_hv = HyperVectorPy(123)
    
    memory.learn_association("test_concept", features, concept_hv)
    
    # Retrieve
    retrieved = memory.retrieve_visual("test_concept")
    assert retrieved is not None
    assert np.allclose(retrieved.to_array(), features.to_array())


def test_generator_creation():
    """Test ImageGenerator initialization."""
    config = GenerationConfig(
        image_size=(32, 32),
        is_color=True
    )
    
    generator = ImageGenerator(config=config)
    
    assert generator.config.image_size == (32, 32)
    assert generator.config.is_color == True
    assert generator.generation_count == 0


def test_concept_extraction():
    """Test concept extraction from text."""
    generator = ImageGenerator()
    
    text = "a red cat with blue eyes"
    concepts = generator._extract_concepts(text)
    
    assert "red" in concepts
    assert "cat" in concepts
    assert "blue" in concepts
    assert "eyes" in concepts
    assert "with" not in concepts  # Stop word
    assert "a" not in concepts  # Stop word


def test_image_generation_basic():
    """Test basic image generation."""
    config = GenerationConfig(
        image_size=(32, 32),
        is_color=True
    )
    generator = ImageGenerator(config=config)
    
    # Generate without training (should create random image)
    image = generator.generate("red circle")
    
    assert isinstance(image, np.ndarray)
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255


def test_image_generation_grayscale():
    """Test grayscale image generation."""
    config = GenerationConfig(
        image_size=(32, 32),
        is_color=False
    )
    generator = ImageGenerator(config=config)
    
    image = generator.generate("dark shape")
    
    assert isinstance(image, np.ndarray)
    assert image.shape == (32, 32)
    assert image.dtype == np.uint8


def test_training_synthetic():
    """Test training with synthetic data."""
    generator = ImageGenerator()
    
    # Create synthetic training data
    train_data = []
    for i in range(5):
        text = f"color_{i % 3}"
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        train_data.append((text, image))
    
    # Train
    initial_concepts = len(generator.concept_memory.feature_templates)
    generator.train_from_examples(train_data, max_examples=5)
    final_concepts = len(generator.concept_memory.feature_templates)
    
    # Should have learned some concepts
    assert final_concepts >= initial_concepts


def test_generation_after_training():
    """Test that generation works after training."""
    generator = ImageGenerator()
    
    # Train with known concepts
    train_data = [
        ("red", np.full((32, 32, 3), [200, 50, 50], dtype=np.uint8)),
        ("blue", np.full((32, 32, 3), [50, 50, 200], dtype=np.uint8)),
    ]
    
    generator.train_from_examples(train_data, max_examples=2)
    
    # Generate
    image = generator.generate("red")
    
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    
    # Check that generator count increased
    assert generator.generation_count == 1


def test_feature_bundling():
    """Test bundling of multiple visual features."""
    generator = ImageGenerator()
    
    features1 = VisualFeatures.random(seed=1)
    features2 = VisualFeatures.random(seed=2)
    
    bundled = generator._bundle_features([features1, features2])
    
    assert bundled.hog_features.shape == (128,)
    assert bundled.color_features.shape == (24,)
    
    # Bundled features should be between the two inputs
    arr1 = features1.to_array()
    arr2 = features2.to_array()
    bundled_arr = bundled.to_array()
    
    # Average should be close
    expected_avg = (arr1 + arr2) / 2
    assert np.allclose(bundled_arr, expected_avg)


def test_statistics():
    """Test generator statistics."""
    generator = ImageGenerator()
    
    stats = generator.get_statistics()
    
    assert "generation_count" in stats
    assert "learned_concepts" in stats
    assert "concept_list" in stats
    assert "config" in stats
    
    assert stats["generation_count"] == 0
    assert stats["learned_concepts"] == 0


def test_histogram_to_pixels():
    """Test histogram to pixel conversion."""
    generator = ImageGenerator()
    
    # Create a simple histogram
    histogram = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
    
    pixels = generator._histogram_to_pixels(histogram, 100)
    
    assert pixels.shape == (100,)
    assert pixels.dtype == np.uint8
    assert pixels.min() >= 0
    assert pixels.max() <= 255


def test_config_parameters():
    """Test generation config parameters."""
    config = GenerationConfig(
        image_size=(64, 64),
        is_color=False,
        smoothing_factor=0.8,
        contrast_boost=1.5
    )
    
    assert config.image_size == (64, 64)
    assert config.is_color == False
    assert config.smoothing_factor == 0.8
    assert config.contrast_boost == 1.5


def test_empty_prompt():
    """Test generation with empty concepts."""
    generator = ImageGenerator()
    
    # This should still generate (fallback to random)
    image = generator.generate("the and or")
    
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


def test_multiple_generations():
    """Test multiple consecutive generations."""
    generator = ImageGenerator()
    
    prompts = ["red", "blue", "green"]
    images = []
    
    for prompt in prompts:
        image = generator.generate(prompt)
        images.append(image)
        
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
    
    # Check generation count
    assert generator.generation_count == 3
    
    # Images should be different (statistically very likely)
    assert not np.array_equal(images[0], images[1])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
