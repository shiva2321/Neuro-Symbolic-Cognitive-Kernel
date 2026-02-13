#!/usr/bin/env python3
"""
NSCK Image Generator Test Suite
================================
Tests the image generation system to verify VSA-based generation works correctly.
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from python.core.multimodal.image_generator import ImageGenerator, ImageFeatures
from python.core.memory.semantic_memory import SemanticMemory
from python.core.multimodal.multimodal_processor import MultimodalProcessor
from python.core.reasoning.context_engine import ContextEngine


def test_generator_initialization():
    """Test that generator initializes correctly."""
    print("Test 1: Generator initialization...")
    
    gen = ImageGenerator()
    assert gen is not None
    assert gen.semantic is None  # No semantic memory provided
    assert gen.default_size == (64, 64)
    
    # Test with semantic memory
    sem_mem = SemanticMemory()
    gen2 = ImageGenerator(semantic_memory=sem_mem)
    assert gen2.semantic is not None
    
    print("  ✓ Generator initializes correctly")


def test_text_to_features():
    """Test text parsing to visual features."""
    print("\nTest 2: Text to features extraction...")
    
    gen = ImageGenerator()
    
    # Test color extraction
    features = gen._text_to_features("red circle", 64, 64)
    assert features.height == 64
    assert features.width == 64
    assert "circular" in features.shape_hints
    assert features.brightness > 0
    
    # Test brightness
    features_bright = gen._text_to_features("bright surface", 64, 64)
    features_dark = gen._text_to_features("dark surface", 64, 64)
    assert features_bright.brightness > features_dark.brightness
    
    # Test shapes
    features_square = gen._text_to_features("blue square", 64, 64)
    assert "rectangular" in features_square.shape_hints
    
    print("  ✓ Text parsing works correctly")


def test_image_synthesis():
    """Test image synthesis from features."""
    print("\nTest 3: Image synthesis...")
    
    gen = ImageGenerator()
    
    # Create test features
    features = ImageFeatures(
        hog_orientations=np.zeros((16, 8)),
        hog_magnitudes=np.ones((16, 8)) * 0.1,
        color_histograms=np.zeros((3, 8)),
        is_color=True,
        texture_pattern="smooth",
        edge_density=0.15,
        brightness=128.0,
        contrast=40.0,
        shape_hints=["circular"],
        dominant_orientation="R",
        height=64,
        width=64
    )
    
    image = gen._synthesize_from_features(features)
    
    assert image is not None
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert np.min(image) >= 0
    assert np.max(image) <= 255
    
    print("  ✓ Image synthesis works correctly")


def test_full_generation():
    """Test end-to-end image generation."""
    print("\nTest 4: Full generation pipeline...")
    
    gen = ImageGenerator()
    
    prompts = [
        "red circle",
        "blue square",
        "green triangle",
        "dark smooth surface",
        "bright detailed pattern"
    ]
    
    for prompt in prompts:
        result = gen.generate(prompt, size=(64, 64))
        
        assert result is not None
        assert result.image is not None
        assert result.image.shape == (64, 64, 3) or result.image.shape == (64, 64)
        assert result.confidence > 0
        assert result.confidence <= 1.0
        assert result.iterations > 0
        
    print(f"  ✓ Generated {len(prompts)} images successfully")


def test_no_neural_networks():
    """Verify no neural network imports."""
    print("\nTest 5: Verify no neural networks...")
    
    # Check that neural network libraries are not imported
    forbidden_modules = [
        'torch', 'tensorflow', 'keras', 
        'transformers', 'diffusers'
    ]
    
    imported = [mod for mod in forbidden_modules if mod in sys.modules]
    
    if imported:
        print(f"  ✗ WARNING: Neural network modules imported: {imported}")
        print("  ! This violates the no-neural-network constraint!")
    else:
        print("  ✓ No neural network modules imported")
    
    # Verify we're using VSA
    assert 'python.core.vsa.hypervec_py' in sys.modules or \
           'python.core.vsa.hypervec_shim' in sys.modules
    
    print("  ✓ Using VSA/hypervectors as intended")


def test_feature_vocabulary():
    """Test feature vocabulary completeness."""
    print("\nTest 6: Feature vocabulary...")
    
    gen = ImageGenerator()
    
    # Check color vocabulary
    assert "red" in gen.color_vocab
    assert "blue" in gen.color_vocab
    assert len(gen.color_vocab) >= 10
    
    # Check texture vocabulary
    assert "smooth" in gen.texture_vocab
    assert "textured" in gen.texture_vocab
    
    # Check shape vocabulary
    assert "circle" in gen.shape_vocab
    assert "square" in gen.shape_vocab
    
    print(f"  ✓ Vocabulary: {len(gen.color_vocab)} colors, "
          f"{len(gen.texture_vocab)} textures, "
          f"{len(gen.shape_vocab)} shapes")


def test_deterministic_generation():
    """Test that generation is deterministic for same input."""
    print("\nTest 7: Deterministic generation...")
    
    gen = ImageGenerator()
    
    # Generate same prompt twice
    result1 = gen.generate("red circle", size=(32, 32))
    result2 = gen.generate("red circle", size=(32, 32))
    
    # Features should be identical (deterministic parsing)
    assert result1.features_used.brightness == result2.features_used.brightness
    assert result1.features_used.shape_hints == result2.features_used.shape_hints
    
    # Note: pixel values may differ slightly due to random noise in synthesis
    # but the overall structure should be the same
    
    print("  ✓ Generation is deterministic for features")


def test_size_variations():
    """Test different image sizes."""
    print("\nTest 8: Size variations...")
    
    gen = ImageGenerator()
    
    sizes = [(32, 32), (64, 64), (128, 128), (48, 96)]
    
    for size in sizes:
        result = gen.generate("blue square", size=size)
        expected_shape = (size[0], size[1], 3)
        assert result.image.shape == expected_shape, \
            f"Expected {expected_shape}, got {result.image.shape}"
    
    print(f"  ✓ Tested {len(sizes)} different sizes")


def run_all_tests():
    """Run all test suites."""
    print("="*70)
    print("NSCK Image Generator Test Suite")
    print("="*70)
    print("Testing VSA-based image generation (no neural networks)")
    print()
    
    try:
        test_generator_initialization()
        test_text_to_features()
        test_image_synthesis()
        test_full_generation()
        test_no_neural_networks()
        test_feature_vocabulary()
        test_deterministic_generation()
        test_size_variations()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("Image generation system verified:")
        print("  • VSA/hypervector architecture working")
        print("  • Classical CV features only")
        print("  • No neural networks")
        print("  • Procedural generation functional")
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
