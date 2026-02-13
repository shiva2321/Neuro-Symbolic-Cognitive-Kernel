#!/usr/bin/env python3
"""
NSCK Image Generator Demo
==========================
Interactive demonstration of VSA-based image generation.
Shows the generation process step-by-step.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from python.core.multimodal.image_generator import ImageGenerator
from python.core.memory.semantic_memory import SemanticMemory
from python.core.multimodal.multimodal_processor import MultimodalProcessor
from python.core.reasoning.context_engine import ContextEngine

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def print_banner():
    """Print demo banner."""
    print("\n" + "="*80)
    print(" "*20 + "NSCK IMAGE GENERATION DEMO")
    print(" "*15 + "VSA-Based Generation (NO Neural Networks)")
    print("="*80 + "\n")


def print_features(features):
    """Pretty print feature details."""
    print("\n📊 Extracted Features:")
    print(f"  • Shape: {', '.join(features.shape_hints)}")
    print(f"  • Brightness: {features.brightness:.1f}/255")
    print(f"  • Contrast: {features.contrast:.1f}")
    print(f"  • Edge Density: {features.edge_density:.2f}")
    print(f"  • Dominant Orientation: {features.dominant_orientation}")
    print(f"  • Texture: {features.texture_pattern}")
    print(f"  • Color Mode: {'Color' if features.is_color else 'Grayscale'}")


def print_result(result):
    """Pretty print generation result."""
    print("\n✨ Generation Complete!")
    print(f"  • Iterations: {result.iterations}")
    print(f"  • Confidence: {result.confidence:.1%}")
    print(f"  • Similarity: {result.similarity_to_target:.1%}")
    print(f"  • Image Size: {result.image.shape}")
    print(f"  • Data Type: {result.image.dtype}")
    print(f"  • Value Range: [{result.image.min()}, {result.image.max()}]")


def show_image_ascii(image_array, width=40):
    """Display a simple ASCII representation of the image."""
    if image_array.ndim == 3:
        # Convert to grayscale for ASCII
        gray = image_array.mean(axis=2)
    else:
        gray = image_array
    
    # Downsample to target width
    h, w = gray.shape
    aspect = h / w
    height = int(width * aspect * 0.5)  # 0.5 to account for character aspect ratio
    
    from scipy.ndimage import zoom
    scale_h = height / h
    scale_w = width / w
    downsampled = zoom(gray, (scale_h, scale_w), order=1)
    
    # ASCII characters from dark to light
    chars = " .:-=+*#%@"
    
    print("\n🖼️  ASCII Preview:")
    for row in downsampled:
        line = ""
        for val in row:
            idx = int((val / 255.0) * (len(chars) - 1))
            line += chars[idx]
        print("  " + line)


def demo_single_generation(generator, prompt, size=(64, 64)):
    """Demonstrate single image generation."""
    print("\n" + "-"*80)
    print(f"📝 Prompt: '{prompt}'")
    print(f"🎯 Target Size: {size[0]}×{size[1]}")
    print("-"*80)
    
    print("\n⚙️  Step 1: Parsing text prompt...")
    features = generator._text_to_features(prompt, size[0], size[1])
    print("  ✓ Extracted visual concepts from text")
    print_features(features)
    
    print("\n⚙️  Step 2: Synthesizing initial image...")
    image = generator._synthesize_from_features(features)
    print(f"  ✓ Generated {image.shape} array using procedural methods")
    print("    • No neural networks used")
    print("    • Classical CV features only")
    
    print("\n⚙️  Step 3: VSA refinement...")
    if generator.multimodal:
        print("  • Encoding image as hypervector")
        print("  • Comparing to target concept")
        print("  • Iteratively adjusting...")
    else:
        print("  • Skipped (no multimodal processor)")
    
    # Full generation with result
    result = generator.generate(prompt, size=size)
    print_result(result)
    
    # Try to show ASCII preview
    try:
        show_image_ascii(result.image)
    except:
        print("\n  (ASCII preview not available)")
    
    return result


def demo_vocabulary():
    """Show available vocabulary."""
    print("\n" + "="*80)
    print("📚 Available Vocabulary")
    print("="*80)
    
    generator = ImageGenerator()
    
    print("\n🎨 Colors:")
    colors = list(generator.color_vocab.keys())
    for i in range(0, len(colors), 7):
        print("  " + ", ".join(colors[i:i+7]))
    
    print("\n🎭 Textures:")
    print("  " + ", ".join(generator.texture_vocab.keys()))
    
    print("\n⬜ Shapes:")
    print("  " + ", ".join(generator.shape_vocab.keys()))
    
    print("\n💡 Brightness modifiers:")
    print("  bright, light, dark, dim")
    
    print("\n📐 Contrast modifiers:")
    print("  high_contrast, low_contrast, sharp, soft")


def demo_comparison():
    """Compare multiple generations."""
    print("\n" + "="*80)
    print("🔬 Comparison Demo")
    print("="*80)
    
    generator = ImageGenerator()
    
    prompts = [
        ("red circle", "Basic color + shape"),
        ("blue square", "Different color + shape"),
        ("bright green texture", "Brightness + texture"),
        ("dark smooth surface", "Brightness + texture contrast"),
    ]
    
    print("\nGenerating multiple images for comparison...\n")
    
    results = []
    for prompt, description in prompts:
        print(f"• {prompt:25s} → {description}")
        result = generator.generate(prompt, size=(48, 48))
        results.append((prompt, result))
        print(f"  Confidence: {result.confidence:.1%}, "
              f"Iterations: {result.iterations}")
    
    print("\n✓ All generations complete!")
    print(f"  Average confidence: {sum(r.confidence for _, r in results) / len(results):.1%}")
    print(f"  Total iterations: {sum(r.iterations for _, r in results)}")


def main():
    """Run the demo."""
    print_banner()
    
    print("Initializing NSCK Image Generator...")
    
    # Create generator with multimodal support
    semantic_memory = SemanticMemory()
    context_engine = ContextEngine(semantic_memory)
    multimodal_processor = MultimodalProcessor(
        context_engine=context_engine,
        semantic_memory=semantic_memory
    )
    
    generator = ImageGenerator(
        semantic_memory=semantic_memory,
        multimodal_processor=multimodal_processor
    )
    
    print("✓ Generator initialized\n")
    
    # Show vocabulary
    demo_vocabulary()
    
    # Demo generations
    print("\n" + "="*80)
    print("🎨 Generation Demonstrations")
    print("="*80)
    
    demo_prompts = [
        "red circle",
        "blue square",
        "bright green texture"
    ]
    
    for prompt in demo_prompts:
        result = demo_single_generation(generator, prompt, size=(64, 64))
    
    # Comparison
    demo_comparison()
    
    # Final summary
    print("\n" + "="*80)
    print("📋 Demo Summary")
    print("="*80)
    print("\n✓ Demonstrated NSCK image generation capabilities:")
    print("  • Text parsing and feature extraction")
    print("  • Classical CV feature mapping (HOG, color, texture)")
    print("  • Procedural image synthesis")
    print("  • VSA-based refinement")
    print("  • No neural networks used")
    print("\n✓ All processing done with:")
    print("  • Vector Symbolic Architecture (VSA)")
    print("  • Classical computer vision")
    print("  • Rule-based reasoning")
    print("  • Procedural generation")
    
    print("\n" + "="*80)
    print("Demo complete! Try generate_images.py for more features.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
