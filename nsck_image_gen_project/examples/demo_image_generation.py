#!/usr/bin/env python3
"""
NSCK Image Generation - Quick Demo
===================================
Demonstrates the image generation system with a simple example.
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

import numpy as np
from PIL import Image

from image_generator import ImageGenerator, GenerationConfig

def main():
    print("=" * 60)
    print("NSCK Image Generation - Quick Demo")
    print("=" * 60)
    print("NO NEURAL NETWORKS - Pure VSA/Hypervector approach")
    print()
    
    # Create generator
    print("1. Creating image generator...")
    config = GenerationConfig(
        image_size=(32, 32),
        is_color=True,
        smoothing_factor=0.3,
        contrast_boost=1.1
    )
    generator = ImageGenerator(config=config)
    print("   ✓ Generator created")
    print()
    
    # Create simple training data
    print("2. Creating training data (synthetic)...")
    train_data = []
    
    # Red images
    red_img = np.full((32, 32, 3), [200, 50, 50], dtype=np.uint8)
    train_data.append(("red", red_img))
    
    # Blue images
    blue_img = np.full((32, 32, 3), [50, 50, 200], dtype=np.uint8)
    train_data.append(("blue", blue_img))
    
    # Green images
    green_img = np.full((32, 32, 3), [50, 200, 50], dtype=np.uint8)
    train_data.append(("green", green_img))
    
    print(f"   ✓ Created {len(train_data)} training examples")
    print()
    
    # Train
    print("3. Training generator...")
    generator.train_from_examples(train_data, max_examples=len(train_data))
    stats = generator.get_statistics()
    print(f"   ✓ Training complete")
    print(f"   ✓ Learned {stats['learned_concepts']} concepts")
    print()
    
    # Generate images
    print("4. Generating images...")
    prompts = [
        "red circle",
        "blue square",
        "green triangle"
    ]
    
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"   [{i+1}/{len(prompts)}] Generating: '{prompt}'")
        
        # Generate
        image = generator.generate(prompt)
        
        # Save
        pil_img = Image.fromarray(image)
        filename = f"{output_dir}/demo_{i+1}_{prompt.replace(' ', '_')}.png"
        pil_img.save(filename)
        
        print(f"       ✓ Saved to: {filename}")
    
    print()
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Generated images saved to: {output_dir}/")
    print(f"Generator statistics:")
    print(f"  - Images generated: {generator.generation_count}")
    print(f"  - Concepts learned: {stats['learned_concepts']}")
    print(f"  - Learned concepts: {', '.join(stats['concept_list'])}")
    print()
    print("To view the images, check the demo_outputs/ directory")
    print()
    print("For interactive generation, run:")
    print("  python3 launch_image_dashboard.py")
    print()


if __name__ == "__main__":
    main()
