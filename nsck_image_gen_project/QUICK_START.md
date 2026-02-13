# Quick Start Guide

Get started with NSCK image generation in 5 minutes!

## Installation

```bash
cd /path/to/Node_network
pip install numpy pillow networkx
```

## 1. Generate Your First Image

```bash
cd nsck_image_gen_project

# Generate a red circle
python generate_images.py "red circle"

# Output: results/generated_red_circle.png
```

## 2. Try Different Prompts

```bash
# Basic shapes and colors
python generate_images.py "blue square"
python generate_images.py "green triangle"

# With attributes
python generate_images.py "bright red texture"
python generate_images.py "dark smooth surface"

# Custom size
python generate_images.py "orange circle" --size 128
```

## 3. Interactive Mode

```bash
python generate_images.py --interactive

# Then type prompts:
> red circle
> blue airplane
> bright textured pattern
> quit
```

## 4. Run the Demo

```bash
python demo_image_generation.py

# Shows:
# - Available vocabulary
# - Step-by-step generation process
# - Feature extraction details
# - Multiple examples
```

## 5. Run Tests

```bash
python test_image_generator.py

# Verifies:
# ✓ Generator initialization
# ✓ Text parsing
# ✓ Image synthesis
# ✓ No neural networks
# ✓ All 8 tests pass
```

## 6. Train on Data (Optional)

```bash
# Train on CIFAR-10 dataset
python train_image_generator.py --samples 100

# This learns associations between:
# - Text labels (cat, dog, airplane, etc.)
# - Visual features from real images
```

## 7. Generate with Trained Model

```bash
# Use learned associations
python generate_images.py "cat" --model models/image_gen_system.pkl

# Or in interactive mode
python generate_images.py --interactive --model models/image_gen_system.pkl
```

## What You Can Generate

### Supported Keywords

**Colors:** red, blue, green, yellow, orange, purple, pink, brown, black, white, gray

**Shapes:** circle, square, rectangle, triangle, line, blob

**Textures:** smooth, textured, rough, detailed, simple

**Brightness:** bright, light, dark, dim

**Trained Labels** (after training): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Example Prompts

```
"red circle"              → Red circular shape
"blue square"             → Blue rectangular shape  
"bright green texture"    → Bright, textured pattern
"dark smooth surface"     → Dark, low-contrast image
"orange detailed pattern" → Orange with high detail
```

## Understanding the Output

Generated images are:
- Size: 64×64 by default (customize with --size)
- Format: PNG with RGB color
- Style: Abstract/procedural (not photorealistic)
- Method: Classical CV + VSA (no neural networks!)

## Next Steps

1. **Read the full README** for detailed documentation
2. **Check PROJECT_SUMMARY.md** for technical details
3. **Explore the code** in `nsck-demo/python/core/multimodal/image_generator.py`
4. **Train on more data** to expand vocabulary
5. **Customize vocabularies** to add new concepts

## Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'X'"
- **Solution:** `pip install X` (numpy, pillow, or networkx)

**Problem:** "Images are very simple/abstract"
- **Expected:** This system uses classical CV, not neural networks
- **Note:** Output is procedural, not photorealistic

**Problem:** "Training gives mock data warning"
- **Solution:** `pip install datasets` for real CIFAR-10 data
- **Alternative:** Use untrained generator (still works!)

## Architecture Note

This system uses:
- ✅ Vector Symbolic Architecture (VSA/hypervectors)
- ✅ Classical computer vision (HOG, color histograms)
- ✅ Procedural generation (rule-based)
- ❌ NO neural networks
- ❌ NO deep learning
- ❌ NO transformers/diffusion models

This is intentional! The NSCK architecture demonstrates cognitive capabilities without neural networks.

## Quick Reference

```bash
# Generate
python generate_images.py "PROMPT"              # Single image
python generate_images.py "PROMPT" --size N     # Custom size
python generate_images.py --interactive         # Interactive

# Train  
python train_image_generator.py --samples N     # Train on N samples

# Test
python test_image_generator.py                  # Run test suite
python demo_image_generation.py                 # Interactive demo

# Help
python generate_images.py --help                # Show all options
```

Happy generating! 🎨
