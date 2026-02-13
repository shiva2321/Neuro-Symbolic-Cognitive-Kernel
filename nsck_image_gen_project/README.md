# NSCK Image Generation Project

A complete image generation system based on the NSCK (Neuro-Symbolic Cognitive Kernel) architecture using Vector Symbolic Architecture (VSA) and classical signal processing.

**IMPORTANT:** This system does NOT use neural networks or LLMs. All image generation is performed using VSA/hypervectors and classical computer vision techniques.

## Overview

This project provides:
- **Image generation from text** using VSA/hypervectors
- **Training pipeline** to learn text-to-image associations from CIFAR-10
- **Interactive generation** with real-time text-to-image conversion
- **Classical CV features** (HOG, color histograms, texture patterns, edge detection)
- **NO neural networks** - pure symbolic/VSA approach

## Architecture

### Core Components

1. **Image Generator** (`nsck-demo/python/core/multimodal/image_generator.py`)
   - Converts text prompts to visual features
   - Uses rule-based parsing and vocabulary matching
   - Synthesizes images from classical CV features
   - Iterative refinement with VSA similarity feedback

2. **Training Pipeline** (`train_image_generator.py`)
   - Learns text-image associations from CIFAR-10
   - Stores associations in semantic memory using hypervectors
   - Builds vocabulary of visual concepts

3. **Generation Tool** (`generate_images.py`)
   - Command-line and interactive image generation
   - Loads trained associations from semantic memory
   - Supports custom sizes and batch generation

### How It Works (No Neural Networks!)

```
Text Input → Text Features (VSA) → Visual Features → Image Synthesis
     ↓              ↓                    ↓                ↓
  "red circle"   HyperVector      HOG + Color +     Procedural
                 (10,240 dims)    Texture Hints     Generation
```

1. **Text Parsing**: Rule-based extraction of color, shape, texture keywords
2. **Feature Mapping**: Map concepts to classical CV features (HOG, color histograms)
3. **Image Synthesis**: Procedurally generate pixels from features
4. **Refinement**: Use VSA similarity to iteratively improve results

## Quick Start

### 1. Installation

```bash
cd /path/to/Node_network
pip install -r requirements.txt

# Key dependencies: numpy, pillow, datasets (for HuggingFace)
```

### 2. Generate Images (Untrained)

```bash
cd nsck_image_gen_project

# Generate single image
python generate_images.py "red circle"

# Custom size
python generate_images.py "blue airplane" --size 128

# Save to specific path
python generate_images.py "green tree" --output my_tree.png

# Interactive mode
python generate_images.py --interactive
```

### 3. Train on CIFAR-10

```bash
# Quick training (50 samples, ~1 minute)
python train_image_generator.py --samples 50

# Full training (500 samples, ~5-10 minutes)
python train_image_generator.py --samples 500
```

**What happens during training:**
- Downloads CIFAR-10 dataset (streaming)
- Processes each image to extract visual features
- Associates text labels with hypervector representations
- Stores learned associations in semantic memory
- Saves trained model to `models/image_gen_system.pkl`

### 4. Generate with Trained Model

```bash
# Use trained associations
python generate_images.py "cat" --model models/image_gen_system.pkl

# Interactive with trained model
python generate_images.py --interactive --model models/image_gen_system.pkl
```

## Supported Text Concepts

The generator understands:

### Colors
- Basic: red, blue, green, yellow, orange, purple, pink, brown, black, white, gray
- Brightness: bright, dark, light, dim

### Shapes
- circle, square, rectangle, triangle, line, blob

### Textures
- smooth, textured, rough, detailed, simple
- High/low contrast

### Objects (with training)
- CIFAR-10 labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Example Prompts

```bash
# Basic shapes and colors
"red circle"
"blue square"
"green triangle"

# Complex descriptions
"bright red detailed pattern"
"dark smooth surface"
"high contrast textured circle"

# Trained concepts (after training on CIFAR-10)
"airplane"
"blue automobile"
"green bird"
"orange cat"
```

## Output

Generated images are saved to:
- Single generation: Specified path or `results/generated_*.png`
- Interactive mode: `results/generated_images/image_001.png`, etc.
- Training logs: `logs/training.log`
- Training metrics: `logs/metrics.json`
- Trained model: `models/image_gen_system.pkl`

## Technical Details

### No Neural Networks - How?

Instead of using convolutional neural networks or diffusion models, this system uses:

1. **Vector Symbolic Architecture (VSA)**
   - 10,240-dimensional binary hypervectors
   - XOR binding, bundling operations
   - Deterministic encoding/decoding

2. **Classical Computer Vision**
   - Histogram of Oriented Gradients (HOG)
   - Color histograms (8 bins per channel)
   - Local Binary Patterns (LBP) for texture
   - Edge detection via Sobel gradients

3. **Procedural Generation**
   - Rule-based pixel synthesis
   - Gradient patterns
   - Shape templates (circles, rectangles)
   - Noise addition for texture

4. **Semantic Memory**
   - Graph-based concept storage
   - Hypervector indexing
   - Associative retrieval

### Performance

- **Generation Speed**: ~0.1-0.5 seconds per image (CPU only)
- **Training Speed**: ~50-100 samples/minute (depends on download)
- **Memory**: ~100-500 MB for trained system
- **Image Quality**: Simple, abstract representations (not photorealistic)

### Limitations

This is NOT a replacement for neural network-based image generators like Stable Diffusion or DALL-E. It produces:
- Abstract, stylized images
- Simple geometric patterns
- Basic color and texture variations

It does NOT produce:
- Photorealistic images
- Complex scenes with multiple objects
- Fine-grained details
- Perspective or 3D rendering

## Project Structure

```
nsck_image_gen_project/
├── train_image_generator.py      # Training pipeline
├── generate_images.py             # Generation tool
├── README.md                      # This file
├── logs/                          # Training logs and metrics
├── models/                        # Trained model checkpoints
└── results/                       # Generated images

nsck-demo/python/core/multimodal/
└── image_generator.py            # Core image generation module
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Input: "red circle"                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Knowledge Learner (VSA)                    │
│  • Rule-based parsing: extract "red", "circle"               │
│  • Map to hypervectors (10,240 dims)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Semantic Memory Query                         │
│  • Retrieve learned associations                             │
│  • Bundle concept hypervectors                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Decoding (Classical CV)                 │
│  • Color: RGB=(200, 50, 50) from "red"                       │
│  • Shape: circular mask from "circle"                        │
│  • Texture: edge_density=0.15 (smooth default)               │
│  • HOG: uniform orientations for circle                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Image Synthesis (Procedural)                      │
│  • Initialize with base color (red)                          │
│  • Apply circular mask                                       │
│  • Add procedural noise for texture                          │
│  • Apply HOG gradient patterns                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Refinement Loop (VSA Feedback)                       │
│  • Encode generated image as hypervector                     │
│  • Compare similarity to target concept                      │
│  • Adjust brightness, contrast iteratively                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: 64×64 RGB Image                         │
│              (or custom size)                                │
└─────────────────────────────────────────────────────────────┘
```

## Comparison to nsck_train_project

Like `nsck_train_project`, this system:
- ✓ Uses VSA/hypervectors for representation
- ✓ Learns from real datasets (CIFAR-10)
- ✓ Stores knowledge in semantic/episodic memory
- ✓ Provides training and inference scripts
- ✓ No neural networks or LLMs
- ✓ Real-time monitoring and logging

Unlike `nsck_train_project`, this system:
- Generates images (not just understands them)
- Uses inverse operations (features → pixels)
- Procedural synthesis instead of classification

## Testing

```bash
# Test basic generator (no training needed)
cd nsck-demo/python/core/multimodal
python image_generator.py

# Test full pipeline
cd nsck_image_gen_project
python train_image_generator.py --samples 10
python generate_images.py "red circle" --output test.png
```

## Advanced Usage

### Batch Generation

```python
from generate_images import ImageGeneratorApp

app = ImageGeneratorApp(model_path="models/image_gen_system.pkl")

prompts = ["red circle", "blue square", "green triangle"]
for i, prompt in enumerate(prompts):
    result = app.generate(prompt, size=(128, 128), 
                         output_path=f"batch_{i}.png")
```

### Custom Vocabulary

Edit `image_generator.py` to add new concepts:

```python
# In _init_feature_vocabulary()
self.color_vocab["teal"] = np.array([0, 128, 128])
self.shape_vocab["star"] = {"shape": "star", "orientations": ["U"]}
```

## Future Enhancements

Possible extensions (maintaining no-neural-network constraint):
- [ ] Multi-object composition using spatial binding
- [ ] Animation via temporal hypervector sequences
- [ ] Style transfer using feature permutation
- [ ] Higher resolution through hierarchical generation
- [ ] 3D shape hints via depth cues

## License

Same as parent NSCK project.

## Citation

If you use this system, please cite the NSCK architecture and note that it uses classical CV + VSA, not neural networks.

---

**Remember:** This is a research prototype demonstrating that image generation is possible without neural networks using VSA and classical signal processing. It's not meant to compete with modern neural generative models in quality, but rather to explore alternative cognitive architectures.
