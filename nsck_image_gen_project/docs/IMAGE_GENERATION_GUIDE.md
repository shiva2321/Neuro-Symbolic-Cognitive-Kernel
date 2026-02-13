# NSCK Image Generation System - User Guide

## Overview

The NSCK Image Generation System is a **neural network-free** image generator that uses **Vector Symbolic Architecture (VSA)** and hypervectors to create images from text descriptions. This system is part of the NSCK cognitive architecture and demonstrates how complex tasks like image generation can be accomplished without deep learning.

## Key Features

- **✨ No Neural Networks**: Pure VSA-based approach using 10,240-bit hypervectors
- **🎨 Text-to-Image Generation**: Generate images from natural language descriptions
- **📚 Trainable**: Learn concept-visual associations from image datasets
- **🚀 Interactive Dashboard**: Web-based interface for easy interaction
- **💾 Persistent Models**: Save and load trained generators
- **🔬 Transparent**: All operations are explainable (no black-box models)

## Architecture

### Core Components

1. **LinguaCortex (Semantic Folding)**
   - Converts text concepts to Sparse Distributed Representations (SDRs)
   - Uses 2D topographical semantic maps (128×128 grid)
   - 2% sparsity for efficient computation

2. **ConceptVisualMemory**
   - Associative memory mapping concepts to visual features
   - Uses hypervector binding for associations
   - Enables semantic similarity search

3. **Visual Feature Encoding**
   - HOG (Histogram of Oriented Gradients) - 128 features
   - Color histograms - 24 features
   - Edge density - 1 feature
   - LBP (Local Binary Patterns) - 40 features
   - Spatial statistics - 8 features
   - **Total: 201 features per image**

4. **Image Decoder**
   - Inverse process of feature extraction
   - Reconstructs pixel values from feature vectors
   - Applies smoothing and contrast enhancement

### How It Works

```
Text Prompt
    ↓
Concept Extraction (e.g., "red", "cat", "fur")
    ↓
Concept → Hypervector (via LinguaCortex)
    ↓
Hypervector → Visual Features (via ConceptVisualMemory)
    ↓
Feature Bundling (combine multiple concepts)
    ↓
Visual Features → Pixels (via decoder)
    ↓
Generated Image
```

## Installation

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Install dependencies
pip install numpy torch torchvision pillow flask flask-cors datasets
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Launch interactive dashboard
python3 launch_image_dashboard.py
```

The dashboard will be available at `http://localhost:5556`

## Usage

### 1. Interactive Dashboard

**Starting the Dashboard:**
```bash
python3 launch_image_dashboard.py --port 5556
```

**Features:**
- Text prompt input
- Real-time image generation
- Training interface
- Generation history
- Statistics display
- Image download

**Example Prompts:**
- "red cat with blue eyes"
- "flying airplane in sky"
- "green tree with leaves"
- "blue car on road"

### 2. Command-Line Training

**Train with Synthetic Data (Quick Test):**
```bash
python3 nsck-demo/python/training/train_image_generation.py \
    --synthetic \
    --max-examples 100 \
    --output-dir ./models/image_gen
```

**Train with CIFAR-10 (Real Images):**
```bash
python3 nsck-demo/python/training/train_image_generation.py \
    --dataset cifar10 \
    --max-examples 1000 \
    --output-dir ./models/image_gen
```

**Test Generation Only:**
```bash
python3 nsck-demo/python/training/train_image_generation.py \
    --test-only \
    --output-dir ./models/image_gen
```

### 3. Python API

**Basic Usage:**
```python
from python.core.multimodal.image_generator import ImageGenerator, GenerationConfig

# Create generator
config = GenerationConfig(
    image_size=(32, 32),
    is_color=True,
    smoothing_factor=0.3,
    contrast_boost=1.1
)
generator = ImageGenerator(config=config)

# Generate image
image = generator.generate("red circle")
# image is numpy array (32, 32, 3)

# Save image
from PIL import Image
Image.fromarray(image).save("output.png")
```

**Training:**
```python
# Prepare training data
train_data = [
    ("red object", red_image_array),
    ("blue object", blue_image_array),
    # ... more pairs
]

# Train
generator.train_from_examples(train_data, max_examples=100)

# Save model
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(generator.concept_memory, f)
```

**Loading Trained Model:**
```python
import pickle

# Load concept memory
with open("model.pkl", "rb") as f:
    generator.concept_memory = pickle.load(f)

# Generate with learned concepts
image = generator.generate("trained concept")
```

### 4. REST API

When the dashboard is running, you can use the REST API:

**Generate Image:**
```bash
curl -X POST http://localhost:5556/api/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"red cat"}'
```

**Train Model:**
```bash
curl -X POST http://localhost:5556/api/train \
    -H "Content-Type: application/json" \
    -d '{"dataset":"synthetic","num_examples":50}'
```

**Get Statistics:**
```bash
curl http://localhost:5556/api/stats
```

## Training Details

### Training Process

1. **Load Dataset**: Load text-image pairs from HuggingFace or synthetic data
2. **Extract Features**: Use multimodal processor to extract visual features
3. **Extract Concepts**: Parse text into key concepts
4. **Learn Associations**: Bind concept hypervectors to visual feature hypervectors
5. **Store in Memory**: Save associations in ConceptVisualMemory

### Datasets Supported

- **CIFAR-10**: 10 object classes with labels
- **Synthetic**: Generated colored shapes with text labels
- **Custom**: Any text-image pairs can be used

### Training Parameters

```python
trainer.train(
    dataset_name="cifar10",     # Dataset to use
    split="train",               # Dataset split
    max_examples=1000,           # Number of examples
    save_interval=200            # Save checkpoint frequency
)
```

## Configuration Options

### GenerationConfig

```python
config = GenerationConfig(
    image_size=(32, 32),         # Output image size (H, W)
    is_color=True,                # Color or grayscale
    smoothing_factor=0.3,         # 0-1, higher = smoother
    contrast_boost=1.1            # 1.0 = no change, >1 = more contrast
)
```

### Recommended Settings

**For Detailed Images:**
```python
config = GenerationConfig(
    image_size=(32, 32),
    is_color=True,
    smoothing_factor=0.1,  # Less smoothing
    contrast_boost=1.3     # More contrast
)
```

**For Smooth Abstract Images:**
```python
config = GenerationConfig(
    image_size=(32, 32),
    is_color=True,
    smoothing_factor=0.7,  # More smoothing
    contrast_boost=1.0     # Normal contrast
)
```

## Examples

### Example 1: Generate Simple Shapes

```python
generator = ImageGenerator()

# Train with shapes
train_data = [
    ("red circle", create_circle_image([255, 0, 0])),
    ("blue square", create_square_image([0, 0, 255])),
    ("green triangle", create_triangle_image([0, 255, 0]))
]

generator.train_from_examples(train_data)

# Generate
image = generator.generate("red circle")
```

### Example 2: Generate from Descriptions

```python
# After training on CIFAR-10
generator.train(dataset="cifar10", max_examples=500)

# Generate various objects
images = {
    "airplane": generator.generate("airplane flying in sky"),
    "car": generator.generate("blue car on road"),
    "cat": generator.generate("orange cat with whiskers")
}
```

### Example 3: Batch Generation

```python
prompts = [
    "red apple",
    "blue ocean",
    "green forest",
    "yellow sun"
]

for prompt in prompts:
    image = generator.generate(prompt)
    Image.fromarray(image).save(f"{prompt.replace(' ', '_')}.png")
```

## Performance

### Speed

- **Generation**: ~0.1-0.2 seconds per image (CPU only)
- **Training**: ~0.5 seconds per 100 examples (synthetic)
- **Memory**: ~200 MB for trained model with 1000 concepts

### Quality

The generated images are **conceptual representations** rather than photorealistic renderings. Quality improves with:
- More training examples
- Better concept-visual associations
- Appropriate smoothing and contrast settings

## Limitations

1. **Resolution**: Limited to 32×32 pixels by design (can be changed in config)
2. **Realism**: Generated images are abstract/conceptual, not photorealistic
3. **Training Data**: Quality depends on training data diversity
4. **Concept Learning**: Requires explicit concept-visual associations

## Troubleshooting

### Issue: Generated images are random noise
**Solution**: Train the generator first with relevant examples

### Issue: All images look similar
**Solution**: Increase training data diversity and check concept extraction

### Issue: Dashboard not starting
**Solution**: Check port availability and Flask installation
```bash
pip install flask flask-cors
```

### Issue: Out of memory during training
**Solution**: Reduce `max_examples` parameter

## Technical Details

### Hypervector Operations

- **Dimension**: 10,240 bits per hypervector
- **Binding**: XOR operation (element-wise)
- **Bundling**: Majority vote across vectors
- **Similarity**: Hamming distance / dimension

### Feature Dimensions

- HOG: 128 (4×4 grid, 8 orientations)
- Color: 24 (3 channels × 8 bins)
- Edge: 1 (density value)
- LBP: 40 (4 quadrants × 10 bins)
- Spatial: 8 (4 quadrants × mean/std)

## References

- [NSCK Architecture Documentation](../ARCHITECTURE.md)
- [VSA Theory](../VSA_THEORY.md)
- [Multimodal Processing](../core/multimodal/multimodal_processor.py)

## Support

For issues or questions:
- GitHub Issues: https://github.com/shiva2321/Node_network/issues
- Documentation: See `docs/` folder

## License

Same as the NSCK project license.
