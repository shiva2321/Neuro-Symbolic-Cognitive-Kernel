# NSCK Multimodal Training Project

A complete, executable setup for training and testing the NSCK cognitive architecture on real HuggingFace datasets.

## Overview

This project provides:
- **Training pipeline** for text (WikiText-2) and images (CIFAR-10)
- **Interactive chat** with the trained model
- **Comprehensive testing** of system capabilities
- **Real-time monitoring** and logging
- **Internal state introspection** to see how the model thinks

## Quick Start

### 1. Install Dependencies

```bash
cd /workspaces/Node_network
pip install -r requirements.txt
```

If you get errors, install essential packages:
```bash
pip install datasets pillow numpy torch torchvision scikit-learn
```

### 2. Train the System

```bash
cd nsck_train_project

# Quick test (50 text samples, 50 image samples)
python train_multimodal.py --text-samples 50 --image-samples 50

# Full training (500 samples each)
python train_multimodal.py --text-samples 500 --image-samples 500

# Text-only training
python train_multimodal.py --text-samples 250 --skip-images
```

**What happens during training:**
- Downloads WikiText-2 dataset (streaming, minimal disk space)
- Downloads CIFAR-10 images (streaming)
- Extracts concepts and relations from text
- Encodes images as hypervectors
- Stores learned knowledge in semantic and episodic memory
- Logs all metrics and outputs

**Training time:**
- Quick test: ~2-3 minutes
- Full training: ~15-20 minutes (depends on internet speed)

### 3. Test the System

```bash
# Run comprehensive test suite
python test_trained_system.py

# Results saved to: results/test_results.json
```

Tests verify:
- ✓ Semantic memory (concepts, relations)
- ✓ Text learning (concept extraction)
- ✓ Episodic memory (experience storage)
- ✓ Multimodal processing (text & images)
- ✓ Knowledge queries
- ✓ Integrated workflows

### 4. Interactive Chat

```bash
# Interactive mode
python chat_with_trained_model.py

# Test mode (automated)
python chat_with_trained_model.py --mode test
```

**Chat Commands:**
```
stats              Show memory statistics
facts              Show learned facts
state              Show internal cognitive state
learn <file>       Learn from a text file
query <text>       Query learned knowledge
quit               Exit
```

## Project Structure

```
nsck_train_project/
├── train_multimodal.py          # Main training script
├── test_trained_system.py       # Comprehensive testing suite
├── chat_with_trained_model.py   # Interactive chat interface
├── logs/                        # Training logs and metrics
├── models/                      # Trained system checkpoints
├── results/                     # Test results (JSON)
└── README.md                    # This file
```

## What You Can Observe

### During Training (in `logs/training.log`)

```
[START] PHASE 1: TEXT LEARNING
  - Sentence processing
  - Concept extraction (nouns, entities)
  - Relation discovery (verb patterns)
  - Encoding to hypervectors
  
[START] PHASE 2: IMAGE LEARNING
  - Feature extraction (HOG, color, texture)
  - Image-to-hypervector conversion
  - Visual concept formation
```

### After Training (in chat/test mode)

1. **Memory Statistics**
   ```
   Episodic Memory: 1000 episodes
   Semantic Memory: 250 concepts, 1500 relations
   Learned Facts: 150
   Learning Sessions: 2
   ```

2. **Learned Facts**
   ```
   dog -is_a-> animal (confidence: 0.95)
   cat -has_property-> furry (confidence: 0.88)
   animal -related_to-> mammal (confidence: 0.75)
   ```

3. **Query Results**
   ```
   Q: "What is a dog?"
   
   Retrieved concepts: animal, mammal, pet
   Learned facts: 
   - dog is a animal
   - dog can be loyal
   
   Confidence: 0.82
   ```

4. **Internal States**
   - Current concepts active
   - Memory access patterns
   - Reasoning trace
   - Decision confidence

## Files Generated

### Training Outputs

- **logs/training.log** - Complete training log with timestamps
- **logs/metrics.json** - Structured metrics (JSON)
- **models/trained_system.pkl** - Saved model state

### Test Outputs

- **results/test_results.json** - Test results and metrics

## Understanding the Output

### Training Log Example

```
[2026-02-13T10:30:45] [INFO] Initializing NSCK cognitive systems...
[2026-02-13T10:30:46] [SUCCESS] All systems ready
[2026-02-13T10:30:47] [START] PHASE 1: TEXT LEARNING
[2026-02-13T10:31:05] [PROGRESS] [TEXT] Processed 10/50 samples | Concepts: 45 | Relations: 120
...
[2026-02-13T10:35:22] [SUCCESS] TEXT PHASE COMPLETE in 248.1s
    Samples processed: 50
    Concepts learned: 152
    Relations learned: 487
    Facts stored: 89
```

### Test Results Example

```json
{
  "total": 8,
  "passed": 8,
  "failed": 0,
  "elapsed": 12.54,
  "results": [
    {
      "test": "Semantic memory handles concepts",
      "status": "✓ PASS",
      "metrics": {"concepts": 250}
    },
    {
      "test": "Text learning extracts concepts",
      "status": "✓ PASS",
      "metrics": {
        "concepts": 152,
        "relations": 487,
        "facts": 89
      }
    }
  ]
}
```

## Troubleshooting

### Out of Memory
```bash
# Train with fewer samples
python train_multimodal.py --text-samples 25 --image-samples 25
```

### Internet Issues
```bash
# The system uses streaming mode - it will retry automatically
# Check your internet connection and try again
```

### Import Errors
```bash
# Make sure you're in the right directory
cd /workspaces/Node_network/nsck_train_project

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Models Not Found for Chat
```bash
# First run training
python train_multimodal.py

# Then run chat
python chat_with_trained_model.py
```

## Advanced Usage

### Custom Training Parameters

```bash
# Very large training
python train_multimodal.py --text-samples 2000 --image-samples 1000

# Skip certain phases
python train_multimodal.py --skip-images        # Text only
python train_multimodal.py --skip-text          # Images only

# Custom output directory
python train_multimodal.py --output-dir my_logs
```

### Analyzing Results

```bash
# View pretty-printed metrics
python -m json.tool logs/metrics.json

# View training log
cat logs/training.log | grep "PROGRESS\|SUCCESS\|ERROR"
```

## Key Concepts

### What NSCK Does Differently

1. **No LLM Dependency**: Text processing uses semantic folding, not transformers
2. **Fast Learning**: Concepts learned in seconds, not hours
3. **Interpretable**: Full decision traces visible in logs
4. **Memory-Aware**: Explicit episodic and semantic memory systems
5. **Multimodal**: Unified representation for text, images, audio

### The Cognitive Pipeline

```
Input (Text/Image)
  ↓
Preprocessing & Feature Extraction
  ↓
Semantic Folding → Hypervectors
  ↓
Semantic Memory (concepts, relations, graph)
  ↓
Episodic Memory (experiences, context)
  ↓
Query/Reasoning
  ↓
Output (answers, actions, explanations)
```

## Next Steps

1. ✓ Run training with small dataset
2. ✓ Review logs in `logs/training.log`
3. ✓ Run tests with `test_trained_system.py`
4. ✓ Chat with model using `chat_with_trained_model.py`
5. → Extend with custom data
6. → Integrate with other systems

## Documentation

For detailed information, see:
- [NSCK Architecture](../docs/ARCHITECTURE.md)
- [Text Learning System](../docs/TEXT_LEARNING_ARCHITECTURE.md)
- [Dashboard Guide](../docs/DASHBOARD_GUIDE.md)
- [Module Reference](../docs/MODULE_REFERENCE.md)

## Issues & Support

- Check the main README: `../README.md`
- Review NSCK documentation: `../docs/`
- Examine logs: `logs/training.log`
- Run tests: `python test_trained_system.py`

## License

Same as parent NSCK project.

---

**Ready to train?** Run:
```bash
python train_multimodal.py --text-samples 100 --image-samples 100
```

Then explore the logs and chat with your trained model!
