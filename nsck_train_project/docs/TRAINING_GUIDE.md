# NSCK AI Training Guide

How to train custom models from scratch or extend the production model.

## Training from Scratch

### Using Web Data Collector

```python
from training.web_data_collector import WebDataCollector
from training.training_pipeline import ExtendedTrainingPipeline

# Collect training data
collector = WebDataCollector()
data = collector.collect_all(target_count=500)

# Run full training pipeline
pipeline = ExtendedTrainingPipeline()
results = pipeline.run_full_pipeline()

print(f"Training complete!")
print(f"Base model: {results['base_pass_rate']:.1f}%")
print(f"Improved model: {results['improved_pass_rate']:.1f}%")
```

### Using Custom Data

```python
from core.neural_chat_backend import TrainableChatBackend
from core.improved_backend import ImprovedBackend
import pickle

# Load your training data
with open('my_training_data.txt') as f:
    sentences = [line.strip() for line in f if line.strip()]

# Initialize and train base backend
backend = TrainableChatBackend()

print(f"Training on {len(sentences)} sentences...")
for i, sentence in enumerate(sentences):
    backend.learn_from_text(sentence)
    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{len(sentences)}")

# Wrap with improvements
improved = ImprovedBackend(backend)

# Save model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(improved, f)

print("Training complete!")
```

## Extending Production Model

### Add New Knowledge

```python
import pickle

# Load production model
with open('models/production_model.pkl', 'rb') as f:
    ai = pickle.load(f)

# Add new domain knowledge
new_facts = [
    "Mars is the fourth planet from the Sun.",
    "Mars has two moons named Phobos and Deimos.",
    "Mars has a thin atmosphere mostly of carbon dioxide.",
    "Mars rovers have explored the surface since 1997."
]

for fact in new_facts:
    ai.learn_from_text(fact)

# Save updated model
with open('models/production_model_v2.pkl', 'wb') as f:
    pickle.dump(ai, f)
```

## Data Collection Strategies

### From Wikipedia

The system includes Simple Wikipedia integration:

```python
from training.web_data_collector import SimpleWikipediaSource
from pathlib import Path

# Collect from specific topics
source = SimpleWikipediaSource(Path('cache/'))
source.topics = [
    "Artificial_Intelligence",
    "Machine_Learning",
    "Neural_Network"
]

data = source.fetch_data(limit=100)
print(f"Collected {len(data)} sentences")
```

### From Documents

```python
import re

def extract_sentences(file_path):
    """Extract clean sentences from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean and filter
    clean = []
    for s in sentences:
        s = s.strip()
        # Keep informative sentences (10-200 chars)
        if 10 <= len(s) <= 200:
            clean.append(s)
    
    return clean

# Use it
sentences = extract_sentences('knowledge_base.txt')
for s in sentences:
    backend.learn_from_text(s)
```

### Quality Guidelines

**Good Training Sentences:**
- ✓ "The speed of light is 299,792,458 m/s in vacuum."
- ✓ "Photosynthesis converts light energy into chemical energy."
- ✓ "World War II lasted from 1939 to 1945."

**Poor Training Sentences:**
- ✗ "It's really interesting how things work." (vague)
- ✗ "I think maybe possibly it could be..." (uncertain)
- ✗ "Click here to learn more!" (not informative)

## Testing Your Model

### Run Comprehensive Tests

```python
from testing.manual_test import test_model_manually

# Test your model
results = test_model_manually(ai, "My Custom Model")

print(f"Pass Rate: {results['pass_rate']:.1f}%")
print(f"Passed: {results['passed']}/{results['total']}")

# Check failed tests
for test in results['details']:
    if not test['passed']:
        print(f"Failed: {test['question']}")
        print(f"Response: {test['response']}")
```

### Custom Test Questions

```python
test_questions = [
    "What is your domain-specific question 1?",
    "What is your domain-specific question 2?",
    # Add more...
]

for q in test_questions:
    response = ai.query(q)
    print(f"Q: {q}")
    print(f"A: {response}")
    print(f"Length: {len(response)} chars")
    print()
```

## Troubleshooting

### Model Not Learning

**Problem:** AI gives generic responses after training.

**Solutions:**
1. Check training data quality (clear, factual statements)
2. Increase training data (aim for 300+ sentences)
3. Verify facts are being stored: `print(ai.training_metadata)`

### Poor Retrieval

**Problem:** AI retrieves wrong concepts.

**Solutions:**
1. Use ImprovedBackend (adaptive retrieval)
2. Add more examples of the concept
3. Check for concept name ambiguity

### OutOfMemory Errors

**Problem:** Training crashes with large datasets.

**Solutions:**
1. Process in batches (50-100 sentences)
2. Increase system RAM
3. Use incremental training (train, save, load, continue)

## Performance Optimization

### Batch Training

```python
batch_size = 50
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    for s in batch:
        backend.learn_from_text(s)
    
    # Save checkpoint
    if (i + batch_size) % 200 == 0:
        with open(f'checkpoint_{i}.pkl', 'wb') as f:
            pickle.dump(backend, f)
```

### Using Rust Acceleration

If available, hypervec_rs provides 10-100x speedup:

```bash
# Install Rust acceleration (optional)
pip install hypervec-rs
```

The system automatically uses it if installed.

## Best Practices

1. **Start Small:** Begin with 100-300 sentences, test, then expand
2. **Curate Data:** Quality > quantity (one good sentence > ten mediocre)
3. **Test Frequently:** Run tests after every 100-200 sentences added
4. **Save Checkpoints:** Don't lose progress if training crashes
5. **Version Models:** Keep previous versions when making changes

## Production Deployment

### Model Versioning

```python
from datetime import datetime

version = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'models/production_model_{version}.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(improved_model, f)

print(f"Saved: {model_path}")
```

### Load Balancing

For high-traffic:

```python
import multiprocessing

def query_worker(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Handle queries...
    
# Spawn workers
workers = []
for i in range(4):
    p = multiprocessing.Process(target=query_worker, 
                                 args=('models/production_model.pkl',))
    p.start()
    workers.append(p)
```

See [ARCHITECTURE.md](../ARCHITECTURE.md) for system design details.
