# NSCK AI User Guide

Quick reference for using the NSCK AI production model.

## Quick Start

### Load and Query

```python
import pickle

# Load model
with open('models/production_model.pkl', 'rb') as f:
    ai = pickle.load(f)

# Ask a question  
response = ai.query("What is artificial intelligence?")
print(response)
```

### Learn New Information

```python
# Teach the AI new facts
ai.learn_from_text("Python is a programming language created by Guido van Rossum.")

# Query the new knowledge
response = ai.query("Who created Python?")
print(response)
```

### Multi-Turn Conversations

```python
# First question
r1 = ai.query("What is the speed of light?")

# Follow-up (AI remembers context)
r2 = ai.query("How does that compare to sound?")

# Another follow-up
r3 = ai.query("Why is there a difference?")

# Clear context when changing topics
ai.clear_context()
```

## Common Use Cases

### Knowledge Base QA

```python
# Load documents
with open('knowledge_source.txt') as f:
    for line in f:
        if line.strip():
            ai.learn_from_text(line.strip())

# Answer questions
questions = [
    "What is machine learning?",
    "How does neural network work?",
    "What is supervised learning?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ai.query(q)}\n")
```

### Educational Assistant

```python
# Teach concepts progressively
ai.learn_from_text("Photosynthesis is the process plants use to make food.")
ai.learn_from_text("Photosynthesis requires sunlight, water, and carbon dioxide.")
ai.learn_from_text("Photosynthesis produces glucose and oxygen.")

# Student asks question
q = "What do plants need for photosynthesis?"
answer = ai.query(q)
```

### Customer Support Bot

```python
# Load FAQ knowledge
faqs = [
    "Returns are accepted within 30 days.",
    "Shipping takes 3-5 business days.",
    "Customer service available 24/7."
]

for faq in faqs:
    ai.learn_from_text(faq)

# Handle customer queries
customer_q = "How long does shipping take?"
response = ai.query(customer_q)
```

## Advanced Features

### Checking Model Stats

```python
# Get metadata
metadata = ai.training_metadata
print(f"Texts learned: {metadata['texts_learned']}")
print(f"Facts stored: {metadata['total_facts']}")
print(f"Last trained: {metadata['last_trained']}")
```

### Image Understanding (Optional)

```python
from PIL import Image

# Load image
img = Image.open('photo.jpg')

# Learn image-caption pair
ai.learn_image_caption(img, "A sunset over mountains")

# Later retrieve similar images
results = ai.search_similar_images(img)
```

## Tips & Best Practices

**For Best Results:**
- Keep training sentences clear and factual
- Avoid overly long sentences (aim for 10-200 chars)
- Use `clear_context()` when changing topics
- Train incrementally rather than all at once

**Common Issues:**
- Vague responses → Add more training data
- Wrong answers → Check training data accuracy
- Slow queries → Check model size (>10,000 concepts may need optimization)

See [TEST_RESULTS.md](TEST_RESULTS.md) for performance metrics.
