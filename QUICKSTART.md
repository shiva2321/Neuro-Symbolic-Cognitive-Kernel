# Quick Start Guide

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Basic Usage

### 1. Run the Complete Demo

```bash
python main.py
```

This will:
- Initialize the modular system
- Create sample training data
- Train all modules
- Test with various queries
- Generate visualizations
- Save the trained models

### 2. Use Programmatically

```python
from core.central_controller import CentralController

# Initialize
controller = CentralController()

# Train on your data
controller.train(data_path="path/to/your/books/")

# Process queries
result = controller.process("Write a story about AI")
print(result['response'])

# Save trained system
controller.save_system("saved_models")
```

### 3. Load Pre-trained Models

```python
from core.central_controller import CentralController

# Load existing models
controller = CentralController()
controller.load_system("saved_models")

# Start using immediately
result = controller.process("Calculate 25 * 4")
print(result['response'])
```

### 4. Interactive Mode

```python
from core.central_controller import CentralController

controller = CentralController()
controller.load_system("saved_models")
controller.interactive_mode()
```

## Module-Specific Usage

### Text Generation Module

```python
from modules.text_module import TextModule

text_mod = TextModule()
text_mod.train(text_data_list, epochs=5)

result = text_mod.process("Write a short story")
print(result['response'])
```

### Math Module

```python
from modules.math_module import MathModule

math_mod = MathModule()
math_mod.train(math_texts)

result = math_mod.process("Calculate 15 + 27")
print(result['response'])
```

### Code Module

```python
from modules.code_module import CodeModule

code_mod = CodeModule()
code_mod.train(code_samples)

result = code_mod.process("Write a Python function to sort a list")
print(result['response'])
```

## Advanced Features

### Custom Traversal Configuration

```python
from core.traversal_engine import TraversalConfig, TraversalStrategy

config = TraversalConfig(
    strategy=TraversalStrategy.TEMPERATURE,
    max_tokens=200,
    temperature=0.9,
    repetition_penalty=1.5
)

# Use with text module
text_module.traversal_engine.config = config
```

### Performance Monitoring

```python
from utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.monitor_operation("training"):
    controller.train(data_path)

monitor.print_report()
monitor.export_metrics("performance.json")
```

### Graph Visualization

```python
from utils.visualizer import GraphVisualizer

visualizer = GraphVisualizer(controller.module_map['text_module'].graph)

# Create interactive HTML visualization
visualizer.create_html_visualization("graph.html", max_nodes=500)

# Export to other formats
visualizer.export_to_graphml("graph.graphml")
visualizer.export_to_dot("graph.dot")
visualizer.generate_statistics_report("stats.txt")
```

## Training on Your Own Data

### Prepare Your Dataset

Organize your data in a directory:
```
my_dataset/
├── books/
│   ├── book1.txt
│   ├── book2.txt
│   └── ...
├── code/
│   ├── script1.py
│   ├── script2.js
│   └── ...
└── math/
    ├── textbook1.txt
    └── ...
```

### Load and Train

```python
from training.dataset_loader import DatasetLoader
from core.central_controller import CentralController

# Load dataset
loader = DatasetLoader("my_dataset")
loader.scan_directory()
data = loader.load_all_data()

# Train
controller = CentralController()
results = controller.train(
    data_path="my_dataset",
    epochs=10,
    learning_rate=0.1
)

print(results['summary'])
```

## Configuration Options

### Training Configuration

```python
from core.training_engine import TrainingConfig

config = TrainingConfig(
    learning_rate=0.1,
    batch_size=32,
    num_epochs=10,
    weight_decay=0.0001,
    momentum=0.9
)
```

### Text Processing Options

```python
from utils.text_processor import TextProcessor

processor = TextProcessor(
    use_spacy=True,  # Use spaCy for better NLP
    ngram_sizes=[1, 2, 3],  # N-gram sizes to extract
    min_word_frequency=2  # Minimum word frequency
)
```

## Troubleshooting

### Out of Memory
- Reduce batch size in training config
- Process data in smaller chunks
- Use graph pruning more frequently

### Slow Training
- Reduce number of epochs
- Limit training data size
- Use parallel processing (future enhancement)

### Poor Quality Results
- Increase training data
- Adjust learning rate
- Modify traversal temperature
- Train for more epochs

## Examples

See the `examples/` directory for more detailed examples:
- `example_text_generation.py` - Advanced text generation
- `example_code_assistant.py` - Code generation examples
- `example_math_solver.py` - Mathematical problem solving
- `example_custom_module.py` - Creating custom modules

## API Reference

See `docs/API.md` for complete API documentation.

