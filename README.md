# Modular Graph-Based Neural Network Architecture

A sophisticated graph-based neural network system designed for advanced text processing, with modular specialized networks for text generation, mathematical computations, and code assistance.

## Architecture Overview

This system implements a graph-based neural network where:
- **Nodes** represent text elements (words, phrases, concepts)
- **Edges** represent weighted relationships and contextual connections
- **Modules** provide specialized processing for different domains
- **Central Controller** routes requests to appropriate specialized modules

## Phases

### Phase 1: Graph-Based Neural Network Architecture
- Core graph structures (nodes, edges, graph)
- Graph traversal algorithms with stop criteria
- Dynamic weight update mechanisms

### Phase 2: Modularity and Integration
- Specialized modules (text, math, code)
- Central controller for intelligent routing
- Inter-module communication

### Phase 3: Scalability and Performance
- Performance monitoring and profiling
- Optimized graph operations
- Memory-efficient data structures

### Phase 4: Training and Updating
- Dataset processing (10,000 books)
- Continuous learning mechanisms
- Feedback loop integration

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

## Usage
## Usage

### Option 1: Web Dashboard (Recommended) 🌐

Launch the interactive web dashboard:

```bash
python launch_dashboard.py
```

Then open http://localhost:5000 in your browser.

**Features**:
- 📁 Upload PDF, TXT, DOC, Code files via drag-and-drop
- 🎓 Configure and monitor training in real-time
- 💬 Interactive query interface
- 🕸️ Visual graph exploration with D3.js
- 📊 Real-time statistics and performance metrics

See `DASHBOARD_GUIDE.md` for complete tutorial.

### Option 2: Command Line

```python
from core.central_controller import CentralController

# Initialize the system
controller = CentralController()
controller.load_training_data("path/to/books/")
controller.train()

# Generate text
response = controller.process("Write a short story about AI")

# Perform math
response = controller.process("Calculate the derivative of x^2 + 3x")

# Code assistance
response = controller.process("Write a Python function to sort a list")
```

See `QUICKSTART.md` for command-line usage.

## Project Structure

```
Node_network/
├── core/
│   ├── graph_network.py      # Core graph data structures
│   ├── traversal_engine.py   # Graph traversal algorithms
│   ├── training_engine.py    # Training and weight updates
│   └── central_controller.py # Central routing controller
├── modules/
│   ├── text_module.py         # Text generation module
│   ├── math_module.py         # Mathematical computation module
│   └── code_module.py         # Code assistance module
├── utils/
│   ├── text_processor.py      # Text preprocessing utilities
│   ├── visualizer.py          # Graph visualization tools
│   └── performance_monitor.py # Performance tracking
├── training/
│   ├── dataset_loader.py      # Dataset loading and management
│   └── feedback_loop.py       # Continuous learning system
└── tests/
    └── test_*.py              # Unit tests
```

## License

MIT License

