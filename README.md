# NCGN: Neuromorphic Cognitive Graph Network

NCGN is a "Cognitive Control System" that operates on sparse, event-driven graph dynamics rather than dense matrix multiplication. It implements a dual-process architecture inspired by biological neural systems, where high-speed reactive processing (System 1) is overseen by a deliberative, schema-based monitor (System 2).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Core Philosophy](#-core-philosophy)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** (Required)
- **pip** package manager

### Basic Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shiva2321/Node_network.git
   cd Node_network
   ```

2. **Install Dependencies** (Optional - only `pytest` is used for external testing):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Everything**:
   Run the unified script to execute all unit tests and the "Dog Eat Metal" demo:
   ```bash
   python run_all.py
   ```

4. **Check Results**:
   Results are saved to `ncgn_complete_results.txt`.

### Quick Demo
```bash
# Run the basic demo
python main.py

# Launch the web dashboard (requires Flask)
python run_dashboard.py

# Start training mode
python run_training.py
```

## 🧠 Core Philosophy

NCGN is built on three pillars:
1.  **Sparsity**: Only active nodes are processed, adhering to a biological "energy budget."
2.  **Causal Verification**: The system doesn't just predict; it validates predictions against logical schemas.
3.  **Surprise-Driven Attention**: Computational resources (System 2) are only allocated when observations violate high-confidence expectations.

### Key Features
- ⚡ **Event-Driven Processing**: Sparse graph computation, not matrix multiplication
- 🧠 **Dual-Process Architecture**: Fast reactive (System 1) + Slow deliberative (System 2)
- 🎯 **K-Winners-Take-All Attention**: Biological attention mechanism
- 🔄 **8-Phase Tick Pipeline**: Deterministic, serialized execution
- 🚨 **Surprise Detection**: Monitors prediction failures
- 📊 **Schema Validation**: Symbolic constraint checking
- 💬 **Interactive Learning**: Dialogue-based knowledge acquisition
- 📈 **Real-time Visualization**: Web-based brain dashboard

## 🏗 System Architecture

NCGN consists of three main layers working together:

```
┌─────────────────────────────────────────────────────────┐
│                    USER / ENVIRONMENT                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   UI LAYER (Optional)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Brain      │  │    Chat      │  │    Graph     │  │
│  │  Dashboard   │  │  Interface   │  │  Visualizer  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CORTEX LAYER (The "Mind")                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Dialogue   │  │  Ingestion   │  │   Staging    │  │
│  │   Manager    │  │   Pipeline   │  │   Buffer     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               CORE LAYER (The "Brain")                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Graph Memory (Storage)                 │  │
│  │   • ConceptNodes (Energy, Threshold, Novelty)    │  │
│  │   • Synapses (Weight, Confidence, Type)          │  │
│  │   • EventSchemas (Logic Templates)               │  │
│  └──────────────┬──────────────┬────────────────────┘  │
│                 │              │                         │
│  ┌──────────────▼──────────┐  ┌▼────────────────────┐  │
│  │   System 1 Engine       │◄─┤  System 2 Controller│  │
│  │   (Fast/Reactive)       │  │  (Slow/Deliberative)│  │
│  │                         │  │                     │  │
│  │  • 8-Phase Tick Loop    │  │  • Schema Checking │  │
│  │  • Energy Propagation   │  │  • Diagnosis       │  │
│  │  • K-WTA Attention      │  │  • Intervention    │  │
│  │  • Surprise Detection   ├─►│  • LTD/LTP         │  │
│  └─────────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Architecture Overview

#### Core Layer (The "Brain")
The foundational layer implementing the neuromorphic engine:

- **Graph Memory** (`core/memory.py`): Stores concepts, associations, and schemas
  - **ConceptNode**: Atomic unit with energy, threshold, and novelty
  - **Synapse**: Weighted connection with confidence scores
  - **EventSchema**: JSON-based logic templates

- **System 1 Engine** (`core/system1.py`): Fast, reactive processing
  - Executes 8-phase tick pipeline
  - Handles energy propagation
  - Implements K-WTA attention mechanism
  - Monitors surprise levels

- **System 2 Controller** (`core/system2.py`): Slow, deliberative reasoning
  - Validates against schemas
  - Diagnoses constraint violations
  - Plans interventions
  - Modifies synaptic weights

#### Cortex Layer (The "Mind")
High-level cognitive functions:

- **Dialogue Manager** (`cortex/dialogue.py`): Conversation state machine
  - Handles user interactions
  - Manages contradiction resolution
  - Learns from explanations

- **Ingestion Pipeline** (`cortex/ingestion.py`): Bulk data processing
  - Parses text into triples
  - Extracts subject-verb-object patterns

- **Staging Buffer** (`cortex/staging.py`): The "Hippocampus"
  - Validates before committing to memory
  - Detects conflicts and contradictions
  - Prevents hallucinations

#### UI Layer (Optional)
Visualization and interaction interfaces:

- **Brain Dashboard** (`ui/brain_dashboard.py`): Real-time neural graph visualization
- **Chat Interface** (`ui/chat_interface.py`): Interactive dialogue system
- **Graph Visualizer** (`ui/graph_visualizer.py`): Network topology display

## 📁 Project Structure

```
Node_network/
├── core/                      # The Logic Engine (Brain Hardware)
│   ├── memory.py              # ConceptNodes, Synapses, GraphMemory
│   ├── system1.py             # 8-phase tick-based physics engine
│   ├── bridge.py              # Surprise monitoring & prediction tracking
│   ├── system2.py             # Deliberative diagnosis & intervention
│   ├── trainer.py             # Training session management
│   ├── query_engine.py        # Graph query utilities
│   ├── datasets.py            # Dataset handling
│   ├── data_acquisition.py    # External data acquisition
│   └── schemas/               # Symbolic knowledge templates (JSON)
│       ├── eat.json           # Eating action schema
│       ├── move.json          # Movement schema
│       └── ...                # Other action schemas
│
├── cortex/                    # High-Level Cognition (The Mind)
│   ├── dialogue.py            # Conversation state machine
│   ├── ingestion.py           # Text-to-triple parsing
│   └── staging.py             # Verification buffer (Hippocampus)
│
├── ui/                        # User Interface Layer
│   ├── brain_dashboard.py     # Web-based real-time visualization
│   ├── chat_interface.py      # Interactive dialogue interface
│   ├── graph_visualizer.py    # Network graph renderer
│   └── templates/             # HTML templates for web UI
│
├── training/                  # Training Infrastructure
│   ├── curricula/             # Training curricula definitions
│   └── evaluation/            # Evaluation metrics and tests
│
├── tests/                     # Comprehensive Test Suite
│   ├── test_physics.py        # System 1 physics tests
│   ├── test_system2.py        # System 2 logic tests
│   ├── test_conversation.py   # Dialogue tests
│   └── capability_tests/      # End-to-end capability tests
│
├── docs/                      # Extended Documentation
│   ├── NCGN_Architecture_Overview.md   # Architecture details
│   ├── NCGN_Technical_Reference.md     # Technical specs
│   └── NCGN_User_Guide.md              # User manual
│
├── main.py                    # System demo and scenarios
├── run_all.py                 # Unified test + demo runner
├── run_training.py            # Training entry point
├── run_dashboard.py           # Dashboard launcher
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── DOCUMENTATION.md           # Detailed technical specification
```

### Directory Details

#### `/core` - The Neural Engine
Contains the fundamental neuromorphic processing components:
- **Pure Python implementation** - No ML frameworks required
- **Graph-based memory** - O(1) node access
- **Deterministic execution** - Strictly serialized pipeline

#### `/cortex` - Cognitive Layer
Higher-level cognitive functions built on top of core:
- **Natural language processing** - Text to knowledge graph
- **Dialogue management** - Multi-turn conversations
- **Knowledge staging** - Safe bulk data ingestion

#### `/ui` - User Interfaces
Optional visualization and interaction tools:
- **Flask-based web dashboard** - Real-time brain visualization
- **CLI chat interface** - Interactive teaching
- **Graph visualization** - NetworkX-based rendering

#### `/training` - Learning Infrastructure
Curriculum-based training system:
- **Structured curricula** - Progressive complexity
- **Evaluation metrics** - Performance tracking
- **Benchmark scenarios** - Standard test cases

## 💻 Installation

### Basic Installation (Core Only)
The core system has **no external dependencies** except Python 3.11+:

```bash
# Clone the repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Run immediately - no pip install needed!
python main.py
```

### Full Installation (All Features)
For web dashboard, visualization, and NLP features:

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import flask, networkx, matplotlib; print('All packages installed!')"
```

### Optional Components

#### Web Dashboard (Visualization)
```bash
pip install flask flask-socketio
python run_dashboard.py
# Open http://127.0.0.1:5000
```

#### Natural Language Processing
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### Data Acquisition
```bash
pip install datasets requests
```

### System Requirements

| Component       | Requirement      | Notes                          |
|----------------|------------------|--------------------------------|
| Python         | 3.11+            | Required                       |
| RAM            | 512MB minimum    | For basic demos                |
| Storage        | 50MB             | Core system                    |
| OS             | Any              | Windows, Linux, macOS          |
| GPU            | Not required     | Pure CPU implementation        |

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=cortex
```

## 🎯 Usage

### Running Different Modes

#### 1. Demo Mode (Quick Start)
Run the comprehensive demo showcasing all features:
```bash
python main.py
```
This will demonstrate:
- Basic energy propagation
- K-WTA attention mechanism
- The "Dog Eat Metal" scenario with surprise detection

#### 2. Web Dashboard (Visual)
Launch the interactive brain visualization:
```bash
python run_dashboard.py --demo
```
Then open your browser to: http://127.0.0.1:5000

**Dashboard Controls:**
- **Inject Energy**: Click on any node to add energy
- **Step**: Execute one tick manually
- **Run/Pause**: Auto-run the simulation
- **Surprise Meter**: Watch for System 2 interventions
- **Graph View**: Real-time network topology

#### 3. Training Mode
Train the system with curriculum-based learning:
```bash
# Default curriculum
python run_training.py

# Custom curriculum
python run_training.py --curriculum advanced --epochs 200

# With visualization
python run_training.py --visualize
```

#### 4. Interactive Dialogue (CLI)
Teach the system through conversation:
```bash
python -m cortex.dialogue
```

Example interaction:
```
You: "Dogs eat meat."
NCGN: ✓ Learned: dogs eats meat

You: "Dogs eat metal."
NCGN: 🤔 My knowledge says: Object 'metal' violates constraints: ['is_edible']
      Why do you say 'dogs eats metal'?

You: "It's a robot dog."
NCGN: ✓ Understood. I've learned:
      • robot dog is a type of dog
      • robot dog can eat metal
```

#### 5. Testing
Run the complete test suite:
```bash
# All tests
python run_all.py

# Specific test modules
pytest tests/test_physics.py -v
pytest tests/test_system2.py -v
pytest tests/test_conversation.py -v

# With coverage
pytest tests/ --cov=core --cov=cortex --cov-report=html
```

### Programming API

#### Basic Usage Example
```python
from core.memory import GraphMemory, EventSchema
from core.system1 import System1Engine
from core.system2 import System2Controller

# Initialize the system
memory = GraphMemory()
engine = System1Engine(memory, k_winners=10)
controller = System2Controller(memory)

# Create concepts
memory.add_node("dog", threshold=0.5)
memory.add_node("meat", threshold=0.5)

# Add associations
memory.add_synapse("dog", "meat", weight=0.9, confidence=0.9)

# Inject energy and run
engine.inject_energy("dog", 1.0)
for _ in range(5):
    engine.tick()
    print(f"Active: {memory.get_active_nodes()}")
```

#### Adding Custom Schemas
```python
# Define a new action schema
schema = EventSchema.from_dict({
    "id": "schema_fly",
    "action": "fly",
    "confidence": 0.95,
    "roles": {
        "agent": "animate_object"
    },
    "constraints": {
        "agent": ["can_fly", "has_wings"]
    }
})

# Add to controller
controller.add_schema(schema)

# Define properties
controller.set_property("bird", "can_fly", True)
controller.set_property("bird", "has_wings", True)
controller.set_property("penguin", "has_wings", True)
controller.set_property("penguin", "can_fly", False)
```

## 🔄 Workflow Diagrams

### System 1: The 8-Phase Tick Pipeline

Each discrete time step executes these phases in strict order:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM 1 TICK PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

Input Energy ───┐
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: TRANSDUCTION                                           │
│ • External energy → sensory_buffer                              │
│ • E_new = min(1.0, E_old + Input)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: PASSIVE DECAY                                          │
│ • All nodes leak energy                                         │
│ • E_new = E_old × decay_alpha (0.9)                            │
│ • Nodes with E < 0.001 → inactive                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: FIRING DETERMINATION                                   │
│ • Check: E > Threshold AND RefractoryTimer == 0                │
│ • Mark firing nodes                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: REFRACTORY & RESET                                     │
│ • Firing nodes: E = 0.0                                         │
│ • RefractoryTimer = 3 ticks                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: PROPAGATION                                            │
│ • Spikes travel via synapses                                    │
│ • Signal = Weight × Spike                                       │
│ • Signals buffered (not applied yet)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: INTEGRATION                                            │
│ • Apply buffered signals                                        │
│ • E_target += Signal                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: K-WTA INHIBITION (Attention)                          │
│ • Select top K most energetic nodes                             │
│ • Suppress all others: E = 0                                    │
│ • Implementation: Min-Heap O(N log K)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: SURPRISE MONITOR                                       │
│ • Compare Expected vs Observed state                            │
│ • Calculate: S = sqrt(Σ((E_pred - E_obs) × Conf)²)           │
│ • If S > threshold → Trigger System 2                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 8: MAINTENANCE                                            │
│ • Decrement refractory timers                                   │
│ • Decay novelty scores                                          │
│ • Cleanup inactive nodes                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                         Complete
```

### System 2: Surprise-Driven Intervention

```
┌───────────────────────────────────────────────────────────────┐
│                SYSTEM 2 INTERVENTION PROTOCOL                  │
└───────────────────────────────────────────────────────────────┘

System 1 Tick ───┐
                 │
                 ▼
          ┌──────────────┐
          │   Surprise   │
          │  Calculated  │
          └──────┬───────┘
                 │
         ┌───────▼────────┐
         │ S > Threshold? │
         └───┬────────┬───┘
             │ NO     │ YES
             │        │
             ▼        ▼
        ┌─────────┐  ┌──────────────────┐
        │Continue │  │  PAUSE SYSTEM 1  │
        │Normally │  │  Trigger System 2│
        └─────────┘  └────────┬─────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Load EventSchema│
                     │ (e.g., eat.json)│
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────────────┐
                     │   DIAGNOSIS PHASE       │
                     │ Check constraints:      │
                     │ • is_edible?            │
                     │ • is_animate?           │
                     │ • exists in ontology?   │
                     └────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │ CONSTRAINT   │  │   MISSING   │  │   NOVEL      │
    │  VIOLATION   │  │  KNOWLEDGE  │  │   CONCEPT    │
    └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │Apply LTD     │  │Inject Query │  │Increase      │
    │(Weaken bad   │  │Goal Energy  │  │Novelty Score │
    │ synapse)     │  │             │  │              │
    └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
           │                 │                 │
           └────────┬────────┴─────────┬───────┘
                    │                  │
                    ▼                  ▼
            ┌────────────────┐  ┌──────────────┐
            │Generate Query  │  │Update Memory │
            │for User        │  │   State      │
            └───────┬────────┘  └──────┬───────┘
                    │                  │
                    └────────┬─────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ RESUME SYSTEM 1  │
                   │ with modifications│
                   └──────────────────┘
```

### Dialogue Flow: The Contradiction Loop

```
┌──────────────────────────────────────────────────────────────┐
│              DIALOGUE STATE MACHINE                           │
└──────────────────────────────────────────────────────────────┘

User Input ───┐
              │
              ▼
       ┌─────────────┐
       │    IDLE     │◄──────────────────────┐
       │  (Waiting)  │                       │
       └──────┬──────┘                       │
              │                              │
              │ Parse Input                  │
              ▼                              │
       ┌──────────────────┐                 │
       │   PROCESSING     │                 │
       │ (System 1 Tick)  │                 │
       └──────┬───────────┘                 │
              │                              │
       ┌──────▼──────┐                      │
       │  Surprise?  │                      │
       └──┬────────┬─┘                      │
          │ NO     │ YES                    │
          │        │                        │
          ▼        ▼                        │
     ┌────────┐  ┌────────────────────┐    │
     │Success │  │ CLARIFICATION      │    │
     │Output  │  │ PENDING            │    │
     └───┬────┘  │ (Waiting for user) │    │
         │       └──────┬─────────────┘    │
         │              │                   │
         │              │ User Explains     │
         │              ▼                   │
         │       ┌──────────────────┐      │
         │       │ Parse Explanation│      │
         │       └──────┬───────────┘      │
         │              │                   │
         │       ┌──────▼───────────────┐  │
         │       │ Proposal Generated:  │  │
         │       │ • NEW_SUBCLASS       │  │
         │       │ • EXCEPTION          │  │
         │       │ • NEW_PROPERTY       │  │
         │       │ • REJECTION          │  │
         │       └──────┬───────────────┘  │
         │              │                   │
         │       ┌──────▼──────────────┐   │
         │       │  System 2 Decision  │   │
         │       │ • Validate proposal │   │
         │       │ • Update ontology   │   │
         │       │ • Modify synapses   │   │
         │       └──────┬──────────────┘   │
         │              │                   │
         │              │ Commit Changes    │
         │              ▼                   │
         │       ┌──────────────┐          │
         │       │ Confirmation │          │
         │       │   Message    │          │
         │       └──────┬───────┘          │
         │              │                   │
         └──────────────┴───────────────────┘
```

### Data Ingestion Pipeline

```
┌───────────────────────────────────────────────────────────────┐
│                  FILE INGESTION WORKFLOW                       │
└───────────────────────────────────────────────────────────────┘

Text File ───┐
             │
             ▼
      ┌──────────────┐
      │   Document   │
      │    Reader    │
      └──────┬───────┘
             │ Parse sentences
             ▼
      ┌──────────────────┐
      │  Text-to-Triple  │
      │  • Subject       │
      │  • Predicate     │
      │  • Object        │
      └──────┬───────────┘
             │
             ▼
      ┌──────────────────┐
      │ Staging Buffer   │
      │ (Hippocampus)    │
      └──────┬───────────┘
             │
             │ For each triple
             ▼
      ┌─────────────────────────┐
      │  Schema Validation      │
      │  • Check constraints    │
      │  • Detect conflicts     │
      │  • Flag violations      │
      └──────┬──────────────────┘
             │
     ┌───────▼────────┐
     │ Valid? Conflict?│
     └───┬────────┬───┘
         │        │
    OK   │        │ CONFLICT
         │        │
         ▼        ▼
  ┌──────────┐  ┌────────────────┐
  │ Add to   │  │ CONFIRMATION   │
  │ Staging  │  │ PENDING        │
  └────┬─────┘  │ (Ask user)     │
       │        └────────┬───────┘
       │                 │
       │         ┌───────▼────────┐
       │         │ User Decision: │
       │         │ • Accept       │
       │         │ • Reject       │
       │         │ • Modify       │
       │         └────────┬───────┘
       │                  │
       └──────────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │  Merge Phase  │
          │ • Nodes       │
          │ • Synapses    │
          │ • Properties  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Graph Memory  │
          │   (Updated)   │
          └───────────────┘
```

## 📚 Examples

### Example 1: Teaching Basic Facts
```python
from cortex.dialogue import DialogueManager
from core.memory import GraphMemory
from core.system1 import System1Engine
from core.system2 import System2Controller

# Setup
memory = GraphMemory()
engine = System1Engine(memory)
controller = System2Controller(memory)
dialogue = DialogueManager(memory, engine, controller)

# Teach
response = dialogue.process_input("Dogs eat meat")
print(response)  # "✓ Learned: dogs eats meat"

# Query
response = dialogue.process_input("What do dogs eat?")
print(response)  # "Dogs are associated with: meat"
```

### Example 2: Handling Contradictions
```python
# First teach normal behavior
dialogue.process_input("Dogs eat meat")

# Now introduce contradiction
response = dialogue.process_input("Dogs eat metal")
# NCGN: "🤔 My knowledge says: Object 'metal' violates 
#        constraints: ['is_edible']. Why?"

# Provide explanation
response = dialogue.process_input("It's a robot dog")
# NCGN: "✓ Understood. Learned: robot dog is_a dog, 
#        robot dog can eat metal"
```

### Example 3: Real-time Monitoring
```python
from ui.brain_dashboard import BrainDashboard

# Create dashboard
dashboard = BrainDashboard(memory, engine, controller)

# Start server (Flask)
dashboard.run(host='0.0.0.0', port=5000)

# Access at http://localhost:5000
# Watch energy flow in real-time
# See surprise spikes when contradictions occur
```

### Example 4: Custom Training Curriculum
```python
from core.trainer import TrainingSession, Curriculum, Episode

# Define curriculum
curriculum = Curriculum(name="Animals")
curriculum.add_episode(Episode(
    input_pattern=["dog", "cat"],
    target_pattern=["pet"],
    label="Pet Classification"
))

# Train
session = TrainingSession(memory, engine, curriculum)
results = session.train(epochs=100)
print(f"Accuracy: {results.accuracy:.2%}")
```

## 📖 Documentation

Comprehensive documentation is available in the `/docs` directory:

- **[NCGN_Architecture_Overview.md](docs/NCGN_Architecture_Overview.md)** - High-level system design
- **[NCGN_Technical_Reference.md](docs/NCGN_Technical_Reference.md)** - Detailed algorithms and math
- **[NCGN_User_Guide.md](docs/NCGN_User_Guide.md)** - Usage tutorials and examples
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete technical specification

### Additional Resources
- API Reference: See docstrings in source code
- Test Examples: Browse `/tests` for usage patterns
- Schema Templates: Check `/core/schemas` for logic examples

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Workflow
1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch**: `git checkout -b feature/your-feature`
4. **Make changes** with clear commit messages
5. **Test**: `pytest tests/ -v`
6. **Push**: `git push origin feature/your-feature`
7. **Pull Request**: Open a PR with description

### Coding Standards
- Follow **PEP 8** style guidelines
- Add **docstrings** to all public functions
- Write **unit tests** for new features
- Keep **functions small** and focused
- Use **type hints** where appropriate

### Areas for Contribution
- 🧠 **New schemas**: Add action templates to `/core/schemas`
- 📚 **Curricula**: Create training scenarios in `/training/curricula`
- 🎨 **UI improvements**: Enhance dashboard visualization
- 🧪 **Test coverage**: Add tests in `/tests`
- 📖 **Documentation**: Improve guides and examples
- 🐛 **Bug fixes**: Check GitHub Issues

### Testing Your Changes
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_physics.py -v

# Run with coverage
pytest tests/ --cov=core --cov=cortex --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: Import Errors
```
ModuleNotFoundError: No module named 'core'
```
**Solution**: Ensure you're running from the repository root:
```bash
cd /path/to/Node_network
python main.py  # Not: cd core && python main.py
```

#### Issue: Dashboard Won't Start
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Install optional dependencies:
```bash
pip install flask flask-socketio
```

#### Issue: System 2 Not Triggering
**Symptoms**: No surprise detection, no queries
**Causes**:
1. Surprise threshold too high
2. Confidence scores too low
3. No schemas loaded

**Solution**:
```python
# Lower surprise threshold
engine = System1Engine(memory, surprise_threshold=0.3)

# Increase synapse confidence
memory.add_synapse("A", "B", weight=0.9, confidence=0.9)

# Verify schema loaded
print(controller.schemas)  # Should not be empty
```

#### Issue: Energy Disappears ("Ghost Signals")
**Cause**: Natural decay without reinforcement

**Solution**: Adjust decay rate or inject energy periodically:
```python
# Lower decay rate (energy lasts longer)
engine = System1Engine(memory, decay_alpha=0.95)

# Or reinject energy
for _ in range(10):
    engine.inject_energy("concept", 0.1)
    engine.tick()
```

#### Issue: Tests Fail
```
ImportError during import
```
**Solution**: Install pytest:
```bash
pip install pytest
pytest tests/ -v
```

### Performance Tips

1. **Reduce K (k-WTA)**: Fewer active nodes = faster execution
   ```python
   engine = System1Engine(memory, k_winners=5)  # Instead of 10
   ```

2. **Prune inactive nodes**: Remove concepts with low energy
   ```python
   memory.prune_inactive_nodes(threshold=0.001)
   ```

3. **Limit tick rate**: Add delays in real-time mode
   ```python
   import time
   for _ in range(100):
       engine.tick()
       time.sleep(0.1)  # 10 ticks/second
   ```

### Getting Help

- 📧 **Email**: shiva2321@github.com
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/shiva2321/Node_network/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/shiva2321/Node_network/discussions)
- 📖 **Documentation**: See `/docs` directory

### Debug Mode

Enable verbose logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code
engine.tick()  # Will print detailed phase information
```

## 🐕 The "Dog Eat Metal" Scenario

The canonical test demonstrating cognitive surprise detection:

### Scenario Flow
```
Step 1: PREDICTION
─────────────────
User activates: "dog"
↓
System 1 propagates: dog → eats → meat
↓
Expected state: {meat: 0.85}

Step 2: OBSERVATION  
────────────────────
User injects: "metal" (not "meat")
↓
Observed state: {metal: 1.0, meat: 0.0}
↓
Surprise = sqrt(((0.85 - 0.0) × 0.9)²) = 0.765

Step 3: INTERRUPT
─────────────────
Surprise (0.765) > Threshold (0.45)
↓
System 1 PAUSED
↓
System 2 ACTIVATED

Step 4: DIAGNOSIS
─────────────────
Load schema: eat.json
Check constraints on "metal"
↓
Result: metal.is_edible = False
↓
Diagnosis: CONSTRAINT_VIOLATION

Step 5: INTERVENTION
────────────────────
Action: LTD (weaken dog→metal synapse)
Generate query: "My physics say metal isn't edible"
↓
State: CLARIFICATION_PENDING

Step 6: RESOLUTION
──────────────────
User: "It's a robot dog"
↓
Parse: NEW_SUBCLASS proposal
↓
System 2 decides:
  • Create: robot_dog --[is_a]--> dog
  • Allow: eat(robot_dog, metal) = True
↓
Resume System 1
```

### Key Insights
- **No hallucination**: System queries instead of accepting impossible input
- **Causal reasoning**: Uses schema logic, not just statistics
- **Learning**: Adds exception without breaking general rule
- **Surprise-driven**: Only activates expensive reasoning when needed

## 📊 Performance Characteristics

| Metric                  | Value              | Notes                        |
|-------------------------|--------------------| -----------------------------|
| Tick Execution Time     | ~0.5-2ms           | For 100 nodes                |
| Memory Usage            | ~1MB per 1000 nodes| Pure Python overhead         |
| Surprise Detection      | <0.1ms             | RMS calculation              |
| K-WTA Selection         | O(N log K)         | Min-heap implementation      |
| Graph Query             | O(1)               | Dictionary-based             |
| Schema Check            | O(constraints)     | Linear in constraint count   |
| Max Nodes (Tested)      | 10,000+            | Performance degrades slowly  |

### Scalability Notes
- **Sparse activation**: Only active nodes processed
- **Pruning**: Inactive nodes removed automatically
- **Lazy evaluation**: Schemas loaded on-demand
- **No GPU required**: Pure CPU implementation

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

Inspired by:
- **Dual-Process Theory** (Kahneman, 2011)
- **Neuromorphic Computing** (Mead, 1990)
- **Cognitive Architectures** (Laird et al., SOAR)
- **Graph Neural Networks** (Scarselli et al., 2009)

---

*Created as part of the Advanced Agentic Coding project.*
