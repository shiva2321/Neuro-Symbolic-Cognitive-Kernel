# NCGN Installation Guide

Complete installation instructions for the Neuromorphic Cognitive Graph Network.

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Optional Components](#optional-components)
- [Development Setup](#development-setup)
- [Docker Installation](#docker-installation)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 512 MB available
- **Storage**: 50 MB free space
- **OS**: Windows, Linux, or macOS

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 2 GB available
- **Storage**: 500 MB (for datasets and logs)
- **OS**: Linux or macOS for best performance

### Supported Platforms
- ✅ Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+)
- ✅ macOS (11.0+)
- ✅ Windows (10, 11)
- ✅ WSL2 (Windows Subsystem for Linux)

## Quick Start

### Option 1: Minimal Installation (Core Only)

```bash
# Clone the repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Run immediately (no dependencies!)
python main.py
```

The core NCGN system has **zero external dependencies** except Python 3.11+.

### Option 2: Full Installation (All Features)

```bash
# Clone the repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import flask, networkx, matplotlib, pytest; print('✓ All packages installed')"

# Run the demo
python main.py
```

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Create virtual environment (recommended)
python -m venv ncgn_env

# Activate virtual environment
# On Linux/macOS:
source ncgn_env/bin/activate
# On Windows:
ncgn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -m pytest tests/ --collect-only
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n ncgn python=3.11
conda activate ncgn

# Install dependencies
pip install -r requirements.txt

# Verify
python main.py
```

### Method 3: System-wide Installation

```bash
# Install system-wide (not recommended for development)
sudo pip install -r requirements.txt

# Run
python main.py
```

## Verification

### Basic Verification

```bash
# Test Python version
python --version  # Should be 3.11+

# Test core system
python -c "from core.memory import GraphMemory; print('✓ Core imports OK')"

# Run quick test
python main.py
```

### Full Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output:
# tests/test_physics.py::test_energy_propagation PASSED
# tests/test_physics.py::test_kwta PASSED
# tests/test_system2.py::test_surprise PASSED
# ... (more tests)
# =============== X passed in Y.YYs ===============
```

### Component Verification

```bash
# Test each component
python -c "from core.system1 import System1Engine; print('✓ System 1')"
python -c "from core.system2 import System2Controller; print('✓ System 2')"
python -c "from cortex.dialogue import DialogueManager; print('✓ Cortex')"
python -c "from ui.brain_dashboard import BrainDashboard; print('✓ UI')"
```

## Optional Components

### Web Dashboard (Visualization)

Required for real-time brain visualization:

```bash
# Install Flask and dependencies
pip install flask flask-socketio

# Test installation
python run_dashboard.py --demo

# Open browser to http://127.0.0.1:5000
```

**Package Details:**
- `flask`: Web framework (2.3.0+)
- `flask-socketio`: WebSocket support (5.3.0+)

### Graph Visualization

Required for network topology rendering:

```bash
# Install visualization packages
pip install networkx matplotlib

# Test
python -c "import networkx as nx; import matplotlib.pyplot as plt; print('✓ Visualization OK')"
```

**Package Details:**
- `networkx`: Graph algorithms (3.0+)
- `matplotlib`: Plotting library (3.7.0+)

### Natural Language Processing

Required for text ingestion and dialogue:

```bash
# Install spaCy
pip install spacy

# Download language model
python -m spacy download en_core_web_sm

# Test
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ NLP ready')"
```

**Package Details:**
- `spacy`: NLP library (3.5.0+)
- `en_core_web_sm`: English language model

### Data Acquisition

Required for dataset integration:

```bash
# Install data packages
pip install datasets requests

# Test
python -c "import datasets, requests; print('✓ Data tools ready')"
```

**Package Details:**
- `datasets`: HuggingFace datasets library (2.14.0+)
- `requests`: HTTP library (2.31.0+)

### Terminal UI

Required for rich CLI output:

```bash
# Install rich
pip install rich

# Test
python -c "from rich import print; print('[bold green]✓ Rich terminal ready[/]')"
```

**Package Details:**
- `rich`: Terminal formatting library (13.0.0+)

## Development Setup

### Complete Development Environment

```bash
# Clone repository
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Verify setup
pytest tests/ -v --cov=core --cov=cortex
```

### IDE Setup

#### VS Code

1. Install Python extension
2. Create `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure interpreter: `File > Settings > Project > Python Interpreter`
3. Select the virtual environment
4. Enable pytest: `File > Settings > Tools > Python Integrated Tools`

## Docker Installation

### Using Docker

```dockerfile
# Dockerfile (create this file)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "run_dashboard.py"]
```

```bash
# Build image
docker build -t ncgn:latest .

# Run container
docker run -p 5000:5000 ncgn:latest

# Access at http://localhost:5000
```

### Using Docker Compose

```yaml
# docker-compose.yml (create this file)
version: '3.8'

services:
  ncgn:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
    environment:
      - FLASK_ENV=development
```

```bash
# Start services
docker-compose up

# Stop services
docker-compose down
```

## Troubleshooting

### Python Version Issues

**Problem**: `python --version` shows Python 2.x or Python < 3.11

**Solution**:
```bash
# Try python3 instead
python3 --version

# If still < 3.11, install latest Python
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# macOS:
brew install python@3.11

# Windows:
# Download from https://www.python.org/downloads/
```

### pip Not Found

**Problem**: `pip: command not found`

**Solution**:
```bash
# Try python -m pip
python -m pip --version

# Or install pip
python -m ensurepip --upgrade

# Ubuntu/Debian:
sudo apt install python3-pip

# macOS:
python3 -m ensurepip
```

### Virtual Environment Issues

**Problem**: Cannot activate virtual environment

**Solution**:
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv

# On Windows, if activation is blocked:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'core'`

**Solution**:
```bash
# Ensure you're in the repository root
pwd  # Should show: /path/to/Node_network

# Don't run from subdirectories
cd /path/to/Node_network
python main.py  # ✓ Correct
# NOT: cd core && python main.py  # ✗ Wrong
```

### Installation Fails on ARM (M1/M2 Mac)

**Problem**: Some packages won't install on Apple Silicon

**Solution**:
```bash
# Use Rosetta Python
arch -x86_64 /usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use native ARM packages
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Slow Installation

**Problem**: pip install takes very long

**Solution**:
```bash
# Use faster mirror (China)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or update pip and use binary wheels
pip install --upgrade pip wheel
pip install -r requirements.txt --prefer-binary
```

### Permission Denied

**Problem**: `Permission denied` when installing

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or install to user directory
pip install --user -r requirements.txt

# Or use sudo (not recommended)
sudo pip install -r requirements.txt
```

## Dependency Reference

### Core Dependencies (Optional)
```
pytest>=7.0.0           # Testing framework
```

### Visualization Dependencies
```
flask>=2.3.0            # Web framework
flask-socketio>=5.3.0   # WebSocket support
networkx>=3.0           # Graph algorithms
matplotlib>=3.7.0       # Plotting
```

### NLP Dependencies
```
spacy>=3.5.0            # Natural language processing
```

### Data Dependencies
```
datasets>=2.14.0        # Dataset library
requests>=2.31.0        # HTTP requests
```

### UI Dependencies
```
rich>=13.0.0            # Terminal formatting
```

### Development Dependencies (Not in requirements.txt)
```
black                   # Code formatter
flake8                  # Linter
mypy                    # Type checker
pytest-cov              # Coverage reporting
```

## Next Steps

After installation:

1. **Run the demo**: `python main.py`
2. **Start the dashboard**: `python run_dashboard.py --demo`
3. **Read the User Guide**: [docs/NCGN_User_Guide.md](docs/NCGN_User_Guide.md)
4. **Try the examples**: See [README.md#examples](README.md#examples)
5. **Run the tests**: `pytest tests/ -v`

## Getting Help

If you encounter issues not covered here:

- 📖 Check [README.md](README.md) for troubleshooting section
- 🐛 Open an issue on [GitHub](https://github.com/shiva2321/Node_network/issues)
- 💬 Join discussions on [GitHub Discussions](https://github.com/shiva2321/Node_network/discussions)

---

*Last updated: 2026-01-21*
