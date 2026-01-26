# NGCN v7 Setup Guide

## System Requirements

### Hardware
- **CPU**: x86_64 with AVX2 support (Intel Haswell or AMD Zen+)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU (Optional)**: NVIDIA GPU with CUDA support for LLM acceleration
  - Phi-3-Mini: ~2.5GB VRAM
  - Mistral-7B: ~6GB VRAM
  - Llama-3-8B: ~6GB VRAM

### Software
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11 (NOT 3.12 due to some package compatibility)
- **C++ Compiler**: Required for llama-cpp-python
  - Windows: Visual Studio Build Tools 2019+
  - Linux: gcc 9+
  - macOS: Xcode Command Line Tools

---

## Step-by-Step Installation

### 1. Create Virtual Environment

```powershell
# From the NGCN project root
cd d:\NGCN

# Create new venv for v7
python -m venv .venv_v7

# Activate (Windows PowerShell)
.\.venv_v7\Scripts\Activate.ps1

# Activate (Linux/macOS)
# source .venv_v7/bin/activate
```

### 2. Install Core Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install numerical computing
pip install numpy>=1.24.0 scipy>=1.10.0

# Install Rustworkx (Rust graph library)
pip install rustworkx>=0.14.0

# Install Pydantic for schemas
pip install pydantic>=2.0.0
```

### 3. Install LLM Dependencies (Optional but Recommended)

#### Option A: CPU-Only (Slower, No GPU Required)
```powershell
pip install llama-cpp-python>=0.2.50
pip install instructor>=1.0.0
```

#### Option B: CUDA GPU Acceleration (Recommended)
```powershell
# Set environment variable before installing
$env:CMAKE_ARGS="-DGGML_CUDA=on"

# Install with CUDA support
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip install instructor>=1.0.0
```

### 4. Install Embedding Model
```powershell
pip install sentence-transformers>=2.2.0

# The model will auto-download on first use (~90MB)
# Model used: all-MiniLM-L6-v2
```

### 5. Install Development Dependencies
```powershell
pip install pytest>=7.0.0 pytest-benchmark>=4.0.0
pip install flask>=2.3.0 flask-socketio>=5.3.0
pip install rich>=13.0.0  # For terminal UI
```

### 6. Verify Installation

```powershell
python -c "
import numpy as np
import scipy.sparse as sp
import rustworkx as rx
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print('✓ NumPy:', np.__version__)
print('✓ SciPy:', sp.__version__)
print('✓ Rustworkx:', rx.__version__)
print('✓ Pydantic: OK')
print('✓ SentenceTransformers: OK')

# Test Rustworkx graph
g = rx.PyDiGraph()
idx = g.add_node('test')
print(f'✓ Rustworkx graph works, got index: {idx}')

# Test sparse matrix
W = sp.csr_matrix((3, 3))
print(f'✓ Sparse matrix works, shape: {W.shape}')

print('\n=== All core dependencies installed! ===')
"
```

---

## LLM Model Setup

### Option 1: Phi-3-Mini (Recommended for Lower VRAM)

1. Download from HuggingFace:
   - [Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
   - Recommended file: `Phi-3-mini-4k-instruct-q4.gguf` (~2GB)

2. Place in models directory:
   ```powershell
   mkdir d:\NGCN\models
   # Copy downloaded .gguf file to d:\NGCN\models\
   ```

### Option 2: Mistral-7B (Better Reasoning)

1. Download from HuggingFace:
   - [Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
   - Recommended file: `mistral-7b-instruct-v0.2.Q5_K_M.gguf` (~5GB)

### Option 3: No LLM (System 1 Only Mode)

If you don't want to use LLMs, v7 can run in pure System 1 mode:
- Fast associative propagation still works
- No structured reasoning
- Good for games and reflex tasks

---

## Directory Structure After Setup

```
d:/NGCN/
├── .venv_v7/                   # New virtual environment
├── models/                     # LLM model files (GGUF)
│   └── Phi-3-mini-4k-instruct-q4.gguf
├── ncgn_v7/                    # New v7 package (created by agent)
│   ├── __init__.py
│   ├── config.py
│   ├── topology.py
│   ├── state.py
│   ├── engine.py
│   ├── learner.py
│   ├── embeddings.py
│   ├── reasoner.py
│   ├── brain.py
│   └── schemas/
│       ├── __init__.py
│       └── cognitive.py
├── core/                       # Legacy v6 (preserved)
├── tests/
│   ├── test_v7/                # New v7 tests
│   │   ├── test_topology.py
│   │   ├── test_dynamics.py
│   │   └── test_integration.py
│   └── ... (existing v6 tests)
├── requirements_v7.txt         # v7 dependencies
└── docs/
    ├── V7_SETUP_GUIDE.md      # This file
    └── V7_ARCHITECTURE.md     # Technical architecture
```

---

## Quick Verification Test

After setup, run this test to verify everything works:

```powershell
cd d:\NGCN
python -c "
# Test v7 system (after implementation)
from ncgn_v7.brain import Brain

# Create brain without LLM (System 1 only)
brain = Brain(llm_model_path=None)

# Add concepts
brain.add_concept('dog', initial_energy=0.8)
brain.add_concept('cat', initial_energy=0.3)
brain.connect('dog', 'cat', weight=0.5)

# Think
active = brain.think(steps=5)
print('Active concepts:', active)
"
```

---

## Troubleshooting

### Error: `llama_cpp` won't install

**Windows**: Install Visual Studio Build Tools:
1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Select "Desktop development with C++"
3. Restart terminal and retry

### Error: `rustworkx` import fails

This usually means the Rust compiler wasn't available during install. Try:
```powershell
pip uninstall rustworkx
pip install rustworkx --no-cache-dir
```

### Error: CUDA not detected for llama-cpp-python

1. Verify CUDA installation: `nvidia-smi`
2. Reinstall with explicit CUDA flag:
   ```powershell
   $env:CMAKE_ARGS="-DGGML_CUDA=on"
   pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

### Out of Memory errors

- Reduce LLM context window: `n_ctx=2048`
- Use smaller model (Phi-3-Mini instead of Mistral-7B)
- Run in CPU mode (slower but uses system RAM)
