# NCGN Installation Guide

Complete installation instructions for the Neuromorphic Cognitive Graph Network.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Hardware-Specific Setup](#hardware-specific-setup)

---

## 🖥️ System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8 or higher
- **CPU**: Modern 6-core processor
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+)
- **RAM**: 16GB
- **Storage**: 256GB SSD

### Recommended Requirements (Tested)

- **OS**: Windows 11
- **Python**: 3.10 or 3.11
- **CPU**: AMD Ryzen 5 7600 (6C/12T) or Intel equivalent
- **GPU**: NVIDIA RTX 3060 12GB VRAM
- **RAM**: 32GB DDR5
- **Storage**: 1TB NVMe SSD
- **CUDA**: 11.8 or 12.1

---

## 🚀 Quick Installation

### Windows (PowerShell)

```powershell
# Navigate to project directory
cd "D:\development project\Node_network"

# Create virtual environment
python -m venv ncgn_env

# Activate environment
.\ncgn_env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (for RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DGL with CUDA 11.8
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### Linux/Mac (Bash)

```bash
# Navigate to project directory
cd /path/to/Node_network

# Create virtual environment
python3 -m venv ncgn_env

# Activate environment
source ncgn_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DGL
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

---

## 📦 Detailed Installation

### Step 1: Prerequisites

#### Python Installation

1. Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version  # Should show Python 3.10.x or 3.11.x
   ```

#### NVIDIA Drivers & CUDA

1. **Check GPU**:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

2. **Install/Update NVIDIA Drivers**:
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install the latest Game Ready or Studio Driver

3. **CUDA Toolkit** (Optional):
   - PyTorch includes CUDA runtime
   - For development, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### Step 2: Virtual Environment

**Why use virtual environments?**
- Isolates dependencies
- Prevents conflicts
- Easy to recreate

**Create and activate**:

```bash
# Windows
python -m venv ncgn_env
.\ncgn_env\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv ncgn_env
source ncgn_env/bin/activate

# Verify activation (should show path to ncgn_env)
which python  # Linux/Mac
where python  # Windows
```

### Step 3: Core Dependencies

#### PyTorch (with CUDA support)

**For CUDA 11.8** (Recommended for RTX 3060):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1** (Newer systems):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only** (No GPU):
```bash
pip install torch torchvision torchaudio
```

**Verify PyTorch**:
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Deep Graph Library (DGL) or PyTorch Geometric (PyG)

**Important**: You can choose either DGL or PyTorch Geometric (or install both). 
- **PyTorch Geometric (PyG)** - Recommended for new installations
- **DGL** - Legacy support, fully compatible with existing code

**PyTorch Geometric (Recommended)**:

For CUDA 11.8:
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

For CUDA 12.1:
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

For CPU only:
```bash
pip install torch-geometric
```

**Verify PyG**:
```python
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

**DGL (Legacy/Alternative)**:

For CUDA 11.8:
```bash
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html
```

For CUDA 12.1:
```bash
pip install dgl-cu121 -f https://data.dgl.ai/wheels/repo.html
```

For CPU only:
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

**Verify DGL**:
```python
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

**Check Available Backends**:
```bash
python utils/check_backend.py
```

See [BACKEND_COMPATIBILITY_GUIDE.md](BACKEND_COMPATIBILITY_GUIDE.md) for detailed information.

### Step 4: Additional Dependencies

#### HuggingFace Transformers

```bash
pip install transformers>=4.35.0
pip install sentencepiece>=0.1.99
pip install tokenizers>=0.15.0
pip install accelerate>=0.25.0
```

#### Spiking Neural Networks

```bash
pip install snntorch>=0.7.0
```

#### Dashboard Dependencies

```bash
pip install flask>=3.0.0
pip install flask-socketio>=5.3.0
pip install python-socketio>=5.10.0
pip install eventlet>=0.33.0
```

#### Monitoring Tools

```bash
pip install py3nvml>=0.2.7
pip install nvitop>=1.0.0
pip install psutil>=5.9.0
```

#### All at Once

```bash
pip install -r requirements.txt
```

### Step 5: Optional Dependencies


#### Weights & Biases (Optional logging)

```bash
pip install wandb
wandb login  # Follow prompts
```

#### MLflow (Optional logging)

```bash
pip install mlflow
```

---

## ✅ Verification

### Run Verification Script

```bash
python verify_installation.py
```

**Expected Output**:
```
================================================================================
  NCGN Installation Verification
================================================================================

>>> Checking Python Version
--------------------------------------------------------------------------------
Python 3.10.x
✅ Python version OK

>>> Checking Dependencies
--------------------------------------------------------------------------------
✅ PyTorch
✅ Deep Graph Library
✅ HuggingFace Transformers
✅ NumPy
✅ NetworkX

>>> Checking CUDA/GPU
--------------------------------------------------------------------------------
✅ CUDA available
  Device: NVIDIA GeForce RTX 3060
  VRAM: 12.00 GB
  CUDA Version: 11.8

>>> Checking NCGN Module
--------------------------------------------------------------------------------
✅ NCGN module found
  Version: 0.1.0
✅ All core components importable

================================================================================
  Verification Summary
================================================================================
✅ PASS - Python Version
✅ PASS - Dependencies
✅ PASS - CUDA/GPU
✅ PASS - NCGN Module

🎉 All checks passed! You're ready to use NCGN.
```

### Manual Verification

```python
# Test imports
python
>>> import torch
>>> import dgl
>>> import transformers
>>> from ncgn import LinguisticGraph, DualSystemArchitecture
>>> print("✅ All imports successful!")
```

### Quick Test Run

```bash
# Run Phase 1 demo
python scripts/ncgn_demo.py --phase 1
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: "ImportError: No module named 'dgl'"

**Solution**:
```bash
# Make sure you installed the correct DGL version
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# Verify CUDA version matches
python -c "import torch; print(torch.version.cuda)"
```

#### Issue 2: "CUDA out of memory"

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   # configs/ncgn_config.yaml
   memory_optimization:
     batch_size: 16  # Reduce from 32
   ```

2. **Enable memory optimizations**:
   ```yaml
   hardware:
     use_mixed_precision: true
     gradient_checkpointing: true
   memory_optimization:
     offload_to_cpu: true
   ```

3. **Clear CUDA cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### Issue 3: "ModuleNotFoundError: No module named 'ncgn'"

**Solution**:
```bash
# Make sure you're in the project directory
cd "D:\development project\Node_network"

# And virtual environment is activated
.\ncgn_env\Scripts\Activate.ps1

# Verify current directory
pwd  # Should show Node_network path
```

#### Issue 4: "Permission denied when activating venv (Windows)"

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\ncgn_env\Scripts\Activate.ps1
```

#### Issue 5: "Slow training/inference"

**Diagnoses**:

1. **Check if using GPU**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.current_device())  # Should be 0
   ```

2. **Enable cuDNN benchmarking**:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

3. **Check device in config**:
   ```yaml
   hardware:
     device: "cuda"  # Not "cpu"
   ```

#### Issue 6: "Import errors for transformers"

**Solution**:
```bash
pip install --upgrade transformers sentencepiece tokenizers
```

---

## 🎛️ Hardware-Specific Setup

### RTX 3060 12GB (Recommended Configuration)

```yaml
# configs/ncgn_config.yaml
hardware:
  device: "cuda"
  gpu_memory_gb: 12
  use_mixed_precision: true
  gradient_checkpointing: true

memory_optimization:
  batch_size: 32
  accumulation_steps: 4
  offload_to_cpu: false
```

### RTX 4090 24GB (High Performance)

```yaml
hardware:
  gpu_memory_gb: 24
  use_mixed_precision: false  # Can use FP32

memory_optimization:
  batch_size: 64
  accumulation_steps: 1
  offload_to_cpu: false
```

### GTX 1660 6GB (Budget Setup)

```yaml
hardware:
  gpu_memory_gb: 6
  use_mixed_precision: true
  gradient_checkpointing: true

memory_optimization:
  batch_size: 8
  accumulation_steps: 8
  offload_to_cpu: true
```

### CPU Only (No GPU)

```yaml
hardware:
  device: "cpu"
  
memory_optimization:
  batch_size: 4
  offload_to_cpu: false  # Already on CPU
```

---

## 🔄 Updating

### Update NCGN

```bash
# Activate environment
.\ncgn_env\Scripts\Activate.ps1

# Pull latest code (if from git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Update PyTorch

```bash
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu118
```

### Update DGL

```bash
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html --upgrade
```

---

## 🗂️ Directory Setup

After installation, your directory structure should look like:

```
Node_network/
├── ncgn_env/              # Virtual environment
├── ncgn/                  # Core NCGN modules
├── dashboard_utils/       # Dashboard backend
├── templates/             # Dashboard HTML
├── static/                # CSS and JavaScript
├── configs/               # Configuration files
├── scripts/               # Demo scripts
├── saved_models/          # Model checkpoints
├── uploads/               # Uploaded datasets
├── logs/                  # Training logs
└── *.md                   # Documentation
```

**Create missing directories**:
```bash
mkdir -p saved_models/ncgn
mkdir -p uploads
mkdir -p logs
mkdir -p cache
```

---

## 🔍 Version Compatibility

| Component | Version | Required |
|-----------|---------|----------|
| Python | 3.8-3.11 | ✓ |
| PyTorch | 2.0+ | ✓ |
| DGL | 1.1+ | ✓ |
| CUDA | 11.8+ | Recommended |
| Transformers | 4.35+ | ✓ |
| NumPy | 1.24+ | ✓ |
| Flask | 3.0+ | ✓ |

---

## 📞 Getting Help

**Installation Issues**:
1. Check this guide thoroughly
2. Verify all prerequisites
3. Run `python verify_installation.py`
4. Check error messages carefully

**Still stuck?**
- Review error messages for clues
- Check Python and CUDA versions
- Ensure virtual environment is activated
- Try reinstalling in a fresh virtual environment

---

## ⏱️ Installation Time

**Typical installation time**:
- Prerequisites check: 2-3 minutes
- Virtual environment: 1 minute
- PyTorch + DGL: 3-5 minutes (depending on internet speed)
- Other dependencies: 2-3 minutes
- Verification: 1 minute

**Total**: ~10-15 minutes

**Disk Space Required**: ~5GB (including dependencies and models)

---

## ✅ Post-Installation

### Next Steps

1. **Run the demo**:
   ```bash
   python scripts/ncgn_demo.py
   ```

2. **Launch dashboard**:
   ```bash
   python ncgn_dashboard.py
   ```

3. **Read the user guide**:
   - See `USER_GUIDE.md` for usage instructions

4. **Configure for your hardware**:
   - Edit `configs/ncgn_config.yaml`

### Learning Path

1. **Beginner**: Run demos, use dashboard
2. **Intermediate**: Configure parameters, upload data
3. **Advanced**: Modify code, extend functionality

---

**Installation Guide Version**: 1.0  
**Last Updated**: January 8, 2026  
**Compatible with**: NCGN v0.1.0+

---

*Happy installing! 🚀*

