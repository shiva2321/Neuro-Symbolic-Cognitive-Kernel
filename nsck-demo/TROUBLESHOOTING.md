# NSCK Troubleshooting Guide

> **Common issues, solutions, and debugging strategies for the Neuro-Symbolic Cognitive Kernel**

This guide covers the most common problems users encounter when installing, configuring, or running NSCK, along with step-by-step solutions.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Problems](#performance-problems)
4. [Learning Issues](#learning-issues)
5. [GPU/CUDA Problems](#gpucuda-problems)
6. [Dashboard & UI Issues](#dashboard--ui-issues)
7. [Debugging Strategies](#debugging-strategies)
8. [FAQ](#frequently-asked-questions)

---

## Installation Issues

### Problem: `maturin: command not found`

**Symptom**:
```bash
$ maturin develop --release
bash: maturin: command not found
```

**Cause**: Maturin build tool is not installed.

**Solution**:
```bash
pip install maturin
# or
pip3 install maturin

# Verify installation
maturin --version
```

**Alternative**: Use `pip install` directly if maturin fails:
```bash
cd rust_vsa
pip install .
```

---

### Problem: `error: linker 'link.exe' not found` (Windows)

**Symptom**:
```
error: linking with `link.exe` failed: exit code: 1
note: LINK : fatal error LNK1104: cannot open file 'kernel32.lib'
```

**Cause**: Windows C++ build tools are missing.

**Solution**:
1. Download [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. Run installer
3. Select "Desktop development with C++"
4. Install (requires ~7GB disk space)
5. Restart terminal
6. Retry `maturin develop --release`

**Quick test**:
```bash
# Check if C++ compiler is available
cl.exe
# Should output "Microsoft (R) C/C++ Optimizing Compiler Version..."
```

---

### Problem: `ld: library not found for -lSystem` (macOS)

**Symptom**:
```
error: linking with `cc` failed: exit code: 1
ld: library not found for -lSystem
```

**Cause**: Xcode Command Line Tools are missing.

**Solution**:
```bash
# Install Xcode CLI tools
xcode-select --install

# If that fails, install full Xcode from App Store
# Then retry:
sudo xcode-select --reset
```

**Verification**:
```bash
# Check if compiler works
gcc --version
# Should output "Apple clang version..."
```

---

### Problem: `ImportError: cannot import name 'HyperVector'`

**Symptom**:
```python
>>> from hypervec_rs import HyperVector
ImportError: cannot import name 'HyperVector' from 'hypervec_rs'
```

**Cause**: Rust extension didn't build or install correctly.

**Solution**:
```bash
# Clean and rebuild
cd rust_vsa
cargo clean
maturin develop --release
cd ..

# Test import
python -c "from hypervec_rs import HyperVector; print('✓ Success')"
```

**If still failing**, check Python version compatibility:
```bash
python --version  # Must be 3.11+

# If using virtual environment, ensure it's activated
which python      # Should show venv path, not system Python
```

---

### Problem: `ModuleNotFoundError: No module named 'snntorch'`

**Symptom**:
```python
ModuleNotFoundError: No module named 'snntorch'
```

**Cause**: Python dependencies not installed.

**Solution**:
```bash
# Ensure you're in nsck-demo directory
cd Node_network/nsck-demo

# Install all requirements
pip install -r requirements.txt

# If specific package fails, install individually
pip install snntorch torch matplotlib pygame zmq Pillow numpy
```

**Check installed packages**:
```bash
pip list | grep -E "(torch|snntorch|pygame)"
```

---

## Runtime Errors

### Problem: `RuntimeError: Expected all tensors to be on the same device`

**Symptom**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Cause**: Mixing GPU and CPU tensors.

**Solution Option 1** (Force CPU):
```python
# At start of script
import torch
torch.set_default_device('cpu')

# OR explicitly move model to CPU
snn = TaskAwareSNN().cpu()
```

**Solution Option 2** (Force GPU):
```python
# Move everything to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
snn = TaskAwareSNN().to(device)
state = state.to(device)
```

---

### Problem: `pygame.error: No available video device`

**Symptom**:
```
pygame.error: No available video device
```

**Cause**: Running on headless server (no display) or missing graphics drivers.

**Solution** (Headless mode):
```bash
# Set headless display
export SDL_VIDEODRIVER=dummy
python python/snake_ui.py
```

**Solution** (Remote server):
```bash
# Use X11 forwarding
ssh -X user@server
python python/snake_ui.py
```

**Solution** (Docker):
```dockerfile
# Add to Dockerfile
ENV SDL_VIDEODRIVER=dummy
```

---

### Problem: `OSError: [Errno 98] Address already in use`

**Symptom**:
```
OSError: [Errno 98] Address already in use
# ZMQ error binding to tcp://127.0.0.1:5565
```

**Cause**: Another NSCK server is already running or port not released.

**Solution**:
```bash
# Find and kill process using port 5565
lsof -i :5565
kill -9 <PID>

# Or kill all Python processes (careful!)
pkill -9 python

# Restart server
python python/python_server.py
```

**Prevention**: Always close dashboard properly (don't force-quit).

---

### Problem: `FileNotFoundError: [Errno 2] No such file or directory: 'codebook.pkl'`

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'codebook.pkl'
```

**Cause**: VSA codebook not generated yet.

**Solution**:
```bash
# Generate codebook
cd nsck-demo
python python/build_codebook.py

# Verify it was created
ls -lh codebook.pkl
# Should show ~17KB file
```

**Alternative**: The codebook is included in the repo, but if missing:
```bash
# Download pre-built codebook
curl -O https://github.com/shiva2321/Node_network/raw/main/nsck-demo/codebook.pkl
```

---

## Performance Problems

### Problem: Training is extremely slow (>5 minutes per episode)

**Symptoms**:
- Episode takes 5+ minutes
- CPU usage stuck at 100%
- Terminal shows "Processing frame..." repeatedly

**Diagnosis**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Solution 1** (Enable GPU):
```python
# In your training script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
snn = snn.to(device)
print(f"Using device: {device}")
```

**Solution 2** (Reduce complexity):
```python
# In snn_qat.py, reduce simulation steps
# Change from 8 to 4 steps
for step in range(4):  # Was: range(8)
    # ... LIF dynamics
```

**Solution 3** (Increase batch size):
```python
# In train_snn.py
BATCH_SIZE = 128  # Increase from 64 for better GPU utilization
```

**Expected performance**:
- CPU: 1-2 episodes/second
- GPU (RTX 3060): 10-15 episodes/second

---

### Problem: Out of memory (OOM) errors

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 8.00 GiB total capacity; ...)
```

**Cause**: Batch size too large or memory leak.

**Solution 1** (Reduce batch size):
```python
# In train_snn.py
BATCH_SIZE = 16  # Reduce from 64
```

**Solution 2** (Clear cache):
```python
import torch
torch.cuda.empty_cache()

# Add this periodically during training
if episode % 10 == 0:
    torch.cuda.empty_cache()
```

**Solution 3** (Reduce replay buffer):
```python
# In learning.py
REPLAY_BUFFER_SIZE = 500  # Reduce from 1000
```

---

### Problem: VSA operations are slow (>1ms per query)

**Symptom**:
```python
import time
start = time.time()
similarity = hv1.similarity(hv2)
print(f"Time: {(time.time() - start) * 1000:.2f}ms")
# Output: Time: 5.23ms (too slow!)
```

**Cause**: Rust extension not compiled with optimizations.

**Solution**:
```bash
# Rebuild with release mode
cd rust_vsa
cargo clean
maturin develop --release  # Note: --release is critical
cd ..
```

**Verification**:
```python
# Should be <0.2ms
import time
from hypervec_rs import HyperVector

hv1 = HyperVector.random()
hv2 = HyperVector.random()

start = time.time()
for _ in range(1000):
    sim = hv1.similarity(hv2)
avg_time = (time.time() - start) / 1000 * 1000
print(f"Average time: {avg_time:.3f}ms")
# Should output: <0.1ms
```

---

## Learning Issues

### Problem: AI doesn't learn (loss stays flat, accuracy doesn't improve)

**Symptoms**:
- After 50+ episodes, accuracy still <50%
- Loss fluctuates randomly, no downward trend
- Agent keeps making same mistakes

**Diagnosis**:
```python
# Check if weights are updating
print("Initial weights:", snn.conv1.weight[:2, :2, 0, 0])
# ... train for 10 episodes ...
print("After 10 episodes:", snn.conv1.weight[:2, :2, 0, 0])
# If identical → weights not updating
```

**Solution 1** (Check learning rate):
```python
# In learning.py
LEARNING_RATE = 0.001  # Might be too low
# Try: 0.01 for faster learning (less stable)
# Or: 0.0001 for slower learning (more stable)
```

**Solution 2** (Verify gradients):
```python
# Add gradient monitoring
for name, param in snn.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")  # Problem here
```

**Solution 3** (Check reward signal):
```python
# Ensure rewards are actually being provided
print(f"Average reward: {np.mean(rewards)}")
# Should be non-zero. If always 0 → reward logic broken
```

---

### Problem: Catastrophic forgetting (loses Snake skills after training Pong)

**Symptom**:
- Train Snake to 85% accuracy
- Train Pong to 80% accuracy
- Test Snake again → 45% accuracy (forgot!)

**Cause**: Not using late fusion properly or task_id is wrong.

**Solution**:
```python
# Verify task_id switching
print(f"Training Snake with task_id=0")
snn_output = snn(state, task_id=0)  # Correct

print(f"Training Pong with task_id=1")  
snn_output = snn(state, task_id=1)  # Correct

# WRONG (will cause forgetting):
# snn_output = snn(state, task_id=0) for both tasks
```

**Test isolation**:
```python
# After training Pong, freeze Pong head
snn.head_pong.requires_grad = False

# Continue training Snake → should not affect Pong
```

---

### Problem: System 2 never triggers (all decisions are System 1)

**Symptom**:
```
Episode logs show:
Decision: system1_trusted (100% of the time)
No VETO or IMPROVE messages
```

**Cause**: Surprise threshold too high.

**Solution**:
```python
# In brain_fusion.py
SURPRISE_THRESHOLD = 0.5  # Lower from 0.7 to invoke System 2 more often
```

**Diagnosis**:
```python
# Add logging
surprise = compute_surprise(expected, observed)
print(f"Surprise: {surprise:.3f}, Threshold: {SURPRISE_THRESHOLD}")
# If surprise always < threshold → lower threshold
```

---

## GPU/CUDA Problems

### Problem: `AssertionError: Torch not compiled with CUDA enabled`

**Symptom**:
```python
>>> torch.cuda.is_available()
False
>>> torch.cuda.get_device_name(0)
AssertionError: Torch not compiled with CUDA enabled
```

**Cause**: PyTorch CPU-only version installed.

**Solution**:
```bash
# Uninstall CPU version
pip uninstall torch

# Install CUDA version (for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Check CUDA version**:
```bash
nvidia-smi
# Look for "CUDA Version: X.X"
```

---

### Problem: GPU utilization is low (<30%)

**Symptom**:
```bash
nvidia-smi
# Shows GPU usage at 15-20%, expected 80%+
```

**Cause**: Batch size too small or CPU bottleneck.

**Solution**:
```python
# Increase batch size
BATCH_SIZE = 128  # Up from 32

# Use multiple workers for data loading
DataLoader(..., num_workers=4, pin_memory=True)

# Profile to find bottleneck
import torch.profiler
with torch.profiler.profile() as prof:
    # ... training loop ...
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Dashboard & UI Issues

### Problem: Dashboard starts but games don't connect

**Symptom**:
- Dashboard window opens successfully
- Click "START Server" → works
- Click "Start Snake" → new window opens but freezes or shows black screen

**Cause**: ZMQ communication failure.

**Solution**:
```bash
# Check if ports are open
netstat -an | grep 5565
netstat -an | grep 5566
netstat -an | grep 5567

# If not listed, server didn't start properly
# Check server logs
python python/python_server.py
# Look for "Listening on tcp://127.0.0.1:5565"
```

---

### Problem: "TEACHER mode not working" (arrow keys ignored)

**Symptom**:
- Turn TEACHER mode ON
- Press arrow keys in game window
- AI continues autonomous play, doesn't follow keys

**Cause**: Game window doesn't have focus or ZMQ message not sent.

**Solution**:
```python
# Ensure game window is active (click on it)
# Verify message is sent
import zmq
context = zmq.Context()
sock = context.socket(zmq.PUSH)
sock.connect("tcp://127.0.0.1:5565")
sock.send_json({"cmd": "TEACHER_ON"})
print("Teacher mode enabled")
```

---

### Problem: Dashboard freezes when loading large logs

**Symptom**:
- After 100+ episodes, dashboard becomes unresponsive
- Scrolling decision log is very slow

**Solution**:
```python
# In dashboard.py, limit log size
MAX_LOG_ENTRIES = 500  # Cap at 500 entries

if len(self.event_log) > MAX_LOG_ENTRIES:
    self.event_log = self.event_log[-MAX_LOG_ENTRIES:]  # Keep recent
```

---

## Debugging Strategies

### Enable Verbose Logging

```python
# At top of any script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger('symbol_grounding').setLevel(logging.DEBUG)
```

### Inspect Neural Network Activations

```python
# Add hooks to see intermediate values
def activation_hook(module, input, output):
    print(f"{module.__class__.__name__}: input={input[0].shape}, output={output.shape}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")

snn.conv1.register_forward_hook(activation_hook)
snn.conv2.register_forward_hook(activation_hook)

# Run inference
output = snn(state, task_id=0)
```

### Visualize Hypervector Similarities

```python
import matplotlib.pyplot as plt
import numpy as np
from hypervec_rs import HyperVector

# Load codebook
import pickle
with open('codebook.pkl', 'rb') as f:
    codebook = pickle.load(f)

# Compute similarity matrix
concepts = list(codebook.keys())[:20]  # First 20 concepts
n = len(concepts)
sim_matrix = np.zeros((n, n))

for i, c1 in enumerate(concepts):
    for j, c2 in enumerate(concepts):
        sim_matrix[i, j] = codebook[c1].similarity(codebook[c2])

# Plot heatmap
plt.imshow(sim_matrix, cmap='viridis')
plt.xticks(range(n), concepts, rotation=90)
plt.yticks(range(n), concepts)
plt.colorbar()
plt.title('Concept Similarity Matrix')
plt.tight_layout()
plt.savefig('concept_similarities.png')
```

### Trace Decision Path

```python
# Enable decision tracing
from brain_fusion import fuse_decisions

action, decision_type, explanation = fuse_decisions(
    snn_output, None, predicates, verbose=True  # Add verbose flag
)

print("=" * 60)
print("DECISION TRACE")
print("=" * 60)
print(f"Predicates: {predicates}")
print(f"System 1 proposal: {explanation['system1_proposal']}")
print(f"System 1 confidence: {explanation['system1_confidence']:.3f}")
print(f"Surprise: {explanation['surprise']:.3f}")
if explanation['system2_invoked']:
    print(f"System 2 reason: {explanation['system2_reason']}")
    print(f"Counterfactual outcomes:")
    for act, outcome in explanation['counterfactuals'].items():
        print(f"  {act}: reward={outcome['reward']}, risk={outcome['risk']:.2%}")
print(f"Final action: {action} ({decision_type})")
print("=" * 60)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# ... run training ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 Memory Consumers ]")
for stat in top_stats[:10]:
    print(stat)
```

---

## Frequently Asked Questions

### Q: Can I run NSCK without a GPU?

**A**: Yes! NSCK is designed for edge devices. CPU-only mode works fine, just slower.
- Expected speed: 1-2 episodes/second (CPU) vs 10-15 episodes/second (GPU)
- For production deployment on Raspberry Pi or similar, CPU is the intended platform

### Q: How much memory does NSCK need?

**A**: Minimum requirements:
- Training: 8GB RAM (16GB recommended)
- Inference only: 4GB RAM
- GPU VRAM: 2GB (if using GPU acceleration)

### Q: Can I train on custom datasets?

**A**: Yes! See [API_REFERENCE.md](./API_REFERENCE.md) for examples of custom task creation.

Key steps:
1. Create game environment class with `reset()` and `step()` methods
2. Define symbolic predicates in `symbol_grounding.py`
3. Add task-specific head to `snn_qat.py`
4. Train using standard loop

### Q: Is NSCK compatible with neuromorphic hardware?

**A**: Yes, with some modifications:
- **Intel Loihi**: SNN architecture maps directly, but requires Nengo or Intel's toolchain
- **SpiNNaker**: Supported via PyNN interface (community contribution needed)
- **BrainChip Akida**: Requires weight conversion (ternary weights compatible)

### Q: How do I cite NSCK in research?

**A**: Use this BibTeX entry:
```bibtex
@software{nsck2026,
  title={Neuro-Symbolic Cognitive Kernel: Energy-Efficient Continuous Learning},
  author={shiva2321},
  year={2026},
  url={https://github.com/shiva2321/Node_network}
}
```

---

## Still Having Issues?

If your problem isn't covered here:

1. **Check GitHub Issues**: https://github.com/shiva2321/Node_network/issues
2. **Search Discussions**: https://github.com/shiva2321/Node_network/discussions
3. **Open New Issue**: Provide:
   - OS and Python version
   - Full error message
   - Minimal code to reproduce
   - What you've already tried

---

## Diagnostic Checklist

Before opening an issue, run this diagnostic script:

```python
#!/usr/bin/env python3
import sys
import torch
import platform

print("=" * 60)
print("NSCK DIAGNOSTIC REPORT")
print("=" * 60)

print(f"\n[System Info]")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")

print(f"\n[PyTorch]")
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\n[Dependencies]")
try:
    import snntorch
    print(f"✓ snntorch: {snntorch.__version__}")
except ImportError:
    print(f"✗ snntorch: NOT INSTALLED")

try:
    from hypervec_rs import HyperVector
    print(f"✓ hypervec_rs: Rust extension loaded")
    # Speed test
    import time
    hv1 = HyperVector.random()
    hv2 = HyperVector.random()
    start = time.time()
    for _ in range(1000):
        hv1.similarity(hv2)
    avg = (time.time() - start) / 1000 * 1000
    print(f"  VSA speed: {avg:.3f}ms per operation")
except ImportError:
    print(f"✗ hypervec_rs: NOT BUILT")

print(f"\n[Files]")
import os
if os.path.exists('codebook.pkl'):
    size = os.path.getsize('codebook.pkl') / 1024
    print(f"✓ codebook.pkl: {size:.1f} KB")
else:
    print(f"✗ codebook.pkl: MISSING")

if os.path.exists('snn_task_aware.pth'):
    size = os.path.getsize('snn_task_aware.pth') / 1024
    print(f"✓ snn_task_aware.pth: {size:.1f} KB")
else:
    print(f"✗ snn_task_aware.pth: MISSING")

print("=" * 60)
print("Copy this output when reporting issues")
print("=" * 60)
```

Save as `diagnose.py` and run:
```bash
python diagnose.py
```

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Contributors**: NSCK Development Team
