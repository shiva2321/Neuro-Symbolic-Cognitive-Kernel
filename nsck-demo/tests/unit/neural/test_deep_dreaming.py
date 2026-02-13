
import pytest
import torch
import time
import os
import sys
from pathlib import Path

# Add python/ directory to sys.path

from python.core.memory.intelligent_buffer import IntelligentReplayBuffer, Experience
from python.servers.python_server import perform_dreaming_cycle, REPLAY_BUFFER
import python.core.neural.snn_qat
from python.core.neural.snn_qat import TaskAwareSNN, UniversalEncoder

# Mock global variables
import python.servers.python_server
import threading
python.servers.python_server.REPLAY_BUFFER = REPLAY_BUFFER 
python.servers.python_server.REPLAY_BATCH_SIZE = 2
python.servers.python_server.model_lock = threading.Lock() # Mock Lock

def test_deep_dreaming_cycle():
    """Verify that the dreaming cycle runs without error and computes loss."""
    
    # Setup Logic
    device = torch.device("cpu")
    model = TaskAwareSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. Register Task Heads
    model.register_task("test_dream", num_actions=4)
    
    # 2. Populate Buffer with High Priority items (forcing archival)
    # We need enough to flush to disk.
    # We will manually archive to test persistence reading
    state = torch.randn(4, 1, 1).float() # Dummy 1D state
    exp = Experience(
        state=state, action_idx=1, reward=1.0, next_state=state, 
        done=False, task_name="test_dream", priority=10.0, timestamp=time.time()
    )
    
    # Add to buffer and FORCE archival by adding dummy items
    # Buffer capacity is 10000 by default, so we call _archive_to_disk processing directly
    REPLAY_BUFFER._archive_to_disk(exp)
    REPLAY_BUFFER.store.flush_episodes()
    
    # 3. Trigger Dreaming Cycle
    # This should:
    # a) Sample from disk
    # b) Run forward pass
    # c) Backprop
    
    try:
        perform_dreaming_cycle(model, optimizer, device)
        success = True
    except Exception as e:
        pytest.fail(f"Dreaming cycle failed with error: {e}")
        success = False
        
    assert success
