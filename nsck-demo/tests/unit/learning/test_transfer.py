
import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import copy

# Add python/ directory to sys.path

from python.core.neural.snn_qat import TaskAwareSNN, UniversalEncoder

def check_weights_changed(state_dict_before, state_dict_after, layer_name):
    """Returns True if weights in the specified layer have changed."""
    w_before = state_dict_before[layer_name]
    w_after = state_dict_after[layer_name]
    return not torch.equal(w_before, w_after)

def test_transfer_learning_freeze():
    """
    Verify Zero-Shot/Transfer capability:
    1. Train Task A (Encoder + Head A updates)
    2. Freeze Encoder
    3. Train Task B (Only Head B updates, Encoder stays fixed)
    """
    device = torch.device("cpu")
    model = TaskAwareSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # 1. Setup Task A
    model.register_task("task_a", num_actions=2)
    
    # Snapshot 1
    state_dict_1 = copy.deepcopy(model.state_dict())
    
    # Train Task A
    # Input: [1, 4, 10, 10]
    x = torch.randn(1, 4, 10, 10).to(device)
    logits, val = model(x, task_name="task_a")
    # Use MSE loss to force output towards 10.0 -> ensures gradients exist (unless output is already 10)
    loss = (logits - 10.0).pow(2).mean() 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Verify Task A updated Encoder
    # We check a specific encoder weight (e.g. encoder.visual_conv1.weight)
    encoder_layer = "encoder.visual_conv1.weight"
    current_state = model.state_dict()
    assert check_weights_changed(state_dict_1, current_state, encoder_layer), "Encoder should change during Task A training"
    
    # 2. Setup Task B & FREEZE ENCODER
    model.register_task("task_b", num_actions=2)
    
    # Freeze Encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # Snapshot 2
    state_dict_2 = copy.deepcopy(model.state_dict())
    
    # Train Task B
    # We must re-create optimizer or update param groups, but easier to just make a new one filtering frozen params
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    
    # Train Task B for multiple steps to ensure update
    for _ in range(10):
        logits_b, val_b = model(x, task_name="task_b")
        loss_b = (logits_b - 10.0).pow(2).mean()
        
        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()
    
    # 3. Assertions
    current_state_final = model.state_dict()
    
    # Encoder should NOT change (Frozen)
    assert not check_weights_changed(state_dict_2, current_state_final, encoder_layer), "Encoder MUST remain frozen during Transfer"
    
    # Task B Head SHOULD change
    head_layer = "heads.task_b.actor.weight"
    assert check_weights_changed(state_dict_2, current_state_final, head_layer), "Task B Head MUST learn even if Encoder is frozen"

    print(">> Verified: Encoder Frozen, New Head Learned.")
