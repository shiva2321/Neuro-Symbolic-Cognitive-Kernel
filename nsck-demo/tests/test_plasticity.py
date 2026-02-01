
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plastic_snn import PlasticSNN, SparseLinear

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_plasticity():
    set_seed(42)
    print("Initializing Plastic SNN...")
    
    # Simple Regression Task
    model = PlasticSNN(input_dim=10, hidden_dim=20, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Dummy Data
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    
    print("\n[Test 1] Sparsity Maintenance & Deep Rewiring")
    layer = model.fc1
    initial_mask = layer.get_mask()
    
    initial_density = initial_mask.mean().item()
    print(f"Initial Density: {initial_density:.2%} (Target: {layer.sparsity:.0%})")
    
    # Train Loop with L1 Reg (Forces Pruning)
    active_drift_detected = False
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # L1 Regularization on Theta (to drive pruning)
        l1_reg = 0.0
        for name, param in model.named_parameters():
            if 'theta' in name:
                l1_reg += 1e-2 * param.abs().sum()
                
        total_loss = loss + l1_reg
        total_loss.backward()
        optimizer.step()
        
        # REWIRE Step
        model.fc1.rewire()
        model.fc2.rewire()
        
        # measure drift
        if epoch % 10 == 0:
            current_mask = layer.get_mask()
            drift = (initial_mask != current_mask).float().sum().item()
            density = current_mask.mean().item()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Drift={drift}, Density={density:.2%}")
            
            if drift > 0:
                active_drift_detected = True
            
            # Update initial mask for next delta
            initial_mask = current_mask.clone()

    if active_drift_detected:
        print(">> PASSED: Connectivity is evolving (Drift > 0).")
    else:
        print(">> FAILED: Weights are static.")
        
    final_density = layer.get_mask().mean().item()
    if abs(final_density - layer.sparsity) < 0.05:
         print(f">> PASSED: Sparsity maintained ({final_density:.2%} vs {layer.sparsity:.0%})")
    else:
         print(f">> FAILED: Sparsity diverged.")
         
    print("\n[Test 2] Neurogenesis (Plateau Trigger)")
    # Force a plateau by feeding static trash
    # Actually, we can just manually trigger it to test the mechanism logic
    # But let's verify the detector logic.
    
    print("Simulating Plateau...")
    # Feed constant loss to history
    for _ in range(30):
        triggered = model.check_plateau(0.5) # Constant loss = Plateau
        if triggered:
            print("Plateau Detected! Triggering Growth...")
            # Capture size before
            size_before = model.fc1.out_features
            
            # GROW
            model.expand_capacity(new_neurons=10)
            
            # Need to Re-Init Optimizer after parameter change
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            size_after = model.fc1.out_features
            print(f"Growth: {size_before} -> {size_after} neurons.")
            
            if size_after == size_before + 10:
                print(">> PASSED: Neurogenesis resized layer.")
            else:
                 print(">> FAILED: Layer size did not change.")
            break
            
    # Sanity Check: Forward pass still works
    try:
        y = model(inputs)
        print(">> PASSED: Forward pass survival after growth.")
    except Exception as e:
        print(f">> FAILED: Crash after growth: {e}")

if __name__ == "__main__":
    test_plasticity()
