
import torch
import sys
import os

# Add the directory to sys.path
sys.path.append(r'd:\NSCK_v1\nsck-demo\python')

try:
    from snn_qat import TaskAwareSNN
    model = TaskAwareSNN()
    model.register_task("char_recognition", 62)
    
    print("--- Model Parameters ---")
    found = False
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        if "heads.char_recognition" in name:
            found = True
            
    if found:
        print("\nSUCCESS: Found 'heads.char_recognition' in parameter names.")
        
        # Test the filtering logic that was failing
        trainable_params = [p for name, p in model.named_parameters() if "heads.char_recognition" in name]
        print(f"Number of trainable parameters for char head: {len(trainable_params)}")
        
        if len(trainable_params) > 0:
            optimizer = torch.optim.Adam(trainable_params, lr=0.001)
            print("Optimizer initialized successfully!")
    else:
        print("\nFAILURE: Could not find 'heads.char_recognition' in parameter names.")

except Exception as e:
    print(f"Error: {e}")
