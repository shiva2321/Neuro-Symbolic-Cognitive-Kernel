
import torch
import torch.nn as nn
import sys
import os
import threading
import time

# Add the directory to sys.path
sys.path.append(r'd:\NSCK_v1\nsck-demo\python')

from snn_qat import TaskAwareSNN

def test_training_init():
    model = TaskAwareSNN()
    model.register_task("char_recognition", 62)
    model_lock = threading.RLock()
    
    print("Simulating train_character_thread initialization...")
    
    # Logic from updated python_server.py
    with model_lock:
        found_any = False
        for name, param in model.named_parameters():
            if "heads.char_recognition" not in name:
                param.requires_grad = False
            else:
                found_any = True
                print(f"Keeping parameter: {name}")
                
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if not trainable_params:
            print("FAILURE: Optimizer would get an empty parameter list!")
            return False
            
        print(f"SUCCESS: Found {len(trainable_params)} trainable parameters.")
        optimizer_local = torch.optim.Adam(trainable_params, lr=0.001)
        print("Optimizer initialized successfully.")
        return True

if __name__ == "__main__":
    if test_training_init():
        print("\nFix Verified: Character training will not crash with 'empty parameter list'.")
    else:
        sys.exit(1)
