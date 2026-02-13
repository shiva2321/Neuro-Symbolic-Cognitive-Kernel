import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from python.core.neural.snn_qat import QuantizedSNN
import numpy as np
import os

# Constants
GRID_SIZE = 10
NUM_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 5

def generate_synthetic_data():
    """
    Generates synthetic 10x10 grids.
    Labels: 0: No Danger/Neutral, 1: Apple Ahead, 2: Wall/Danger Ahead
    For simplicity, let's assume the snake head is always at center (5,5) 
    and we classify what is directly in front (let's say NORTH).
    
    Representation:
    0: Empty
    1: Snake Body (Self) - Treat as Wall
    2: Apple
    3: Wall (Boundary)
    """
    X = []
    y = []
    
    for _ in range(NUM_SAMPLES):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Random Apple position
        ax, ay = np.random.randint(0, GRID_SIZE, 2)
        grid[ax, ay] = 2.0 # Apple
        
        # Wall boundaries are implicit logic, but visual representation:
        # Let's say we input the local view or full grid.
        # User said "10x10 dataset (Apple, Wall, Empty)"
        
        # Let's just create random patterns classified by a simple rule 
        # to ensure the network learns something.
        # Rule: If pixel (5, 4) (North of center) is Apple -> Label 0 (Apple)
        #       If pixel (5, 4) is Wall -> Label 1 (Wall)
        #       Else -> Label 2 (Empty)
        
        # Place "Northern neighbor"
        target_pos = (5, 4)
        val = np.random.choice([0, 2, 3], p=[0.6, 0.2, 0.2]) # 0=Empty, 2=Apple, 3=Wall
        grid[target_pos] = val
        
        # Labeling
        if val == 2:
            label = 0 # Apple
        elif val == 3:
            label = 1 # Wall
        else:
            label = 2 # Empty
            
        X.append(grid.flatten())
        y.append(label)
        
    return torch.tensor(np.array(X)), torch.tensor(np.array(y), dtype=torch.long)

def train():
    X, y = generate_synthetic_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = QuantizedSNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in loader:
            optimizer.zero_grad()
            
            # Forward pass: [Time, Batch, Out]
            spikes = model(data)
            
            # Aggregate spikes (sum over time) to get rate/logit
            # Sum predictions over time
            sum_spikes = spikes.sum(dim=0)
            
            loss = criterion(sum_spikes, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(sum_spikes.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}, Acc: {100 * correct / total:.2f}%")

    # Export
    torch.save(model.state_dict(), "snn_model.pth")
    # model.export_onnx("snn_qat.onnx") # Optional export
    print("Model saved to snn_model.pth")

if __name__ == "__main__":
    train()
