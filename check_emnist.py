
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def check_emnist():
    transform = transforms.Compose([
        transforms.Resize((10, 10)),
        transforms.ToTensor(),
    ])
    
    try:
        ds = datasets.EMNIST(root='./data_mnist', split='byclass', train=True, download=True, transform=transform)
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            img_tensor, label = ds[i]
            img = img_tensor.squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {label}")
        
        plt.savefig('emnist_samples.png')
        print("Saved emnist_samples.png. Please check if characters are upright or transposed.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_emnist()
