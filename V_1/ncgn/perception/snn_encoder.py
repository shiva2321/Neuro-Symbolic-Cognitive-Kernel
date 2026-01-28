import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Tuple

class SNNEncoder(nn.Module):
    """
    Spiking Convolutional Neural Network (SCNN) Encoder.
    
    This module acts as the "Perception" layer. It ingests spatial inputs (images),
    transduces them into temporal spike trains, and processes them through LIF neurons.
    The output is a 'Rate Vector' representing the firing frequency of the output layer,
    which is then fed to the VSA Bridge.
    
    Architecture:
    Conv2d -> MaxPool -> LIF -> Conv2d -> MaxPool -> LIF -> Flatten -> Linear -> LIF
    """
    
    def __init__(self, num_steps: int = 25, beta: float = 0.5, output_size: int = 128):
        """
        Args:
            num_steps (int): Number of time steps to simulate.
            beta (float): Decay rate of the LIF neurons (0.0 to 1.0).
            output_size (int): Dimension of the output feature vector.
        """
        super().__init__()
        self.num_steps = num_steps
        self.output_size = output_size
        
        # Surrogate gradient (Fast Sigmoid) for backprop
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1)
        
        # Layer 2
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1)
        
        # Layer 3 (Output)
        self.flatten = nn.Flatten()
        # Calculate size: 28x28 -> conv1(5x5) -> 24x24 -> mp1 -> 12x12
        # 12x12 -> conv2(5x5) -> 8x8 -> mp2 -> 4x4
        # 32 channels * 4 * 4 = 512
        self.fc = nn.Linear(32 * 4 * 4, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1, output=True) 
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (Tensor): Input image batch. Shape: (Batch, 1, 28, 28)
            
        Returns:
            Tensor: Spike Rate Vector. Shape: (Batch, output_size)
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer spikes
        spk3_rec = []
        
        # Time Loop
        for step in range(self.num_steps):
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc(self.flatten(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            
        # Stack time steps: (Steps, Batch, Output)
        spk3_rec = torch.stack(spk3_rec, dim=0)
        
        # Calculate Rate: Mean firing rate over time window
        # Shape: (Batch, Output)
        rate_vector = torch.mean(spk3_rec.float(), dim=0)
        
        return rate_vector
