import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Adapted to standard nn.Conv2d to ensure compatibility without Brevitas dependency
class TaskAwareSNN(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # 1. Shared Visual Cortex (2 Channels -> Features)
        # Input: [Batch, 4, 10, 10] (Expanded temporal window: 4 frames)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5)
        
        self.flatten = nn.Flatten()
        
        # Shared Latent Space (32 channels * 3x3 spatial = 288 flat)
        # LATE FUSION: We add +1 for the Task ID injection here
        self.fc_shared = nn.Linear((32 * 3 * 3) + 1, 64)
        self.lif_shared = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5)

        # 2. Specialized Heads (The "Task Experts")
        self.head_snake = nn.Linear(64, 4) # UP, DOWN, LEFT, RIGHT
        self.head_pong  = nn.Linear(64, 2) # UP, DOWN
        self.lif_out    = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True, threshold=0.5)

    def forward(self, x, task_id):
        # x shape: [Batch, 4, 10, 10] (Frames)
        # task_id: 0 (Pong) or 1 (Snake)
        
        # Init State
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_shared = self.lif_shared.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        
        # Simulation Steps (T=8)
        for step in range(8):
            # Layer 1
            cur1 = self.conv1(x)
            # BN skipped for stability as before
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Shared Linear + Late Fusion
            flat = self.flatten(spk2) # [Batch, 288]
            
            # Inject Context Here (Late Fusion)
            # Create (Batch, 1) tensor for task_id
            task_tensor = torch.full((x.size(0), 1), float(task_id), device=x.device)
            combined = torch.cat([flat, task_tensor], dim=1) # [Batch, 289]
            
            cur_shared = self.fc_shared(combined)
            spk_shared, mem_shared = self.lif_shared(cur_shared, mem_shared)

            # Task Switching Head
            if task_id == 1: # Snake
                cur_out = self.head_snake(spk_shared)
            else: # Pong
                cur_out = self.head_pong(spk_shared)
            
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec, dim=0).sum(0) # Sum spikes = Rate Code

if __name__ == "__main__":
    model = TaskAwareSNN()
    print("TaskAwareSNN Initialized (Late Fusion)")
