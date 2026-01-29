import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# --- TERNARY QUANTIZATION LOGIC ---
class TernaryQuantize(torch.autograd.Function):
    """
    Ternary Weight Quantization: Maps weights to {-1, 0, 1}
    Uses Straight-Through Estimator (STE) for gradients.
    """
    @staticmethod
    def forward(ctx, input, delta=0.1):
        # Scale to max magnitude
        scale = input.abs().max() + 1e-6
        x = input / scale
        
        # Ternarize: -1 if < -delta, 1 if > delta, 0 otherwise
        out = torch.zeros_like(x)
        out[x > delta] = 1.0
        out[x < -delta] = -1.0
        
        return out * scale # Multiply back scale for magnitude preservation

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through as is
        return grad_output, None

def ternarize_weight(w):
    return TernaryQuantize.apply(w)

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
        # LATE FUSION:
        # +1 for Task ID
        # +4 for Compass (Above, Below, Left, Right)
        self.fc_shared = nn.Linear((32 * 3 * 3) + 1 + 4, 64)
        self.lif_shared = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5)

        # 2. Specialized Heads (The "Task Experts")
        self.head_snake = nn.Linear(64, 4) # UP, DOWN, LEFT, RIGHT
        self.head_pong  = nn.Linear(64, 2) # UP, DOWN
        self.head_chars = nn.Linear(64, 62) # 0-9, A-Z, a-z
        self.lif_out    = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True, threshold=0.5)

    def forward(self, x, task_id, compass=None):
        """
        Forward pass.
        Args:
            x: Visual input [Batch, 4, 10, 10]
            task_id: Scalar task index (0=Pong, 1=Snake, 2=Chars)
            compass: Optional 4-bit direction vector [Batch, 4] (Above, Below, Left, Right)
        """
        # 1. APPLY TERNARY QUANTIZATION TO WEIGHTS (ON THE FLY)
        # This keeps the float weights for gradients but uses Ternary for inference
        w_conv1 = ternarize_weight(self.conv1.weight)
        w_conv2 = ternarize_weight(self.conv2.weight)
        w_fc_s  = ternarize_weight(self.fc_shared.weight)
        w_h_sn  = ternarize_weight(self.head_snake.weight)
        w_h_po  = ternarize_weight(self.head_pong.weight)
        w_h_ch  = ternarize_weight(self.head_chars.weight)

        # Init State
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_shared = self.lif_shared.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        
        # Prepare Compass Tensor
        if compass is None:
            compass_tensor = torch.zeros((x.size(0), 4), device=x.device)
        else:
            compass_tensor = compass.to(x.device).float()

        # Simulation Steps (T=8)
        for step in range(8):
            # Layer 1 (Functional to use quantized W)
            cur1 = torch.nn.functional.conv2d(x, w_conv1, stride=2, padding=1)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = torch.nn.functional.conv2d(spk1, w_conv2, stride=2, padding=1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Shared Linear + Late Fusion
            flat = self.flatten(spk2)
            
            # Inject Context (Task ID + Compass)
            task_tensor = torch.full((x.size(0), 1), float(task_id), device=x.device)
            combined = torch.cat([flat, task_tensor, compass_tensor], dim=1) 
            
            cur_shared = torch.nn.functional.linear(combined, w_fc_s)
            spk_shared, mem_shared = self.lif_shared(cur_shared, mem_shared)

            # Task Switching Head (Functional to use quantized W)
            if task_id == 1: # Snake
                cur_out = torch.nn.functional.linear(spk_shared, w_h_sn, self.head_snake.bias)
            elif task_id == 0: # Pong
                cur_out = torch.nn.functional.linear(spk_shared, w_h_po, self.head_pong.bias)
            else: # Characters (Task 2)
                cur_out = torch.nn.functional.linear(spk_shared, w_h_ch, self.head_chars.bias) 
            
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec, dim=0).sum(0) # Sum spikes = Rate Code

if __name__ == "__main__":
    model = TaskAwareSNN()
    print("TaskAwareSNN Initialized (Late Fusion)")
