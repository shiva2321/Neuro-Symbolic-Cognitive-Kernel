import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from universal_encoder import UniversalEncoder

# --- TERNARY QUANTIZATION LOGIC ---
class TernaryQuantize(torch.autograd.Function):
    """
    Ternary Weight Quantization: Maps weights to {-1, 0, 1}
    Uses Straight-Through Estimator (STE) for gradients.
    """
    @staticmethod
    def forward(ctx, input, delta=0.1):
        scale = input.abs().max() + 1e-6
        x = input / scale
        out = torch.zeros_like(x)
        out[x > delta] = 1.0
        out[x < -delta] = -1.0
        return out * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ternarize_weight(w):
    return TernaryQuantize.apply(w)

class TaskAwareSNN(nn.Module):
    """
    NSCK v2.0: Universal Actor-Critic Brain.
    - Decoupled from Sensor Format (uses UniversalEncoder).
    - Decoupled from Action Space (uses DynamicHeads).
    - Returns: (PolicyLogits, ValueEstimate)
    """
    def __init__(self, beta=0.5):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # 1. Universal Cortex (The "Eye/Ear")
        self.encoder = UniversalEncoder(latent_dim=128)
        
        # 2. Association Area (SNN Core)
        # Input: 128 (Latent) -> Hidden: 128 -> Output: 128
        self.fc_shared = nn.Linear(128, 128)
        self.lif_shared = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5)

        # 3. Dynamic Heads (The "Motor Cortex")
        # Stores "Actor" (Policy) and "Critic" (Value) for each task
        # Format: {"snake": nn.ModuleDict({"actor": ..., "critic": ...})}
        self.heads = nn.ModuleDict() 
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True, threshold=0.5)

    def register_task(self, task_name, num_actions):
        """
        Grow new neurons for a new task.
        """
        if task_name not in self.heads:
            print(f"[Brain] Growing new Neocortex segment for task: {task_name}")
            self.heads[task_name] = nn.ModuleDict({
                "actor": nn.Linear(128, num_actions),   # [Batch, Actions]
                "critic": nn.Linear(128, 1)             # [Batch, 1] - How good is this state?
            })
            
            # Send to device if model is already on GPU
            if next(self.parameters()).is_cuda:
                self.heads[task_name].to("cuda")

    def forget_task(self, task_name):
        """
        Synaptic Pruning: Removes a task's neural circuitry continuously.
        Useful for forgetting obsolete tasks or freeing memory.
        """
        if task_name in self.heads:
            del self.heads[task_name]
            print(f"[Brain] Pruned Cortex segment for task: {task_name}")

    def forward(self, x, task_name="snake", modality_hint=None):
        """
        Universal Forward Pass.
        Returns: (Actor_Logits, Critic_Value)
        """
        # A. PERCEPTION (Universal Encoder)
        # Output: [Batch, 256] (Latent Thought)
        latent = self.encoder(x, modality_hint)

        # B. COGNITION (SNN Association Loop)
        # We quantize weights on the fly for efficiency
        w_shared = ternarize_weight(self.fc_shared.weight)

        mem_shared = self.lif_shared.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        val_rec = []

        # Ensure we have a head for this task
        if task_name not in self.heads:
            # Auto-register with default 4 actions if unknown (failsafe)
            self.register_task(task_name, 4)
            
        head = self.heads[task_name]
        w_actor = ternarize_weight(head["actor"].weight)
        # Critic is NOT quantized (needs high precision for value estimation)
        
        # Use analog readout for char recognition (62-way) to avoid dead/flat spike codes.
        use_analog_readout = (task_name == "char_recognition")

        # SNN Loop (T=8)
        # Note: Encoder run once (static perception), SNN runs over time (processing)
        for step in range(8):
            # 1. Association
            cur_shared = torch.nn.functional.linear(latent, w_shared, self.fc_shared.bias)
            spk_shared, mem_shared = self.lif_shared(cur_shared, mem_shared)

            if use_analog_readout:
                # Actor/Critic read out from membrane (dense analog feature)
                feat = mem_shared
                cur_actor = torch.nn.functional.linear(feat, w_actor, head["actor"].bias)
                # Still pass through lif_out so the rest of the system expectations stay intact
                spk_actor, mem_out = self.lif_out(cur_actor, mem_out)
                value = head["critic"](feat)
            else:
                # Default spiking readout (works well for low-action tasks like snake/pong/maze)
                cur_actor = torch.nn.functional.linear(spk_shared, w_actor, head["actor"].bias)
                spk_actor, mem_out = self.lif_out(cur_actor, mem_out)
                value = head["critic"](spk_shared)

            spk_rec.append(spk_actor)
            val_rec.append(value)

        # Rate Coding: Sum spikes for action strength
        action_logits = torch.stack(spk_rec, dim=0).sum(0)
        
        # Value: Average value estimate over time
        value_estimate = torch.stack(val_rec, dim=0).mean(0)
        
        return action_logits, value_estimate

if __name__ == "__main__":
    model = TaskAwareSNN()
    model.register_task("snake", 4)
    model.register_task("chat", 5000) # Text task
    print("Universal Actor-Critic Brain Initialized")
