
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class SaliencyVisualizer:
    def __init__(self, model):
        """
        Initialize Grad-CAM Visualizer.
        Args:
            model: The TaskAwareSNN model instance.
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the last convolutional layer of the UniversalEncoder
        # Target: model.encoder.visual_conv2
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0] # [Batch, Channel, H, W]

        def forward_hook(module, input, output):
            self.activations = output # [Batch, Channel, H, W]

        # Locate layer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'visual_conv2'):
            target_layer = self.model.encoder.visual_conv2
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        else:
            print("[SALIENCY] Warning: Could not find visual_conv2 in model.encoder")

    def generate_heatmap(self, input_tensor, task_name, action_idx=None):
        """
        Generate Grad-CAM heatmap for a given input.
        Args:
            input_tensor: [1, C, H, W] input image.
            task_name: 'snake', 'pong', etc.
            action_idx: Index of action to explain. If None, explains max output.
        Returns:
            heatmap_color: [H, W, 3] uint8 numpy array (JET colormap) applied to original size.
        """
        # 1. Forward Pass
        self.model.zero_grad()
        logits, _ = self.model(input_tensor, task_name=task_name)
        
        # 2. Select Target (Action)
        if action_idx is None:
            action_idx = logits.argmax(dim=1).item()
            
        # 3. Backward Pass
        # We want to maximize the output of the chosen action neuron
        target = logits[0, action_idx]
        target.backward(retain_graph=True)
        
        # 4. Compute Grad-CAM
        # Gradients: [1, 32, 10, 10] (if padded) or [1, 32, 10, 10]
        # Activations: [1, 32, 10, 10]
        
        if self.gradients is None or self.activations is None:
            return None
            
        # Global Average Pooling of Gradients -> Weights alpha
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # [32]
        
        # Weight the activations (Clone to avoid In-Place Error on live graph)
        # [1, 32, H, W] * [32, 1, 1]
        weighted_activations = self.activations.clone()
        for i in range(len(pooled_gradients)):
             weighted_activations[:, i, :, :] *= pooled_gradients[i]
             
        # Average the channels -> Heatmap
        heatmap = torch.mean(weighted_activations, dim=1).squeeze() # [H, W]
        
        # ReLU (Focus on Positive influence only)
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.detach().cpu().numpy()
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        # Resize to Match Input
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(heatmap, (input_w, input_h))
        
        # Apply Color Map
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        return heatmap_color
