
import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path



from python.core.neural.snn_qat import TaskAwareSNN, UniversalEncoder
from python.core.perception.saliency import SaliencyVisualizer

def test_saliency_heatmap_generation():
    """Verify Grad-CAM heatmap generation."""
    
    # Setup
    device = torch.device("cpu")
    model = TaskAwareSNN().to(device)
    model.register_task("test_saliency", num_actions=4)
    model.eval() # Hooks work in eval mode too usually, but for Grad-CAM we need gradients.
    
    # Saliency Visualizer depends on gradients, so we might need train mode or manually enable grad
    visualizer = SaliencyVisualizer(model)
    
    # Dummy Input [1, 4, 10, 10]
    input_tensor = torch.randn(1, 4, 10, 10, requires_grad=True).to(device)
    
    # Run Generation
    # We expect a (10, 10, 3) image
    heatmap = visualizer.generate_heatmap(input_tensor, task_name="test_saliency")
    
    assert heatmap is not None
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (10, 10, 3)
    assert heatmap.dtype == np.uint8
    
    # Check that hooks captured something
    assert visualizer.gradients is not None
    assert visualizer.activations is not None

def test_saliency_hooks_registered():
    """Verify hooks are attached to the model."""
    model = TaskAwareSNN()
    visualizer = SaliencyVisualizer(model)
    
    # Check PyTorch hooks
    # model.encoder.visual_conv2._forward_hooks should not be empty
    assert len(model.encoder.visual_conv2._forward_hooks) > 0
    assert len(model.encoder.visual_conv2._backward_hooks) > 0 # Full backward hook
