"""
NSCK Perception Module
Handles SNN inference, frame processing, and entropy calculation.
"""
import torch
import numpy as np
from typing import Tuple, Optional, List
from collections import deque


def calculate_entropy(probs_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy of probability distribution.
    
    H(p) = -sum(p * log(p))
    
    Args:
        probs_tensor: Probability distribution [batch, classes]
        
    Returns:
        Entropy per sample [batch]
    """
    p = torch.clamp(probs_tensor, 1e-6, 1.0)
    entropy = -torch.sum(p * torch.log(p), dim=1)
    return entropy


def preprocess_frame(img: np.ndarray, target_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """
    Preprocess a game frame for SNN input.
    
    Args:
        img: Grayscale image (any size)
        target_size: Target dimensions (width, height)
        
    Returns:
        Normalized float32 array [H, W]
    """
    import cv2
    
    if img.shape != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img.astype(np.float32) / 255.0


def stack_frames(
    current_frame: np.ndarray,
    history: deque,
    num_frames: int = 4
) -> np.ndarray:
    """
    Stack frames for temporal input to SNN.
    
    Args:
        current_frame: Current preprocessed frame [H, W]
        history: Deque of previous frames
        num_frames: Number of frames to stack
        
    Returns:
        Stacked frames [num_frames, H, W]
    """
    # Bootstrap history if empty
    if len(history) == 0:
        for _ in range(num_frames - 1):
            history.append(current_frame)
    
    history.append(current_frame)
    
    frames_list = list(history)
    while len(frames_list) < num_frames:
        frames_list.insert(0, frames_list[0])
    
    return np.stack(frames_list[-num_frames:], axis=0)


class PerceptionEngine:
    """
    Handles SNN-based perception for NSCK.
    
    Encapsulates:
    - Frame preprocessing
    - Frame stacking
    - SNN forward pass
    - Entropy calculation
    """
    
    def __init__(self, model, device: str = "cpu"):
        """
        Initialize perception engine.
        
        Args:
            model: TaskAwareSNN model
            device: Torch device string
        """
        self.model = model
        self.device = torch.device(device)
        self.frame_histories = {
            "snake": deque(maxlen=4),
            "pong": deque(maxlen=4),
        }
    
    def infer(
        self,
        frame: np.ndarray,
        task_id: int,
        game_type: str,
        model_lock=None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run SNN inference on a frame.
        
        Args:
            frame: Preprocessed frame [H, W]
            task_id: Task ID (0=pong, 1=snake, 2=chars)
            game_type: Game type string for frame history
            model_lock: Optional threading lock
            
        Returns:
            Tuple of (logits, probabilities, entropy)
        """
        # Stack frames
        history = self.frame_histories.get(game_type, deque(maxlen=4))
        stacked = stack_frames(frame, history)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(stacked).float().to(self.device)
        input_tensor = input_tensor.unsqueeze(0)  # [1, 4, H, W]
        
        # Forward pass
        if model_lock:
            with model_lock:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(input_tensor, task_id)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor, task_id)
        
        probs = torch.softmax(logits, dim=1)
        entropy = calculate_entropy(probs).item()
        
        return logits, probs, entropy
    
    def get_action(self, probs: torch.Tensor) -> int:
        """Get argmax action from probabilities."""
        return probs.argmax(dim=1).item()
    
    def reset_history(self, game_type: str):
        """Clear frame history for a game."""
        if game_type in self.frame_histories:
            self.frame_histories[game_type].clear()
