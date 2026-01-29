"""
NSCK Configuration Module
Centralized configuration for hyperparameters and settings.
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class NSCKConfig:
    """Central configuration for NSCK system."""
    
    # === Hardware ===
    device: str = "cpu"  # "cpu" or "cuda"
    
    # === SNN Hyperparameters ===
    learning_rate: float = 1e-3
    beta: float = 0.5  # LIF neuron decay
    
    # === Training ===
    sleep_epochs: int = 5
    replay_batch_size: int = 32
    save_interval: float = 60.0  # seconds
    
    # === VSA ===
    vsa_strength: float = 5.0
    confidence_threshold: float = 0.6  # Entropy threshold for VSA rescue
    
    # === Game ===
    grid_size: int = 10
    adversarial_rate: float = 0.05
    
    # === Ablation Flags ===
    enable_snn: bool = True
    enable_vsa: bool = True
    enable_sleep: bool = True
    freeze_pong: bool = False
    no_teacher: bool = False
    
    # === Paths ===
    model_path: str = "snn_task_aware.pth"
    persistence_db: str = "nsck_brain.db"
    
    # === ZMQ Ports ===
    zmq_pull_port: int = 5565
    zmq_pub_port: int = 5566
    zmq_stats_port: int = 5567
    
    # === Episode Memory (Phase 2) ===
    episode_capacity: int = 10000
    full_state_capacity: int = 1000
    
    @classmethod
    def from_env(cls) -> "NSCKConfig":
        """Create config from environment variables with defaults."""
        return cls(
            device=os.getenv("NSCK_DEVICE", "cpu"),
            learning_rate=float(os.getenv("NSCK_LR", "1e-3")),
            model_path=os.getenv("NSCK_MODEL_PATH", "snn_task_aware.pth"),
            persistence_db=os.getenv("NSCK_DB_PATH", "nsck_brain.db"),
        )


# Global default config (can be overridden)
DEFAULT_CONFIG = NSCKConfig()
