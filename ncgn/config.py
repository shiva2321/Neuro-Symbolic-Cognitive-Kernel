"""
NCGN v7 Configuration Module

Central configuration for all system parameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Central configuration for NCGN v7.
    
    All parameters are tunable from a single location.
    """
    
    # Capacity and sizing
    initial_capacity: int = 10000
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Physics parameters
    decay_delta: float = 0.1          # Energy decay per tick
    flow_alpha: float = 0.8           # Propagation conductivity
    norm_beta: float = 0.01           # Divisive normalization constant
    default_threshold: float = 0.75   # Firing threshold
    firing_threshold: float = 0.9     # Actual fire trigger
    refractory_period: int = 10       # Ticks after firing
    seizure_threshold: float = 50.0   # Global energy cap
    seizure_damping: float = 0.5      # Damping factor when exceeded
    
    # Learning parameters
    learning_rate: float = 0.01
    weight_min: float = 0.0
    weight_max: float = 1.0
    genesis_threshold: float = 0.8    # Hebbian term for new edges
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM settings
    llm_model_path: Optional[str] = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    llm_context_window: int = 4096
    llm_gpu_layers: int = -1  # -1 = all layers on GPU
    llm_max_retries: int = 3
    
    # Activity thresholds
    active_energy_threshold: float = 0.05   # Minimum to be "active"
    top_k_concepts: int = 20                # Context for LLM


# Default singleton instance
DEFAULT_CONFIG = Config()
