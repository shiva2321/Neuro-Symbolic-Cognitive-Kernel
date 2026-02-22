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
    novelty_threshold: float = 0.5 # [AGI] Curiosity threshold
    
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
    memory_capacity: int = 2500 # [AGI] Recent memory capacity
    
    # === Rule Learning ===
    min_rule_support: int = 5
    min_rule_confidence: float = 0.7
    min_success_rate: float = 0.6  # [AGI] Added for consistency

    # === V3 Feature Flags (all False by default — preserves existing behaviour) ===
    enable_construction_grammar: bool = False
    enable_frame_semantics: bool = False
    enable_coreference: bool = False
    enable_contextual_encoding: bool = False
    enable_free_energy_beliefs: bool = False
    enable_distributional_semantics: bool = False
    enable_incremental_concept_refinement: bool = False
    enable_dual_process: bool = False
    system1_confidence_threshold: float = 0.75
    enable_conceptual_blending: bool = False
    enable_hnsw_index: bool = False
    enable_homeostasis: bool = False
    enable_stigmergy: bool = False
    enable_auto_categories: bool = False

    # === V4 Feature Flags — new cognitive capabilities ===
    # Schema induction: generalise episodic experiences into abstract schemas
    enable_schema_induction: bool = False
    schema_similarity_threshold: float = 0.55
    schema_min_support: int = 2

    # Predictive processing: active-inference-style prediction-error learning
    enable_predictive_processing: bool = False
    predictive_learning_rate: float = 0.15

    # Abductive reasoning: Inference-to-Best-Explanation over causal graph
    enable_abductive_reasoning: bool = False
    abductive_max_depth: int = 4

    # Temporal reasoning: Allen interval algebra for event ordering
    enable_temporal_reasoning: bool = False

    @classmethod
    def from_env(cls) -> "NSCKConfig":
        """Create config from environment variables with defaults."""
        return cls(
            device=os.getenv("NSCK_DEVICE", "cpu"),
            learning_rate=float(os.getenv("NSCK_LR", "1e-3")),
            model_path=os.getenv("NSCK_MODEL_PATH", "snn_task_aware.pth"),
            persistence_db=os.getenv("NSCK_DB_PATH", "nsck_brain.db"),
        )

    @classmethod
    def minimal(cls) -> "NSCKConfig":
        """Minimal config: all V3 flags off (original behaviour preserved)."""
        return cls()

    @classmethod
    def research(cls) -> "NSCKConfig":
        """Research config: all V3 + V4 flags on for maximum capability exploration."""
        return cls(
            enable_construction_grammar=True,
            enable_frame_semantics=True,
            enable_coreference=True,
            enable_contextual_encoding=True,
            enable_free_energy_beliefs=True,
            enable_distributional_semantics=True,
            enable_incremental_concept_refinement=True,
            enable_dual_process=True,
            enable_conceptual_blending=True,
            enable_hnsw_index=True,
            enable_homeostasis=True,
            enable_stigmergy=True,
            enable_auto_categories=True,
            # V4
            enable_schema_induction=True,
            enable_predictive_processing=True,
            enable_abductive_reasoning=True,
            enable_temporal_reasoning=True,
        )

    @classmethod
    def production(cls) -> "NSCKConfig":
        """Production config: performance + stability flags on, experimental flags off."""
        return cls(
            enable_dual_process=True,
            enable_hnsw_index=True,
            enable_homeostasis=True,
            enable_stigmergy=True,
            enable_incremental_concept_refinement=True,
            # V4 production-safe
            enable_schema_induction=True,
            enable_abductive_reasoning=True,
            enable_temporal_reasoning=True,
        )


# Global default config (can be overridden)
DEFAULT_CONFIG = NSCKConfig()
