"""
NSCK Configuration Module
Centralized configuration for hyperparameters and settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List
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
    # Negation-aware relation extraction: "X does not Y" stored as not_relates_to
    enable_negation_handling: bool = False
    # Temporal ordering extraction: "X before Y" → precedes, "X after Y" → follows
    enable_temporal_reasoning: bool = False
    # Conditional logic extraction: "if X then Y" → implies / conditional_on
    enable_conditional_logic: bool = False
    # Taxonomic/transitive closure: A is_a B + B is_a C → A is_a C
    enable_transitive_inference: bool = False
    # Prototype-based category generalization via VSA bundling
    enable_prototype_generalization: bool = False

    # === V5 Feature Flags — remaining roadmap items ===
    # Spatial reasoning: place entities and query above/below/left/right/near
    enable_spatial_reasoning: bool = False
    # Pragmatics: scalar implicature, Gricean maxims, indirect speech acts
    enable_pragmatics: bool = False

    # === V7 Feature Flags — dialogue quality, corpus training, noise filtering ===
    # Wire FluentNLG into DialogueManager (replaces template-based realize_sentence)
    enable_fluent_dialogue: bool = True   # safe drop-in replacement; on by default
    # Try to download a HuggingFace corpus slice to augment distributional HV training.
    # Requires internet access; falls back to BUILTIN_CORPUS silently when offline.
    enable_hf_corpus: bool = False

    # === V8 Feature Flags — new capabilities (all off by default → zero regressions) ===
    enable_concurrent_multimodal: bool = False
    enable_full_rust_snn: bool = False
    memory_decay_lambda: float = 0.01
    memory_prune_threshold: float = 0.1
    enable_dialogue_state_tracking: bool = False
    enable_hierarchical_srl: bool = False
    enable_multi_agent: bool = False
    enable_active_inference: bool = False
    active_inference_weight: float = 0.2

    # === V9 Feature Flags — substrate transformation ===
    # Automatically transfer rules from existing tasks to newly registered tasks
    enable_auto_transfer: bool = True

    # === V11 Feature Flags ===
    generalization_interval: int = 100
    enable_continuous_generalization: bool = True
    enable_ngram_nlu: bool = True
    enable_cross_modal_learning: bool = False

    # === V14 Feature Flags — Rich Perception ===
    perception_mode: str = "pure"           # "pure" | "bridge" | "hybrid"
    text_bridge_model: str = "all-MiniLM-L6-v2"
    image_bridge_model: str = "mobilenet_v3_small"
    audio_bridge_model: str = "whisper-tiny"
    bridge_cache_embeddings: bool = True
    bridge_dim: int = 384
    distillation_threshold: float = 0.80
    knowledge_packs: List[str] = field(default_factory=list)

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
        """Research config: all V3+V4 flags on for maximum capability exploration."""
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
            enable_negation_handling=True,
            enable_temporal_reasoning=True,
            enable_conditional_logic=True,
            enable_transitive_inference=True,
            enable_prototype_generalization=True,
            # V5
            enable_spatial_reasoning=True,
            enable_pragmatics=True,
            # V7
            enable_fluent_dialogue=True,
            # V8
            enable_concurrent_multimodal=True,
            enable_dialogue_state_tracking=True,
            enable_hierarchical_srl=True,
            enable_multi_agent=True,
            enable_active_inference=True,
            # V11
            enable_continuous_generalization=True,
            enable_ngram_nlu=True,
            enable_cross_modal_learning=True,
            perception_mode="bridge",
            # V16
            enable_ewc=True,
        )

    @classmethod
    def production(cls) -> "NSCKConfig":
        """Production config: performance flags on, experimental flags off."""
        return cls(
            enable_dual_process=True,
            enable_hnsw_index=True,
            enable_homeostasis=True,
            enable_stigmergy=True,
            enable_incremental_concept_refinement=True,
            # V4: stable inference features on in production
            enable_negation_handling=True,
            enable_temporal_reasoning=True,
            enable_conditional_logic=True,
            enable_transitive_inference=True,
            # V7: fluent dialogue on in production
            enable_fluent_dialogue=True,
            perception_mode="pure",
        )

    # === V16 Feature Flags — EWC Wiring ===
    enable_ewc: bool = False
    ewc_lambda: float = 1000.0
    ewc_consolidate_interval: int = 500
    ewc_importance_window: int = 100

    # === V16 Feature Flags — Semantic Seeding ===
    enable_seeding: bool = False
    seed_conceptnet_pack: str = ""
    seed_bert_on_init: bool = False
    seed_bert_model_name: str = "bert-base-uncased"

    # === V15 Feature Flags — Model Transplantation ===
    enable_transplant: bool = False
    transplant_strategy: str = "svd_factored"  # "random", "learned", "svd_factored"
    transplant_calibration_epochs: int = 10
    transplant_validation_threshold: float = 0.80
    transplant_svd_components: int = 128
    transplant_fpe_bins: int = 256
    transplant_batch_size: int = 512
    transplant_sample_pairs: int = 10000
    transplant_auto_live_encoding: bool = True

    @classmethod
    def rich(cls) -> "NSCKConfig":
        """Rich config: all research flags + bridge perception."""
        cfg = cls.research()
        cfg.perception_mode = "bridge"
        return cfg

    @classmethod
    def seeded(cls) -> "NSCKConfig":
        """Seeded config: ConceptNet + auto-seeding enabled."""
        return cls(
            enable_seeding=True,
            seed_conceptnet_pack="nsck/data/knowledge_packs/conceptnet_en_50k.kp",
        )

    @classmethod
    def transplant(cls) -> "NSCKConfig":
        """Transplant config: all transplant flags enabled."""
        return cls(
            enable_transplant=True,
            transplant_strategy="svd_factored",
            transplant_calibration_epochs=10,
            transplant_validation_threshold=0.80,
            transplant_svd_components=128,
            transplant_fpe_bins=256,
            transplant_batch_size=512,
            transplant_sample_pairs=10000,
            transplant_auto_live_encoding=True,
        )

    @classmethod
    def for_scale(cls, n_concepts: int) -> "NSCKConfig":
        """Auto-tune config for a given concept scale."""
        cfg = cls()
        cfg.memory_capacity = max(2500, n_concepts * 2)
        if n_concepts >= 10_000:
            cfg.enable_hnsw_index = True
        return cfg

    # === V18 Feature Flags — Semantic HV Bootstrap ===
    enable_semantic_bootstrap: bool = False
    semantic_bootstrap_strategy: str = "auto"   # "auto", "bridge", "corpus", "prebuilt"
    semantic_bootstrap_model: str = "all-MiniLM-L6-v2"
    semantic_codebook_path: str = ""  # path to pre-built .pkl codebook

    # === V17 Feature Flags — Enrichment & Glass-Box Tracing ===
    enable_causal_enrichment: bool = False
    enable_perceptual_enrichment: bool = False
    enable_semantic_enrichment: bool = False
    enable_glass_box_tracer: bool = False
    enable_crossmodal_enrichment: bool = False
    glass_box_max_history: int = 100
    causal_enrichment_n_context: int = 3
    perceptual_enricher_window: int = 8
    semantic_enrichment_add_inverses: bool = True
    crossmodal_similarity_threshold: float = 0.7

    @classmethod
    def v17(cls) -> "NSCKConfig":
        """Preset enabling all V17 enrichment and glass-box capabilities."""
        cfg = cls()
        cfg.enable_causal_enrichment = True
        cfg.enable_perceptual_enrichment = True
        cfg.enable_semantic_enrichment = True
        cfg.enable_glass_box_tracer = True
        cfg.enable_crossmodal_enrichment = True
        return cfg

    @classmethod
    def semantic(cls) -> "NSCKConfig":
        """Semantic config: full research flags + semantic HV bootstrap enabled."""
        cfg = cls.research()
        cfg.enable_distributional_semantics = True
        cfg.enable_hf_corpus = False  # offline by default
        cfg.enable_semantic_bootstrap = True
        cfg.semantic_bootstrap_strategy = "auto"
        cfg.semantic_bootstrap_model = "all-MiniLM-L6-v2"
        cfg.semantic_codebook_path = ""
        return cfg

    # === V4 Architecture Flags — Procedural Memory ===
    procedural_reward_threshold: float = 0.0
    procedural_familiarity_threshold: float = 0.72

    # === V4 Architecture Flags — Semantic Memory ===
    semantic_hot_cache_size: int = 256
    semantic_hnsw_m: int = 16
    semantic_hnsw_ef: int = 50

    # === NSCK-UPMA Vision Absorption Feature Flags ===
    enable_vision_absorption: bool = False
    vision_reference_model: str = "openai/clip-vit-base-patch32"
    vision_absorption_max_samples: int = 500
    vision_absorption_batch_size: int = 32
    vision_fusion_override_threshold: float = 0.95
    vision_causal_chain_min_depth: int = 1
    vision_domains: List[str] = field(default_factory=list)

    @classmethod
    def vision(cls) -> "NSCKConfig":
        """Vision config: enable NSCK-UPMA universal model absorption."""
        cfg = cls()
        cfg.enable_vision_absorption = True
        cfg.enable_transplant = True
        cfg.transplant_strategy = "svd_factored"
        cfg.transplant_svd_components = 128
        cfg.transplant_fpe_bins = 256
        cfg.transplant_auto_live_encoding = True
        cfg.vision_reference_model = "openai/clip-vit-base-patch32"
        cfg.vision_absorption_max_samples = 500
        cfg.vision_absorption_batch_size = 32
        cfg.vision_fusion_override_threshold = 0.95
        cfg.vision_causal_chain_min_depth = 1
        return cfg

    # === V26 Feature Flags — Societal Hypervector Knowledge Representation ===
    enable_societal: bool = False
    societal_bond_threshold: float = 0.65
    societal_max_bonds: int = 8
    societal_bond_decay: float = 0.01
    societal_activation_decay: float = 0.05
    societal_activation_spread: float = 0.4
    societal_cluster_resolution: float = 1.0
    societal_auto_cluster_interval: int = 10
    societal_top_k: int = 5
    societal_min_similarity: float = 0.55

    @classmethod
    def societal(cls) -> "NSCKConfig":
        """Societal config: enable Societal Hypervector Knowledge Representation."""
        cfg = cls()
        cfg.enable_societal = True
        cfg.societal_bond_threshold = 0.65
        cfg.societal_max_bonds = 8
        cfg.societal_bond_decay = 0.01
        cfg.societal_activation_decay = 0.05
        cfg.societal_activation_spread = 0.4
        cfg.societal_cluster_resolution = 1.0
        cfg.societal_auto_cluster_interval = 10
        cfg.societal_top_k = 5
        cfg.societal_min_similarity = 0.55
        return cfg


# Global default config (can be overridden)
DEFAULT_CONFIG = NSCKConfig()
