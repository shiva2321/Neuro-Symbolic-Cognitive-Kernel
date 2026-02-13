"""
Create __init__.py files for all new package subdirectories
"""
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo")

# Packages that need __init__.py files
packages = [
    # Core packages
    "python/core",
    "python/core/vsa",
    "python/core/memory",
    "python/core/reasoning",
    "python/core/learning",
    "python/core/perception",
    "python/core/language",
    "python/core/multimodal",
    "python/core/cognitive",
    "python/core/neural",
    "python/core/integration",
    
    # Games
    "python/games",
    "python/games/snake",
    "python/games/maze",
    "python/games/pong",
    "python/games/physics",
    "python/games/collector",
    
    # Other top-level packages
    "python/benchmarks",
    "python/benchmarks/reports",  
    "python/training",
    "python/training/demos",
    "python/interfaces",
    "python/servers",
    "python/utilities",
    "python/scripts",
    
    # Test packages
    "tests/unit",
    "tests/integration",
    "tests/benchmarks",
    "tests/experiments",
]

# __init__.py contents for each package
init_contents = {
    "python/core": '''"""
NSCK Core - Cognitive Architecture
===================================
Core neuro-symbolic cognitive components.
"""

__version__ = "2.0.0"
''',
    
    "python/core/vsa": '''"""Vector Symbolic Architecture (VSA) - Hyperdimensional Computing"""
from .hypervec_py import HyperVector
from .hypervec_shim import HVBackend
from .universal_encoder import UniversalEncoder

__all__ = ["HyperVector", "HVBackend", "UniversalEncoder"]
''',
    
    "python/core/memory": '''"""Memory Systems - Episodic, Semantic, Working Memory"""
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .intelligent_buffer import IntelligentBuffer
from .staged_recall import StagedRecall

__all__ = ["EpisodicMemory", "SemanticMemory", "IntelligentBuffer", "StagedRecall"]
''',
    
    "python/core/reasoning": '''"""Reasoning Engines - Cognitive, Causal, Symbolic"""
from .cognitive_engine import CognitiveEngine
from .causal_reasoning import CausalReasoner
from .context_engine import ContextEngine
from .global_workspace import GlobalWorkspace
from .rule_learner import RuleLearner
from .analogy import AnalogyEngine
from .planner import Planner

__all__ = ["CognitiveEngine", "CausalReasoner", "ContextEngine", "GlobalWorkspace", 
           "RuleLearner", "AnalogyEngine", "Planner"]
''',
    
    "python/core/learning": '''"""Learning Systems - RL, Continual, Meta-Learning"""
from .learning import LearningEngine
from .continual_learning import ContinualLearner
from .meta_learning import MetaLearner
from .multi_task_learning import MultiTaskLearner
from .rl_engine import RLEngine
from .curiosity import CuriosityModule

__all__ = ["LearningEngine", "ContinualLearner", "MetaLearner", "MultiTaskLearner",
           "RLEngine", "CuriosityModule"]
''',
    
    "python/core/perception": '''"""Perception Systems - Grounding, Saliency, Symbol-Sensory Interface"""  
from .perception import PerceptionModule
from .symbol_grounding import SymbolGrounding
from .grounding_verifier import GroundingVerifier
from .saliency import SaliencyDetector

__all__ = ["PerceptionModule", "SymbolGrounding", "GroundingVerifier", "SaliencyDetector"]
''',
    
    "python/core/language": '''"""Natural Language Processing - Understanding, Generation, Grounding"""
from .language_module import LanguageModule
from .lingua_cortex import LinguaCortex
from .text_knowledge_learner import TextKnowledgeLearner
from .universal_input import UniversalInput
from .nlg import NaturalLanguageGenerator
from .dialogue_manager import DialogueManager

__all__ = ["LanguageModule", "LinguaCortex", "TextKnowledgeLearner", "UniversalInput",
           "NaturalLanguageGenerator", "DialogueManager"]
''',
    
    "python/core/multimodal": '''"""Multimodal Processing - Vision + Language + Sensor Fusion"""
from .multimodal_processor import MultimodalProcessor

__all__ = ["MultimodalProcessor"]
''',
    
    "python/core/cognitive": '''"""Higher Cognition - Theory of Mind, Emotion, Self-Model"""
from .metacognition import Metacognition
from .theory_of_mind import TheoryOfMind
from .self_model import SelfModel
from .emotion_system import EmotionSystem
from .empathy import EmpathyModule
from .value_alignment import ValueAlignment

__all__ = ["Metacognition", "TheoryOfMind", "SelfModel", "EmotionSystem",
           "EmpathyModule", "ValueAlignment"]
''',
    
    "python/core/neural": '''"""Neural Network Components - SNN, World Models"""
from .plastic_snn import PlasticSNN
from .world_model import WorldModel
from .snn_qat import SNNQAT

__all__ = ["PlasticSNN", "WorldModel", "SNNQAT"]
''',
    
    "python/core/integration": '''"""System Integration - Brain Fusion, Persistence, Lifecycle"""
from .brain_fusion import BrainFusion
from .knowledge_integration import KnowledgeIntegration
from .module_registry import ModuleRegistry
from .persistence import PersistenceManager
from .lifecycle import LifecycleManager

__all__ = ["BrainFusion", "KnowledgeIntegration", "ModuleRegistry",
           "PersistenceManager", "LifecycleManager"]
''',
    
    "python/games": '''"""Game Environments for Training and Benchmarking"""
__all__ = ["snake", "maze", "pong", "physics", "collector"]
''',
    
    "python/benchmarks": '''"""Benchmarking and Performance Evaluation"""
from .benchmark import run_benchmark
from .transfer_experiments import run_transfer_experiments
from .power_monitor import PowerMonitor

__all__ = ["run_benchmark", "run_transfer_experiments", "PowerMonitor"]
''',
    
    "python/training": '''"""Training Scripts and Pipelines"""
__all__ = ["snn_training_pipeline", "train_snn", "train_semantic_folding", "demos"]
''',
    
    "python/interfaces": '''"""User Interfaces - Dashboards, Voice, Chat"""
__all__ = ["unified_dashboard", "testing_dashboard", "voice_interface", "voice_hd", "voice_chatbot"]
''',
    
    "python/servers": '''"""Backend Servers - API, Logging"""
from .python_server import app as server_app
from .logger_service import LoggerService

__all__ = ["server_app", "LoggerService"]
''',
    
    "python/utilities": '''"""Utility Modules"""
__all__ = ["config", "agency", "ai_controller", "teacher_interface", "teaching",
           "homeostasis", "intrinsic_motivation", "explanation", "consciousness_metrics",
           "learning_progress", "curriculum", "spatial_reasoning", "concept_mapper", "self_modifier"]
''',
    
    "python/scripts": '''"""Standalone Scripts and Tools"""
__all__ = ["build_codebook", "download_model", "system_launcher", "char_offline_eval",
           "character_dataset", "debug_char_preprocess", "latent_probe", "rule_extraction",
           "semantic_coherence"]
''',
    
    "tests/experiments": '''"""Experimental Validation Tests"""
__all__ = ["belief_revision_test", "multimodal_test", "text_reasoning_test",
           "transitive_test", "verify_f1"]
''',
}

# Default content for packages without specific content
default_init = '''"""Package initialization"""
'''

print("Creating __init__.py files...")
print("=" * 60)

for package in packages:
    init_file = base / package / "__init__.py"
    content = init_contents.get(package, default_init)
    
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Created: {package}/__init__.py")

print("=" * 60)
print(f"\nCreated {len(packages)} __init__.py files!")
