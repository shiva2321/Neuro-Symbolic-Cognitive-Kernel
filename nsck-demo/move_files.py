"""
File move script for repository reorganization
Moves files from flat python/ structure to organized subdirectories
"""
import shutil
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo\python")

# File moves: (source_filename, destination_subdir)
moves = {
    # VSA modules
    "hypervec_py.py": "core/vsa",
    "hypervec_shim.py": "core/vsa",
    "universal_encoder.py": "core/vsa",
    
    # Memory modules  
    "episodic_memory.py": "core/memory",
    "semantic_memory.py": "core/memory",
    "intelligent_buffer.py": "core/memory",
    "staged_recall.py": "core/memory",
    
    # Reasoning modules
    "cognitive_engine.py": "core/reasoning",
    "causal_reasoning.py": "core/reasoning",
    "context_engine.py": "core/reasoning",
    "global_workspace.py": "core/reasoning",
    "rule_learner.py": "core/reasoning",
    "analogy.py": "core/reasoning",
    "planner.py": "core/reasoning",
    
    # Learning modules
    "learning.py": "core/learning",
    "continual_learning.py": "core/learning",
    "meta_learning.py": "core/learning",
    "multi_task_learning.py": "core/learning",
    "rl_engine.py": "core/learning",
    "curiosity.py": "core/learning",
    
    # Perception modules
    "perception.py": "core/perception",
    "symbol_grounding.py": "core/perception",
    "grounding_verifier.py": "core/perception",
    "saliency.py": "core/perception",
    
    # Language modules        
    "language_module.py": "core/language",
    "lingua_cortex.py": "core/language",
    "text_knowledge_learner.py": "core/language",
    "universal_input.py": "core/language",
    "nlg.py": "core/language",
    "dialogue_manager.py": "core/language",
    
    # Multimodal
    "multimodal_processor.py": "core/multimodal",
    
    # Higher cognition
    "metacognition.py": "core/cognitive",
    "theory_of_mind.py": "core/cognitive",
    "self_model.py": "core/cognitive",
    "emotion_system.py": "core/cognitive",
    "empathy.py": "core/cognitive",
    "value_alignment.py": "core/cognitive",
    
    # Neural components
    "plastic_snn.py": "core/neural",
    "world_model.py": "core/neural",
    "snn_qat.py": "core/neural",
    
    # Integration
    "brain_fusion.py": "core/integration",
    "knowledge_integration.py": "core/integration",
    "module_registry.py": "core/integration",
    "persistence.py": "core/integration",
    "lifecycle.py": "core/integration",
    
    # Games - Snake
    "snake_headless.py": "games/snake",
    "snake_ui.py": "games/snake",
    
    # Games - Maze
    "maze_game.py": "games/maze",
    "maze_ui.py": "games/maze",
    
    # Games - Pong
    "simulation.py": "games/pong",
    "pong_ui.py": "games/pong",
    
    # Games - Physics
    "balancer_game.py": "games/physics",
    "catcher_game.py": "games/physics",
    
    # Games - Collector
    "collector_game.py": "games/collector",
    
    # Benchmarks
    "benchmark.py": "benchmarks",
    "transfer_experiments.py": "benchmarks",
    "power_monitor.py": "benchmarks",
    "visualize_transfer.py": "benchmarks",
    "benchmark_report.json": "benchmarks/reports",
    "benchmark_report.md": "benchmarks/reports",
    "transfer_report.md": "benchmarks/reports",
    
    # Training
    "snn_training_pipeline.py": "training",
    "train_snn.py": "training",
    "train_semantic_folding.py": "training",
    
    # Interfaces
    "unified_dashboard.py": "interfaces",
    "testing_dashboard.py": "interfaces",
    "voice_interface.py": "interfaces",
    "voice_hd.py": "interfaces",
    "voice_chatbot.py": "interfaces",
    
    # Servers
    "python_server.py": "servers",
    "logger_service.py": "servers",
    
    # Utilities
    "config.py": "utilities",
    "agency.py": "utilities",
    "ai_controller.py": "utilities",
    "teacher_interface.py": "utilities",
    "teaching.py": "utilities",
    "homeostasis.py": "utilities",
    "intrinsic_motivation.py": "utilities",
    "explanation.py": "utilities",
    "consciousness_metrics.py": "utilities",
    "learning_progress.py": "utilities",
    "curriculum.py": "utilities",
    "spatial_reasoning.py": "utilities",
    "concept_mapper.py": "utilities",
    "self_modifier.py": "utilities",
    
    # Scripts
    "build_codebook.py": "scripts",
    "download_model.py": "scripts",
    "system_launcher.py": "scripts",
    "char_offline_eval.py": "scripts",
    "character_dataset.py": "scripts",
    "debug_char_preprocess.py": "scripts",
    "latent_probe.py": "scripts",
    "rule_extraction.py": "scripts",
    "semantic_coherence.py": "scripts",
}

# Training demo renames
demo_renames = {
    "train_phase1_demo.py": "demo_neural_learning.py",
    "train_phase2_demo.py": "demo_perception.py",
    "train_phase3_demo.py": "demo_planning.py",
    "train_phase4_demo.py": "demo_reasoning.py",
    "train_phase5_demo.py": "demo_social_cognition.py",
    "train_phase6_demo.py": "demo_meta_learning.py",
    "train_phase7_demo.py": "demo_integration.py",
}

# Test file moves (from python/ to tests/experiments/)
test_moves = {
    "belief_revision_test.py": "../tests/experiments",
    "multimodal_test.py": "../tests/experiments",
    "text_reasoning_test.py": "../tests/experiments",
    "transitive_test.py": "../tests/experiments",
    "verify_f1.py": "../tests/experiments",
}

print("Moving files to new structure...")
print("=" * 60)

moved = 0
errors = []

# Move regular files
for source_name, dest_dir in moves.items():
    source = base / source_name
    dest = base / dest_dir / source_name
    
    if source.exists():
        try:
            shutil.move(str(source), str(dest))
            print(f"  Moved: {source_name} -> {dest_dir}/")
            moved += 1
        except Exception as e:
            errors.append(f"{source_name}: {e}")
            print(f"  ERROR: {source_name} - {e}")
    else:
        print(f"  SKIP (not found): {source_name}")

# Move and rename training demos
print("\nRenaming and moving training demos...")
for old_name, new_name in demo_renames.items():
    source = base / old_name
    dest = base / "training" / "demos" / new_name
    
    if source.exists():
        try:
            shutil.move(str(source), str(dest))
            print(f"  Moved & Renamed: {old_name} -> training/demos/{new_name}")
            moved += 1
        except Exception as e:
            errors.append(f"{old_name}: {e}")
            print(f"  ERROR: {old_name} - {e}")
    else:
        print(f"  SKIP (not found): {old_name}")

# Move test files  
print("\nMoving test files to tests/experiments...")
for source_name, dest_dir in test_moves.items():
    source = base / source_name
    dest = base / dest_dir / source_name
    
    if source.exists():
        try:
            shutil.move(str(source), str(dest))
            print(f"  Moved: {source_name} -> tests/experiments/")
            moved += 1
        except Exception as e:
            errors.append(f"{source_name}: {e}")
            print(f"  ERROR: {source_name} - {e}")
    else:
        print(f"  SKIP (not found): {source_name}")

print("=" * 60)
print(f"\nMoved {moved} files successfully")
if errors:
    print(f"Encountered {len(errors)} errors:")
    for err in errors:
        print(f"  - {err}")
else:
    print("No errors!")
