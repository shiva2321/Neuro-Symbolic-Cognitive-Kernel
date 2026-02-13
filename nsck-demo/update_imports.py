"""
Update all imports to use new package structure
This script updates imports across the entire codebase
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
base = Path(r"d:\Node_network\nsck-demo")
python_dir = base / "python"
tests_dir = base / "tests"

# Import mapping: old_module -> new_module_path
IMPORT_MAP = {
    # VSA
    "hypervec_py": "python.core.vsa.hypervec_py",
    "hypervec_shim": "python.core.vsa.hypervec_shim",
    "universal_encoder": "python.core.vsa.universal_encoder",
    
    # Memory
    "episodic_memory": "python.core.memory.episodic_memory",
    "semantic_memory": "python.core.memory.semantic_memory",
    "intelligent_buffer": "python.core.memory.intelligent_buffer",
    "staged_recall": "python.core.memory.staged_recall",
    
    # Reasoning
    "cognitive_engine": "python.core.reasoning.cognitive_engine",
    "causal_reasoning": "python.core.reasoning.causal_reasoning",
    "context_engine": "python.core.reasoning.context_engine",
    "global_workspace": "python.core.reasoning.global_workspace",
    "rule_learner": "python.core.reasoning.rule_learner",
    "analogy": "python.core.reasoning.analogy",
    "planner": "python.core.reasoning.planner",
    
    # Learning
    "learning": "python.core.learning.learning",
    "continual_learning": "python.core.learning.continual_learning",
    "meta_learning": "python.core.learning.meta_learning",
    "multi_task_learning": "python.core.learning.multi_task_learning",
    "rl_engine": "python.core.learning.rl_engine",
    "curiosity": "python.core.learning.curiosity",
    
    # Perception
    "perception": "python.core.perception.perception",
    "symbol_grounding": "python.core.perception.symbol_grounding",
    "grounding_verifier": "python.core.perception.grounding_verifier",
    "saliency": "python.core.perception.saliency",
    
    # Language
    "language_module": "python.core.language.language_module",
    "lingua_cortex": "python.core.language.lingua_cortex",
    "text_knowledge_learner": "python.core.language.text_knowledge_learner",
    "universal_input": "python.core.language.universal_input",
    "nlg": "python.core.language.nlg",
    "dialogue_manager": "python.core.language.dialogue_manager",
    
    # Multimodal
    "multimodal_processor": "python.core.multimodal.multimodal_processor",
    
    # Cognitive
    "metacognition": "python.core.cognitive.metacognition",
    "theory_of_mind": "python.core.cognitive.theory_of_mind",
    "self_model": "python.core.cognitive.self_model",
    "emotion_system": "python.core.cognitive.emotion_system",
    "empathy": "python.core.cognitive.empathy",
    "value_alignment": "python.core.cognitive.value_alignment",
    
    # Neural
    "plastic_snn": "python.core.neural.plastic_snn",
    "world_model": "python.core.neural.world_model",
    "snn_qat": "python.core.neural.snn_qat",
    
    # Integration
    "brain_fusion": "python.core.integration.brain_fusion",
    "knowledge_integration": "python.core.integration.knowledge_integration",
    "module_registry": "python.core.integration.module_registry",
    "persistence": "python.core.integration.persistence",
    "lifecycle": "python.core.integration.lifecycle",
    
    # Games
    "snake_headless": "python.games.snake.snake_headless",
    "snake_ui": "python.games.snake.snake_ui",
    "maze_game": "python.games.maze.maze_game",
    "maze_ui": "python.games.maze.maze_ui",
    "simulation": "python.games.pong.simulation",
    "pong_ui": "python.games.pong.pong_ui",
    "balancer_game": "python.games.physics.balancer_game",
    "catcher_game": "python.games.physics.catcher_game",
    "collector_game": "python.games.collector.collector_game",
    
    # Benchmarks
    "benchmark": "python.benchmarks.benchmark",
    "transfer_experiments": "python.benchmarks.transfer_experiments",
    "power_monitor": "python.benchmarks.power_monitor",
    "visualize_transfer": "python.benchmarks.visualize_transfer",
    
    # Training
    "snn_training_pipeline": "python.training.snn_training_pipeline",
    "train_snn": "python.training.train_snn",
    "train_semantic_folding": "python.training.train_semantic_folding",
    
    # Interfaces
    "unified_dashboard": "python.interfaces.unified_dashboard",
    "testing_dashboard": "python.interfaces.testing_dashboard",
    "voice_interface": "python.interfaces.voice_interface",
    "voice_hd": "python.interfaces.voice_hd",
    "voice_chatbot": "python.interfaces.voice_chatbot",
    
    # Servers
    "python_server": "python.servers.python_server",
    "logger_service": "python.servers.logger_service",
    
    # Utilities
    "config": "python.utilities.config",
    "agency": "python.utilities.agency",
    "ai_controller": "python.utilities.ai_controller",
    "teacher_interface": "python.utilities.teacher_interface",
    "teaching": "python.utilities.teaching",
    "homeostasis": "python.utilities.homeostasis",
    "intrinsic_motivation": "python.utilities.intrinsic_motivation",
    "explanation": "python.utilities.explanation",
    "consciousness_metrics": "python.utilities.consciousness_metrics",
    "learning_progress": "python.utilities.learning_progress",
    "curriculum": "python.utilities.curriculum",
    "spatial_reasoning": "python.utilities.spatial_reasoning",
    "concept_mapper": "python.utilities.concept_mapper",
    "self_modifier": "python.utilities.self_modifier",
    
    # Scripts
    "build_codebook": "python.scripts.build_codebook",
    "download_model": "python.scripts.download_model",
    "system_launcher": "python.scripts.system_launcher",
    "char_offline_eval": "python.scripts.char_offline_eval",
    "character_dataset": "python.scripts.character_dataset",
    "debug_char_preprocess": "python.scripts.debug_char_preprocess",
    "latent_probe": "python.scripts.latent_probe",
    "rule_extraction": "python.scripts.rule_extraction",
    "semantic_coherence": "python.scripts.semantic_coherence",
}


def update_import_line(line: str) -> Tuple[str, bool]:
    """
    Update a single import line. Returns (updated_line, was_changed)
    """
    original_line = line
    
    # Pattern 1: from module import ...
    from_import_pattern = r'^(\s*)from\s+([a-zA-Z0-9_]+)\s+import\s+(.+)$'
    match = re.match(from_import_pattern, line)
    if match:
        indent, module, imports = match.groups()
        if module in IMPORT_MAP:
            new_module = IMPORT_MAP[module]
            line = f"{indent}from {new_module} import {imports}\n"
            return line, True
    
    # Pattern 2: import module
    import_pattern = r'^(\s*)import\s+([a-zA-Z0-9_]+)(\s+as\s+\w+)?(.*)$'
    match = re.match(import_pattern, line)
    if match:
        indent, module, as_clause, rest = match.groups()
        if module in IMPORT_MAP:
            new_module = IMPORT_MAP[module]
            as_part = as_clause if as_clause else ""
            line = f"{indent}import {new_module}{as_part}{rest}\n"
            return line, True
    
    return original_line, False


def update_file_imports(file_path: Path) -> int:
    """
    Update all imports in a single file.
    Returns number of lines changed.
    """
    if not file_path.exists():
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {file_path.name}: {e}")
        return 0
    
    new_lines = []
    changes = 0
    
    for line in lines:
        new_line, was_changed = update_import_line(line)
        new_lines.append(new_line)
        if was_changed:
            changes += 1
    
    if changes > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"  Error writing {file_path.name}: {e}")
            return 0
    
    return changes


def find_all_python_files(directory: Path) -> List[Path]:
    """Find all Python files recursively."""
    return list(directory.rglob("*.py"))


def main():
    print("=" * 70)
    print("UPDATING IMPORTS TO NEW PACKAGE STRUCTURE")
    print("=" * 70)
    
    # Find all Python files
    python_files = find_all_python_files(python_dir)
    test_files = find_all_python_files(tests_dir)
    
    all_files = python_files + test_files
    
    print(f"\nFound {len(python_files)} Python files in python/")
    print(f"Found {len(test_files)} test files in tests/")
    print(f"Total: {len(all_files)} files to process\n")
    
    # Update imports
    total_changes = 0
    files_modified = 0
    
    print("Processing files...")
    print("-" * 70)
    
    for file_path in all_files:
        # Skip __init__.py files and our utility scripts
        if file_path.name in ["__init__.py", "create_structure.py", "move_files.py", 
                               "create_inits.py", "update_imports.py"]:
            continue
        
        changes = update_file_imports(file_path)
        if changes > 0:
            rel_path = file_path.relative_to(base)
            print(f"  {rel_path}: {changes} import(s) updated")
            files_modified += 1
            total_changes += changes
    
    print("-" * 70)
    print(f"\nSummary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total import lines updated: {total_changes}")
    print("=" * 70)
    
    if files_modified == 0:
        print("\nNo changes needed - all imports already up to date!")
    else:
        print(f"\nSuccessfully updated {files_modified} files!")


if __name__ == "__main__":
    main()
