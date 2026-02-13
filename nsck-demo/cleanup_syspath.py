"""
Clean up sys.path modifications - most are now unnecessary since conftest.py handles it
We'll keep conftest.py as-is and remove redundant sys.path in test files
"""
import re
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo")

# Files where we should REMOVE sys.path modifications entirely (tests - handled by conftest.py)
test_files_to_clean = [
    "tests/test_agency.py",
    "tests/test_benchmark.py",
    "tests/test_brain_versioning.py",
    "tests/test_bug_fixes.py",
    "tests/test_capability_proofs.py",
    "tests/test_collector.py",
    "tests/test_context_engine.py",
    "tests/test_cross_module.py",
    "tests/test_dashboard.py",
    "tests/test_deep_dreaming.py",
    "tests/test_dreaming.py",
    "tests/test_dynamic_brain.py",
    "tests/test_emergence.py",
    "tests/test_external_plugin_integration.py",
    "tests/test_fusion.py",
    "tests/test_homeostasis.py",
    "tests/test_hypothetical_scenarios.py",
    "tests/test_intelligent_archival.py",
    "tests/test_knowledge_integration.py",
    "tests/test_lingua.py",
    "tests/test_maze_reset.py",
    "tests/test_maze_validation.py",
    "tests/test_module_registry.py",
    "tests/test_multimodal_processor.py",
    "tests/test_phase6_social.py",
    "tests/test_phase8_mental_rehearsal.py",
    "tests/test_phase8_permutation.py",
    "tests/test_phase8_universal_input.py",
    "tests/test_plasticity.py",
    "tests/test_rule_learner_interface.py",
    "tests/test_rule_learning_stress.py",
    "tests/test_saliency.py",
    "tests/test_semantic_folding.py",
    "tests/test_semantic_roles.py",
    "tests/test_stability.py",
    "tests/test_testing_dashboard.py",
    "tests/test_text_knowledge_learner.py",
    "tests/test_transfer.py",
    "tests/test_transfer_learning.py",
    "tests/test_transfer_nongrid.py",
    "tests/test_transfer_rigorous.py",
    "tests/test_unified_loop.py",
    "tests/test_universal_encoder.py",
    "tests/test_voice_hd.py",
    "tests/test_workspace.py",
    "tests/experiments/belief_revision_test.py",
    "tests/experiments/multimodal_test.py",
    "tests/experiments/text_reasoning_test.py",
    "tests/experiments/transitive_test.py",
    "tests/experiments/verify_f1.py",
]

# Files where we should UPDATE sys.path to point to nsck-demo root (standalone scripts)
files_to_update_path = {
    "python/benchmarks/benchmark.py": "../../",  # Go up 2 levels to nsck-demo
    "python/benchmarks/transfer_experiments.py": "../../",
    "python/benchmarks/power_monitor.py": "../../",
    "python/interfaces/testing_dashboard.py": "../../",
    "python/interfaces/unified_dashboard.py": "../../",
    "python/games/maze/maze_ui.py": "../../../",  # Go up 3 levels
    "python/games/snake/snake_headless.py": "../../../",
    "python/training/demos/demo_integration.py": "../../../",
    "python/training/demos/demo_meta_learning.py": "../../../",
}

print("=" * 70)
print("CLEANING UP SYS.PATH MODIFICATIONS")
print("=" * 70)

removed_count = 0
updated_count = 0

# Remove sys.path from test files (conftest.py handles it)
print("\nRemoving redundant sys.path from test files...")
for rel_path in test_files_to_clean:
    file_path = base / rel_path
    if not file_path.exists():
        continue
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        removed_lines = 0
        
        for line in lines:
            # Skip lines that modify sys.path
            if 'sys.path.insert' in line or 'sys.path.append' in line:
                removed_lines += 1
                continue
            new_lines.append(line)
        
        if removed_lines > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"  {rel_path}: removed {removed_lines} line(s)")
            removed_count += removed_lines
    except Exception as e:
        print(f"  Error with {rel_path}: {e}")

# Update sys.path in standalone scripts
print("\nUpdating sys.path in standalone scripts...")
for rel_path, levels_up in files_to_update_path.items():
    file_path = base / rel_path
    if not file_path.exists():
        continue
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace old sys.path manipulations with correct one
        original = content
        
        # Pattern to match sys.path.insert/append lines
        pattern = r'sys\.path\.(insert|append)\([^)]+\)'
        replacement = f"sys.path.insert(0, os.path.join(os.path.dirname(__file__), '{levels_up}'))"
        
        content = re.sub(pattern, replacement, content, count=1)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  {rel_path}: updated sys.path")
            updated_count += 1
    except Exception as e:
        print(f"  Error with {rel_path}: {e}")

print("=" * 70)
print(f"\nSummary:")
print(f"  Removed sys.path lines: {removed_count}")
print(f"  Updated sys.path in scripts: {updated_count}")
print("=" * 70)
