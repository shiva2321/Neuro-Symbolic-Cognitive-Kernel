"""
Final import fix - handle "from python.X" patterns that weren't caught
"""
import re
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo")

# Map old "python.X" -> new "python.core.Y.X"
DIRECT_MODULE_MAP = {
    "python.brain_fusion": "python.core.integration.brain_fusion",
    "python.knowledge_integration": "python.core.integration.knowledge_integration",
    "python.persistence": "python.core.integration.persistence",
    "python.lifecycle": "python.core.integration.lifecycle",
    "python.module_registry": "python.core.integration.module_registry",
    
    "python.episodic_memory": "python.core.memory.episodic_memory",
    "python.semantic_memory": "python.core.memory.semantic_memory",
    "python.intelligent_buffer": "python.core.memory.intelligent_buffer",
    "python.staged_recall": "python.core.memory.staged_recall",
    
    "python.cognitive_engine": "python.core.reasoning.cognitive_engine",
    "python.causal_reasoning": "python.core.reasoning.causal_reasoning",
    "python.global_workspace": "python.core.reasoning.global_workspace",
    "python.context_engine": "python.core.reasoning.context_engine",
    "python.rule_learner": "python.core.reasoning.rule_learner",
    "python.analogy": "python.core.reasoning.analogy",
    "python.planner": "python.core.reasoning.planner",
    
    "python.perception": "python.core.perception.perception",
    "python.symbol_grounding": "python.core.perception.symbol_grounding",
    "python.grounding_verifier": "python.core.perception.grounding_verifier",
    "python.saliency": "python.core.perception.saliency",
    
    "python.language_module": "python.core.language.language_module",
    "python.lingua_cortex": "python.core.language.lingua_cortex",
    "python.text_knowledge_learner": "python.core.language.text_knowledge_learner",
    "python.universal_input": "python.core.language.universal_input",
    "python.nlg": "python.core.language.nlg",
    "python.dialogue_manager": "python.core.language.dialogue_manager",
    
    "python.learning": "python.core.learning.learning",
    "python.continual_learning": "python.core.learning.continual_learning",
    "python.meta_learning": "python.core.learning.meta_learning",
    "python.multi_task_learning": "python.core.learning.multi_task_learning",
    "python.rl_engine": "python.core.learning.rl_engine",
    "python.curiosity": "python.core.learning.curiosity",
}

print("=" * 70)
print("FINAL IMPORT FIX - Updating 'from python.X' patterns")
print("=" * 70)

all_files = list((base / "tests").rglob("*.py")) + list((base / "python").rglob("*.py"))
files_modified = 0
total_changes = 0

for file_path in all_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        changes_in_file = 0
        
        # Replace all old module paths with new ones
        for old_path, new_path in DIRECT_MODULE_MAP.items():
            # Pattern: from python.X import ...
            pattern = r'\bfrom\s+' + re.escape(old_path) + r'\s+import\s+'
            replacement = f'from {new_path} import '
            before = content
            content = re.sub(pattern, replacement, content)
            if content != before:
                num_replacements = before.count(f'from {old_path} import')
                changes_in_file += num_replacements
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            rel_path = file_path.relative_to(base)
            print(f"  {rel_path}: {changes_in_file} import(s) updated")
            files_modified += 1
            total_changes += changes_in_file
    except Exception as e:
        print(f"  Error with {file_path.name}: {e}")

print("=" * 70)
print(f"\nSummary:")
print(f"  Files modified: {files_modified}")
print(f"  Total imports updated: {total_changes}")
print("=" * 70)
