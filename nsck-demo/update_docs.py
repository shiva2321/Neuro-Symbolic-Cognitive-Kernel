"""
Update all documentation files with new restructured paths.
"""
import re
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo")

# Path mappings for documentation updates
PATH_REPLACEMENTS = {
    # File paths (with .py extension or as links)
    r'`?python/multimodal_test\.py`?': 'tests/experiments/multimodal_test.py',
    r'`?python/text_reasoning_test\.py`?': 'tests/experiments/text_reasoning_test.py',
    r'`?python/belief_revision_test\.py`?': 'tests/experiments/belief_revision_test.py',
    r'`?python/transitive_test\.py`?': 'tests/experiments/transitive_test.py',
    
    r'`?python/multimodal_processor\.py`?': 'python/core/multimodal/multimodal_processor.py',
    r'`?python/perception\.py`?': 'python/core/perception/perception.py',
    r'`?python/text_knowledge_learner\.py`?': 'python/core/language/text_knowledge_learner.py',
    r'`?python/semantic_memory\.py`?': 'python/core/memory/semantic_memory.py',
    r'`?python/analogy\.py`?': 'python/core/reasoning/analogy.py',
    r'`?python/language_module\.py`?': 'python/core/language/language_module.py',
    r'`?python/episodic_memory\.py`?': 'python/core/memory/episodic_memory.py',
    r'`?python/cognitive_engine\.py`?': 'python/core/reasoning/cognitive_engine.py',
    r'`?python/brain_fusion\.py`?': 'python/core/integration/brain_fusion.py',
    
    # Training demos
    r'python/train_phase\d+_demo\.py': 'python/training/demos/',
}

# Import statement patterns
IMPORT_REPLACEMENTS = {
    # Old module imports → new package-qualified imports
    r'from text_knowledge_learner import': 'from python.core.language.text_knowledge_learner import',
    r'from language_module import': 'from python.core.language.language_module import',
    r'from multimodal_processor import': 'from python.core.multimodal.multimodal_processor import',
    r'from semantic_memory import': 'from python.core.memory.semantic_memory import',
    r'from episodic_memory import': 'from python.core.memory.episodic_memory import',
    r'from cognitive_engine import': 'from python.core.reasoning.cognitive_engine import',
    r'from brain_fusion import': 'from python.core.integration.brain_fusion import',
    r'from perception import': 'from python.core.perception.perception import',
}

print("=" * 70)
print("DOCUMENTATION PATH UPDATER")
print("=" * 70)

# Find all .md files
md_files = list(base.rglob("*.md"))
md_files = [f for f in md_files if '.git' not in str(f) and 'node_modules' not in str(f)]

print(f"\nFound {len(md_files)} markdown files to process")

files_modified = 0
total_replacements = 0

for md_file in md_files:
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        file_replacements = 0
        
        # Apply path replacements
        for pattern, replacement in PATH_REPLACEMENTS.items():
            before = content
            content = re.sub(pattern, replacement, content)
            if content != before:
                matches = len(re.findall(pattern, before))
                file_replacements += matches
        
        # Apply import replacements
        for pattern, replacement in IMPORT_REPLACEMENTS.items():
            before = content
            content = re.sub(pattern, replacement, content)
            if content != before:
                matches = len(re.findall(pattern, before))
                file_replacements += matches
        
        if content != original:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            rel_path = md_file.relative_to(base)
            print(f"  [OK] {rel_path}: {file_replacements} replacement(s)")
            files_modified += 1
            total_replacements += file_replacements
        
    except Exception as e:
        print(f"  [ERROR] Error with {md_file.name}: {e}")

print("=" * 70)
print(f"\nSummary:")
print(f"  Files modified: {files_modified}")
print(f"  Total replacements: {total_replacements}")
print("=" * 70)

# Now update the directory structure in README
readme_path = base / "README.md"
if readme_path.exists():
    print(f"\nUpdating directory structure diagram in README.md...")
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the project structure section
    new_structure ='''## Project Structure

```
nsck-demo/
├── python/
│   ├── core/                         # Core cognitive modules
│   │   ├── vsa/                      # Vector Symbolic Architecture
│   │   ├── memory/                   # Episodic, semantic, working memory
│   │   ├── reasoning/                # Cognitive engine, causal reasoning
│   │   ├── learning/                 # RL, continual, meta-learning
│   │   ├── perception/               # Symbol grounding, saliency
│   │   ├── language/                 # NLP, text understanding
│   │   ├── multimodal/               # Cross-modal processing
│   │   ├── cognitive/                # Theory of mind, metacognition
│   │   ├── neural/                   # SNN, world models
│   │   └── integration/              # Brain fusion, persistence
│   ├── games/                        # Training environments
│   ├── benchmarks/                   # Performance measurement
│   ├── training/                     # Training scripts and demos
│   ├── interfaces/                   # Dashboards and UIs
│   ├── servers/                      # Backend services
│   └── utilities/                    # Helper modules
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── benchmarks/                   # Benchmark tests
│   └── experiments/                  # Experimental tests
├── data/
│   └── test_corpus/
│       └──xylophone_planets.txt     # F1 test corpus
└── nsck_sdk/                         # External module SDK
    ├── README.md                     # Module development guide
    └── MODULE_DEV_GUIDE.md           # 600+ line tutorial
```'''
    
    # Replace the old structure section
    pattern = r'## Project Structure\n\n```[\s\S]*?```'
    content = re.sub(pattern, new_structure, content)
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  [OK] Updated directory structure diagram")

print("\n[OK] Documentation update complete!")
