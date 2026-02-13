"""
Create directory structure for repository reorganization
"""
import os
from pathlib import Path

# Base directory
base = Path(r"d:\Node_network\nsck-demo")

# Directory structure to create
directories = [
    # Core subdirectories
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
    
    # Games subdirectories
    "python/games/snake",
    "python/games/maze",
    "python/games/pong",
    "python/games/physics",
    "python/games/collector",
    
    # Other python subdirectories
    "python/benchmarks/reports",
    "python/training/demos",
    "python/interfaces",
    "python/servers",
    "python/utilities",
    "python/scripts",
    
    # Test subdirectories
    "tests/unit",
    "tests/integration",
    "tests/benchmarks",
    "tests/experiments",
]

# Docs directories (one level up from nsck-demo)
docs_base = Path(r"d:\Node_network\docs")
docs_directories = [
    "getting-started",
    "architecture",
    "development",
    "research",
    "results",
]

print("Creating python directory structure...")
for dir_path in directories:
    full_path = base / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {dir_path}")

print("\nCreating docs directory structure...")
for dir_path in docs_directories:
    full_path = docs_base / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created: docs/{dir_path}")

print("\nAll directories created successfully!")

