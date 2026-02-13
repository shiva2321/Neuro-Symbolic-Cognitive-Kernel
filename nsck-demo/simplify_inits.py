"""Simple script to empty all __init__.py files in core/ to avoid circular imports"""
from pathlib import Path

base = Path(r"d:\Node_network\nsck-demo\python\core")

# Find all __init__.py files in core subdirectories
init_files = list(base.rglob("__init__.py"))

print(f"Simplifying {len(init_files)} __init__.py files...")

for init_file in init_files:
    # Read current content
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract docstring if it exists
        lines = content.split('\n')
        docstring = ""
        if lines and '"""' in lines[0]:
            for i, line in enumerate(lines):
                if i > 0 and '"""' in line:
                    docstring = '\n'.join(lines[:i+1])
                    break
        
        # Write simplified version
        if docstring:
            new_content = f"{docstring}\n# Empty init to avoid circular imports - import modules directly\n"
        else:
            new_content = "# Empty init to avoid circular imports - import modules directly\n"
        
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        rel_path = init_file.relative_to(base.parent)
        print(f"  Simplified: {rel_path}")
    except Exception as e:
        print(f"  Error with {init_file.name}: {e}")

print(f"\nDone!")
