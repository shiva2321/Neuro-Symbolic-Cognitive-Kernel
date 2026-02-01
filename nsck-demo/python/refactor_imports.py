
import os
import glob

TARGET_DIR = "d:\\NSCK_v1\\nsck-demo\\python"

def refactor_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix __init__.py local import
    content = content.replace("from . import hypervec_rs as _hypervec_rs_compat", "from . import hypervec_shim as _hypervec_rs_compat")
    
    # 2. Fix standard import (careful with 'import hypervec_rs as')
    # If it's already "import hypervec_rs as", we might break it if we are not careful, but usually it's just "import hypervec_rs"
    # We want "import hypervec_rs" -> "import hypervec_shim as hypervec_rs"
    # usage: hypervec_rs.HyperVector works same.
    
    # But wait, what if they do `import hypervec_rs as hv`?
    # grep didn't show that.
    
    if "import hypervec_shim" not in content: # Don't re-patch
        if "import hypervec_rs" in content:
             # Basic replace
             content = content.replace("import hypervec_rs", "import hypervec_shim as hypervec_rs")
             
             # Fix the case where we created "from . import hypervec_shim as hypervec_rs as _hypervec_rs_compat"
             # The first replace handled __init__.py line.
             # If __init__.py had "from . import hypervec_rs ...", replace 1 handles it.
             # But if it triggers replace 2? "import hypervec_rs" is substring of "from . import hypervec_rs".
             
             # Let's render the replacements more carefully.
             pass

    # A better approach: Line by line.
    lines = original_content.splitlines()
    new_lines = []
    modified = False
    
    for line in lines:
        if "hypervec_shim" in line:
             new_lines.append(line)
             continue
             
        if line.strip().startswith("from . import hypervec_rs as _hypervec_rs_compat"):
            new_lines.append(line.replace("hypervec_rs", "hypervec_shim"))
            modified = True
        elif line.strip() == "import hypervec_rs":
            new_lines.append("import hypervec_shim as hypervec_rs")
            modified = True
        elif line.strip().startswith("import hypervec_rs as"):
            # e.g. import hypervec_rs as hv -> import hypervec_shim as hv
            new_lines.append(line.replace("hypervec_rs", "hypervec_shim"))
            modified = True
        elif "from hypervec_shim import" in line:
            new_lines.append(line.replace("hypervec_rs", "hypervec_shim"))
            modified = True
        else:
            new_lines.append(line)
            
    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n" if lines else "")
        print(f"Updated {filepath}")

for filepath in glob.glob(os.path.join(TARGET_DIR, "*.py")):
    if filepath.endswith("hypervec_shim.py"):
        continue  # Don't patch the shim itself to import itself!
    refactor_file(filepath)
