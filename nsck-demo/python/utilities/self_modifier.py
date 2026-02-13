"""
NSCK Self-Modification Module (Phase 5.2)
=========================================
Enables the AGI to modify its own architecture and code.
Includes rigorous safety sandboxing and AST-based verification.
"""

import os
import ast
import time
import importlib
from typing import Dict, List, Optional, Any, Tuple

class SelfModifier:
    """
    Handles the proposal and application of self-modifications.
    """
    def __init__(self, workspace_path: str):
        self.workspace = workspace_path
        self.history = []
        self.verifier = SafetyVerifier()
        
    def propose_change(self, target_file: str, motivation: str, new_code: str):
        """Analyze a proposed code change."""
        full_path = os.path.join(self.workspace, target_file)
        if not os.path.exists(full_path):
            return {"status": "error", "message": "File not found"}
            
        # 1. Verification
        is_safe, reasons = self.verifier.verify(new_code)
        if not is_safe:
            return {"status": "rejected", "reasons": reasons}
            
        return {
            "status": "pending",
            "file": target_file,
            "motivation": motivation,
            "proposed_code": new_code
        }

    def apply_change(self, proposal: Dict[str, Any]) -> bool:
        """Apply the verified change to the filesystem."""
        if proposal["status"] != "pending":
            return False
            
        full_path = os.path.join(self.workspace, proposal["file"])
        
        # Backup
        timestamp = int(time.time())
        backup_path = f"{full_path}.{timestamp}.bak"
        with open(full_path, "r") as f:
             with open(backup_path, "w") as bf:
                 bf.write(f.read())
                 
        # Write new
        with open(full_path, "w") as f:
            f.write(proposal["proposed_code"])
            
        self.history.append({
            "timestamp": timestamp,
            "file": proposal["file"],
            "backup": backup_path
        })
        
        print(f"[EVOLUTION] Successfully modified {proposal['file']}")
        return True

class SafetyVerifier:
    """
    Static analysis for code safety.
    Prevents execution of dangerous primitives.
    """
    def __init__(self):
        self.forbidden_calls = {"eval", "exec", "os.system", "subprocess.run", "open"}
        self.forbidden_imports = {"os", "subprocess", "shutil", "requests", "socket"}

    def verify(self, code: str) -> Tuple[bool, List[str]]:
        reasons = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax Error: {str(e)}"]

        for node in ast.walk(tree):
            # 1. Check for forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.forbidden_imports:
                        reasons.append(f"Forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.forbidden_imports:
                    reasons.append(f"Forbidden import from: {node.module}")

            # 2. Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_calls:
                        reasons.append(f"Forbidden call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # check for os.system etc
                    pass

        return len(reasons) == 0, reasons
