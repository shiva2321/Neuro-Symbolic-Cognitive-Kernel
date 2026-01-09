"""
Automated Bug Detection and Fixing System
Scans codebase for common issues and attempts automatic fixes.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import importlib.util


class BugDetector:
    """Detects common bugs and code issues"""

    def __init__(self, root_dir="D:/development project/Node_network"):
        self.root_dir = Path(root_dir)
        self.issues = []
        self.fixes_applied = []

    def scan_all_files(self):
        """Scan all Python files in the project"""
        print("=" * 70)
        print("AUTOMATED BUG DETECTION SYSTEM")
        print("=" * 70)

        python_files = list(self.root_dir.rglob("*.py"))
        print(f"\nScanning {len(python_files)} Python files...")

        for file_path in python_files:
            if '__pycache__' in str(file_path):
                continue
            self.scan_file(file_path)

        return self.issues

    def scan_file(self, file_path: Path):
        """Scan a single file for issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.issues.append({
                    'file': file_path,
                    'type': 'SyntaxError',
                    'line': e.lineno,
                    'message': str(e),
                    'severity': 'CRITICAL'
                })
                return  # Can't do further checks if syntax is broken

            # Check for common issues
            self.check_imports(file_path, content)
            self.check_undefined_variables(file_path, content)
            self.check_return_statements(file_path, content)
            self.check_exception_handling(file_path, content)
            self.check_tensor_operations(file_path, content)

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

    def check_imports(self, file_path: Path, content: str):
        """Check for import issues"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for unused imports (simple heuristic)
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                # Extract imported names
                match = re.search(r'import\s+(\w+)', line)
                if match:
                    imported_name = match.group(1)
                    # Count occurrences (excluding import line itself)
                    count = sum(1 for j, l in enumerate(lines) if j != i-1 and imported_name in l)
                    if count == 0:
                        self.issues.append({
                            'file': file_path,
                            'type': 'UnusedImport',
                            'line': i,
                            'message': f"Unused import: {imported_name}",
                            'severity': 'WARNING',
                            'fix': 'remove_line'
                        })

    def check_undefined_variables(self, file_path: Path, content: str):
        """Check for potentially undefined variables"""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    # Check for common typos
                    name = node.id
                    if name.lower() in ['none', 'ture', 'flase']:
                        self.issues.append({
                            'file': file_path,
                            'type': 'TypoInConstant',
                            'line': node.lineno,
                            'message': f"Potential typo: {name}",
                            'severity': 'ERROR'
                        })
        except:
            pass

    def check_return_statements(self, file_path: Path, content: str):
        """Check for missing return statements"""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has return type annotation but no return
                    if node.returns is not None:
                        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                        if not has_return and node.name != '__init__':
                            self.issues.append({
                                'file': file_path,
                                'type': 'MissingReturn',
                                'line': node.lineno,
                                'message': f"Function '{node.name}' has return type but no return statement",
                                'severity': 'WARNING'
                            })
        except:
            pass

    def check_exception_handling(self, file_path: Path, content: str):
        """Check for bare except clauses"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if re.match(r'\s*except\s*:', line):
                self.issues.append({
                    'file': file_path,
                    'type': 'BareExcept',
                    'line': i,
                    'message': "Bare 'except:' clause catches all exceptions",
                    'severity': 'WARNING',
                    'suggestion': 'Use "except Exception:" instead'
                })

    def check_tensor_operations(self, file_path: Path, content: str):
        """Check for common PyTorch/DGL issues"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for potential device mismatch
            if 'torch.tensor' in line and '.to(' not in line and 'device=' not in line:
                self.issues.append({
                    'file': file_path,
                    'type': 'DeviceMismatch',
                    'line': i,
                    'message': "Tensor created without explicit device",
                    'severity': 'INFO',
                    'suggestion': 'Consider specifying device explicitly'
                })

            # Check for in-place operations that might cause issues
            if re.search(r'\.data\s*=', line):
                self.issues.append({
                    'file': file_path,
                    'type': 'InPlaceOperation',
                    'line': i,
                    'message': "Direct .data assignment can break autograd",
                    'severity': 'WARNING',
                    'suggestion': 'Use .detach() and clone() instead'
                })

    def print_report(self):
        """Print bug detection report"""
        if not self.issues:
            print("\n✅ No issues found!")
            return

        print(f"\n{'='*70}")
        print(f"FOUND {len(self.issues)} ISSUES")
        print(f"{'='*70}")

        # Group by severity
        critical = [i for i in self.issues if i['severity'] == 'CRITICAL']
        errors = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        info = [i for i in self.issues if i['severity'] == 'INFO']

        if critical:
            print(f"\n🔴 CRITICAL ({len(critical)}):")
            for issue in critical[:5]:  # Show first 5
                print(f"  {issue['file'].name}:{issue['line']} - {issue['message']}")

        if errors:
            print(f"\n🟠 ERRORS ({len(errors)}):")
            for issue in errors[:5]:
                print(f"  {issue['file'].name}:{issue['line']} - {issue['message']}")

        if warnings:
            print(f"\n🟡 WARNINGS ({len(warnings)}):")
            for issue in warnings[:5]:
                print(f"  {issue['file'].name}:{issue['line']} - {issue['message']}")

        if info:
            print(f"\n🔵 INFO ({len(info)}):")
            for issue in info[:5]:
                print(f"  {issue['file'].name}:{issue['line']} - {issue['message']}")

        print(f"\n{'='*70}")
        print(f"Summary: {len(critical)} critical, {len(errors)} errors, {len(warnings)} warnings, {len(info)} info")
        print(f"{'='*70}")

    def save_report(self, output_path="./test_results/bug_report.json"):
        """Save bug report to file"""
        import json
        from datetime import datetime

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.issues),
            'by_severity': {
                'CRITICAL': len([i for i in self.issues if i['severity'] == 'CRITICAL']),
                'ERROR': len([i for i in self.issues if i['severity'] == 'ERROR']),
                'WARNING': len([i for i in self.issues if i['severity'] == 'WARNING']),
                'INFO': len([i for i in self.issues if i['severity'] == 'INFO']),
            },
            'issues': [
                {
                    'file': str(i['file']),
                    'type': i['type'],
                    'line': i['line'],
                    'message': i['message'],
                    'severity': i['severity']
                }
                for i in self.issues
            ]
        }

        with open(output, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Bug report saved to: {output}")


class AutoFixer:
    """Attempts to automatically fix common issues"""

    def __init__(self, detector: BugDetector):
        self.detector = detector
        self.fixes_applied = []

    def auto_fix_issues(self, apply=False):
        """Attempt to automatically fix issues"""
        print(f"\n{'='*70}")
        print("AUTOMATIC FIXES")
        print(f"{'='*70}")

        fixable_issues = [i for i in self.detector.issues if 'fix' in i]

        if not fixable_issues:
            print("\nNo automatically fixable issues found.")
            return

        print(f"\nFound {len(fixable_issues)} fixable issues")

        if not apply:
            print("\nRun with --apply to apply fixes automatically")
            return

        for issue in fixable_issues:
            try:
                if issue['fix'] == 'remove_line':
                    self.fix_remove_line(issue)
            except Exception as e:
                print(f"Error fixing {issue['file']}:{issue['line']}: {e}")

        print(f"\n✅ Applied {len(self.fixes_applied)} fixes")

    def fix_remove_line(self, issue):
        """Remove a line (e.g., unused import)"""
        file_path = issue['file']

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Remove the line (1-indexed to 0-indexed)
        del lines[issue['line'] - 1]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        self.fixes_applied.append(issue)
        print(f"Fixed: {file_path.name}:{issue['line']} - Removed {issue['message']}")


def run_bug_detection(apply_fixes=False):
    """Main function to run bug detection"""
    detector = BugDetector()
    issues = detector.scan_all_files()
    detector.print_report()
    detector.save_report()

    if issues:
        fixer = AutoFixer(detector)
        fixer.auto_fix_issues(apply=apply_fixes)

    return len([i for i in issues if i['severity'] in ['CRITICAL', 'ERROR']]) == 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NCGN Bug Detection System')
    parser.add_argument('--apply', action='store_true', help='Apply automatic fixes')

    args = parser.parse_args()

    success = run_bug_detection(apply_fixes=args.apply)

    if success:
        print("\n✅ No critical issues found!")
    else:
        print("\n⚠️  Critical issues detected. Please review.")

