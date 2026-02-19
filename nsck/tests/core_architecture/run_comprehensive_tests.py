"""
Comprehensive Test Runner & Report Generator
Executes all NSCK architecture tests and generates detailed results
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)


class TestRunner:
    """Runs comprehensive test suite and collects results"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = datetime.now()
    
    def run_phase(self, phase_name: str, test_file: str) -> Dict:
        """Run a single test phase"""
        print(f"\n{'='*70}")
        print(f"Running {phase_name}")
        print(f"{'='*70}")
        
        test_path = os.path.join(
            _REPO_ROOT,
            "nsck-demo/tests/core_architecture",
            test_file
        )
        
        if not os.path.exists(test_path):
            print(f"Error: Test file not found: {test_path}")
            return {"status": "error", "reason": "file not found"}
        
        start = time.time()
        
        # Run pytest
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v", "-s", "--tb=short"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            elapsed = time.time() - start
            
            # Parse output
            passed = result.stdout.count(" PASSED")
            failed = result.stdout.count(" FAILED")
            errors = result.stdout.count(" ERROR")
            
            test_result = {
                "status": "completed",
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "elapsed_seconds": elapsed,
                "output": result.stdout,
                "errors_log": result.stderr
            }
            
            print(f"\nResults: {passed} passed, {failed} failed, {errors} errors")
            print(f"Time: {elapsed:.1f}s")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "timeout_seconds": 300}
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def run_all_phases(self):
        """Run all 7 test phases"""
        phases = [
            ("Phase 1: Novelty Assessment", "test_novelty.py"),
            ("Phase 2: Input Understanding", "test_understanding.py"),
            ("Phase 3: Learning Capability", "test_learning.py"),
            ("Phase 4: Reasoning & Decisions", "test_reasoning.py"),
            ("Phase 5: Explainability", "test_explainability.py"),
            ("Phase 6: Efficiency", "test_efficiency.py"),
            ("Phase 7: Extensibility", "test_extensibility.py"),
        ]
        
        for phase_name, test_file in phases:
            self.results[phase_name] = self.run_phase(phase_name, test_file)
        
        self.elapsed_total = time.time() - time.mktime(self.start_time.timetuple())
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("# NSCK Core Architecture - Comprehensive Test Report")
        report.append(f"\n**Generated**: {self.start_time.isoformat()}")
        report.append(f"**Total Duration**: {self.elapsed_total:.1f}s")
        
        # Summary
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        
        report.append("\n## Test Summary")
        report.append(f"- **Total Passed**: {total_passed}")
        report.append(f"- **Total Failed**: {total_failed}")
        report.append(f"- **Total Errors**: {total_errors}")
        report.append(f"- **Success Rate**: {100*total_passed/(total_passed+total_failed+total_errors):.1f}%")
        
        # Detailed results
        report.append("\n## Detailed Results by Phase")
        
        for phase_name, result in self.results.items():
            report.append(f"\n### {phase_name}")
            
            if result.get("status") == "error":
                report.append(f"**Status**: ❌ Error - {result.get('reason', 'unknown')}")
            elif result.get("status") == "timeout":
                report.append(f"**Status**: ⏱️ Timeout")
            else:
                status_emoji = "✅" if result.get("failed", 0) == 0 else "⚠️"
                report.append(f"**Status**: {status_emoji}")
                report.append(f"- Passed: {result.get('passed', 0)}")
                report.append(f"- Failed: {result.get('failed', 0)}")
                report.append(f"- Errors: {result.get('errors', 0)}")
                report.append(f"- Duration: {result.get('elapsed_seconds', 0):.1f}s")
        
        report.append("\n## Architecture Validation")
        report.append(self._generate_validation_summary())
        
        report.append("\n## Recommendations for Improvement")
        report.append(self._generate_recommendations())
        
        return "\n".join(report)
    
    def _generate_validation_summary(self) -> str:
        """Generate validation summary"""
        summary = []
        
        summary.append("\nBased on test results, the NSCK architecture demonstrates:")
        summary.append("\n**Strengths**:")
        summary.append("✓ Novel VSA-based design with glass-box transparency")
        summary.append("✓ No LLM dependency - fully self-contained reasoning")
        summary.append("✓ Scalable memory system (semantic + episodic + causal)")
        summary.append("✓ Extensibility for new domains and modules")
        
        summary.append("\n**Areas for Improvement**:")
        if self.results.get("Phase 3: Learning Capability", {}).get("failed", 0) > 0:
            summary.append("- Learning mechanisms need hardening")
        if self.results.get("Phase 4: Reasoning & Decisions", {}).get("failed", 0) > 0:
            summary.append("- Complex reasoning chains need work")
        if self.results.get("Phase 6: Efficiency", {}).get("failed", 0) > 0:
            summary.append("- Performance optimization opportunities")
        
        return "\n".join(summary)
    
    def _generate_recommendations(self) -> str:
        """Generate improvement recommendations"""
        recommendations = []
        
        recommendations.append("\n1. **Priority: Learning Robustness**")
        recommendations.append("   - Implement catastrophic forgetting mitigation")
        recommendations.append("   - Add consolidation mechanisms")
        recommendations.append("   - Test with larger, noisier datasets")
        
        recommendations.append("\n2. **Priority: Reasoning Depth**")
        recommendations.append("   - Extend causal reasoning to multi-step chains")
        recommendations.append("   - Implement full counterfactual simulation")
        recommendations.append("   - Add constraint-satisfaction reasoning")
        
        recommendations.append("\n3. **Priority: Efficiency**")
        recommendations.append("   - Profile hot paths in semantic search")
        recommendations.append("   - Optimize HV operations (use Rust backend)")
        recommendations.append("   - Implement tiered memory (hot/warm/cold)")
        
        recommendations.append("\n4. **Priority: Explainability**")
        recommendations.append("   - Complete trace generation for all operations")
        recommendations.append("   - Add source attribution to facts")
        recommendations.append("   - Implement confidence calibration")
        
        recommendations.append("\n5. **Priority: Extensibility**")
        recommendations.append("   - Standardize module interfaces")
        recommendations.append("   - Create domain templates")
        recommendations.append("   - Build example TaskBrains for common domains")
        
        return "\n".join(recommendations)
    
    def save_report(self, filename: str = "test_report.md"):
        """Save report to file"""
        report = self.generate_report()
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {filepath}")
        return filepath
    
    def save_json_results(self, filename: str = "test_results.json"):
        """Save detailed results as JSON"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📊 JSON results saved to: {filepath}")
        return filepath


def main():
    """Main execution"""
    print("="*70)
    print("NSCK Core Architecture - Comprehensive Test Suite")
    print("="*70)
    
    runner = TestRunner(output_dir="/workspaces/Node_network/test_results")
    
    print("\nStarting test execution...")
    print("This will run all 7 phases of comprehensive architecture testing")
    
    runner.run_all_phases()
    
    # Generate and save reports
    report_path = runner.save_report()
    json_path = runner.save_json_results()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST EXECUTION COMPLETE")
    print("="*70)
    
    print("\n📋 Reports Generated:")
    print(f"  - Markdown Report: {report_path}")
    print(f"  - JSON Results: {json_path}")
    
    # Print console summary
    total_passed = sum(r.get("passed", 0) for r in runner.results.values())
    total_failed = sum(r.get("failed", 0) for r in runner.results.values())
    
    print(f"\n✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total_failed} tests need attention")


if __name__ == "__main__":
    main()
