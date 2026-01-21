"""
NCGN Unified Runner

Runs all unit tests and the system demo, capturing all output into a single file.
"""

import unittest
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import tests
from tests.test_physics import TestEnergyPropagation, TestSpikePropagation, TestActiveNodeTracking
from tests.test_wta import TestKWTAAlgorithm, TestKWTAIntegration
from tests.test_surprise import TestSurpriseCalculation, TestSurpriseMonitor, TestSurpriseInPipeline
from tests.test_system2 import TestSchemaValidation, TestDiagnosis, TestInterventionPlanning, TestDogEatMetal, TestInterventionApplication

# Import demo
import main

def run_everything():
    output_file = "ncgn_complete_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("      NCGN COMPLETE IMPLEMENTATION TEST & DEMO RESULTS      \n")
        f.write("============================================================\n\n")
        
        # 1. Run Tests
        f.write("--- PHASE 1: UNIT TESTS ---\n")
        
        # Collect all test classes
        test_classes = [
            TestEnergyPropagation, TestSpikePropagation, TestActiveNodeTracking,
            TestKWTAAlgorithm, TestKWTAIntegration,
            TestSurpriseCalculation, TestSurpriseMonitor, TestSurpriseInPipeline,
            TestSchemaValidation, TestDiagnosis, TestInterventionPlanning, 
            TestDogEatMetal, TestInterventionApplication
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
            
        # Capture test output
        test_out = io.StringIO()
        runner = unittest.TextTestRunner(stream=test_out, verbosity=2)
        result = runner.run(suite)
        
        f.write(test_out.getvalue())
        f.write("\n\n")
        
        # 2. Run Demo
        f.write("--- PHASE 2: SYSTEM DEMO ---\n")
        
        demo_out = io.StringIO()
        with redirect_stdout(demo_out), redirect_stderr(demo_out):
            try:
                main.main()
            except Exception as e:
                print(f"\nDemo failed with error: {e}")
        
        f.write(demo_out.getvalue())
        f.write("\n\n")
        f.write("============================================================\n")
        f.write("                  END OF RESULTS                            \n")
        f.write("============================================================\n")

    print(f"All tests and demo completed. Results saved to {output_file}")

if __name__ == "__main__":
    run_everything()
