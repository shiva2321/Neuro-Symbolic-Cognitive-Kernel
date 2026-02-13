#!/usr/bin/env python3
"""
NSCK Image Generation Project - Verification Script
====================================================
Verifies all components of the image generation project.
"""

import sys
import os

# Add parent directory paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo'))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo/python'))

# Add project source
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

import subprocess
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_file_exists(filepath, description):
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def main():
    print_section("NSCK Image Generation Project - Verification")
    
    all_passed = True
    
    # Check directory structure
    print_section("1. Directory Structure")
    dirs = [
        ('src', 'Source directory'),
        ('tests', 'Tests directory'),
        ('examples', 'Examples directory'),
        ('docs', 'Documentation directory'),
        ('models', 'Models directory')
    ]
    
    for dir_name, desc in dirs:
        all_passed &= check_file_exists(dir_name, desc)
    
    # Check core files
    print_section("2. Core Files")
    files = [
        ('src/image_generator.py', 'Core image generator'),
        ('src/train_image_generation.py', 'Training pipeline'),
        ('src/image_generation_dashboard.py', 'Dashboard'),
        ('tests/test_image_generation.py', 'Test suite'),
        ('examples/demo_image_generation.py', 'Demo script'),
        ('launch_image_dashboard.py', 'Dashboard launcher'),
        ('README.md', 'Project README'),
        ('requirements.txt', 'Requirements file'),
        ('.gitignore', 'Git ignore file')
    ]
    
    for filepath, desc in files:
        all_passed &= check_file_exists(filepath, desc)
    
    # Check documentation
    print_section("3. Documentation")
    docs = [
        ('docs/IMAGE_GENERATION_GUIDE.md', 'User guide'),
        ('docs/IMAGE_GENERATION_ARCHITECTURE.md', 'Architecture docs'),
        ('docs/IMAGE_GENERATION_README.md', 'Quick reference'),
        ('docs/IMAGE_GENERATION_SUMMARY.md', 'Summary')
    ]
    
    for filepath, desc in docs:
        all_passed &= check_file_exists(filepath, desc)
    
    # Test imports
    print_section("4. Import Tests")
    try:
        from image_generator import ImageGenerator, GenerationConfig
        print("✅ Can import ImageGenerator")
    except Exception as e:
        print(f"❌ Cannot import ImageGenerator: {e}")
        all_passed = False
    
    try:
        from train_image_generation import ImageGenerationTrainer
        print("✅ Can import ImageGenerationTrainer")
    except ImportError as e:
        if 'torch' in str(e) or 'datasets' in str(e):
            print("⚠️  ImageGenerationTrainer needs torch/datasets (optional for training)")
        else:
            print(f"❌ Cannot import ImageGenerationTrainer: {e}")
            all_passed = False
    except Exception as e:
        print(f"❌ Cannot import ImageGenerationTrainer: {e}")
        all_passed = False
    
    # Run tests
    print_section("5. Running Tests")
    try:
        result = subprocess.run(
            ['python3', '-m', 'pytest', 'tests/test_image_generation.py', '-v'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Count passed tests
            output = result.stdout
            if '15 passed' in output:
                print("✅ All 15 tests passing")
            else:
                print(f"⚠️  Tests output: {output[-200:]}")
        else:
            print(f"❌ Tests failed")
            print(result.stderr[-500:])
            all_passed = False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        all_passed = False
    
    # Final summary
    print_section("6. Summary")
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nThe NSCK Image Generation Project is properly organized and functional!")
        print("\nQuick Start:")
        print("  - Run demo: python3 examples/demo_image_generation.py")
        print("  - Run tests: python3 -m pytest tests/ -v")
        print("  - Launch dashboard: python3 launch_image_dashboard.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
