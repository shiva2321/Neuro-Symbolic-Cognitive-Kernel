#!/usr/bin/env python3
"""
NSCK Image Generation Dashboard Launcher
=========================================
Quick launcher for the image generation web interface.
"""

import sys
import os

# Add parent directory paths to access NSCK core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo'))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo/python'))

# Add project source directory
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

from image_generation_dashboard import main

if __name__ == '__main__':
    main()
