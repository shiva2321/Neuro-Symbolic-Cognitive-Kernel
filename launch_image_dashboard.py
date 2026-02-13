#!/usr/bin/env python3
"""
NSCK Image Generation Dashboard Launcher
=========================================
Quick launcher for the image generation web interface.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsck-demo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsck-demo/python'))

from python.interfaces.image_generation_dashboard import main

if __name__ == '__main__':
    main()
