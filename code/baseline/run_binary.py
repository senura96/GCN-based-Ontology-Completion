#!/usr/bin/env python3
"""
Wrapper script to run binary prediction.

This script provides easy access to the binary prediction functionality
after the reorganization into the binary/ folder.
"""

import sys
import os

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary.binary_predict import main

if __name__ == '__main__':
    main()
