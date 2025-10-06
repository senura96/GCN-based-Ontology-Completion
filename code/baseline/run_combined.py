#!/usr/bin/env python3
"""
Wrapper script to run combined features experiment.

This script provides easy access to the combined features functionality
after the reorganization into the shared/ folder.
"""

import sys
import os

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.combined_features import main

if __name__ == '__main__':
    main()
