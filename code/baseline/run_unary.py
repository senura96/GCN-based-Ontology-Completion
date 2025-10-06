#!/usr/bin/env python3
"""
Wrapper script to run unary classification.

This script provides easy access to the unary classification functionality
after the reorganization into the unary/ folder.
"""

import sys
import os

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unary.unary_classify import main

if __name__ == '__main__':
    main()
