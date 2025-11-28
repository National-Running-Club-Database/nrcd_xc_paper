"""
Helper module to set up paths when running scripts from main directory.

This ensures imports work correctly whether scripts are run from scripts/ or main directory.
"""

import os
import sys

def setup_paths():
    """Add scripts directory to path and return script directory."""
    # Get the directory where this file is located (scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add to path if not already there
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    return script_dir

# Auto-setup when imported
setup_paths()

