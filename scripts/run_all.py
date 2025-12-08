"""
Run All Research Questions

This script runs all three research questions (RQ1, RQ2, RQ3) sequentially.

All outputs are saved to their respective output directories:
- output/rq1/
- output/rq2/
- output/rq3/

Run from main directory: python scripts/run_all.py
"""

import os
import sys
import time

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

def main():
    """Run all research questions sequentially."""
    print("="*80)
    print("RUNNING ALL RESEARCH QUESTIONS")
    print("="*80)
    print("\nThis will run:")
    print("  1. RQ1: Performance improvement patterns across race positions")
    print("  2. RQ2: Multi-season performance analysis with race count consistency filter")
    print("  3. RQ3: Gender fairness analysis")
    print("\nAll outputs will be saved to their respective output/rqX/ directories.")
    print("\n" + "="*80)
    
    start_time = time.time()
    
    # Run RQ1
    print("\n" + "="*80)
    print("STARTING RQ1")
    print("="*80)
    from rq1 import main as rq1_main
    rq1_main()
    
    # Run RQ2
    print("\n" + "="*80)
    print("STARTING RQ2")
    print("="*80)
    from rq2 import main as rq2_main
    rq2_main()
    
    # Run RQ3
    print("\n" + "="*80)
    print("STARTING RQ3")
    print("="*80)
    from rq3 import main as rq3_main
    rq3_main()
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("ALL RESEARCH QUESTIONS COMPLETE")
    print("="*80)
    print(f"\nTotal execution time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
    print("\nOutput directories:")
    print("  - output/rq1/")
    print("  - output/rq2/")
    print("  - output/rq3/")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()


